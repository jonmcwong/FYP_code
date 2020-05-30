# # code based off of 
# # https://github.com/mandubian/pytorch_math_dataset and
# # https://github.com/lucidrains/reformer-pytorch

import math
import numpy as np
import torch
from torch import nn
from torch.utils import data
import torch.optim as optim
import tqdm as tqdm
import random
from datetime import datetime
from apex import amp
import pickle
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import gc

import mandubian.math_dataset
from mandubian.math_dataset import MathDatasetManager
from mandubian.transformer import Constants
from mandubian.math_dataset import (random_split_dataset,
    question_answer_to_position_batch_collate_fn
)
import mandubian.checkpoints
from mandubian.checkpoints import rotating_save_checkpoint, build_checkpoint, restore_checkpoint
from mandubian.math_dataset import np_encode_string, np_decode_string
import mandubian.model_process
from mandubian.utils import build_transformer
from mandubian.tensorboard_utils import Tensorboard
from mandubian.tensorboard_utils import tensorboard_event_accumulator
from mandubian.model_process import predict_single
from mandubian.transformer.Generator import Generator
from mandubian.loss import compute_performance

from mandubian.math_dataset import (
    VOCAB_SZ, MAX_QUESTION_SZ, MAX_ANSWER_SZ
)

print("Torch Version", torch.__version__)

from lucidrains_reformer.reformer_pytorch import ReformerLM, Autopadder, Recorder
from lucidrains_reformer.reformer_pytorch import ReformerEncDec
from lucidrains_reformer.reformer_pytorch.generative_tools import TrainingWrapper

# # Random seed

seed = 1
torch.manual_seed(seed)

# # Check hardware

print(torch.cuda.device_count(), "detected CUDA devices")
cuda_device = torch.cuda.current_device()
print("Using CUDA device: ", cuda_device)
print(torch.cuda.get_device_name(cuda_device))
device = torch.device("cuda")

# # Initialize Math Dataset Manager

mdsmgr = MathDatasetManager(
  "/home/jonathan/Repos/final_year_at_ic/awesome_project/mathematics_dataset-v1.0/"
)

# Examine dataset structure

# print("mdsmgr structure", dir(mdsmgr))

# ### Check availables types, problem categories and problem subcategories

# print("types", list(mdsmgr.get_types()))
# print("categories", list(mdsmgr.get_categories()))
# print("modules of arithmetic", mdsmgr.get_modules_for_category('arithmetic'))

# # Ways to manipulate dataset

# # Build Dataset from a single module in a category
# ds = mdsmgr.build_dataset_from_module('arithmetic', 'add_or_sub', 'train-easy')
# print("size", len(ds))

# # Build Dataset from a single module in a category with limited number of elements
# ds = mdsmgr.build_dataset_from_module('arithmetic', 'add_or_sub', 'train-easy', max_elements=1000)
# print("size", len(ds))

# # Build Dataset from several modules in a category
# ds = mdsmgr.build_dataset_from_modules('arithmetic', ['add_or_sub', 'add_sub_multiple'], 'train-easy')
# print("size", len(ds))

# # Build Dataset from all modules in a category
# ds = mdsmgr.build_dataset_from_category('arithmetic', 'train-easy')
# ds = mdsmgr.build_dataset_from_category('arithmetic', 'interpolate')
# print("size", len(ds))

# # Build Dataset from all modules in several categories
# ds = mdsmgr.build_dataset_from_categories(['arithmetic', 'polynomials'], 'train-easy')
# print("size", len(ds))

# # Experiment ID --------------------------------------------------------------

exp_name = "activations_collection_batch_size_3072"
# exp_name = "test"
now = datetime.now()
unique_id = now.strftime("%m-%d-%Y_%H-%M-%S")
base_dir = "/home/jonathan/Repos/final_year_at_ic/awesome_project/code/tests/"

# # Training constants ---------------------------------------------------------

NUM_CPU_THREADS = 12             # dataloader
BATCH_SIZE = 256                 # size of every accumulatino
GRADIENT_ACCUMULATE_EVERY = 12  # number of accumulation
LEARNING_RATE = 6e-6            # 
VALIDATE_EVERY  = 200            # number of batches between validations
GENERATE_EVERY  = 500            # number of batechs between sequences generated when training
GENERATE_LENGTH = 32            # how many characters to generate
COLLECT_ACTIVATIONS_EVERY = 500

# # model hyperparameters ------------------------------------------------------

Q_SEQ_LEN = 256
A_SEQ_LEN = 30 # unused due to requirements of axial_positon_shape
NUM_TOKENS = VOCAB_SZ + 1
D_MODEL = 512
EMB_DIM = D_MODEL
NUM_HEADS = 8
QKV_DIM = D_MODEL / NUM_HEADS
NUM_LAYERS = 6
D_FF = 2048

# helpers ----------------------------------------------------------------------

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

def get_non_pad_mask(seq):
    # returns True when token is not PAD and false otherwise
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

# # Get training and test data -------------------------------------------------

# training_data = mdsmgr.build_dataset_from_category('arithmetic','train-easy') # for now
training_data = mdsmgr.build_dataset_from_modules('arithmetic', ['add_or_sub', 'add_sub_multiple','div','mul', 'mixed', 'mul_div_multiple'], 'train-easy')

# testing_data_interpolate = mdsmgr.build_dataset_from_category('arithmetic','interpolate')
# testing_data_extrapolate = mdsmgr.build_dataset_from_category('arithmetic','extrapolate')

# interpolate_data = mdsmgr.build_dataset_from_module('arithmetic', 'add_sub_multiple', 'interpolate')
# extrapolate_data = mdsmgr.build_dataset_from_module('arithmetic', 'add_sub_multiple', 'extrapolate')

train_ds, val_ds = mandubian.math_dataset.random_split_dataset(training_data,split_rate=0.9)

# # get pytorch dataloaders ----------------------------------------------------

# # Questions are padded in question_answer_to_position_batch_collate_fn

train_loader = data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_CPU_THREADS,
    collate_fn=question_answer_to_position_batch_collate_fn, pin_memory = True)
train_loader = cycle(train_loader)

val_loader = data.DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_CPU_THREADS,
    collate_fn=question_answer_to_position_batch_collate_fn, pin_memory = True)
val_loader = cycle(val_loader)

# #  for viewing output sequences

gen_loader = data.DataLoader(
    val_ds, batch_size=1, shuffle=False, num_workers=NUM_CPU_THREADS,
    collate_fn=question_answer_to_position_batch_collate_fn, pin_memory = True)
gen_loader = cycle(gen_loader)

# interpolate_loader = data.DataLoader(
#     interpolate_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_CPU_THREADS,
#     collate_fn=question_answer_to_mask_batch_collate_fn, pin_memory = True)
# interpolate_loader = cycle(interpolate_loader)

# extrapolate_loader = data.DataLoader(
#     extrapolate_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_CPU_THREADS,
#     collate_fn=question_answer_to_mask_batch_collate_fn)
# extrapolate_loader = cycle(extrapolate_loader)


# # Model ----------------------------------------------------------------------

model = build_transformer()
# print(model)
# model = Recorder(model)
model.to(device)


# # Optimizer learning rate scheduler, mixed precision setup

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.9995), eps=1e-9)

# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.85, patience=50, verbose=True, threshold=1e-4, min_lr=1e-7)
# scheduler = optim.lr_scheduler.StepLR(optimizer, 400)
# # mixed precision
model, optimizer = amp.initialize(model, optimizer, opt_level='O2')


# # Train
i = 0
train_loss_list = []
val_loss_list = []
best_val_loss = 10.0
restore = False
filename = "./checkpoints/test_05-22-2020_05-22-01_log_0.pth"
if (restore):
    state = restore_checkpoint(filename=filename, model=model, optimizer=optimizer)
    i = state["batch"]
    train_loss_list = state["train_loss"]
    val_loss_list = state["val_loss"]
    best_val_loss = val_loss_list[-1][1]
    amp.load_state_dict(state["amp"])



while True:
    
    # exclude the 0th element as it is BOS

    if (i % GENERATE_EVERY) - 1 == 0:
        model.eval()
        gen_qs, gen_qs_pos, gen_as, gen_as_pos = next(gen_loader)
        prime = np_decode_string(gen_qs.numpy())
        print('*' * 100, "\nQuestion: ", prime)
        print("Actual Answer: ", np_decode_string(gen_as.numpy()))
        # gen_qs = gen_qs.to(device, non_blocking=True)
        # gen_as = gen_as.to(device, non_blocking=True)
        # gen_qs_pos = gen_qs_pos.to(device, non_blocking=True)
        # print(prime)
        with torch.no_grad():
            # print("test question", prime)
            response = predict_single(prime[1:-1], model, device)
            # sample = model.generate(gen_qs, gen_as[:,0:1], GENERATE_LENGTH, enc_input_mask = gen_qs_pos, dec_eos_token=Constants.EOS)
        # sample = sample.cpu().numpy()
        # output_str = np_decode_string(sample)
        print("Decoded Prediction: ", response[0]["resp"])
        # print(response)
        # np.savetxt(base_dir + "logs/" + exp_name + "_" + unique_id + "-train_loss.txt", train_loss_list, fmt="%f")
        # np.savetxt(base_dir + "logs/" + exp_name + "_" + unique_id + "-val_loss.txt", val_loss_list, fmt="%f")
            

    model.train()
    train_loss_record = 0
    batch_activations = []
    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        batch_qs, batch_qs_pos, batch_as, batch_as_pos = map(lambda x: x.to(device, non_blocking=True), next(train_loader))
        gold_as = batch_as[:, 1:]
        # print(batch_qs_pos)

        # print(batch_as)
        pred_as, enc_output = model(batch_qs, batch_qs_pos, batch_as, batch_as_pos)
        # print("prediction: ", pred_as.size())
        train_loss, n_correct, n_correct_answers = compute_performance(pred_as, gold_as, smoothing=False)    
        # train_loss = model(batch_qs, batch_as, return_loss = True, enc_input_mask = batch_qs_mask)
        del batch_qs, batch_qs_pos, batch_as, batch_as_pos, gold_as, n_correct_answers, n_correct
        with amp.scale_loss(train_loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        # train_loss.backward()
        train_loss_record += float(train_loss)
        # print(train_loss_record)
        del train_loss
        if (i % COLLECT_ACTIVATIONS_EVERY) - 1 == 0:
            enc_output = enc_output.detach().to('cpu', non_blocking=True)
            batch_activations.append(enc_output)
        del enc_output

    if (i % COLLECT_ACTIVATIONS_EVERY) - 1 == 0:
        with open(f"./checkpoints/{exp_name}_{unique_id}_encoder_outputs_step_{i}", 'wb+') as log_file:
            pickle.dump({i: batch_activations}, log_file)
        gc.collect()   
    del batch_activations

#     if i % GRADIENT_ACCUMULATE_EVERY == 0:
    train_loss_record /= GRADIENT_ACCUMULATE_EVERY
    print("Step ", i, "\t", f'training loss: {train_loss_record}', "\t", datetime.now().time() )
    train_loss_list.append((i, train_loss_record))
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()
    optimizer.zero_grad()
    # scheduler.step(train_loss_record)

    if i % VALIDATE_EVERY == 0:
        model.eval()
        batch_qs, batch_qs_pos, batch_as, batch_as_pos = map(lambda x: x.to(device), next(val_loader))
        gold_as = batch_as[:, 1:]
        # val_batch_qs, val_batch_qs_mask, val_batch_as, val_batch_as_mask = map(lambda x: x.to(device, non_blocking=True), next(val_loader))
        with torch.no_grad():

            # forward
            pred_as, enc_output = model(batch_qs, batch_qs_pos, batch_as, batch_as_pos)
            # pred_as = nn.LogSoftmax(pred_as, dim=1)

            val_loss, n_correct, n_correct_answers = compute_performance(pred_as, gold_as, smoothing=False)
            del enc_output, n_correct_answers, pred_as
            # val_loss = model(val_batch_qs, val_batch_as, return_loss = True, enc_input_mask = val_batch_qs_mask)
            print(f'validation loss: {val_loss.item()}')
            val_loss_list.append((i, val_loss.item()))

            #accuracy
            non_pad_mask = batch_as.ne(Constants.PAD)
            n_char = non_pad_mask.sum().item()
            print("Proportion of correct characters: ", float(n_correct)/float(n_char))

            # # log model
            print("Checkpointing model to ", f"{exp_name}_{unique_id}_log")
            state = build_checkpoint(exp_name, unique_id, "log", model, optimizer, val_loss_list, train_loss_list, i, amp)
            rotating_save_checkpoint(state, prefix=f"{exp_name}_{unique_id}_log", path="./checkpoints", nb=1)

            # if we have a good val model, save it 
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                print("Best validation! Checkpointing model to ", f"{exp_name}_{unique_id}_best_val")
                state = build_checkpoint(exp_name, unique_id, "best_val", model, optimizer, val_loss_list, train_loss_list, i, amp)
                rotating_save_checkpoint(state, prefix=f"{exp_name}_{unique_id}_best_val", path="./checkpoints", nb=1)    
        gc.collect()
    i+=1
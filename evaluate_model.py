import math
import numpy as np
import torch
from torch.utils import data
import torch.optim as optim
from tqdm import tqdm
import random
from datetime import datetime
from apex import amp
import pickle
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

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
from mandubian.utils import build_transformer, build_just_decoder
from mandubian.tensorboard_utils import Tensorboard
from mandubian.tensorboard_utils import tensorboard_event_accumulator
from mandubian.math_dataset import (
	VOCAB_SZ, MAX_QUESTION_SZ, MAX_ANSWER_SZ
)
from mandubian.loss import compute_performance

print("Torch Version", torch.__version__)

from lucidrains_reformer.reformer_pytorch import ReformerLM, Autopadder, Recorder
from lucidrains_reformer.reformer_pytorch import ReformerEncDec
from lucidrains_reformer.reformer_pytorch.generative_tools import TrainingWrapper

def get_perf(model, loader, desc=None):
	model.eval()
	count = 0
	total_loss = 0
	total_acc_char = 0
	total_acc_answer = 0
	for batch in tqdm(loader, desc=desc):
		batch_qs, batch_qs_pos, batch_as, batch_as_pos = map(lambda x: x.to(device, non_blocking=True), batch)
		gold_as = batch_as[:, 1:]
		with torch.no_grad():

			# forward
			pred_as = model(batch_qs, batch_qs_pos, batch_as, batch_as_pos)
			# del enc_output

			loss, n_correct, n_correct_answers = compute_performance(pred_as, gold_as, smoothing=False)
			total_loss += loss.item()

			#char accuracy
			non_pad_mask = batch_as.ne(Constants.PAD)
			n_char = non_pad_mask.sum().item()
			total_acc_char += float(n_correct)/float(n_char)

			#answer accuracy
			total_acc_answer += float(n_correct_answers)/float(len(batch_qs))
		count += 1
	total_loss /= count
	total_acc_char /= count
	total_acc_answer /= count
	return total_loss, total_acc_char, total_acc_answer

# # import model

filename = "./checkpoints/final_transformer_06-04-2020_18-55-27/final_transformer_06-04-2020_18-55-27_log_0.pth"
# model = build_just_decoder()
model = build_transformer()
state = restore_checkpoint(filename=filename, model=model)
# print("this is model: ", model)
i = state["batch"]
	# train_loss_list = state["train_loss"]
	# val_loss_list = state["val_loss"]
	# best_val_loss = val_loss_list[-1][1]
# # actual testing
modules = ['add_or_sub', 'add_sub_multiple', 'div', 'mixed', 'mul', 'mul_div_multiple']
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

# # Get data for train and val easy
training_data_easy, breakdown = mdsmgr.build_dataset_from_category('arithmetic','train-easy', label=True) # for now
train_ds, val_ds = mandubian.math_dataset.random_split_dataset(training_data_easy,split_rate=0.9)

# # reset Random seed to recreate exact splitting
seed = 1
torch.manual_seed(seed)
train_labels, val_labels = mandubian.math_dataset.random_split_dataset(breakdown,split_rate=0.9)

# # view that both have been split correctly
# print("train_ds")
# for i in range(5):
#     print(train_ds[i])
# print("train_labels")
# for i in range(5):
#     print(train_labels[i])


print("created train indicies")
# separated_data = {}	# contains data data separated by module, difficulty and train and val
indicies_data = {}	# contains the indicies of the train and val set separated by module, difficulty.
for i in range(len(train_ds)):
	if "train-"+train_labels[i] not in indicies_data:
		# separated_data["train-"+train_labels[i]] = []
		indicies_data["train-"+train_labels[i]] = []
	indicies_data["train-"+train_labels[i]].append(i)	# which index of train_ds belongs to which module

	# separated_data["train-"+train_labels[i]].append(train_ds[i])
print("created val indicies")
for i in range(len(val_ds)):
	if "val-"+val_labels[i] not in indicies_data:
		# separated_data["val-"+val_labels[i]] = []
		indicies_data["val-"+val_labels[i]] = []
	# separated_data["val-"+val_labels[i]].append(val_ds[i])
	indicies_data["val-"+val_labels[i]].append(i)

# # get back the train and val for each module


for module in modules:

	# # Get training and test data -------------------------------------------------

	# # data for spceific module
	print("creating subsets")

	train_easy_ds = data.Subset(train_ds, indicies_data["train-"+module+"-train-easy"][0:10000])
	val_easy_ds  = data.Subset(val_ds, indicies_data["val-"+ module+"-train-easy"][0:10000])


	# training_data = mdsmgr.build_dataset_from_module('arithmetic', 'add_sub_multiple', 'train-easy')
	training_data_medium = mdsmgr.build_dataset_from_module('arithmetic',module,'train-medium', max_elements= 10000) # for now
	training_data_hard = mdsmgr.build_dataset_from_module('arithmetic',module,'train-hard', max_elements= 10000) # for now

	testing_data_interpolate = mdsmgr.build_dataset_from_module('arithmetic',module,'interpolate', max_elements= 10000)

	if module in ["mixed", "mul_div_multiple", "add_sub_multiple"]:

		testing_data_extrapolate = mdsmgr.build_dataset_from_module('arithmetic',f'{module}_longer','extrapolate', max_elements= 10000)
	else:
		testing_data_extrapolate = mdsmgr.build_dataset_from_module('arithmetic',f'{module}_big','extrapolate', max_elements= 10000)

	# interpolate_data = mdsmgr.build_dataset_from_module('arithmetic', 'add_sub_multiple', 'interpolate')
	# extrapolate_data = mdsmgr.build_dataset_from_module('arithmetic', 'add_sub_multiple', 'extrapolate')


	# # get pytorch dataloaders ----------------------------------------------------

	# # Questions are padded in question_answer_to_position_batch_collate_fn

	NUM_CPU_THREADS = 1	            # dataloader
	BATCH_SIZE = 512                 # size of every accumulatino
	GRADIENT_ACCUMULATE_EVERY = 4   # number of accumulation
	LEARNING_RATE = 6e-6            # 
	VALIDATE_EVERY  = 20            # number of batches between validations
	GENERATE_EVERY  = 200            # number of batechs between sequences generated when training
	GENERATE_LENGTH = 32            # how many characters to generate
	train_easy_loader = data.DataLoader(
		train_easy_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_CPU_THREADS,
		collate_fn=question_answer_to_position_batch_collate_fn, pin_memory = True)

	val_easy_loader = data.DataLoader(
		val_easy_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_CPU_THREADS,
		collate_fn=question_answer_to_position_batch_collate_fn, pin_memory = True)

	medium_loader = data.DataLoader(
		training_data_medium, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_CPU_THREADS,
		collate_fn=question_answer_to_position_batch_collate_fn, pin_memory = True)

	hard_loader = data.DataLoader(
		training_data_hard, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_CPU_THREADS,
		collate_fn=question_answer_to_position_batch_collate_fn, pin_memory = True)

	interpolate_loader = data.DataLoader(
		testing_data_interpolate, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_CPU_THREADS,
		collate_fn=question_answer_to_position_batch_collate_fn, pin_memory = True)

	extrapolate_loader = data.DataLoader(
		testing_data_extrapolate, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_CPU_THREADS,
		collate_fn=question_answer_to_position_batch_collate_fn, pin_memory = True)

	# # interpolation accuracy
	model = model.to(device)
	save_to_file = {}
	# # train easy
	train_loss, char_acc, answer_acc = get_perf(model, train_easy_loader, desc="train_easy")
	scores = {}
	scores['char acc'] = char_acc
	scores['loss'] 	   = train_loss
	scores['answer acc'] = answer_acc
	save_to_file['train_easy'] = scores
	print("Proportion of correct characters: ", char_acc)
	print('loss: ', train_loss)
	print("Proportion of correct answers: ", answer_acc)

	# # val easy
	train_loss, char_acc, answer_acc = get_perf(model,val_easy_loader, desc="val_easy")
	scores = {}
	scores['char acc'] = char_acc
	scores['loss'] 	   = train_loss
	scores['answer acc'] = answer_acc
	save_to_file['val_easy'] = scores
	print("Proportion of correct characters: ", char_acc)
	print('loss: ', train_loss)
	print("Proportion of correct answers: ", answer_acc)

	# # medium
	train_loss, char_acc, answer_acc = get_perf(model,medium_loader, desc="medium")
	scores = {}
	scores['char acc'] = char_acc
	scores['loss'] 	   = train_loss
	scores['answer acc'] = answer_acc
	save_to_file['medium'] = scores	
	print("Proportion of correct characters: ", char_acc)
	print('loss: ', train_loss)
	print("Proportion of correct answers: ", answer_acc)

	# # hard
	train_loss, char_acc, answer_acc = get_perf(model,hard_loader, desc="hard")
	scores = {}
	scores['char acc'] = char_acc
	scores['loss'] 	   = train_loss
	scores['answer acc'] = answer_acc
	save_to_file['hard'] = scores
	print("Proportion of correct characters: ", char_acc)
	print('loss: ', train_loss)
	print("Proportion of correct answers: ", answer_acc)

	# # interpolate
	train_loss, char_acc, answer_acc = get_perf(model,interpolate_loader, desc="interpolate")
	scores = {}
	scores['char acc'] = char_acc
	scores['loss'] 	   = train_loss
	scores['answer acc'] = answer_acc
	save_to_file['interpolate'] = scores
	print("Proportion of correct characters: ", char_acc)
	print('loss: ', train_loss)
	print("Proportion of correct answers: ", answer_acc)

	# # extrapolate
	train_loss, char_acc, answer_acc = get_perf(model,extrapolate_loader, desc="extrapolate")
	scores = {}
	scores['char acc'] = char_acc
	scores['loss'] 	   = train_loss
	scores['answer acc'] = answer_acc
	save_to_file['extrapolate'] = scores
	print("Proportion of correct characters: ", char_acc)
	print('loss: ', train_loss)
	print("Proportion of correct answers: ", answer_acc)

	with open(f"./checkpoints/final_transformer_06-04-2020_18-55-27/best_val_eval/{module}.txt", 'wb+') as log_file:
		pickle.dump(save_to_file, log_file, 0)


# # extrapolation accuracy

# # interactive test with individual sequences
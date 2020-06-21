# # generic transformer model


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
import numpy as np
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import time

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

from ut_pytorch.utils import build_UTransformer

# helpers ----------------------------------------------------------------------

def cycle(loader):
	while True:
		for data in loader:
			yield data

def map_fn(index, flags):



	# # Random seed

	seed = flags["seed"]
	torch.manual_seed(seed)

	# # Check hardware

	device = xm.xla_device()

	# # Experiment ID --------------------------------------------------------------

	exp_name = "baseline_rerun"
	# exp_name = "test"
	now = datetime.now()
	unique_id = now.strftime("%m-%d-%Y_%H-%M-%S")
	base_dir = "/home/jonmcwong/tests/"

	# # Training constants ---------------------------------------------------------

	NUM_CPU_THREADS = flags["num_cpu_threads"]          # dataloader
	BATCH_SIZE = flags["batch_size"]                    # size of every accumulatino
	LEARNING_RATE = flags["learning_rate"]              # 
	VALIDATE_EVERY  = flags["validate_every"]			# number of batches between validations
	STEPS = flags["steps"]								# number of weight updates

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

	# # Get training and test data -------------------------------------------------

	# # Initialize Math Dataset Manager



	modules = ['add_or_sub', 'add_sub_multiple', 'div', 'mixed', 'mul', 'mul_div_multiple', 'add_or_sub_in_base', 'nearest_integer_root', 'simplify_surd']
	val_modules = ['add_or_sub', 'add_sub_multiple', 'div', 'mixed', 'mul', 'mul_div_multiple']

	mdsmgr = MathDatasetManager(
	  "/home/jonmcwong/mathematics_dataset-v1.0/"
	)

	train_module_data = {}
	val_module_data = {}
	for module in modules:
		tmp_data = mdsmgr.build_dataset_from_module('arithmetic', module, 'train-easy')
		tmp_train, tmp_val = mandubian.math_dataset.random_split_dataset(tmp_data,split_rate=0.97)
		train_module_data[module] = tmp_train
		val_module_data[module] = tmp_val

	train_ds = data.ConcatDataset(train_module_data.values())
	val_dss = val_module_data

	# # Get the distributed train samplers -----------------------------------------
	train_sampler = data.distributed.DistributedSampler(
		train_ds,
		num_replicas=xm.xrt_world_size(),
		rank=xm.get_ordinal(),
		shuffle=True
	)

	val_samplers = {}
	for val_module in val_modules:
		val_samplers[val_module] = data.distributed.DistributedSampler(
		val_dss[val_module],
		num_replicas=xm.xrt_world_size(),
		rank=xm.get_ordinal(),
		shuffle=False
	)

	# # Get dataloaders ------------------------------------------------------------

	train_loader = data.DataLoader(
		train_ds,
		batch_size=BATCH_SIZE,
		sampler=train_sampler,
		num_workers=NUM_CPU_THREADS,
		collate_fn=question_answer_to_position_batch_collate_fn,
		pin_memory = True,
		drop_last=True)

	val_loaders = {}
	for val_module in val_modules:
		val_loaders[val_module] = data.DataLoader(
			val_dss[val_module],
			batch_size=BATCH_SIZE,
			sampler=val_samplers[val_module],
			shuffle=False,
			collate_fn=question_answer_to_position_batch_collate_fn,
			pin_memory=True,
			num_workers=NUM_CPU_THREADS,
			drop_last=True)


	# #  for viewing output sequences

	# gen_loader = data.DataLoader(
	#     val_ds, batch_size=1, shuffle=False, num_workers=NUM_CPU_THREADS,
	#     collate_fn=question_answer_to_position_batch_collate_fn, pin_memory = True)
	# gen_loader = cycle(gen_loader)

	# interpolate_loader = data.DataLoader(
	#     interpolate_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_CPU_THREADS,
	#     collate_fn=question_answer_to_mask_batch_collate_fn, pin_memory = True)
	# interpolate_loader = cycle(interpolate_loader)

	# extrapolate_loader = data.DataLoader(
	#     extrapolate_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_CPU_THREADS,
	#     collate_fn=question_answer_to_mask_batch_collate_fn)
	# extrapolate_loader = cycle(extrapolate_loader)


	# # Model ----------------------------------------------------------------------
	model = build_UTransformer()

	# model = build_transformer()
	# model = build_just_decoder()
	model.to(device).train()


	# # Optimizer learning rate scheduler

	optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.995), eps=1e-9)

	# # restore model
	restore = False
	filename = "./checkpoints/transformer_baseline_continued_06-02-2020_07-50-17/transformer_baseline_continued_06-02-2020_07-50-17_log_0.pth"
	if (restore):
		state = restore_checkpoint(filename=filename, model=model, optimizer=optimizer)
		i = state["batch"]
		logs = state["logs"]


	# # logging variables ----------------------------------------------------------
	i = 0
	logs = {}
	metrics = ["loss", "char_acc", "ans_acc"]

	for module in val_modules:
		for metric in metrics:
			logs[module + "_val_" + metric] = []
	for metric in metrics:
		logs["train_" + metric] = []

	best_val_ans_acc = 0
	last_milestone_val_ans_acc = 0

	# # parallel loaders
	para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
	para_train_loader = cycle(para_train_loader)
	para_val_loaders = {}
	for module, loader in val_loaders.items():
		para_val_loaders[module] = pl.ParallelLoader(loader, [device]).per_device_loader(device)
		para_val_loaders[module] = cycle(para_val_loaders[module])
	print("reached this print")
	# # Train ----------------------------------------------------------------------
	model.train()
	train_start=time.time()
	for i in range(STEPS):

		# if (i % GENERATE_EVERY) - 1 == 0:
		#     model.eval()
		#     gen_qs, gen_qs_pos, gen_as, gen_as_pos = next(gen_loader)
		#     prime = np_decode_string(gen_qs.numpy())
		#     print('*' * 100, "\nQuestion: ", prime)
		#     print("Actual Answer: ", np_decode_string(gen_as.numpy()))
		#     with torch.no_grad():
		#         response = predict_single(prime[1:-1], model, device)
		#     print("Decoded Prediction: ", response[0]["resp"])
		#     gc.collect()

		batch_qs, batch_qs_pos, batch_as, batch_as_pos = next(para_train_loader)
		gold_as = batch_as[:, 1:]

		# # output
		pred_as = model(batch_qs, batch_qs_pos, batch_as, batch_as_pos, return_emb=False)

		# # loss
		train_loss, n_correct, n_correct_answers = compute_performance(pred_as, gold_as, smoothing=False)    
		train_loss.backward()

		torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
		xm.optimizer_step(optimizer)
		optimizer.zero_grad()

		print("step: ", i)


		# # validate model save snapshots
		if i % VALIDATE_EVERY == 0:
			print("-----------------------------------------------------------------")
			# # log train metrics
			# # calculate char acc
			non_pad_mask = batch_as.ne(Constants.PAD)
			n_char = non_pad_mask.sum().item()
			n_correct = float(n_correct) / float(n_char)

			# # calculate ans acc
			n_correct_answers = float(n_correct_answers)/float(len(batch_qs))

			logs["train_loss"], logs["train_char_acc"], logs["train_ans_acc"] = float(train_loss), n_correct, n_correct_answers

			print("Step ", i, "\t", 
				"train loss: " + str(train_loss), "\t", 
				"char acc: " + str(n_correct), "\t", 
				"ans acc: " + str(n_correct_answers), "\t", 
				datetime.now().time() )

			# # log val metrics
			model.eval()
			average_ans_acc = 0
			for module, para_val_loader in para_val_loaders.items():
				batch_qs, batch_qs_pos, batch_as, batch_as_pos = next(para_val_loader)
				gold_as = batch_as[:, 1:]

				with torch.no_grad():

					# forward
					pred_as = model(batch_qs, batch_qs_pos, batch_as, batch_as_pos)

					val_loss, n_correct, n_correct_answers = compute_performance(pred_as, gold_as, smoothing=False)

					# # calculate char acc
					non_pad_mask = batch_as.ne(Constants.PAD)
					n_char = non_pad_mask.sum().item()
					n_correct = float(n_correct) / float(n_char)

					# # calculate ans acc
					n_correct_answers = float(n_correct_answers)/float(len(batch_qs))

					logs[module + "_val_loss"], logs[module + "_val_char_acc"], logs[module + "_val_ans_acc"] = val_loss, n_correct, n_correct_answers
					average_ans_acc += float(n_correct_answers)
					print(module + " val loss: " + str(val_loss), "\t", 
					"char acc: " + str(n_correct), "\t", 
					"ans acc: " + str(n_correct_answers))
			average_ans_acc = average_ans_acc / len(para_val_loaders)
			model.train()

			# # save model
			print("Checkpointing model to ", exp_name + "_" + unique_id + "_log")
			state = build_checkpoint_detailed(
				exp_name,
				unique_id,
				"log",
				model,
				optimizer,
				logs,
				i)
			rotating_save_checkpoint(
				state,
				prefix=exp_name + "_" + unique_id + "_latest",
				path="./checkpoints",
				nb=1)

			# if we have a good val model, save it
			if average_ans_acc > best_val_ans_acc:
				best_val_ans_acc = average_ans_acc
				print("Reached best validation answer accuracy!")
				rotating_save_checkpoint(state,
					prefix=exp_name + "_" + unique_id + "_best_val_ans_acc",
					path="./checkpoints", nb=1)


				# if we reached an accuracy milestone
				if average_ans_acc > last_milestone_val_ans_acc + 20:
					last_milestone_val_ans_acc = average_ans_acc
					print("Milestone Reached")
					rotating_save_checkpoint(state,
						prefix=exp_name + "_" + unique_id + "_training_milestone",
						path="./checkpoints", nb=5)

	elapsed_train_time = time.time() - train_start
	print("Process", index, "finished training. Train time was:", elapsed_train_time)

#        def run():
 #               torch.multiprocessing.freeze_support()
  #              print('loop')

if __name__ == '__main__':
 #               run()

	# # run program
	flags = {}
	flags["batch_size"] = 128
	flags["steps"] = 1000
	flags["seed"] = 1
	flags["num_cpu_threads"] = 1
	flags["validate_every"] = 1000
	flags["learning_rate"] = 6e-6

	xmp.spawn(map_fn, args=(flags,), nprocs=8, start_method='fork')

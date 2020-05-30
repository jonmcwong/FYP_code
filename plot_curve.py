
import math
import numpy as np
import torch
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

import mandubian.math_dataset
from mandubian.math_dataset import MathDatasetManager
from mandubian.transformer import Constants
from mandubian.math_dataset import (random_split_dataset,
    question_answer_to_mask_batch_collate_fn
)
import mandubian.checkpoints
from mandubian.checkpoints import rotating_save_checkpoint, build_checkpoint, restore_checkpoint
from mandubian.math_dataset import np_encode_string, np_decode_string
import mandubian.model_process
import mandubian.utils
from mandubian.tensorboard_utils import Tensorboard
from mandubian.tensorboard_utils import tensorboard_event_accumulator
from mandubian.math_dataset import (
    VOCAB_SZ, MAX_QUESTION_SZ, MAX_ANSWER_SZ
)

print("Torch Version", torch.__version__)

from lucidrains_reformer.reformer_pytorch import ReformerLM, Autopadder, Recorder
from lucidrains_reformer.reformer_pytorch import ReformerEncDec
from lucidrains_reformer.reformer_pytorch.generative_tools import TrainingWrapper




# # restore model
filenames = [
"./checkpoints/transformer_1024_arithmetic_traineasy_lr_6e-6_05-24-2020_18-12-55/transformer_1024_arithmetic_traineasy_lr_6e-6_05-24-2020_18-12-55_log_0.pth",
"./checkpoints/activations_collection_batch_size_3072_05-29-2020_19-53-45/activations_collection_batch_size_3072_05-29-2020_19-53-45_log_0.pth",
"./checkpoints/activations_collection_batch_size_1024_1e-4_05-30-2020_08-07-48_log_0.pth",
"./checkpoints/activations_collection_batch_size_1024_1e-4_05-30-2020_07-51-08_log_0.pth",
]

def smooth_series(series, factor=0.9):
	# factor = 0 for no smoothing
	if factor == 0:
		return series
	prev = series[0]
	for i in range(len(series)):
		if math.isnan(prev):
			prev = series[i]
		series[i] = prev*factor + series[i]*(1-factor)
		prev = series[i]
	return series

for filename in filenames:
	model = None
	state = restore_checkpoint(filename=filename, model=model)
	i = state["batch"]
	train_loss_list = state["train_loss"]
	val_loss_list = state["val_loss"]
	best_val_loss = val_loss_list[-1][1]

	# # calculate smoothed lines
	smooth_train_list = smooth_series([x[1] for x in train_loss_list], factor=0.95)
	smooth_val_list = smooth_series([x[1] for x in val_loss_list], factor=0.95)

	# # plot training graphs
	plt.plot([x[0] for x in train_loss_list], smooth_train_list, label="train", color='blue' )
	plt.plot([x[0] for x in val_loss_list], smooth_val_list, label="validation", color='orange')

	# # plot variance
	plt.plot([x[0] for x in train_loss_list], [x[1] for x in train_loss_list], color='blue', alpha=0.2 )
	plt.plot([x[0]
	 for x in val_loss_list], [x[1] for x in val_loss_list], color='orange', alpha=0.2)
	plt.axis([0,4500,1.2,6])
plt.xlabel('Steps')
plt.ylabel('Mean Cross Entropy Loss')
plt.title('Training Curve for arithmetic, train-easy')
# ax = plt.subplot(111)
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


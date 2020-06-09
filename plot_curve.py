
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
from mandubian.checkpoints import rotating_save_checkpoint, build_checkpoint, restore_checkpoint, restore_checkpoint_cpu
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
"./checkpoints/final_transformer_06-04-2020_18-55-27/final_transformer_06-04-2020_18-55-27_log_0.pth",
# "./checkpoints/activations_collection_batch_size_3072_05-29-2020_19-53-45/activations_collection_batch_size_3072_05-29-2020_19-53-45_log_0.pth",
"./checkpoints/final_decoder_only_06-03-2020_21-39-11/final_decoder_only_06-03-2020_21-39-11_log_0.pth",
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
labels = ["full_transformer", "decoder_only"]
colors = ["blue", "brown"]
for filename, color, label in zip(filenames, colors, labels):
	model = None
	state = restore_checkpoint_cpu(filename=filename)
	i = state["batch"]
	train_loss_list = state["train_loss"]
	val_loss_list = state["val_loss"]
	best_val_loss = val_loss_list[-1][1]

	# # calculate smoothed lines
	smooth_train_list = smooth_series([x[1] for x in train_loss_list], factor=0.95)
	smooth_val_list = smooth_series([x[1] for x in val_loss_list], factor=0.95)

	# # plot training graphs
	# plt.plot([x[0] for x in train_loss_list], smooth_train_list, label="train", color='blue' )
	plt.plot([x[0] for x in val_loss_list], smooth_val_list, label=label, color=color, alpha=0.9)

	# # plot variance
	# plt.plot([x[0] for x in train_loss_list], [x[1] for x in train_loss_list], color='blue', alpha=0.2 )
	plt.plot([x[0] for x in val_loss_list], [x[1] for x in val_loss_list], color=color, alpha=0.2)
	plt.axis([-5000,190000,0.0,1.4])
plt.xlabel('Steps')
plt.ylabel('Mean Cross Entropy Loss')
plt.title('Validation Curve for arithmetic, train-easy')
# ax = plt.subplot(111)
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


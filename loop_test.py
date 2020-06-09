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
from mandubian.model_process import predict_single, predict_multiple
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

# # get the model
# filename = "./checkpoints/transformer_baseline_continued_06-02-2020_07-50-17/transformer_baseline_continued_06-02-2020_07-50-17_best_val_0.pth"
filename = "./checkpoints/final_decoder_only_06-03-2020_21-39-11/final_decoder_only_06-03-2020_21-39-11_best_val_0.pth"
model = build_just_decoder()
# model = build_transformer()
state = restore_checkpoint(filename=filename, model=model)
# print("this is model: ", model)
i = state["batch"]
	# train_loss_list = state["train_loss"]
	# val_loss_list = state["val_loss"]
	# best_val_loss = val_loss_list[-1][1]
# # actual testing


# # Check hardware

print(torch.cuda.device_count(), "detected CUDA devices")
cuda_device = torch.cuda.current_device()
print("Using CUDA device: ", cuda_device)
print(torch.cuda.get_device_name(cuda_device))
device = torch.device("cuda")

model.eval()
count = 0
question_list = []
answer_list = []
# # get next question
for i in range(1000):
	print(i)
	q = "Multiply " + str(i) + " and " + str(13.4) + "."
	answer_list.append(str(i+0.19)[-2:])
	question_list.append(q)		
with torch.no_grad():
	response = predict_multiple(question_list, model, device)
assert len(response) == len(answer_list)
for i, resp in tqdm(enumerate(response)):
	if resp["resp"][-3:-1] == answer_list[i]:
		print(resp["resp"],"\t", answer_list[i])
		count += 1
print("accuracy: ", str(count/1000))	
y_data_transformer = [float(resp["resp"][:-1]) for resp in response]
#########################################################################################################################

x = range(1000)
y_expected = [13.2*i for i in x]

plt.plot(y_data_transformer, label="decoder only prediction", alpha=0.7)
plt.plot(y_expected, alpha=0.7, label="ground truth")
plt.xlabel('i')
plt.ylabel('Predicted Value')
plt.title('Decoder Only prediction of y=13.4i')
plt.grid(True)
plt.legend(loc='upper left')
plt.locator_params(axis='y', nbins=10)
plt.show()
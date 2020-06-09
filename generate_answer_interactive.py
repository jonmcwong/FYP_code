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
from mandubian.model_process import predict_single
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
# filename = "./checkpoints/final_decoder_only_06-03-2020_21-39-11/final_decoder_only_06-03-2020_21-39-11_best_val_0.pth"
filename = "./checkpoints/transformer_baseline_continued_06-02-2020_07-50-17/transformer_baseline_continued_06-02-2020_07-50-17_best_val_0.pth"
# model = build_just_decoder()
model = build_transformer()
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

# # get next question
while(True):
	q = input("You gonna enter another question or what?\n")
	print("Thank you very much.")


	# # generate top 5
	# encoded_q = np_encode_string(q)
	print("You asked: " + q)
	with torch.no_grad():
	    # print("test question", prime)
	    response = predict_single(q, model, device)
	print("The models top 5 responses are:")
	for resp in response:
		print("score: " + str(resp["score"]), "\tresponse: " + str(resp["resp"]))
	gc.collect()
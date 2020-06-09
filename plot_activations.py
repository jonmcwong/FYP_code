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
from mpl_toolkits.mplot3d import Axes3D

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import pprint
pp = pprint.PrettyPrinter(indent=4)

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

# # constants
NUM_CPU_THREADS = 6             # dataloader
BATCH_SIZE = 256                 # size of every accumulatino
GRADIENT_ACCUMULATE_EVERY = 8   # number of accumulation
LEARNING_RATE = 6e-6            # 
VALIDATE_EVERY  = 20            # number of batches between validations
GENERATE_EVERY  = 200            # number of batechs between sequences generated when training
GENERATE_LENGTH = 32            # how many characters to generate

def get_accept_mask(batch, accepted_chars, device=None):
	'''returns mask of the same size as batch which is True for letters in accepted_chars'''
	mask = torch.tensor((), device=device)
	mask = mask.new_full(batch.size(), False, dtype=bool)
	encoded_accepted_chars = np_encode_string(accepted_chars)
	encoded_accepted_chars = encoded_accepted_chars[1:-1]
	for c in encoded_accepted_chars:
		mask |= batch.eq(c)
	return mask


def get_reject_mask(batch, rejected_chars):
	mask = torch.tensor((), device=device)
	mask = mask.new_full(batch.size(), False)
	encoded_rejected_chars = np_encode_string(accepted_chars)
	encoded_rejected_chars = encoded_rejected_chars[1:-1]
	for c in encoded_accepted_chars:
		mask |= batch.ne(c)
	return mask
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


# # import model
filename = "./checkpoints/just_decoder_1024_6e-6_06-01-2020_00-21-51/just_decoder_1024_6e-6_06-01-2020_00-21-51_best_val_0.pth"
model = build_just_decoder()
state = restore_checkpoint(filename=filename, model=model)
model = model.to(device)
i = state["batch"]
# untrained_model = build_transformer()


# # get data
training_data = mdsmgr.build_dataset_from_modules('arithmetic', ['add_or_sub', 'add_sub_multiple','div','mul', 'mixed', 'mul_div_multiple'], 'train-easy', max_elements=1000)
train_loader = data.DataLoader(
    training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_CPU_THREADS,
    collate_fn=question_answer_to_position_batch_collate_fn, pin_memory = True)




accept_string = "+-*/"
# # lits of random colors for each c
colors = []
for i in range(len(accept_string)):
	colors.append((random.random(),
		random.random(),
		random.random()))
# # get the embeddings from the encoder
embeddings = []
for i in range(len(accept_string)):
	embeddings.append([])
for batch in tqdm(train_loader):
	batch_qs, batch_qs_pos, batch_as, batch_as_pos = map(lambda x: x.to(device, non_blocking=True), batch)
	gold_as = batch_as[:, 1:]
	with torch.no_grad():
		pred_as, enc_output = model(batch_qs, batch_qs_pos, batch_as, batch_as_pos, return_emb=True)
		del pred_as

		# # filter for desired embeddings
		batch_qs = batch_qs.view(-1)
		enc_output = enc_output.view(-1, 512)
		for i, c in enumerate(accept_string):
			accepted_mask = get_accept_mask(batch_qs, c, device=device)
			filtered_embs = enc_output[accepted_mask]
			filtered_embs = filtered_embs.to('cpu',non_blocking=True).numpy()
			embeddings[i].append(filtered_embs)
# # get list of np.arrays
for i in range(len(embeddings)):
	embeddings[i] = np.concatenate(embeddings[i], axis=0)
classes = []
for i in range(len(embeddings)):
	classes.append(np.full((embeddings[i].shape[0]),i))
print(classes)
classes = np.concatenate(classes, axis=0)
print(classes)
# currently embeddings is a list of lists
# each list in embeddings refers to a single character and contains the embeddings for those characters

# # PCA the embeddings
# pca = PCA(n_components=2)

lda = LinearDiscriminantAnalysis(n_components=3)
all_data = np.concatenate(embeddings, axis=0)
# pca.fit(all_data)
print("all_data shape: ", all_data.shape)
print("classes shape: ", classes.shape)
lda.fit(all_data, classes)
for i, c in enumerate(embeddings):
	# pca.transform(c)
	embeddings[i] = lda.transform(c)
print("len", len(embeddings))
print("len of each embs for each c", len(embeddings[0]))


# # plot transformed embeddings



# # creating plot
print("plotting now")
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
for embeddings, color,c in zip(embeddings,colors, accept_string):

	x,y,z = [x[0] for x in list(embeddings)], [x[1] for x in list(embeddings)], [x[2] for x in list(embeddings)]
	ax.scatter(x, y, z, alpha=0.7, c=color, label=c, marker=',', lw=0, s=4)
plt.show()

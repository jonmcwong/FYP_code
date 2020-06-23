import numpy as np
import torch
from torch.utils import data
import torch.nn.functional as F
from mandubian.transformer import Constants
from mandubian.math_dataset import np_decode_string


def compute_performance(pred, gold, smoothing, log=False):
    loss = compute_loss(pred, gold, smoothing)

    # pred max returns a tuple of max, index
    pred_max = pred.max(2)[1]
    # print("Argmax of pred (along dim(2)) has shape: ", pred_max.size())
    # pred_max = pred.max(1)[1]

    # gold = gold.contiguous().view(-1)
    # if log:
    #  print("pred", pred)
    #  print("pred", pred_max)
    #  print("gold", gold)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred_max.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    # # number of correct whole answers
    pad_mask = gold.eq(Constants.PAD)

    char_incorrect = pred_max.ne(gold)
    # print("Incorrect chars in pred: ", char_incorrect)
    char_incorrect[pad_mask] = 0
    char_incorrect = char_incorrect.sum(dim=1)
    # print("Number of incorrect chars per sequence: ", char_incorrect)
    char_incorrect = (char_incorrect == 0).sum()
    # print("Total number of correct chars: ", n_correct_answers)

    return loss, n_correct, char_incorrect

def compute_loss(pred, gold, smoothing):
    gold = gold.contiguous().view(-1)
    pred = pred.reshape(-1, pred.size(2)) 

    # print("In compute_loss: size of gold is: ", gold.size())
    # print('Size of pred is: ', pred.size())
    # print("gold: ", gold)
    if smoothing:
      eps = 0.1
      n_class = pred.size(1)

      # create array of 0s same shape as pred. Put 1 at teh indices indicated by gold.
      one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)

      # smooth
      one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1) # 0.1, 0.9, 0.1, 0.1

      # log probabilities
      log_prb = F.log_softmax(pred, dim=1)

      non_pad_mask = gold.ne(Constants.PAD)
      loss = -(one_hot * log_prb).sum(dim=1)
      loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
      # print("pred: ", pred[0])
      # print("gold: ", gold[0])
      # print("pred size: ", pred.size())
      # print("gold: ", gold.size())
      #gold indicates the correct class
      loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='mean')    
    return loss
  
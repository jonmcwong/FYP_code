from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from lucidrains_reformer.reformer_pytorch.reformer_pytorch import ReformerLM
from lucidrains_reformer.reformer_pytorch.autopadder import Autopadder
from mandubian.transformer import Constants

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.gather(1, sorted_indices)

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

class TrainingWrapper(nn.Module):
    def __init__(self, net, ignore_index = Constants.PAD, pad_value = 0):
        super().__init__()
        assert isinstance(net, ReformerLM), 'generative trainer wrapper can only accept ReformerLM class'
        assert(pad_value == Constants.PAD)
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = Autopadder(net)
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token = None, temperature = 1., filter_logits_fn = top_k, filter_thres = 0.9, **kwargs):
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        # if batch is 1
        if num_dims == 1:
            #transpose? get 1 by t
            start_tokens = start_tokens[None, :]
        
        # batch, token number
        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        # no input mask given during generate
        input_mask = kwargs.pop('input_mask', None)

        if input_mask is None:
            # create mask all ones
            input_mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

        # 32 times
        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            input_mask = input_mask[:, -self.max_seq_len:]

            # get next letters for whole batch
            logits = self.net(x, input_mask=input_mask, **kwargs)[:, -1, :]
            filtered_logits = filter_logits_fn(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
            # pad the last dim once on one side with True
            input_mask = F.pad(input_mask, (0, 1), value=True)

            if eos_token is not None and (sample == eos_token).all():
                break

        # get rid of all the tokens that were used as prompt
        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out

    def forward(self, x, return_loss = False, **kwargs):

        # function for padding with zeros
        pad = partial(pad_sequence, batch_first = True, padding_value = self.pad_value)

        # encoder exits here
        if not return_loss:
            if not isinstance(x, torch.Tensor):
                x = pad(x)
            return self.net(x, **kwargs)

        # if necessary, pad with 0s
        if isinstance(x, torch.Tensor):
            # everything in batch excluding last token of each sequence
            xi = x[:, :-1]
            # everything in batch excluding first token of each sequence
            xo = x[:, 1:]
        else:
            xi = pad(list(map(lambda t: t[:-1], x)))
            xo = pad(list(map(lambda t: t[1:], x)))

        out = self.net(xi, **kwargs)
        # print("generative_tools.py: line 101: shape of encoder output: ", out.size())


        # only decoder reaches here
        # ignore loss from pad tokens which have index of "ignore_index" = 0
        loss = F.cross_entropy(out.transpose(1, 2), xo, ignore_index = self.ignore_index)
        return loss

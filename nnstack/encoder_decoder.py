## code based off of https://github.com/TIANHAO-WANG/nnstack

import numpy as np
import sys
from nnstack.nnstack import Controller
import torch
import torch.nn as nn
import torch.optim as optim
from mandubian.transformer import Constants as Constants


# Define Seq2Seq Models
class Encoder(nn.Module):
	def __init__(self, n_src_vocab, input_dim, hid_dim, n_layers, device, type='RNN',mem_width=128):
		super().__init__()
		
		self.hid_dim = hid_dim
		self.n_layers = n_layers
		
		self.type = type
		
		self.embed = nn.Embedding(n_src_vocab, input_dim, padding_idx=Constants.PAD)
		
		if type=='RNN':
			self.rnn = nn.RNN(input_dim, hid_dim, n_layers)
		elif type=='LSTM':
			self.rnn = nn.LSTM(input_dim, hid_dim, n_layers)
		elif type=='NNstack':
			self.rnn = Controller(input_dim, hid_dim, n_layers, device, mem_width=mem_width)
		
	def forward(self, src):
		
		#src.shape = (seq_len, vocab_size) [batch_size, seq_len]
		batch_size, seq_len,  = src.shape
		src = torch.transpose(src, 0,1)
		src = self.embed(src)
		
		# src = src.unsqueeze(1)
		
		#src.shape = (seq_len, batch_size, d_word_vec)

		# if we use neural stack, both encoder and decoder use the same stack. 
		if self.type=='NNstack':
			input = src[0:1] # [1, batch_size, d_word_vec]
			hidden = None

			#loops
			for i in range(1, seq_len):
				output, hidden = self.rnn(input, hidden)
				input = src[i:(i+1)]
		else:
			outputs, hidden = self.rnn(src)
		
		#hidden.shape = [n_layers, 1, hid_dim]
		#cell.shape = [n_layers, 1, hid_dim]
		
		return hidden
	
	
class Decoder(nn.Module):
	def __init__(self, n_src_vocab, output_dim, hid_dim, n_layers, device, type='RNN',mem_width=128):
		super().__init__()
		
		self.output_dim = output_dim
		self.hid_dim = hid_dim
		self.n_layers = n_layers
		
		if type=='RNN':
			self.rnn = nn.RNN(output_dim, hid_dim, n_layers)
		elif type=='LSTM':
			self.rnn = nn.LSTM(output_dim, hid_dim, n_layers)
		elif type=='NNstack':
			self.rnn = Controller(output_dim, hid_dim, n_layers, device, mem_width=mem_width)
				
	
	def forward(self, input, hidden):
		#input.shape = [1, vocab_len]
		
		#hidden.shape = [n_layers, 1, hid_dim]
		#cell.shape = [n_layers, 1, hid_dim]
				
		#input.shape = [1, batch_size, d_word_vec]
		
				
		prediction, hidden = self.rnn(input, hidden)
		
		#output.shape = (1, 1, hid_dim)
				
		return prediction, hidden
	
	
class Seq2seq(nn.Module):
	def __init__(self, n_src_vocab, n_tgt_vocab, device, 
		d_word_vec=512, d_model=512, n_layers=6, mem_width=128):
		super().__init__()

		self.device = device
		self.encoder = Encoder(n_src_vocab, d_word_vec, d_model, n_layers, device, type='NNstack', mem_width=mem_width)
		self.decoder = Decoder(n_src_vocab, d_word_vec, d_model, n_layers, device, type='NNstack', mem_width=mem_width)
		
		# projecting the output of the decoder
		self.tgt_word_prj = nn.Linear(d_word_vec, n_tgt_vocab, bias=False).to(device)
		nn.init.xavier_normal_(self.tgt_word_prj.weight)

		if (self.encoder.hid_dim!=self.decoder.hid_dim) or (self.encoder.n_layers!=self.decoder.n_layers):
			sys.exit('ENCODER AND DECODER MUST HAVE SAME DIM!')
		
		# used when saving outputs of decoder
		self.d_model = d_model
		self.d_word_vec = d_word_vec

		self.decoder_emb = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)
		#sharing weights between the decoder embedding and output projection

		#sharing weights between the decoder embedding and the encoder embedding

	def forward(self, src, tgt, teacher_force = 0.75, return_emb=False):
		del return_emb
		# BATCH SIZE = 1
		# src = [batch, seq_len]
		# tgt = [batch, seq_len]

		# leaving out ending token for tgt sequence so dims match
		tgt = tgt[:, :-1]

		
		batch_size, seq_len = tgt.shape[0], tgt.shape[1]
		
		# tgt_vocab_size = self.decoder.output_dim
		
		# tensor to store outputs&
		output_embs = torch.zeros(seq_len, batch_size, self.d_word_vec).to(self.device)
		
		#last hidden state of the encoder is used as the initial hidden state of the decoder
		hidden = self.encoder(src) # lstm takes size [seq_len, batch_size, d_word_vec]
		
		#first input to the decoder is the <sos> tokens
		tgt = torch.transpose(tgt, 0, 1) # [seq_len, batch_size]
		input_emb = self.decoder_emb(tgt[0:1])  # [1, batchs_size, d_word_vec]
		
		# produce values from decoder one by one
		tf = torch.full((seq_len,1),teacher_force)
		tf = torch.bernoulli(tf).to(self.device)
		for t in range(1, seq_len):
			
			#insert input_emb token embedding, previous hidden and previous cell states
			#receive output tensor (predictions) and new hidden and cell states
			output, hidden = self.decoder(input_emb, hidden)
			# print("decoder output has size: ", output.size()) # should be [batch_size, d_word_vec]
			
			#saving embedding from decdoer
			output_embs[t] = output

			# IN EVAL PHASE, stop training / prediction if top1 is EOS
			# if self.training == False:

			# 	#get the highest predicted token from our predictions
			# 	top1 = self.tgt_word_prj(output).argmax(1) #may need to unsqueeze
				
			# 	if top1==Constants.EOS: break
			
			# 	top1 = torch.Tensor(top1)
			# 	top1 = top1.unsqueeze(0)
			
			input_emb = tf[t]*self.decoder_emb(tgt[t:t+1]) + (1-tf[t])*output # may need to slice instead

			
		# project outputs

		return torch.transpose(self.tgt_word_prj(output_embs), 0, 1) # return [batch_size, seq_len, tgt_vocab_size]


### Define Training & Evaluation Process

def train(model, train_set, optimizer, criterion):
	model.train()
	
	epoch_loss = 0
	
	train_src, train_tgt = train_set['src'], train_set['tgt']
	
	BATCH_SIZE = len(train_src)
		
	for i in range(BATCH_SIZE):
		
		src = torch.Tensor(train_src[i]).to(model.device)
		tgt = torch.Tensor(train_tgt[i]).to(model.device)
				
		optimizer.zero_grad()
				
		output = model(src, tgt)
				
		#tgt.shape = [seq_len, vocab_size]
		#output.shape = [seq_len, vocab_size]
		
		output = output[1:]
		tgt = tgt[1:]
		
		#trg.shape = [seq_len-1, vocab_size]
		#output.shape = [seq_len-1, vocab_size]
		
		loss = criterion(output, tgt.argmax(dim=1))
		loss.backward()
								
		optimizer.step()
				
		epoch_loss += loss.item()

	return epoch_loss / BATCH_SIZE


def evaluate(model, test_set, criterion):
	model.eval()
	
	epoch_loss = 0
	coarse = 0
	fine_lst = []
	
	test_src = test_set['src']
	test_tgt = test_set['tgt']
	
	BATCH_SIZE = len(test_src)
		
	with torch.no_grad():
		
		for i in range(BATCH_SIZE):
			
			src = torch.Tensor(test_src[i]).to(model.device)
			tgt = torch.Tensor(test_tgt[i]).to(model.device)

			output = model(src, tgt, 0) #turn off teacher forcing
			
			output = output[1:]
			tgt = tgt[1:]
			
			output = output.data.tolist()
			tgt = tgt.data.tolist()
			
			fine_i = 0
			
			output_len = len(output)
			seq_len = len(tgt)
			
			for i in range(min(seq_len, output_len)):
				output_i = np.argmax(output[i])
				tgt_i = np.argmax(tgt[i])
				if output_i == tgt_i: fine_i += 1
				else: break
			
			fine_lst.append(fine_i/seq_len)
			if fine_i == seq_len: coarse += 1
				
	return coarse / BATCH_SIZE, np.mean(fine_lst)
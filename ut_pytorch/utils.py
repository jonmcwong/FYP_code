import torch
from ut_pytorch.models.UTransformer import UTransformer

from mandubian.math_dataset import (
    VOCAB_SZ, MAX_QUESTION_SZ, MAX_ANSWER_SZ
)

def build_UTransformer(num_vocab=VOCAB_SZ + 1, embedding_size=512, 
	hidden_size=512, num_layers=6, num_heads=8, total_key_depth=64, 
	total_value_depth=64, filter_size=2048, 
	enc_seq_max_length=256, dec_seq_max_length=32, 
	input_dropout=0.0, layer_dropout=0.0, attention_dropout=0.0, 
	relu_dropout=0.0, use_mask=False, act=True
):
	return UTransformer(num_vocab=num_vocab, 
		embedding_size=embedding_size, 
		hidden_size=hidden_size, num_layers=num_layers, 
		num_heads=num_heads, total_key_depth=total_key_depth, 
		total_value_depth=total_value_depth, filter_size=filter_size, 
		enc_seq_max_length=enc_seq_max_length, dec_seq_max_length=dec_seq_max_length, 
		input_dropout=input_dropout, layer_dropout=layer_dropout, 
		attention_dropout=attention_dropout, 
		relu_dropout=relu_dropout, 
		use_mask=use_mask, 
		act=act
		)
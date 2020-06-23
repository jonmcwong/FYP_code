import torch
from mandubian.math_dataset import (
    VOCAB_SZ, MAX_QUESTION_SZ, MAX_ANSWER_SZ
)
from nnstack.encoder_decoder import Seq2seq

from mandubian.math_dataset import (
    VOCAB_SZ, MAX_QUESTION_SZ, MAX_ANSWER_SZ
)


def build_neural_stack(n_src_vocab=128, n_tgt_vocab=128, device="cuda",
				d_word_vec=512, d_model=512, n_layers=4, mem_width=512
):
	return Seq2seq(n_src_vocab=n_src_vocab, n_tgt_vocab=n_tgt_vocab, device=device,
				d_word_vec=d_word_vec, d_model=d_model, n_layers=n_layers, mem_width=mem_width)

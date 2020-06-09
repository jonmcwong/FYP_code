import torch
from mandubian.math_dataset import (
    VOCAB_SZ, MAX_QUESTION_SZ, MAX_ANSWER_SZ
)
from mandubian.transformer.Models import Transformer, Just_Decoder

from mandubian.math_dataset import (
    VOCAB_SZ, MAX_QUESTION_SZ, MAX_ANSWER_SZ
)


def one_hot_seq(chars, vocab_size=VOCAB_SZ, char0 = ord(' ')):
  chars = (chars - char0).long()
  return torch.zeros(len(chars), VOCAB_SZ+1).scatter_(1, chars.unsqueeze(1), 1.)


def torch_one_hot_encode_string(s):
    chars = np.array(list(s), dtype='S1').view(np.uint8)
    q = torch.tensor(chars, dtype=torch.uint8)
    q = one_hot_seq(q)
    return q

def build_transformer(
    n_src_vocab=VOCAB_SZ + 1, n_tgt_vocab=VOCAB_SZ + 1,
    len_max_seq_encoder=MAX_QUESTION_SZ, len_max_seq_decoder=MAX_ANSWER_SZ,
    d_word_vec=512, d_model=512, d_inner=2048,
    n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
    tgt_emb_prj_weight_sharing=True,
    emb_src_tgt_weight_sharing=True
):
    return Transformer(
      n_src_vocab=n_src_vocab, # add PAD in vocabulary
      n_tgt_vocab=n_tgt_vocab, # add PAD in vocabulary
      len_max_seq_encoder=len_max_seq_encoder,
      len_max_seq_decoder=len_max_seq_decoder,
      d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
      n_layers=n_layers, n_head=8, d_k=64, d_v=64, dropout=0.1,
      tgt_emb_prj_weight_sharing=True,
      emb_src_tgt_weight_sharing=True
    )

def build_dgl_transformer(
    n_src_vocab=VOCAB_SZ + 1, n_tgt_vocab=VOCAB_SZ + 1,
    len_max_seq_encoder=MAX_QUESTION_SZ, len_max_seq_decoder=MAX_ANSWER_SZ
):
    from dgl_transformer.dgl_transformer import make_model
    return make_model(src_vocab=n_src_vocab, tgt_vocab=n_tgt_vocab)


def build_just_decoder(
    n_src_vocab=VOCAB_SZ + 1, n_tgt_vocab=VOCAB_SZ + 1,
    len_max_seq_encoder=MAX_QUESTION_SZ, len_max_seq_decoder=MAX_ANSWER_SZ,
    d_word_vec=512, d_model=512, d_inner=2048,
    n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
    tgt_emb_prj_weight_sharing=True,
    emb_src_tgt_weight_sharing=True
):
    return Just_Decoder(
      n_src_vocab=n_src_vocab, # add PAD in vocabulary
      n_tgt_vocab=n_tgt_vocab, # add PAD in vocabulary
      len_max_seq_encoder=len_max_seq_encoder,
      len_max_seq_decoder=len_max_seq_decoder,
      d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
      n_layers=n_layers, n_head=8, d_k=64, d_v=64, dropout=0.1,
      tgt_emb_prj_weight_sharing=True,
      emb_src_tgt_weight_sharing=True
    )
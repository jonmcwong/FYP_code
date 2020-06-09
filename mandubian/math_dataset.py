import os
from pathlib import Path
import glob
import pandas as pd

import numpy as np
import torch
from torch.utils import data
from .transformer import Constants

from mandubian.dgl_transformer.dataset.graph import GraphPool

# Math Dataset constants (from paper)

# input chars are selected from basic ASCII chars
VOCAB_SZ = 95
# questions have less than 160 chars
MAX_QUESTION_SZ = 160
# answers have less than 30 chars
MAX_ANSWER_SZ = 30


def random_split_dataset(ds, split_rate):
    """uses Torch utils to split and randomize data into train/val dataset"""
    size = len(ds)
    train_split = int(size * split_rate)
    val_split = size - train_split
    train_ds, val_ds = data.random_split(ds, [train_split, val_split])
    return train_ds, val_ds

def np_encode_string(s, char0 = ord(' ')):
    """converts a string into a numpy array of bytes
    (char0 - 1) is subtracted from all bytes values (0 is used for PAD)
    string is pre-pended with BOS and post-pended with EOS"""
    chars = np.array(list(s), dtype='S1').view(np.uint8)
    # normalize to 1 - 96, 0 being PAD
    chars = chars - char0 + 1

    chars = np.insert(chars, 0, Constants.BOS)
    chars = np.insert(chars, len(chars), Constants.EOS)
    return chars

def np_decode_string(chars, char0 = ord(' ')):
    """converts a numpy array of bytes into a UTF-8 string
    (char0 - 1) is added to all bytes values (0 is used for PAD)
    BOS/EOS are removed before utf-8 decoding"""
    chars = chars.astype(np.uint8)
    chars[chars == 0] = 1
    chars = chars + char0 - 1
    # chars = chars[:-1]
    chars = chars.tobytes()
    s = chars.decode('UTF-8')
    return s


class LazyFileMathDataset(data.Dataset):
    """Stream loads math dataset file in a lazy way (optional)
    pandas is used for naive streaming as Python doesn't provide any better tool for that critical feature"""
    def __init__(self, file, lazy_load=False, max_elements=None, log=False):
        self.file = Path(file)
        self.lazy_load = lazy_load
        self.max_elements = max_elements

        fn = self.file.name.replace(".txt", "")
        self.category, self.module = fn.split("__")

        if not self.lazy_load:
            self.build_dataset()
            if log:
                print(f"Initialized MathDataset with file {self.file} (category:{self.category}, module:{self.module}) containing {self.qas.shape[0]} pairs of questions/answers")
        else:
            self.qas = None
            if log:
                print(f"Initialized MathDataset with file {self.file} (category:{self.category}, module:{self.module}) in lazy mode")

      
    def _read_build_dataset(self):

        # # read file and put into a dataframe
        self.df = pd.read_csv(self.file, header=None, sep='\n', names=['qa'], engine='c')
        self._build_dataset()
    
    def _build_dataset(self):
        if self.max_elements is not None:
            self.df_max = self.df.iloc[0:self.max_elements*2]   # get the top max_elements datapoints
        else:
            self.df_max = self.df
        
        # take every other line
        self.questions = self.df_max[0::2]

        # # get rid of the old indicies, replace with regular 0...
        self.questions.reset_index(inplace=True, drop=True)
        self.questions.rename(columns={ "qa" : "questions" }, inplace=True)
        self.answers = self.df_max[1::2]
        self.answers.reset_index(inplace=True, drop=True)
        self.answers.rename(columns={ "qa" : "answers" }, inplace=True)
        self.qas = pd.concat([self.questions, self.answers], axis=1)
        
    def set_max_elements(self, max_elements):
        self.max_elements = max_elements
        # # lazy load
        if self.qas is None:
            self._read_build_dataset()
        else:
            self._build_dataset()
        
    def __getitem__(self, idx):
        # # dataset is only read upon __getitem
        # #reads the whole file and returns a bunch of question answer pairs based on their idx
        if self.qas is None:
            self._read_build_dataset()            
        question, answer = self.qas.iloc[idx]
        # # returns dict
        return {
            "q": question, 
            "q_enc": np_encode_string(question), # np_encode_string converts the data to numpy arrays
            "a": answer, 
            "a_enc": np_encode_string(answer),
        }

    def __len__(self):
        if self.qas is None:
           self._read_build_dataset() 
        return self.qas.shape[0]

    
    
class MathDatasetManager(data.Dataset):
    """A Math Dataset manager starting at root directory (like v1.0) to extract files and build torch datasets
    in a lazy loading and streamed way based on specific types/categories/modules presented in paper.
    
    It indexes difficulty/use-case types:
        - train-easy
        - train-medium
        - train-hard
        - interpolate
        - extrapolate
    
    and all categories:
        - algebra
        - numbers
        - polynomials
        - arithmetic
        - measurement
        - comparison
        - probability
        - calculus
        
    and all modules in those categories:
        - mul
        - add_or_sub_in_base
        - simplify_surd
        - mul_div_multiple
        - mixed
        - nearest_integer_root
        - div
        - add_or_sub
        - add_sub_multiple
        - add_sub_multiple_longer
        - mul_div_multiple_longer
        - div_big
        - mul_big
        - mixed_longer
        - add_or_sub_big
        - etc...
    """
    def __init__(self, root_dir, log=False):
        self.root_dir = Path(root_dir)

        self.dirs = {
            "train-easy" : self.root_dir / "train-easy",
            "train-medium" : self.root_dir / "train-medium",
            "train-hard" : self.root_dir / "train-hard",
            "interpolate" : self.root_dir / "interpolate",
            "extrapolate" : self.root_dir / "extrapolate",
        }
        
        self.dfs = {}
        
        # # for each difficulty
        for k, dir in self.dirs.items():

            # # for each file in each dir
            files = [ff for ff in glob.glob(str(dir) + "/**/*.txt", recursive=True)]
            for f in files:

                # # create a lazy load object 
                ds = LazyFileMathDataset(f, lazy_load = True, log=log)
                if ds.category not in self.dfs:
                    self.dfs[ds.category] = {}
                if ds.module not in self.dfs[ds.category]:

                    # # add module and initialise with dict
                    self.dfs[ds.category][ds.module] = {
                        "train-easy" : {}, "train-medium" : {}, "train-hard" : {},
                        "interpolate": {}, "extrapolate": {}
                    }

                # # add lazy load object as the value of category/module/difficulty:
                self.dfs[ds.category][ds.module][k] = ds # data is stored in a dataframe

        print(f"initialized MultiFilesMathDataset with categories {list(self.dfs.keys())} and types {list(self.dirs.keys())}")

    def get_types(self):
        """retrieves all math typesfor this multi-file dataset"""
        return self.dirs.keys()            
        
    def get_categories(self):
        """retrieves all math problem categories in this multi-file dataset"""
        return self.dfs.keys()
    
    def get_modules_for_category(self, c):
        """retrieves all mathematical modules in a math problem category"""
        return self.dfs[c].keys()
    
    def _build_datasets_from_category(self, category, typ, max_elements=None, label=False):
        ds = []
        labels = []
        # # module name and dict for everything inside module
        for k, m in self.dfs[category].items():

            # # if the module contains the specified difficulty level
            if typ in m:
                if type(m[typ]) is not dict:
                    m[typ].set_max_elements(max_elements)
                    ds.append(m[typ])   # append a lazy object

                    # # keep track of module names and their sizes
                    labels.append([k+'-'+typ] * len(m[typ]))
                    # print("type of module: ", type(m[typ]))
                    print(f"added module {category}/{k}/{typ}")

        if label:
            return ds, labels    # list lazy datasets , list of lists
        else:
            return ds   # list of np arrays, might be lits of lazy objects
        
    def build_dataset_from_category(self, category, typ, max_elements=None, label=False):
        """Build a dataset for all modules in a category"""
        print(f"adding category {category}/../{typ}")
        ds, labels = self._build_datasets_from_category(category, typ, max_elements=max_elements, label=label)
        
        if label:
            return data.ConcatDataset(ds), data.ConcatDataset(labels) # one single np array
        else:
            return data.ConcatDataset(ds)


    def build_dataset_from_categories(self, categories, typ, max_elements=None):
        """Build a dataset for all modules in several categories"""
        ds = []
        for c in categories:
            print(f"adding category {c}/../{typ}")
            dss = self._build_datasets_from_category(c, typ, max_elements=max_elements)
            ds.extend(dss)
        return data.ConcatDataset(ds)

    def build_dataset_from_module(self, category, module, typ, max_elements=None):
        """Build a dataset from a single module in a category"""
        self.dfs[category][module][typ].set_max_elements(max_elements)
        return self.dfs[category][module][typ]

    def build_dataset_from_modules(self, category, modules, typ, max_elements=None):
        """Build a dataset from several modules in a category"""
        ds = []
        for module in modules:
            self.dfs[category][module][typ].set_max_elements(max_elements)
            ds.append(self.dfs[category][module][typ])
        return data.ConcatDataset(ds)

    def build_dataset_from_difficulties_modules_categories(self, categories, modules, difficulties, max_elements=None):
        """Build a dataset from multiple modules, multiple difficulties and multiple categories"""
        ds = []
        for category in categories:
            for module in modules:
                for difficulty in difficulties:
                    self.dfs[category][module][difficulty].set_max_elements(max_elements)
                    ds.append(self.dfs[category][module][typ])
        return data.ConcatDataset(ds)
    
    
    
def question_answer_to_position_batch_collate_fn(qas):
    ''' Gather + Pad the question/answer to the max seq length in batch '''

    # should probably overide this to 256 each
    max_q_len = max(len(qa["q_enc"]) for qa in qas)
    max_a_len = max(len(qa["a_enc"]) for qa in qas)

    batch_qs = []
    batch_as = []
    batch_pos = []
    for qa in qas:
      batch_qs.append(np.pad(qa["q_enc"], (0, max_q_len - len(qa["q_enc"])), mode='constant', constant_values=Constants.PAD))
      batch_as.append(np.pad(qa["a_enc"], (0, max_a_len - len(qa["a_enc"])), mode='constant', constant_values=Constants.PAD))

    # every letter is given a position from 1 to the end. 0 is given for padding
    batch_qs_pos = np.array([
        [pos_i+1 if w_i != Constants.PAD else 0
         for pos_i, w_i in enumerate(q)] for q in batch_qs])

    batch_as_pos = np.array([
        [pos_i+1 if w_i != Constants.PAD else 0
         for pos_i, w_i in enumerate(a)] for a in batch_as])
    
    batch_qs = torch.LongTensor(batch_qs)
    batch_qs_pos = torch.LongTensor(batch_qs_pos)

    batch_as = torch.LongTensor(batch_as)
    batch_as_pos = torch.LongTensor(batch_as_pos)

    return batch_qs, batch_qs_pos, batch_as, batch_as_pos

def question_answer_to_mask_batch_collate_fn(qas):
    ''' Gather + Pad the question/answer to the max seq length in batch '''

    # should probably overide this to 256 each
    # max_q_len = max(len(qa["q_enc"]) for qa in qas)
    # max_a_len = max(len(qa["a_enc"]) for qa in qas)

    max_q_len = 256
    max_a_len = 256

    batch_qs = []
    batch_as = []
    batch_pos = []
    for qa in qas:
      batch_qs.append(np.pad(qa["q_enc"], (0, max_q_len - len(qa["q_enc"])), mode='constant', constant_values=Constants.PAD))
      batch_as.append(np.pad(qa["a_enc"], (0, max_a_len - len(qa["a_enc"])), mode='constant', constant_values=Constants.PAD))

    # every letter is given a position from 1 to the end. 0 is given for padding
    batch_qs_mask = np.array([
        [1 if w_i != Constants.PAD else 0
         for pos_i, w_i in enumerate(q)] for q in batch_qs])

    batch_as_mask = np.array([
        [1 if w_i != Constants.PAD else 0
         for pos_i, w_i in enumerate(a)] for a in batch_as])
    
    batch_qs = torch.LongTensor(batch_qs)
    batch_qs_mask = torch.LongTensor(batch_qs_mask).bool()

    batch_as = torch.LongTensor(batch_as)
    batch_as_mask = torch.LongTensor(batch_as_mask).bool()

    return batch_qs, batch_qs_mask, batch_as, batch_as_mask


    
def question_answer_to_batch_collate_fn(qas):
    ''' Gather + Pad the question/answer to the max seq length in batch '''

    max_q_len = max(len(qa["q_enc"]) for qa in qas)
    max_a_len = max(len(qa["a_enc"]) for qa in qas)

    batch_qs = []
    batch_as = []
    batch_pos = []
    for qa in qas:
      batch_qs.append(np.pad(qa["q_enc"], (0, max_q_len - len(qa["q_enc"])), mode='constant', constant_values=Constants.PAD))
      batch_as.append(np.pad(qa["a_enc"], (0, max_a_len - len(qa["a_enc"])), mode='constant', constant_values=Constants.PAD))
    
    batch_qs = torch.LongTensor(batch_qs)
    batch_as = torch.LongTensor(batch_as)

    return batch_qs, batch_as


def question_to_position_batch_collate_fn(qs):
    ''' Gather + Pad the question to the max seq length in batch '''

    max_q_len = max(len(q) for q in qs)

    batch_qs = []
    batch_pos = []
    for q in qs:
        batch_qs.append(np.pad(q, (0, max_q_len - len(q)), mode='constant', constant_values=Constants.PAD))

    batch_qs_pos = np.array([
        [pos_i+1 if w_i != Constants.PAD else 0
         for pos_i, w_i in enumerate(q)] for q in batch_qs])
    
    batch_qs = torch.LongTensor(batch_qs)
    batch_qs_pos = torch.LongTensor(batch_qs_pos)

    return batch_qs, batch_qs_pos


class GraphCollate():   
    def __init__(self):
        self.graph_pool = GraphPool(MAX_QUESTION_SZ, MAX_ANSWER_SZ)
        
    def __call__(self, device):
        def collate_fn(qas):
            ''' Gather + Pad the question/answer to the max seq length in batch '''

            max_q_len = max(len(qa["q_enc"]) for qa in qas)
            max_a_len = max(len(qa["a_enc"]) for qa in qas)

            batch_qs = []
            batch_as = []
            batch_pos = []
            for qa in qas:
              batch_qs.append(np.pad(qa["q_enc"], (0, max_q_len - len(qa["q_enc"])), mode='constant', constant_values=Constants.PAD))
              batch_as.append(np.pad(qa["a_enc"], (0, max_a_len - len(qa["a_enc"])), mode='constant', constant_values=Constants.PAD))

            batch_qs = np.array(batch_qs)
            batch_as = np.array(batch_as)
            gold_as = torch.LongTensor(batch_as[:, 1:])
            g = self.graph_pool(batch_qs, batch_as, device='cpu')

            return gold_as, g
        
        return collate_fn

    
    def beam(self, qs, device, max_len, start_sym, beam_size):
        ''' Gather + Pad the question to the max seq length in batch '''

        max_q_len = max(len(q) for q in qs)

        batch_qs = []
        batch_pos = []
        for q in qs:
            batch_qs.append(np.pad(q, (0, max_q_len - len(q)), mode='constant', constant_values=Constants.PAD))

        batch_qs = np.array(batch_qs)
        g = self.graph_pool.beam(batch_qs, start_sym, max_len, beam_size, device=device)

        return g    
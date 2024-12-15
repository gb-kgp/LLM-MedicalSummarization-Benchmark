# import logging
# from dataclasses import dataclass, field
# import os
# import random
# import torch
# from torch.utils.data import DataLoader, Dataset, Sampler, RandomSampler, SequentialSampler
# import torch.distributed as dist
# import torch.multiprocessing as mp
# import torch.nn.functional as F
# import torch.nn as nn
# import torch.optim as optim
# import torch.utils.data.distributed as data_dist
# from datasets import load_dataset,load_from_disk, Dataset
# from transformers import AutoTokenizer, TrainingArguments
# from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     BitsAndBytesConfig,
#     Trainer,
#     TrainingArguments,
#     set_seed,
# )
# from transformers.utils import is_flash_attn_2_available
# from torch.cuda import is_bf16_supported
# import peft
# from peft import LoraConfig, LoftQConfig, get_peft_config, prepare_model_for_kbit_training
# import bitsandbytes as bnb
# import numpy as np
# import time
# from accelerate import PartialState
# import sys
# import datetime
# from transformers import set_seed
# set_seed(42)
# import json
# import math
# import numpy as np
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

import pandas as pd
import numpy as np
import sys

data = sys.argv[1]
n_icl = int(sys.argv[2])

def read_csv_to_list(fn_csv, csv_action='r'):
    ''' given full path to csv file 
        load each element to list '''

    with open(fn_csv, csv_action) as f:
        lines = f.readlines()

    return lines

def prepend_icl_examples(element,element_idx, prefix1,prefix2,suffix):

    print(f'Starting for element', element)
    pre1, pre2,suf = prefix1, prefix2,suffix
    icl_list_prompt, trn_inputs, trn_target = [], [], []

    # select in-context examples from training set, then run over test set
    df = pd.read_csv('../../../../../../Medical/PushToNeumann/CSV-Datasets/EBM-Train.csv')
    trn_inputs = df['input_text'].tolist()
    trn_target = df['target_text'].tolist()
    
    example_indices = np.load('../../../../../../Medical/PushToNeumann/ICL-Examples/closest_neighbors_EBM.npy')
    example_indices = example_indices.astype(int)
    
    new_element = '' 

    # format in-context examples via best practices
    # see example 5, https://shorturl.at/kuGN6
    
    for idx_idx, idx_ex in enumerate(example_indices[element_idx][:n_icl]):
            print('\n-----------Found match', trn_inputs[idx_ex])
            
            query = trn_inputs[idx_ex].split('\n')[0]
            abs = ' '.join(trn_inputs[idx_ex].split('\n')[1:])
            
            inp_query = f'''{pre1} {idx_idx+1}: {query}'''.replace('\n', '')
            
            inp_abs = f'''{pre2} {idx_idx+1}: {abs}'''.replace('\n', '')
            
            tgt_ex = f'''{suf} {idx_idx+1}: {trn_target[idx_ex]}'''.replace('\n', '')
            
            new_element += f'''{inp_query}?\n{inp_abs}\n{tgt_ex}\n##\n'''

    # append formatted input sample (i.e. prompt)
    return new_element,idx_idx

def prep(example,idx):
    instr = '''You are a medical expert. You are given a query and query-relevant information as inputs. Your task is to summarize this information. The summary should be concise, include only non-redundant, query-relevant evidence, and be approximately 100 words long. Use the provided examples to guide word choice.'''
    prefix1 = "Query"
    prefix2 = "Document"
    suffix = "Summary"
    
    icl_part,_ = prepend_icl_examples(example,idx,prefix1,prefix2,suffix)
    
    query = example.split('\n')[0]
    abs = ' '.join(example.split('\n')[1:])
    
    str_ = f'''{instr}\n\n{icl_part.strip()}\n{prefix1}: {query}?\n{prefix2}: {abs}\n{suffix}:'''

    return str_

df_test = pd.read_csv('../../../../../../Medical/PushToNeumann/CSV-Datasets/EBM-Test.csv')
srcs = df_test['input_text'].tolist()
tgts = df_test['target_text'].tolist()

icl_data = []
for idx,src in enumerate(srcs):
    icl_data.append(prep(src,idx))

df = pd.DataFrame({'icl_input':icl_data,'target':tgts})
df.to_csv(f'./ICL-Data/{data}_{n_icl}_ICL.csv',index=False)
#!/usr/bin/env python
import argparse
from transformers import AutoTokenizer
import json
import pandas as pd
import os

parser = argparse.ArgumentParser() 
parser.add_argument('-dataset',type=str,required=True) 
parser.add_argument('-v_size',type=int,required=True)
parser.add_argument('-vpath',type=str,required=True)
parser.add_argument('-PAC_path',type=str,required=True)
parser.add_argument('-TGT_path',type=str,required=True) 
args = parser.parse_args()

print(f'\n------------------------------\nStarting for {args.dataset} {args.v_size//1000}K...')

tok = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
org_vocab = tok.get_vocab()

list_PM_All = list()
vocab_PAC = json.load(open(args.PAC_path))
for term,idx in vocab_PAC.items():
        if idx < 256: continue
        if term in org_vocab: continue
        list_PM_All.append(term)

print('V_PAC not in V_PLM',len(list_PM_All))

PAC_topK = list_PM_All[:args.v_size]

list_TGT_notPLM = list()
vocab_TGT = json.load(open(args.TGT_path))
for term,idx in vocab_TGT.items():
        if idx < 256: continue
        if term in org_vocab: continue
        else: list_TGT_notPLM.append(term)

print('V_TGT not in V_PLM:',len(list_TGT_notPLM))

list_filtered = list()
list_consider = list()
list_TGT_inPAC = list()
for idx,line in enumerate(list_TGT_notPLM):
    term = line
    if args.v_size == 0:
        list_TGT_inPAC.append(line)
        continue

    if term in PAC_topK: 
        list_TGT_inPAC.append(line)
        list_consider.append(idx)

    else: list_filtered.append(idx)
 
print('V_TGT not in V_PLM and in V_PAC:',len(list_TGT_inPAC))

V_TGT = list_TGT_inPAC
FINAL_Vocab = V_TGT

import re
pattern = r"^[A-Za-zĠ]+$"
del_idx = list()
for idx,v in enumerate(FINAL_Vocab):
    if re.match(pattern,v.strip().split()[0]): 
        continue
    else: del_idx.append(idx)

for idx in del_idx[::-1]: del FINAL_Vocab[idx]

print('Final_Vocab:', len(FINAL_Vocab))

if not os.path.exists(f'{args.vpath}'): os.makedirs(f'{args.vpath}')

dump_path = f'{args.vpath}/{args.v_size//1000}K.txt'

with open(dump_path,'w',encoding='utf-8') as f:
    f.write('\n'.join(FINAL_Vocab))
f.close()

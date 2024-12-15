#!/usr/bin/env python
import argparse
from transformers import LlamaTokenizer
import sentencepiece as spm
import pandas as pd
import os

parser = argparse.ArgumentParser() 
parser.add_argument('-dataset',type=str,required=True) 
parser.add_argument('-v_size',type=int,required=True)
parser.add_argument('-vpath',type=str,required=True)
parser.add_argument('-PAC_path',type=str,required=True) 
args = parser.parse_args()

print(f'\n------------------------------\nStarting for {args.dataset} {args.v_size}K...')

tok = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
org_vocab = tok.get_vocab()

list_PM_All = list()
with open(f'{args.PAC_path}','r') as f:
    for idx,line in enumerate(f):
        if idx < 3: continue
        term = line.split()[0]
        if term in org_vocab: continue
        list_PM_All.append(line)

print('V_PAC not in V_PLM',len(list_PM_All))

PAC_topK = [x.split()[0] for x in list_PM_All]

list_TGT_notPLM = list()
with open(f'VocabFiles_Llama2/{args.dataset}_All.vocab','r') as f:
    for idx,line in enumerate(f):
        if idx < 3: continue
        term = line.split()[0]
        if term in org_vocab: continue
        else: list_TGT_notPLM.append(line)
f.close()

print('V_TGT not in V_PLM:',len(list_TGT_notPLM))

list_filtered = list()
list_consider = list()
list_TGT_inPAC = list()
for idx,line in enumerate(list_TGT_notPLM):
    term = line.split()[0]
    if args.v_size == 0:
        list_TGT_inPAC.append(line)
        continue

    if term in PAC_topK: 
        list_TGT_inPAC.append(line)
        list_consider.append(idx)

    else: list_filtered.append(idx)

# if not os.path.exists('Position_Dist'): os.makedirs('Position_Dist')
# with open(f'Position_Dist/Consider_{args.v_size//1000}K_{args.dataset}.txt','w') as f:
#     f.write('\n'.join([str(x) for x in list_consider]))
# f.close()

# with open(f'Position_Dist/Filtered_{args.v_size//1000}K_{args.dataset}.txt','w') as f:
#     f.write('\n'.join([str(x) for x in list_filtered]))
# f.close()
 
print('V_TGT not in V_PLM and in V_PAC:',len(list_TGT_inPAC))

V_TGT = list_TGT_inPAC
FINAL_Vocab = V_TGT

import re
pattern = r"^[A-Za-z▁]+$"
del_idx = list()
for idx,v in enumerate(FINAL_Vocab):
    if re.match(pattern,v.strip().split()[0]): 
        continue
    else: del_idx.append(idx)

for idx in del_idx[::-1]: del FINAL_Vocab[idx]

print('Final_Vocab:', len(FINAL_Vocab))

if not os.path.exists(f'{args.vpath}'): os.makedirs(f'{args.vpath}')

dump_path = f'{args.vpath}/{args.v_size}K.txt'

with open(dump_path,'w') as f:
    f.write(''.join(FINAL_Vocab))
f.close()

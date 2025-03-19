#!/usr/bin/env python
import argparse
from transformers import AutoTokenizer
import json
import pandas as pd
import os

parser = argparse.ArgumentParser() 
parser.add_argument('-dataset',type=str,required=True) 
parser.add_argument('-v_size',type=int,required=True)
parser.add_argument('-frac',type=float,required=True) 
parser.add_argument('-vpath',type=str,required=True)
parser.add_argument('-PAC_path',type=str,required=True)
parser.add_argument('-TGT_path',type=str,required=True) 
args = parser.parse_args()

print(f'\n------------------------------\nStarting for {args.dataset} {args.v_size//1000}K-{args.frac}...')

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

if not os.path.exists('Position_Dist'): os.makedirs('Position_Dist')
with open(f'Position_Dist/Consider_{args.v_size//1000}K_{args.dataset}.txt','w') as f:
    f.write('\n'.join([str(x) for x in list_consider]))
f.close()

with open(f'Position_Dist/Filtered_{args.v_size//1000}K_{args.dataset}.txt','w') as f:
    f.write('\n'.join([str(x) for x in list_filtered]))
f.close()
 
print('V_TGT not in V_PLM and in V_PAC:',len(list_TGT_inPAC))

V_TGT = list_TGT_inPAC

def get_Union_PM(l1,l2,frac):
    ret_list = [x for x in l1]
    ret_list_keys = [x.split()[0] for x in ret_list]
    new_list = list()
    
    v_size = int(frac*len(ret_list))
    
    added = 0
    for row in l2:
        if added >= v_size: break
        
        if row.split()[0]  not in ret_list_keys:
            if row.split()[0] not in org_vocab:
                new_list.append(row)
                added +=1
    
    print('Size of added V_PAC:',len(new_list))
    return ret_list+new_list

if args.frac > 0: FINAL_Vocab = get_Union_PM(V_TGT,list_PM_All,args.frac)
else: FINAL_Vocab = V_TGT
print('Final_Vocab:', len(FINAL_Vocab))

if not os.path.exists(f'{args.vpath}'): os.makedirs(f'{args.vpath}')

dump_path = f'{args.vpath}/{args.v_size//1000}K_{args.frac}_.txt'

with open(dump_path,'w',encoding='utf-8') as f:
    f.write('\n'.join(FINAL_Vocab))
f.close()

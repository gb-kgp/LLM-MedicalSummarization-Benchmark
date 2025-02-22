import argparse
from quickumls import QuickUMLS
import pandas as pd
import glob

from transformers import LlamaTokenizer
import sentencepiece as spm
import pandas as pd
import os

parser = argparse.ArgumentParser() 
parser.add_argument('-umls_path',type=str,required=True) 
parser.add_argument('-src_path',type=int,required=True)
parser.add_argument('-save_name',type=str,required=True)
parser.add_argument('-verbose',type=str,required=True) 
args = parser.parse_args()


umls_path = args.umls_path
matcher = QuickUMLS(umls_path,similarity_name='cosine',threshold=0.95) 

from collections import defaultdict
counter_PAC = defaultdict(int)


print(f'Starting for {args.src_path}.....')
lines_PAC = open(args.src_path).readlines()

for idx,abst in enumerate(lines_PAC):
    abst = abst.strip()
    if args.verbose:
        if idx%100 == 0: print(f'Processed till {idx+1}... {len(counter_PAC)} are considered till now.')

    flag = 0
    d = matcher.match(abst, best_match=True, ignore_syntax=False)
    if len(d) == 0:
        continue
    
    else:
        for l in d:
            counter_PAC[l[0]['ngram']] += 1
            
if args.verbose:
    if idx%100 == 0: print(f'Processed till {idx+1}... {len(counter_PAC)} are considered till now.')

import pandas as pd

list_tokens = list(counter_PAC.keys())
val_tokens = list(counter_PAC.values())

df = pd.DataFrame({'Tokens':list_tokens,'Count':val_tokens})
df.to_csv(f'{args.save_path}',index=False)
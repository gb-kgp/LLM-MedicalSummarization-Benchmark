import os,glob
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"

from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
from tokenization_llama_Lookup import LlamaTokenizer
import argparse
from transformers import AutoTokenizer
from multiprocess import Pool

parser = argparse.ArgumentParser()
parser.add_argument('-dataset',type=str,required=True)
parser.add_argument('-input_path',type=str,required=True) 
parser.add_argument('-dir_dump',type=str,required=True)
parser.add_argument('-message',type=str)
parser.add_argument('-consider',type=int,default=0)
args = parser.parse_args()

import pandas as pd
df = pd.read_csv(args.input_path)
if args.consider ==1: df = df[df['Consider']==1]
freq_ebm = df['Count'].to_list()
terms_EBM = df['Word'].to_list()
split_llama2 = df['Splits'].to_list()

sum_num = 0.
sum_den = 0.
for idx,term in enumerate(terms_EBM):
    sum_num += split_llama2[idx]*freq_ebm[idx]
    sum_den += freq_ebm[idx]

old_score = sum_num/sum_den


import glob
from collections import defaultdict

dict_scores = defaultdict(lambda : defaultdict(dict))
def checkFragment(fname):
    list_tokenized = []
    tokenizer = AutoTokenizer.from_pretrained(fname)
    tokenizer.save_pretrained(fname)
    
    added_vocab_tokens = [k for k,v in tokenizer.get_vocab().items() if v>=32000]
    
    with open(f'{fname}/added_vocab.txt','w') as f:
        for token in added_vocab_tokens:
            f.write(token.strip()+'\n')
    f.close()
    
    domain_tok = LlamaTokenizer.from_pretrained(fname,added_vocab = f'{fname}/added_vocab.txt', use_fast=False)
    sum_num = 0.
    sum_den = 0.
    for idx,term in enumerate(terms_EBM):
        split = len(domain_tok.tokenize(term))
        sum_num += split*freq_ebm[idx]
        sum_den += freq_ebm[idx]
        list_tokenized.append(split)

    key = fname.split('/')[-1]
    dict_scores[key] = [round(sum_num/sum_den,2),len(domain_tok)-32000]
    
    df['Updated_Splits'] = list_tokenized
    
    df.to_csv(f'{fname}/UpdatedSplits.csv',index=False)
    return f'Done with {fname}.... Frag Score:{round(sum_num/sum_den,2)}'


pool = Pool(4)
a_list=[]
for fname in glob.glob(f'{args.dir_dump}/*'):
    a_list.append(fname)
for d in pool.imap_unordered(checkFragment, a_list):
    print(d)

pool.close()
pool.join()

# with open(f'{args.dataset}-FERTILITY','a') as f:
#     f.write(f'-------------\n{args.message}\n--------------------\n')
#     f.write('llama2_Tok: '+str(round(old_score,2))+'\n')

#     for k1 in dict_scores:
#         f.write(k1+'\t')
#         f.write(f'{dict_scores[k1][0]}/{dict_scores[k1][1]}\t')
#         f.write('\n')
# f.close()



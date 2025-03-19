import os,glob
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"

from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
from transformers import LlamaTokenizer
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-dataset',type=str,required=True)
parser.add_argument('-input_path',type=str,required=True) 
parser.add_argument('-vocab_path',type=str,required=True)
parser.add_argument('-dir_dump',type=str,required=True)
parser.add_argument('-message',type=str)
parser.add_argument('-consider',type=int,default=0)
args = parser.parse_args()

if not os.path.exists(args.dir_dump): os.mkdir(args.dir_dump)

for fname in glob.glob(f'{args.vocab_path}/*'):
    llama_tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    llama_spm = sp_pb2_model.ModelProto()
    llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())

    dir_name = f'./{args.dir_dump}/{fname.split("/")[-1][:-4]}'
    
    if not os.path.exists(dir_name): os.mkdir(dir_name)
    
    special_tokens = open(fname, "r").read().split("\n")[:-1]
    for idx,row in enumerate(special_tokens):
        token, score = row.split()
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = token
        new_p.score = 0. 
        llama_spm.pieces.append(new_p)

    with open(f'{dir_name}/tokenizer.model', 'wb') as f:
        f.write(llama_spm.SerializeToString())
    f.close()
    
    tokenizer = LlamaTokenizer(vocab_file=dir_name+'/tokenizer.model')
    tokenizer.save_pretrained(dir_name)

    del llama_tokenizer, tokenizer, llama_spm

import pandas as pd
df = pd.read_csv(args.input_path)
if args.consider ==1: df = df[df['Consider']==1]
freq_ebm = df['Count'].to_list()
terms_EBM = df['Word'].to_list()
split_bart = df['Splits'].to_list()

sum_num = 0.
sum_den = 0.
for idx,term in enumerate(terms_EBM):
    sum_num += split_bart[idx]*freq_ebm[idx]
    sum_den += freq_ebm[idx]

old_score = sum_num/sum_den


import glob
from collections import defaultdict

dict_scores = defaultdict(lambda : defaultdict(dict))
for fname in sorted(glob.glob(f'{args.dir_dump}/*') ,key = lambda x: [int(x.split('/')[-1].split('_')[-3][:-1]),float(x.split('/')[-1].split('_')[-2])]):
    domain_tok = LlamaTokenizer.from_pretrained(fname)
    sum_num = 0.
    sum_den = 0.
    
    for idx,term in enumerate(terms_EBM):
        sum_num += min(len(domain_tok.tokenize(term)),len(domain_tok.tokenize(' '+term)))*freq_ebm[idx]
        sum_den += freq_ebm[idx]

    key = fname.split('/')[-1].split('_')
    dict_scores[key[-3]][key[-2]] = [round(sum_num/sum_den,2),len(domain_tok)-32000]

with open(f'{args.dataset}-FERTILITY','a') as f:
    f.write(f'-------------\n{args.message}\n--------------------\n')
    f.write('BART_Tok: '+str(round(old_score,2))+'\n')
    for k1 in dict_scores:
        if k1 == '0K': continue
        f.write('data\t')
        for k2 in dict_scores[k1]:
            f.write(k2+'\t')
        f.write('\n')
        break

    for k1 in dict_scores:
        f.write(k1+'\t')
        for k2 in dict_scores[k1]:
            f.write(f'{dict_scores[k1][k2][0]}/{dict_scores[k1][k2][1]}\t')
        f.write('\n')
f.close()



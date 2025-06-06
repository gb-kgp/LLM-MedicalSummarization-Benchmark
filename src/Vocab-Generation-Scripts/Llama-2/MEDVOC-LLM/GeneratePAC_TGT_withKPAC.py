import argparse
import sentencepiece as spm
import pandas as pd
import os


if not os.path.exists('VocabFiles_Llama2'): os.makedirs('VocabFiles_Llama2')

parser = argparse.ArgumentParser()
parser.add_argument('-input_path',type=str,required=True)
parser.add_argument('-v_size',type=int,required=True) 
parser.add_argument('-dataset',type=str,required=True) 

args = parser.parse_args()
print(f'\n------------------------------\nStarting for {args.dataset}...')

spm.SentencePieceTrainer.train(f'--input={args.input_path} --model_prefix=VocabFiles_Llama2/{args.dataset}_{args.v_size//1000}K_All --vocab_size={args.v_size} --model_type=bpe --byte_fallback')
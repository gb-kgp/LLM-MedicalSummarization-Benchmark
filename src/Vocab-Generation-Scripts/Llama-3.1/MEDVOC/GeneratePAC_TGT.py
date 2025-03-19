import argparse
import sentencepiece as spm
import pandas as pd
import os

from tokenizers import ByteLevelBPETokenizer


if not os.path.exists('VocabFiles_Llama3'): os.makedirs('VocabFiles_Llama3')

parser = argparse.ArgumentParser()
parser.add_argument('-input_path',type=str,required=True)
parser.add_argument('-v_size',type=int,required=True) 
parser.add_argument('-dataset',type=str,required=True) 

args = parser.parse_args()
print(f'\n------------------------------\nStarting for {args.dataset}...')

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(args.input_path,vocab_size=256000,show_progress=True)

if not os.path.exists(f'VocabFiles_Llama3/{args.dataset}'): 
    os.mkdir(f'VocabFiles_Llama3/{args.dataset}')
    
tokenizer.save_model(f'VocabFiles_Llama3/{args.dataset}')
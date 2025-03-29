#!/bin/bash
<<COMM
This script generates the vocabularies for the MEDVOC dataset using the Llama2 model. 
You need to prepare the dataset first. The format should be .txt file with one document per line.
The dataset for PAC should be the Source documents and for TGT should be the reference summaries for target dataset. 

One thing you need to try out is the value of v_size for TGT dataset. Here it is set to 5000, which might throw an error. However, the error message will show the desired vocabulary size, please set v_size to that.
For PAC this value is set to nearest 5000s compared to the original vocabulary size (32000 for Llama2).
COMM

python GeneratePAC_TGT.py --input_path /path/to/EBM/Dataset.txt \
                          --v_size 5000 \
                          --dataset EBM

python GeneratePAC_TGT.py --input_path /path/to/PAC/Dataset.txt \
                          --v_size 35000 \
                          --dataset PAC




for data in EBM
do
  for v_size in 5000 10000 15000 20000 25000 30000
  do
    for frac in 0 0.25 0.5 0.75 1
    do
      python GenerateSubwords.py -v_size $v_size \
                                 -dataset $data \
                                 -frac $frac \
                                 -vpath $data\
                                 -PAC_path VocabFiles_Llama2/PAC_All.vocab
    done
  done

  python GenerateSubwords.py -v_size 0 \
                             -dataset $data \
                             -frac 0 \
                             -vpath $data\
                             -PAC_path VocabFiles_Llama2/PAC_All.vocab

  python UpdateVocab.py -dataset $data \
                        -input_path ./"$data"_OOV.csv \
                        -vocab_path $data \
                        -dir_dump ./"$data"_Vocabs_MEDVOC_Llama2 \
                        -message "All_OOV" \
                          
done
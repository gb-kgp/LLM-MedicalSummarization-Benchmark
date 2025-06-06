#!/bin/bash

for data in EBM
do
  for v_size in 5 10 15 20 25 30
  do
      python GenerateSubwords_WithKPAC.py -v_size $v_size \
                                 -dataset $data \
                                 -vpath "$data"_KPAC \
                                 -PAC_path VocabFiles_Llama2/PAC_"$v_size"K_All.vocab
  done

  python UpdateVocab-KPAC.py -dataset $data \
                        -input_path ./"$data"_OOV.csv \
                        -vocab_path "$data"_KPAC \
                        -dir_dump ./"$data"_Filter_Vocabs_Llama2 \
                        -message "All_OOV_KPAC" 
done
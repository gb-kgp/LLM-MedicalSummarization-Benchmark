#!/bin/bash

for data in EBM
do
  for v_size in 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000 65000 70000 75000 80000 85000 90000 95000 100000 105000 110000 115000 120000 125000 130000
  do
      python GenerateSubwords.py -v_size $v_size \
                                 -dataset $data \
                                 -vpath $data\
                                 -PAC_path ./VocabFiles_Llama3/PAC/vocab.json \
                                 -TGT_path ./VocabFiles_Llama3/EBM/vocab.json
  done

  python GenerateSubwords.py -v_size 0 \
                             -dataset $data \
                             -vpath $data\
                             -PAC_path ./VocabFiles_Llama3/PAC/vocab.json \
                             -TGT_path ./VocabFiles_Llama3/EBM/vocab.json
  
  # python UpdateVocab.py -dataset $data \
  #                       -input_path ./"$data"_OOV.csv \
  #                       -vocab_path $data \
  #                       -dir_dump ./"$data"_Vocabs_MEDVOC_Llama2 \
  #                       -message "All_OOV" \
                          

done
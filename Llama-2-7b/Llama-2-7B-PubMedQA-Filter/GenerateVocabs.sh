#!/bin/bash

for data in PubMedQA
do
  for v_size in 5 10 15 20 25 30
  do
      python GenerateSubwords_WithKPAC.py -v_size $v_size \
                                 -dataset $data \
                                 -vpath "$data"_Filter \
                                 -PAC_path VocabFiles_Llama2/PAC_"$v_size"K_All.vocab
  done

  python UpdateVocab-KPAC.py -dataset $data \
                        -input_path ../Llama-2-7B-PubMedQA-MedicalLookup-Fragment/PubMedQA_SplitMoreThan1_OOV.csv \
                        -vocab_path "$data"_Filter \
                        -dir_dump ./"$data"_Filter_Vocabs_Llama2 \
                        -message "All_OOV_Filter" 
done
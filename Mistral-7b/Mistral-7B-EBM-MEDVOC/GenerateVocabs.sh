#!/bin/bash

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
                                 -PAC_path VocabFiles_Mistral/PAC_All.vocab
    done
  done

  python GenerateSubwords.py -v_size 0 \
                             -dataset $data \
                             -frac 0 \
                             -vpath $data\
                             -PAC_path VocabFiles_Mistral/PAC_All.vocab

  python UpdateVocab.py -dataset $data \
                        -input_path /Users/gunjanbalde/Documents/SR-NG-MedVoc/SR-NG-MedVoc/Expert-Domain/Codebase_MTP_Ayush_Shakya/Codebase_Medical/Clin-Summ/Vocab-Adapt/ACL-ARR-Expts/Mistral-7b/Mistral-7B-EBM-MedicalLookup-Fragment/EBM_SplitMoreThan1_OOV.csv \
                        -vocab_path $data \
                        -dir_dump ./"$data"_Vocabs_MEDVOC_Mistral \
                        -message "All_OOV" \
                          

done
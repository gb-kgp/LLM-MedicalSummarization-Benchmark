#!/bin/bash

for data in EBM
do
  for split in SplitMoreThan1
  do
    python UpdateVocab-Lookup.py -dataset $data \
                               -input_path ./"$data"_SplitMoreThan1_OOV.csv \
                               -vocab_path EBM_Lookup_SplitMoreThan1_Also_Add_As_Subwords \
                               -dir_dump EBM_Lookup_SplitMoreThan1_Also_Add_As_Subwords_Mistral \
                               -message "EBM_Lookup_SplitMoreThan1_Also_Add_As_Subwords" 

    python CheckFertilityWithAdaptBPE.py -dataset $data \
                               -input_path ./"$data"_SplitMoreThan1_OOV.csv \
                               -dir_dump EBM_Lookup_SplitMoreThan1_Also_Add_As_Subwords_Mistral \
                               -message "EBM_Lookup-$split" 
  done
done
#!/bin/bash

for data in PubMedQA
do
  for split in SplitMoreThan1
  do
    python UpdateVocab-Lookup.py -dataset $data \
                               -input_path ./"$data"_SplitMoreThan1_OOV.csv \
                               -vocab_path "$data"_Lookup_SplitMoreThan1_Also_Add_As_Subwords \
                               -dir_dump "$data"_Lookup_SplitMoreThan1_Also_Add_As_Subwords_Mistral \
                               -message "$data-Lookup_SplitMoreThan1_Also_Add_As_Subwords" 

    python CheckFertilityWithAdaptBPE.py -dataset $data \
                               -input_path ./"$data"_SplitMoreThan1_OOV.csv \
                               -dir_dump "$data"_Lookup_SplitMoreThan1_Also_Add_As_Subwords_Mistral \
                               -message "$data-Lookup_SplitMoreThan1_Also_Add_As_Subwords-AdaptBPE" 
  done
done
# LLM-MedicalSummarization-Benchmark

This is the official codebase for the ACL Submission.

## Models covererd in the study
We cover three models in this study
| Model     | Base Vocabulary | huggingfcae-model-id                                                          | Tokenization Class | #Params |
|-----------|-----------------|-------------------------------------------------------------------------------|--------------------|---------|
| Llama-3.1 | 128256          | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)     | TikToken-BPE       | 8B      |
| Llama-2   | 32000           | [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)   | SentencePiece-BPE  | 7B      |
| Mistral   | 32000           | [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) | SentencePiece-BPE  | 7B      |


## Data Description

We use three datasets in this study: EBM, PubMedQA, and BioASQ.

| Dataset | Test Set  Size | Token Count of Reference Summaries | Token Count of Reference Summaries | OOV Concentration Split more than once (in \%) | OOV Concentration Split more than once (in \%) | OOV Concentration Split more than once (in \%) | OOV Concentration Split more than once (in \%) | OOV Concentration Split more than thrice (in \%) | OOV Concentration Split more than thrice (in \%) | OOV Concentration Split more than thrice (in \%) | OOV Concentration Split more than thrice (in \%) | Unigram Novelty (in \%) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|  |  | Llama-2 | Llama-3.1 | Llama-2 | Llama-2 | Llama-3.1 | Llama-3.1 | Llama-2 | Llama-2 | Llama-3.1 | Llama-3.1 |  |
|  |  |  |  | SD | RS | SD | RS | SD | RS | SD | RS |  |
| PubMedQA | 500 | 63 | 25 | 36.67 | 38.00 | 43.68 | 45.65 | 4.91 | 4.65 | 2.61 | 2.42 | 41.32 |
| EBM | 424 | 112 | 91 | 38.97 | 40.90 | 45.60 | 46.23 | 6.65 | 7.92 | 3.90 | 5.17 | 47.15 |
| BioASQ-$\mathcal{M}$ | 963 | 85 | 69 | 46.20 | 50.64 | 52.03 | 56.61 | 9.12 | 11.04 | 5.55 | 7.09 | 42.58 |
| BioASQ-$\mathcal{S}$ | 496 | 73 | 58 | 47.12 | 50.00 | 52.00 | 57.15 | 8.70 | 9.10 | 4.76 | 4.55 | 4.11 |

## Generating Vocabulary
We provide a sample shell script for generating MEDVOC vocabulary for Llama-2 using EBM dataset.

```
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
```
The output will be a text file EBM-FERTILITY which would have the fragment score matrix. You can pick any setting of your choice. In the paper, we report for 25K_0.5, following the MEDVOC hyperparam search strategy.
```
-------------
All_OOV
--------------------
BART_Tok: 2.65
da	0.0     	0.25    	0.5     	0.75    	1.0	
0K	1.23/24966	
5K	1.87/2789	1.84/3486	1.83/4183	1.82/4880	1.79/5578	
10K	1.73/4441	1.69/5551	1.67/6661	1.66/7771	1.64/8882	
15K	1.65/5562	1.6/6952	1.58/8343	1.56/9733	1.55/11124	
20K	1.61/6480	1.56/8100	1.53/9720	1.51/11340	1.51/12960	
25K	1.56/7224	1.5/9030	1.48/10836	1.47/12642	1.46/14448	
30K	1.54/7817	1.49/9771	1.46/11725	1.44/13679	1.44/15634	
-------------
```

## Training Vocabulary Adaptation Strategy

We use three vocabulary adaptation strategies --MEDVOC, MEDVOC-LLM, and ScafFix. Each of these strategies are trained using two Continual Pretraining stratagies --End-to-End and 2Stage. We provide an example script to run a vocabulary adaptation strategy using a CPT strategy (Eg., MEDVOC using End-to-End for Llama3): 

```
      python train.py --dataset_path /path/to/PAC/dataset.txt \
                      --val_dataset_path /path/to/VAL/dataset.txt \
                      --output_dir /path/to/output/dir \
                      --logging_dir /path/to/output/dir \
                      --tokenizer_name_or_path /path/to/MEDVOC/Tokenizer  \
                      --model_name_or_path meta-llama/Llama-3.1-8B \
                      --model_type llama3 \
                      --seed 42 \
                      --eval_strategy steps \
                      --logging_steps 5 \
                      --learning_rate 1e-4 \
                      --weight_decay 0.01 \
                      --warmup_ratio 0.05 \
                      --num_train_epochs 3 \
                      --per_device_train_batch_size 8 \
                      --per_device_eval_batch_size 8 \
                      --prediction_loss_only \
                      --overwrite_output_dir \
                      --optim adamw_bnb_8bit \
                      --do_train \
                      --do_eval \
                      --lr_scheduler_type cosine \
                      --disable_tqdm False \
                      --label_names labels \
                      --remove_unused_columns False \
                      --save_strategy steps \
                      --save_steps 100 \
                      --bf16 \
                      --tf32 True \
                      --gradient_checkpointing True \
                      --tune_embeddings \
                      --eval_steps 100 \
                      --ddp_find_unused_parameters True \
                      --r 32 \
                      --lora_alpha 64 \
                      --gradient_accumulation_steps 4 \
                      --save_total_limit 1 \
                      --load_best_model_at_end True \
                      --metric_for_best_model eval_loss \
                      --greater_is_better False \
                      --save_only_model True

```

## Baseline Strategy [CPT-Only]
We use a CPT-Only baseline to understand the efficacy of only continual pretraining without any vocabulary adaptation. To run the baseline you can use the following python call:

```
python train_without_Vocab.py --dataset_path /path/to/PAC/dataset \
                                  --val_dataset_path /path/to/val/dataset \
                                  --output_dir ./Llama-2-7B_PAC_WithoutVocab_32_CPT_ACLFeb_Final/ \
                                  --logging_dir ./Llama-2-7B_PAC_WithoutVocab_32_CPT_ACLFeb_Final/ \
                                  --tokenizer_name_or_path meta-llama/Llama-2-7B  \
                                  --model_name_or_path meta-llama/Llama-2-7B \
                                  --model_type mistral \
                                  --seed 42 \
                                  --eval_strategy steps \
                                  --logging_steps 5 \
                                  --learning_rate 1e-4 \
                                  --weight_decay 0.01 \
                                  --warmup_ratio 0.05 \
                                  --num_train_epochs 3 \
                                  --per_device_train_batch_size 8 \
                                  --per_device_eval_batch_size 8 \
                                  --prediction_loss_only \
                                  --overwrite_output_dir \
                                  --optim adamw_bnb_8bit \
                                  --do_train \
                                  --do_eval \
                                  --lr_scheduler_type cosine \
                                  --disable_tqdm False \
                                  --label_names labels \
                                  --remove_unused_columns False \
                                  --save_strategy steps \
                                  --save_steps 100 \
                                  --bf16 \
                                  --tf32 True \
                                  --gradient_checkpointing True \
                                  --tune_embeddings \
                                  --eval_steps 100 \
                                  --ddp_find_unused_parameters True \
                                  --r 32 \
                                  --lora_alpha 64 \
                                  --gradient_accumulation_steps 4 \
                                  --save_total_limit 1 \
                                  --load_best_model_at_end True \
                                  --metric_for_best_model eval_loss \
                                  --greater_is_better False \
                                  --save_only_model True

```

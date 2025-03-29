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

## Vocabulary Adaptation Strategy

We use three vocabulary adaptation strategies --MEDVOC, MEDVOC-LLM, and ScafFix. Each of these strategies are trained using two Continual Pretraining stratagies --End-to-End and 2Stage. We provide an example script to run a vocabulary adaptation strategy using a CPT strategy (Eg., MEDVOC using End-to-End for Llama3): 

```
python3 train_model.py  --dataset_path /path/to/cpt-dataset \
                        --output_dir /path/to/output-dir \
                        --logging_dir /path/to/output-dir \
                        --tokenizer_name_or_path /path/to/MEDVOC_Files/ \
                        --model_type llama3 \
                        --seed 42 \
                        --evaluation_strategy steps\
                        --logging_steps 5 \
                        --learning_rate 1e-4 \
                        --weight_decay 0.01 \
                        --warmup_ratio 0.05 \
                        --num_train_epochs 3 \
                        --per_device_train_batch_size 32 \
                        --prediction_loss_only \
                        --overwrite_output_dir \
                        --optim adamw_bnb_8bit \
                        --do_train \
                        --lr_scheduler_type cosine \
                        --disable_tqdm False \
                        --label_names labels \
                        --remove_unused_columns False \
                        --save_strategy steps \
                        --bf16 \
                        --tf32 True \
                        --gradient_checkpointing True \
                        --tune_embeddings \
                        --model_name_or_path \

```

## Baseline Strategy [CPT-Only]
We use a CPT-Only baseline to understand the efficacy of only continual pretraining without any vocabulary adaptation. To run the baseline you can use the following python call:

```
python3 train_cpt-only.py  --dataset_path /path/to/cpt-dataset \
                        --output_dir /path/to/output-dir \
                        --logging_dir /path/to/output-dir \
                        --model_type llama3 \
                        --seed 42 \
                        --evaluation_strategy steps\
                        --logging_steps 5 \
                        --learning_rate 1e-4 \
                        --weight_decay 0.01 \
                        --warmup_ratio 0.05 \
                        --num_train_epochs 3 \
                        --per_device_train_batch_size 32 \
                        --prediction_loss_only \
                        --overwrite_output_dir \
                        --optim adamw_bnb_8bit \
                        --do_train \
                        --lr_scheduler_type cosine \
                        --disable_tqdm False \
                        --label_names labels \
                        --remove_unused_columns False \
                        --save_strategy steps \
                        --bf16 \
                        --tf32 True \
                        --gradient_checkpointing True \
                        --tune_embeddings \
                        --model_name_or_path \

```

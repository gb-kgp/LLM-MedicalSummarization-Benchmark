import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import datasets
import torch
import torch.nn as nn
from peft import (LoraConfig, TaskType, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from config import CustomArgumentParser

import json
import math
import numpy as np

def group_texts(examples: dict, block_size=512, eos_id=2):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    else:
        # add padding
        for k, v in concatenated_examples.items():
            concatenated_examples[k] = v + [eos_id] * (block_size - len(v))
        total_length = block_size
    
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()    
    return result

def tokenize_function(examples,tokenizer):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)


def main(args, training_args):
    model_id = args.model_name_or_path
    hf_token = args.hf_token

    logger.info(f"Device: {torch.cuda.device_count()}")
    # Load the dataset
    train_dataset = datasets.load_dataset("text",data_files={"train":args.dataset_path},split='train',cache_dir="./")
    #train_dataset = datasets.load_from_disk(args.dataset_path)
    #print("train dataset: ",train_dataset)
    #exit()
    if args.val_dataset_path is not None:
        val_dataset = datasets.load_dataset("text",data_files={"validation":args.val_dataset_path},split='validation',cache_dir="./")
    else:
        val_dataset = None

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    # source_tokenizer = AutoTokenizer.from_pretrained(model_id,token=hf_token)
    
    # Set up the data collator
    tokenizer.pad_token = tokenizer.eos_token
    
    #Tokenize Train set
    train_tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names, fn_kwargs={"tokenizer":tokenizer},num_proc=16)
    train_tokenized_dataset = train_tokenized_dataset.map(
        lambda examples: group_texts(examples, 512, tokenizer.eos_token_id),
        batched=True, 
        num_proc=16
        )

    #Tokenize Train set
    val_tokenized_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names, fn_kwargs={"tokenizer":tokenizer},num_proc=16)
    val_tokenized_dataset = val_tokenized_dataset.map(
        lambda examples: group_texts(examples, 512, tokenizer.eos_token_id),
        batched=True, 
        num_proc=16
        )

    #Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Load the model
    if args.freeze_model:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            #torch_dtype=torch.bfloat16,
            #device_map='sequential',#'auto',
            #max_memory=max_memory,
            token=hf_token
        )        
        model.resize_token_embeddings(len(tokenizer))

    else:
        print("Inside weight training...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token
        )
        
        # model,tokenizer = instantiate_model_by_mean(model, source_tokenizer, tokenizer, False)
        model = prepare_model_for_kbit_training(model)
        
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)
    #model.lm_head = CastOutputToFloat(model.lm_head)
    logger.info(model)

    # Set up LoRA
    if not args.no_lora:
        logger.info(f'Before PEFT applied (Memory): {model.get_memory_footprint()}')
        
        if args.model_type in ("llama3", "mistral"):
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
        else:
            raise ValueError(f"Model type {args.model_type} not supported.")
        
        if args.tune_embeddings:
            if args.model_type in ("llama3", "mistral"):
                modules_to_save = ["lm_head", "embed_tokens"]
            else:
                raise ValueError(f"Model type {args.model_type} not supported.")
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                target_modules=target_modules,
                inference_mode=False, 
                r=args.r,
                lora_alpha=args.lora_alpha, 
                lora_dropout=args.lora_dropout, 
                modules_to_save=modules_to_save
            )
            model = get_peft_model(model, peft_config)

        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=args.r,
                lora_alpha=args.lora_alpha, 
                lora_dropout=args.lora_dropout, 
            )
            model = get_peft_model(model, peft_config)

        logger.info(f'After PEFT applied (Memory): {model.get_memory_footprint()}')

    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for par_name, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                print("Trainable: ",par_name)
                trainable_params += param.numel()
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
        
    print_trainable_parameters(model)
    
    # Set up the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        data_collator=data_collator,
        eval_dataset=val_tokenized_dataset,
    )

    # Train the model
    trainer.train()

    # Save the model
    # trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = CustomArgumentParser()
    args, training_args = parser.parse_args()
    logger.info(args)
    logger.info(training_args)

    main(args, training_args)

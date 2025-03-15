#!/usr/bin/env python
# coding: utf-8

# More details in this article: [Fine-Tuning Gemma 3 on Your Computer with LoRA and QLoRA (+model review)](https://kaitchup.substack.com/p/fine-tuning-gemma-3-on-your-computer)
# 
# This notebook shows how to fine-tune Gemma 3 with a single GPU. Full fine-tuning, LoRA, and QLoRA are supported.
# 
# * Gemma 3 1B and 4B can be fine-tuned with a 24 GB GPU without quantization (LoRA)
# * Gemma 3 12B can be fine-tuned with a 24 GB GPU with quantization (QLoRA)
# * Gemma 3 27B can be fine-tuned with a 40 GB GPU with quantization (QLoRA). Technically, using very short sequences and skipping the retraining of the embeddings would make QLoRA fine-tuning also possible on a 24 GB GPU.

# # Install
# 
# We need a special commit of Transformers. It will probably be pushed to the main branch soon so a simple --upgrade of Transformers might be enough to make it work, soon.

get_ipython().system('pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3 trl peft datasets accelerate bitsandbytes')


# # Fine-Tuning Code

import torch, os, multiprocessing
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, #not supported yet
    AutoTokenizer,
    Gemma3ForConditionalGeneration, #we need this
    BitsAndBytesConfig,
    set_seed
)
from trl import SFTTrainer, SFTConfig
set_seed(1234)

compute_dtype = torch.bfloat16
attn_implementation = 'eager'

def fine_tune(model_name, batch_size=1, gradient_accumulation_steps=32, LoRA=False, QLoRA=False):

  #The tokenizer has a pad token!
  #Need to force adding the bos token according to the technical report
  tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos=True)
  tokenizer.padding_side = 'right'

  #Google followed recent practices which involve not including the chat template in the tokenizer of base model...
  #Let's add it so we can fine-tune the model with the chat template.
  tokenizer.chat_template = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", add_bos=True).chat_template

  ds_train = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:30000]")
  #Add the EOS token
  def process(row):
      row["text"] = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)+tokenizer.eos_token
      return row

  ds_train = ds_train.map(
      process,
      num_proc= multiprocessing.cpu_count(),
      load_from_cache_file=False,
  )

  ds_train = ds_train.remove_columns(["messages","prompt","prompt_id"])

  print(ds_train[0])


  if QLoRA:
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
              model_name, quantization_config=bnb_config, device_map={"": 0}, attn_implementation=attn_implementation
    )
    model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={'use_reentrant':True})
  else:
    model = Gemma3ForConditionalGeneration.from_pretrained(
              model_name, device_map={"": 0}, torch_dtype=compute_dtype, attn_implementation=attn_implementation
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})

  print(model)

  if LoRA or QLoRA:
    peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=16,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"],
            modules_to_save=['embed_tokens', 'lm_head'] #because of the chat template potentially containing untrained special tokens, we need to retrain the embeddings
    )
  else:
      peft_config = None

  if LoRA:
    output_dir = "./LoRA/"
  elif QLoRA:
    output_dir = "./QLoRA/"
  else:
    output_dir = "./FFT/"

  training_arguments = SFTConfig(
          output_dir=output_dir,
          #eval_strategy="steps",
          #do_eval=True,
          optim="paged_adamw_8bit",
          per_device_train_batch_size=batch_size,
          gradient_accumulation_steps=gradient_accumulation_steps,
          #per_device_eval_batch_size=batch_size,
          log_level="debug",
          save_strategy="epoch",
          logging_steps=25,
          learning_rate=1e-5,
          bf16 = True,
          #eval_steps=25,
          num_train_epochs=1,
          warmup_ratio=0.1,
          lr_scheduler_type="linear",
          dataset_text_field="text",
          max_seq_length=1024,
          report_to="none"
  )

  trainer = SFTTrainer(
          model=model,
          train_dataset=ds_train,
          #eval_dataset=ds['test'],
          peft_config=peft_config,
          processing_class=tokenizer,
          args=training_arguments,
  )

  #--code by Unsloth: https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=pCqnaKmlO1U9

  gpu_stats = torch.cuda.get_device_properties(0)
  start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
  max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
  print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
  print(f"{start_gpu_memory} GB of memory reserved.")

  trainer_ = trainer.train()


  used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
  used_memory_for_trainer= round(used_memory - start_gpu_memory, 3)
  used_percentage = round(used_memory         /max_memory*100, 3)
  trainer_percentage = round(used_memory_for_trainer/max_memory*100, 3)
  print(f"{trainer_.metrics['train_runtime']} seconds used for training.")
  print(f"{round(trainer_.metrics['train_runtime']/60, 2)} minutes used for training.")
  print(f"Peak reserved memory = {used_memory} GB.")
  print(f"Peak reserved memory for training = {used_memory_for_trainer} GB.")
  print(f"Peak reserved memory % of max memory = {used_percentage} %.")
  print(f"Peak reserved memory for training % of max memory = {trainer_percentage} %.")
  print("-----")
  #----


# # Example of LoRA Fine-Tuning for Gemma 3 4B

fine_tune("google/gemma-3-4b-pt", batch_size=1, gradient_accumulation_steps=32, LoRA=True)


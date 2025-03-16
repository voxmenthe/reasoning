#!/usr/bin/env python
# coding: utf-8

# # Let's make Gemma 3 1b think! 🍎
# 
# This is another notebook to make Gemma 3 think. This time focusing on the smallest 1b variant. You should be able to download this notebook for Mac silicone.
# 
# ![logo](https://storage.googleapis.com/gweb-uniblog-publish-prod/images/Gemma3_KeywordBlog_RD3_V01b.width-1200.format-webp.webp)
# 
# 👩‍🎓 If you want to learn more about making models think and reason, check out [The Reasoning Course](https://huggingface.co/reasoning-course)

# ### Installation

# # install this release tag of transformers
# !pip install -qqq git+https://github.com/huggingface/trl.git@main \
#                   bitsandbytes

# !pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3

# !pip install git+https://github.com/huggingface/peft.git


import os
from creds import all_creds
os.environ["HUGGING_FACE_HUB_TOKEN"] = all_creds['HUGGINGFACE_ACCESS_TOKEN_Gemma']


from huggingface_hub import notebook_login
notebook_login()


import torch
from transformers import Gemma3ForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

torch_dtype = torch.bfloat16

model = Gemma3ForCausalLM.from_pretrained(
    pretrained_model_name_or_path="google/gemma-3-1b-it",
    device_map="auto" if not torch.mps.is_available() else torch.device("mps"),  # switch to mac silicon
    #attn_implementation="sdpa",
    attn_implementation="eager",
    torch_dtype=torch_dtype
)

# Load LoRA
peft_config = LoraConfig(
    lora_alpha=4,
    lora_dropout=0.05,
    r=4,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],  # make sure to save the lm_head and embed_tokens as you train the special tokens
)

model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())

processor = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")


# ### Process data to create reasoning chains
# 
# Borrowing from [Will Brown's gist](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb) we'll make reasoning chains from GSM8k.

import re
from datasets import load_dataset, Dataset

# Load and prep dataset
SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""

XML_COT_FORMAT = """<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()


# Optionally load data from CSV
from datasets import Dataset
import pandas as pd

def csv_to_gsm8k_format(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Convert to the required format
    formatted_data = {
        'question': [],
        'answer': [],
        'prompt': []
    }

    SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""

    for _, row in df.iterrows():
        formatted_data['question'].append(row['question'])
        formatted_data['answer'].append(row['answer'])
        formatted_data['prompt'].append([
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': row['question']}
        ])

    # Create HuggingFace dataset
    dataset = Dataset.from_dict(formatted_data)
    return dataset

# Example usage:
# dataset = csv_to_gsm8k_format('your_csv_file.csv')


# # Reward Functions
# 
# Now, let's define reward functions. These are the functions we'll need to setup reward chains.
# 
# | Reward Function | Purpose |
# |---|---|
# | `correctness_reward_func` | Rewards the model when its answer matches the correct answer |
# | `int_reward_func` | Rewards the model for providing a numeric answer |
# | `strict_format_reward_func` and `soft_format_reward_func` | Reward the model for following the specified format |
# | `xmlcount_reward_func` | Rewards proper XML tag usage and penalizes extra content after the closing tags |

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


dataset.features


print(dataset[0])


dataset.data[0][0]


# # Train with GRPOTrainer
# 
# Now we'll confgure training with the `GRPOConfig`

from trl import GRPOConfig, GRPOTrainer
from transformers import GenerationConfig

max_prompt_length = 1024
max_seq_length = 2048


training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "constant",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    num_generations = 2,
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    num_train_epochs = 1,
    max_steps = 5,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "none",
    cache_implementation="hybrid"
)


# # Start trainer

from trl.trainer.utils import pad
import torch

trainer = GRPOTrainer(
    model = model,
    processing_class = processor,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)

trainer.train()


trainer.push_to_hub("voxmenthe/gemma3-1b-thinking")


from transformers import pipeline

question = "The school principal decided that she wanted every class to have an equal number of boys and girls in each first-grade classroom. There are 4 classrooms. There are 56 boys and 44 girls. How many total students are in each classroom?"
generator = pipeline("text-generation", model=trainer.model, tokenizer=processor.tokenizer)
input = processor.apply_chat_template([{"role": "user", "content": question}])
input + "<reasoning>"
output = generator(input, max_new_tokens=1024)


output


# # Next Steps!
# 
# Checkout the [The Reasoing Course](https://huggingface.co/reasoning-course) for more info on GRPO.
# 
# In the coming days we'll release a version of this notebook with Unsloth!
# 
# <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>

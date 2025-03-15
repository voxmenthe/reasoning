import re
from datasets import load_dataset, Dataset
import pandas as pd

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

def get_gsm8k_questions(split = "train") -> Dataset:
    """Load GSM8K dataset and format it for reasoning tasks"""
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data

def csv_to_gsm8k_format(csv_path):
    """Convert CSV data to the format required for GSM8K-style reasoning"""
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Convert to the required format using vectorized operations
    formatted_data = {
        'question': df['question'].tolist(),
        'answer': df['answer'].tolist(),
        'prompt': [
            [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': question}
            ] 
            for question in df['question']
        ]
    }

    # Create HuggingFace dataset
    dataset = Dataset.from_dict(formatted_data)
    return dataset

def process_chat_data(dataset, tokenizer):
    """Process chat dataset with tokenizer chat template"""
    def process(row):
        row["text"] = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=False) + tokenizer.eos_token
        return row
    
    import multiprocessing
    return dataset.map(
        process,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )

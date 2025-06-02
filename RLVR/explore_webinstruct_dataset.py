"""
Script to download and explore the TIGER-Lab/WebInstruct-verified dataset.

This script will:
1. Load the dataset from Hugging Face Hub.
2. Print dataset information (features, splits, number of rows).
3. Show a few examples from the 'train' split.
"""

import argparse
from datasets import load_dataset
# Note: pandas is not strictly necessary for this manual counting approach,
# but can be useful for other dataset manipulations.
# import pandas as pd

def explore_dataset(dataset_name: str, num_examples_to_show: int = 5):
    """
    Loads the specified dataset, prints its information, and shows some examples.

    Args:
        dataset_name (str): The name of the dataset on Hugging Face Hub.
        num_examples_to_show (int): Number of examples to print from the 'train' split.
    """
    print(f"Loading dataset: {dataset_name}...")
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}': {e}")
        return

    print("\nDataset information:")
    print(dataset)

    print("\nFeatures (columns) for each split:")
    for split_name, split_data in dataset.items():
        print(f"  Split '{split_name}':")
        print(f"    Number of rows: {len(split_data)}")
        print(f"    Features: {split_data.features}")

    if "train" in dataset:
        train_split = dataset["train"]
        print(f"\nFirst {num_examples_to_show} examples from the 'train' split:")
        for i in range(min(num_examples_to_show, len(train_split))):
            print(f"\nExample {i+1}:")
            # Print all fields for the example
            for col_name, value in train_split[i].items():
                print(f"  {col_name}: {value}")

        # Calculate and print value counts for 'answer_type' and 'category'
        if 'answer_type' in train_split.features:
            print("\nDistribution of 'answer_type' in 'train' split:")
            answer_type_counts = {}
            # Iterate through the full train_split for accurate counts
            for example in train_split:
                at = example['answer_type']
                answer_type_counts[at] = answer_type_counts.get(at, 0) + 1
            for at, count in sorted(answer_type_counts.items(), key=lambda item: item[1], reverse=True):
                print(f"  {at}: {count}")
        
        if 'category' in train_split.features:
            print("\nDistribution of 'category' in 'train' split:")
            category_counts = {}
            # Iterate through the full train_split for accurate counts
            for example in train_split:
                cat = example['category']
                category_counts[cat] = category_counts.get(cat, 0) + 1
            for cat, count in sorted(category_counts.items(), key=lambda item: item[1], reverse=True):
                print(f"  {cat}: {count}")

    else:
        print("\n'train' split not found in the dataset.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and explore a Hugging Face dataset.")
    parser.add_argument("--dataset_name", type=str, default="TIGER-Lab/WebInstruct-verified", 
                        help="Name of the dataset on Hugging Face Hub.")
    parser.add_argument("--num_examples", type=int, default=3, 
                        help="Number of examples to display from the 'train' split.")
    
    args = parser.parse_args()
    
    explore_dataset(args.dataset_name, args.num_examples)

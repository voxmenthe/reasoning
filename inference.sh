# Basic usage
python inference.py --question "Your math question here"

# Append reasoning tag and use greedy decoding (faster)
python inference.py --question "Your math question here" --append_reasoning

# For more creative responses
python inference.py --question "Your math question here" --do_sample --temperature 0.8

# For batch processing
python batch_inference.py --input_file questions.json --output_dir ./results --append_reasoning
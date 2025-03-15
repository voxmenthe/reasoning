import re
from typing import List

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Rewards the model when its answer matches the correct answer"""
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    """Rewards the model for providing a numeric answer"""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format with exact newlines."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has XML tags in any format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    """Counts XML tags and penalizes content after closing tags"""
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
    """Rewards proper XML tag usage and penalizes extra content after closing tags"""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def anti_repetition_reward_func(completions, **kwargs) -> list[float]:
    """Penalizes nonsensical repetitive patterns regardless of the script/language used."""
    contents = [completion[0]["content"] for completion in completions]
    
    # Regular expressions to detect various repetition patterns
    # Match any character from same script repeated many times consecutively
    consecutive_repeats = r'([\p{Script=Han}]{10,}|[\p{Script=Arabic}]{10,}|[\p{Script=Tamil}]{10,}|[\p{Script=Cyrillic}]{10,})'
    
    # Match repeating patterns (same sequence repeated multiple times)
    repeating_pattern = r'(.{2,}?)\1{3,}'  # Any sequence repeated 3+ times
    
    # Match repeating punctuation patterns
    repeating_punctuation = r'([.]{5,}|["][.]["]{'+'3,}|[.]["][.]["]{'+'2,})'
    
    rewards = []
    for text in contents:
        reward = 0.0
        
        # Check for consecutive repetitions of same-script characters (likely nonsensical)
        try:
            import regex
            consecutive_matches = regex.findall(consecutive_repeats, text, regex.UNICODE)
            if consecutive_matches:
                reward -= sum(len(match) for match in consecutive_matches) * 0.05
        except ImportError:
            # Fallback if regex module not available
            pass
            
        # Check for repeating patterns (strongest penalty)
        repetition_matches = re.findall(repeating_pattern, text)
        if repetition_matches:
            reward -= sum(len(pattern) for pattern in repetition_matches) * 0.1
        
        # Check for repetitive punctuation
        punct_matches = re.findall(repeating_punctuation, text)
        if punct_matches:
            reward -= sum(len(match) for match in punct_matches) * 0.03
        
        rewards.append(reward)
    
    return rewards

def extract_xml_answer(text: str) -> str:
    """Extract the answer from XML-formatted text"""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip() 
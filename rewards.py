import re
from typing import List
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('punkt_tab')

from reward_config import (
    CORRECTNESS_REWARD,
    INTEGER_REWARD,
    STRICT_FORMAT_REWARD,
    SOFT_FORMAT_REWARD,
    XML_TAG_WEIGHTS,
    REPETITION_PENALTIES,
    TRAILING_CONTENT_PENALTY,
    REPETITION_SETTINGS,
    TOPIC_SETTINGS,
    TOPIC_PENALTIES
)

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Rewards the model when its answer matches the correct answer"""
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [CORRECTNESS_REWARD if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    """Rewards the model for providing a numeric answer"""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [INTEGER_REWARD if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format with exact newlines."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [STRICT_FORMAT_REWARD if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has XML tags in any format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [SOFT_FORMAT_REWARD if match else 0.0 for match in matches]

def count_xml(text) -> float:
    """Counts XML tags and penalizes content after closing tags"""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += XML_TAG_WEIGHTS["reasoning_open"]
    if text.count("\n</reasoning>\n") == 1:
        count += XML_TAG_WEIGHTS["reasoning_close"]
    if text.count("\n<answer>\n") == 1:
        count += XML_TAG_WEIGHTS["answer_open"]
        count -= len(text.split("\n</answer>\n")[-1]) * TRAILING_CONTENT_PENALTY
    if text.count("\n</answer>") == 1:
        count += XML_TAG_WEIGHTS["answer_close"]
        count -= (len(text.split("\n</answer>")[-1]) - 1) * TRAILING_CONTENT_PENALTY
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Rewards proper XML tag usage and penalizes extra content after closing tags"""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def find_repeated_words(text: str) -> list[tuple[str, int]]:
    """Find words that are repeated in sequence, with or without spaces.
    Returns list of (word, count) tuples."""
    # First find words repeated with spaces
    words = text.split()
    repeated = []
    current_word = None
    count = 1
    
    for word in words:
        if len(word) >= REPETITION_SETTINGS["min_word_length"]:
            if word == current_word:
                count += 1
            else:
                if count >= REPETITION_SETTINGS["min_repeats"]:
                    repeated.append((current_word, count))
                current_word = word
                count = 1
    
    if count >= REPETITION_SETTINGS["min_repeats"]:
        repeated.append((current_word, count))
    
    return repeated

def find_no_space_repetitions(text: str) -> list[tuple[str, int]]:
    """Find word-like sequences that are repeated without spaces."""
    # Pattern to match repeated sequences of letters (at least min_word_length long)
    pattern = f"([a-zA-Z]{{{REPETITION_SETTINGS['min_word_length']},}})\\1{{{REPETITION_SETTINGS['min_repeats']-1},}}"
    matches = re.finditer(pattern, text)
    return [(m.group(1), len(m.group(0))//len(m.group(1))) for m in matches]

def detect_script(char: str) -> str:
    """Detect the Unicode script of a character."""
    try:
        import regex
        for script in REPETITION_SETTINGS["supported_scripts"]:
            if regex.match(fr"\p{{Script={script}}}", char):
                return script
    except ImportError:
        pass
    return "Unknown"

def is_legitimate_multilingual(text: str, script_changes: list[tuple[str, str]]) -> bool:
    """
    Determine if a text segment is legitimate multilingual content rather than spam.
    Returns True if the content appears to be legitimate multilingual text.
    """
    if not script_changes:
        return True
        
    # Count consecutive script changes
    consecutive_changes = 0
    max_consecutive = 0
    prev_script = None
    
    # Track characters per script
    script_chars = {}
    
    for char, script in zip(text, [s[1] for s in script_changes]):
        if script != "Unknown":
            script_chars[script] = script_chars.get(script, 0) + 1
            
            if prev_script and script != prev_script:
                consecutive_changes += 1
            else:
                consecutive_changes = 0
            max_consecutive = max(max_consecutive, consecutive_changes)
            prev_script = script
    
    # Check if any script has too few characters
    for script, count in script_chars.items():
        if count < REPETITION_SETTINGS["min_chars_per_script"]:
            return False
            
    # Check consecutive script changes
    if max_consecutive > REPETITION_SETTINGS["max_consecutive_script_changes"]:
        return False
        
    # Check ratio of characters to script changes
    total_chars = sum(script_chars.values())
    if total_chars / (len(script_changes) + 1) < REPETITION_SETTINGS["legitimate_script_ratio"]:
        return False
        
    return True

def analyze_script_changes(text: str) -> list[tuple[str, str, int]]:
    """
    Analyze script changes in text and return a list of (segment, script, changes) tuples.
    Only returns segments that appear problematic.
    """
    segments = []
    current_segment = ""
    current_script = None
    changes_in_segment = 0
    script_changes = []
    
    for char in text:
        script = detect_script(char)
        if script != "Unknown":
            if current_script and script != current_script:
                changes_in_segment += 1
                script_changes.append((char, script))
            current_segment += char
            current_script = script
        else:
            # End of segment
            if current_segment and changes_in_segment > 0:
                # Only add if it's not legitimate multilingual content
                if not is_legitimate_multilingual(current_segment, script_changes):
                    segments.append((current_segment, current_script, changes_in_segment))
            current_segment = ""
            current_script = None
            changes_in_segment = 0
            script_changes = []
            
    # Handle last segment
    if current_segment and changes_in_segment > 0:
        if not is_legitimate_multilingual(current_segment, script_changes):
            segments.append((current_segment, current_script, changes_in_segment))
            
    return segments

def find_mixed_script_sequences(text: str) -> list[tuple[str, int]]:
    """Find sequences where different scripts are mixed together in a problematic way."""
    problematic_segments = analyze_script_changes(text)
    return [(segment, changes) for segment, _, changes in problematic_segments]

def find_repeated_phrases(text: str) -> list[tuple[str, int]]:
    """Find phrases (3+ words) that are repeated."""
    words = text.split()
    phrases = []
    
    # Look for phrases of different lengths
    for phrase_length in range(REPETITION_SETTINGS["phrase_min_words"], 6):
        for i in range(len(words) - phrase_length + 1):
            phrase = " ".join(words[i:i+phrase_length])
            if len(phrase) <= REPETITION_SETTINGS["max_phrase_length"]:
                count = text.count(phrase)
                if count >= REPETITION_SETTINGS["min_repeats"]:
                    phrases.append((phrase, count))
    
    return phrases

def find_non_latin_repeats(text: str) -> list[tuple[str, int]]:
    """Find problematic repeated sequences in non-Latin scripts."""
    try:
        import regex
        pattern = "".join([
            fr"([\p{{Script={script}}}]{{3,}})"
            for script in REPETITION_SETTINGS["supported_scripts"]
            if script != "Latin"
        ])
        matches = regex.finditer(pattern, text)
        results = []
        for match in matches:
            sequence = match.group(0)
            # Only count if it appears repetitive
            count = text.count(sequence)
            if count >= REPETITION_SETTINGS["min_repeats"]:
                # Check if it's legitimate multilingual content
                if not is_legitimate_multilingual(sequence, [(c, detect_script(c)) for c in sequence]):
                    results.append((sequence, count))
        return results
    except ImportError:
        return []

def anti_repetition_reward_func(completions, **kwargs) -> list[float]:
    """Penalizes nonsensical repetitive patterns regardless of the script/language used."""
    contents = [completion[0]["content"] for completion in completions]
    
    rewards = []
    for text in contents:
        reward = 0.0
        
        # 1. Check for consecutive repetitions of same-script characters
        try:
            import regex
            consecutive_repeats = r'([\p{Script=Han}]{10,}|[\p{Script=Arabic}]{10,}|[\p{Script=Tamil}]{10,}|[\p{Script=Cyrillic}]{10,})'
            consecutive_matches = regex.findall(consecutive_repeats, text, regex.UNICODE)
            if consecutive_matches:
                reward -= sum(len(match) for match in consecutive_matches) * REPETITION_PENALTIES["consecutive_script"]
        except ImportError:
            pass
            
        # 2. Check for repeating patterns
        repeating_pattern = r'(.{2,}?)\1{3,}'
        repetition_matches = re.findall(repeating_pattern, text)
        if repetition_matches:
            reward -= sum(len(pattern) for pattern in repetition_matches) * REPETITION_PENALTIES["pattern"]
        
        # 3. Check for repetitive punctuation
        repeating_punctuation = r'([.]{5,}|["][.]["]{'+'3,}|[.]["][.]["]{'+'2,})'
        punct_matches = re.findall(repeating_punctuation, text)
        if punct_matches:
            reward -= sum(len(match) for match in punct_matches) * REPETITION_PENALTIES["punctuation"]
            
        # 4. Check for repeated words with spaces
        word_repetitions = find_repeated_words(text)
        for word, count in word_repetitions:
            reward -= len(word) * count * REPETITION_PENALTIES["word"]
            
        # 5. Check for repeated sequences without spaces
        no_space_repetitions = find_no_space_repetitions(text)
        for seq, count in no_space_repetitions:
            reward -= len(seq) * count * REPETITION_PENALTIES["no_space_word"]
            
        # 6. Check for problematic mixed script sequences
        mixed_scripts = find_mixed_script_sequences(text)
        for seq, changes in mixed_scripts:
            reward -= len(seq) * changes * REPETITION_PENALTIES["mixed_script_spam"]
            
        # 7. Check for repeated phrases
        repeated_phrases = find_repeated_phrases(text)
        for phrase, count in repeated_phrases:
            reward -= len(phrase) * count * REPETITION_PENALTIES["phrase"]
            
        # 8. Check for problematic non-Latin script repetitions
        non_latin_repeats = find_non_latin_repeats(text)
        for seq, count in non_latin_repeats:
            reward -= len(seq) * count * REPETITION_PENALTIES["non_latin_spam"]
        
        rewards.append(reward)
    
    return rewards

def extract_xml_answer(text: str) -> str:
    """Extract the answer from XML-formatted text"""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_sections(text: str) -> tuple[str, str, str]:
    """Extract reasoning, answer, and any content after the answer."""
    reasoning = ""
    answer = ""
    post_answer = ""
    
    # Extract reasoning
    if "<reasoning>" in text and "</reasoning>" in text:
        reasoning = text.split("<reasoning>")[1].split("</reasoning>")[0].strip()
    
    # Extract answer
    if "<answer>" in text and "</answer>" in text:
        answer = text.split("<answer>")[1].split("</answer>")[0].strip()
        
    # Extract post-answer content
    if "</answer>" in text:
        post_answer = text.split("</answer>")[1].strip()
        
    return reasoning, answer, post_answer

def is_same_script_majority(text1: str, text2: str) -> bool:
    """Check if two texts share the same majority script."""
    def get_majority_script(text):
        script_counts = {}
        for char in text:
            if char.isspace():
                continue
            script = detect_script(char)
            if script != "Unknown":
                script_counts[script] = script_counts.get(script, 0) + 1
        return max(script_counts.items(), key=lambda x: x[1])[0] if script_counts else "Unknown"
    
    return get_majority_script(text1) == get_majority_script(text2)

def compute_text_similarity(text1: str, text2: str) -> float:
    """Compute semantic similarity between two texts, with special handling for multilingual content."""
    if not text1 or not text2:
        return 0.0
    
    # If texts are primarily in different scripts, consider them potentially related
    if not is_same_script_majority(text1, text2):
        return TOPIC_SETTINGS["different_script_base_similarity"]
    
    # For same-script content, use TF-IDF but without English-specific processing
    vectorizer = TfidfVectorizer(stop_words=None)
    try:
        # Normalize texts to handle different scripts
        text1_norm = ''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in text1)
        text2_norm = ''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in text2)
        
        tfidf_matrix = vectorizer.fit_transform([text1_norm, text2_norm])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Boost similarity for texts with shared scripts
        if similarity > 0:
            similarity = min(1.0, similarity * TOPIC_SETTINGS["same_script_boost"])
            
        return similarity
    except:
        return TOPIC_SETTINGS["different_script_base_similarity"]

def topic_relevance_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Rewards responses that stay on topic and penalizes off-topic content, excessive length,
    and content after the answer that isn't part of the reasoning."""
    
    rewards = []
    for prompt, completion in zip(prompts, [c[0]['content'] for c in completions]):
        reward = 0.0
        question = prompt[-1]['content']  # Get the actual question
        
        # Extract sections
        reasoning, answer, post_answer = extract_sections(completion)
        
        # 1. Check reasoning length
        if len(reasoning) > TOPIC_SETTINGS["max_reasoning_length"]:
            excess_chars = len(reasoning) - TOPIC_SETTINGS["max_reasoning_length"]
            reward -= (excess_chars / 100) * TOPIC_PENALTIES["excessive_length"]
            
        # 2. Check answer length
        if len(answer) > TOPIC_SETTINGS["max_answer_length"]:
            excess_chars = len(answer) - TOPIC_SETTINGS["max_answer_length"]
            reward -= (excess_chars / 100) * TOPIC_PENALTIES["excessive_length"]
            
        # 3. Penalize post-answer content
        if post_answer:
            sentences = sent_tokenize(post_answer)
            reward -= len(sentences) * TOPIC_PENALTIES["post_answer_content"]
            
        # 4. Check topic relevance of reasoning
        if reasoning:
            sentences = sent_tokenize(reasoning)
            for sentence in sentences:
                similarity = compute_text_similarity(question, sentence)
                if similarity < TOPIC_SETTINGS["min_similarity_threshold"]:
                    reward -= TOPIC_PENALTIES["off_topic_content"]
                    
        # 5. Check for topic drift
        if reasoning:
            sentences = sent_tokenize(reasoning)
            prev_sentence = sentences[0]
            for sentence in sentences[1:]:
                similarity = compute_text_similarity(prev_sentence, sentence)
                if similarity < TOPIC_SETTINGS["min_similarity_threshold"]:
                    reward -= TOPIC_PENALTIES["topic_drift"]
                prev_sentence = sentence
                
        rewards.append(reward)
        
    return rewards 
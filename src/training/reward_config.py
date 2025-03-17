"""
Configuration for reward functions used in training.

Each reward's weight/scale factor determines its relative contribution to the total reward.
The weights are chosen to create a balanced reward signal that prioritizes:
1. Correctness of the answer (highest weight)
2. Proper formatting and structure (medium weight)
3. Penalization of bad patterns (negative weights)
4. Topic relevance and conciseness (medium-high weight)

Total maximum positive reward possible: ~4.0
- Correctness: 2.0 (50% of total possible)
- Integer answer: 0.5 (12.5% of total)
- Strict format: 0.5 (12.5% of total)
- Soft format: 0.5 (12.5% of total)
- XML structure: 0.5 (12.5% of total)

Negative rewards (penalties) can reduce the total significantly:
- Anti-repetition: Up to -1.0 per pattern type
  - Consecutive script repeats: -0.05 per character
  - Pattern repeats: -0.1 per pattern
  - Word repeats: -0.08 per word
  - No-space word repeats: -0.15 per sequence
  - Punctuation repeats: -0.03 per match
- Topic relevance: Up to -2.0
  - Off-topic content: -0.2 per sentence
  - Excessive length: -0.1 per 100 chars over limit
  - Topic drift penalty: -0.5 per topic change
"""

# Anti-repetition scaling factor - reduces the overwhelming penalty
ANTI_REPETITION_SCALE = 0.075  # Scale down the anti-repetition penalty
MAX_ANTI_REPETITION_PENALTY =  -9.0 # float('-inf') # Cap the maximum anti-repetition penalty

# Core reward weights
CORRECTNESS_REWARD = 4.00  # Primary objective - correct answer
INTEGER_REWARD = 1.0      # Reward for providing numeric answer
STRICT_FORMAT_REWARD = 1.0  # Perfect formatting with newlines
SOFT_FORMAT_REWARD = 0.75    # Basic XML structure present
XML_COUNT_REWARD = 0.125    # Per-tag reward, max 0.5 total

# XML tag weights for granular XML structure reward
XML_TAG_WEIGHTS = {
    "reasoning_open": 0.125,    # <reasoning>\n
    "reasoning_close": 0.125,   # \n</reasoning>\n
    "answer_open": 0.125,       # \n<answer>\n
    "answer_close": 0.125       # \n</answer>
}

# Anti-repetition penalty factors
REPETITION_PENALTIES = {
    "consecutive_script": 0.05,    # Per character in repeated script sequence
    "pattern": 0.1,               # Per repeated pattern found
    "word": 0.08,                # Per repeated word
    "no_space_word": 0.15,       # Per repeated word sequence without spaces
    "punctuation": 0.03,         # Per punctuation repetition
    "mixed_script_spam": 0.2,    # Per mixed script sequence that appears repetitive
    "phrase": 0.12,              # Per repeated phrase (3+ words)
    "non_latin_spam": 0.15       # Per repeated non-latin sequence that appears spammy
}

# Repetition detection settings
REPETITION_SETTINGS = {
    "min_word_length": 2,         # Minimum length for a word to be considered in repetition checks
    "min_repeats": 3,            # Minimum number of repetitions to trigger penalty
    "max_word_gap": 2,           # Maximum number of words between repetitions to be considered repetitive
    "phrase_min_words": 3,       # Minimum words to consider as a phrase
    "max_phrase_length": 50,     # Maximum characters in a phrase to check for repetition
    "supported_scripts": [        # Scripts to check for mixed-script repetition
        "Latin", "Devanagari", "Thai", "Arabic", "Cyrillic", 
        "Han", "Hiragana", "Katakana", "Hangul", 
        "Tamil", "Telugu", "Kannada", "Malayalam"
    ],
    # New settings for legitimate multilingual content
    "max_script_changes_per_sentence": 3,  # Allow up to 3 script changes per sentence before penalizing
    "min_chars_per_script": 3,    # Minimum characters in a script to be considered legitimate content
    "max_consecutive_script_changes": 5,   # Max number of back-to-back script changes before considered spam
    "legitimate_script_ratio": 0.4  # Minimum ratio of chars to script changes to be considered legitimate
}

# Topic relevance and length settings
TOPIC_SETTINGS = {
    "max_reasoning_length": 500,  # Maximum reasonable length for reasoning section
    "max_answer_length": 100,    # Maximum reasonable length for answer section
    "min_similarity_threshold": 0.3,  # Minimum semantic similarity to consider text on-topic
    # New settings for multilingual content
    "different_script_base_similarity": 0.4,  # Base similarity score for content in different scripts
    "same_script_boost": 1.2,    # Boost factor for similarity within same script
}

# Topic relevance penalties
TOPIC_PENALTIES = {
    "off_topic_content": 0.2,    # Per sentence that's off-topic
    "excessive_length": 0.1,     # Per 100 chars over length limit
    "topic_drift": 0.5,         # Per detected topic change
    "post_answer_content": 0.3,  # Per sentence after answer that isn't part of reasoning
}

# Content after closing tag penalty
TRAILING_CONTENT_PENALTY = 0.001  # Per character after final tag 

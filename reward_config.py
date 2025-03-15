"""
Configuration for reward functions used in training.

Each reward's weight/scale factor determines its relative contribution to the total reward.
The weights are chosen to create a balanced reward signal that prioritizes:
1. Correctness of the answer (highest weight)
2. Proper formatting and structure (medium weight)
3. Penalization of bad patterns (negative weights)

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
  - Punctuation repeats: -0.03 per match
"""

# Core reward weights
CORRECTNESS_REWARD = 2.0  # Primary objective - correct answer
INTEGER_REWARD = 0.5      # Reward for providing numeric answer
STRICT_FORMAT_REWARD = 0.5  # Perfect formatting with newlines
SOFT_FORMAT_REWARD = 0.5    # Basic XML structure present
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
    "consecutive_script": 0.05,  # Per character in repeated script sequence
    "pattern": 0.1,             # Per repeated pattern found
    "punctuation": 0.03         # Per punctuation repetition
}

# Content after closing tag penalty
TRAILING_CONTENT_PENALTY = 0.001  # Per character after final tag 
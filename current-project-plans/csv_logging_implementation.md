# CSV Logging Implementation Plan for Gemma Training

## CSV Schema

Implemented columns for the CSV logging file:

```
timestamp,global_step,question_id,question,ground_truth_answer,model_output,reasoning,model_answer,total_reward,correctness_reward_func,anti_repetition_reward_func,soft_format_reward_func,strict_format_reward_func,int_reward_func,xmlcount_reward_func,topic_relevance_reward_func
```

### Column Descriptions:

1. `timestamp` - ISO format timestamp of when the generation was logged
2. `global_step` - Training step when the generation was created
3. `question_id` - Source-identified unique ID (includes source, date, and hash of question+answer)
4. `question` - The actual question/prompt text
5. `ground_truth_answer` - The expected answer from the dataset
6. `model_output` - Full raw model output
7. `reasoning` - Extracted reasoning from XML tags
8. `model_answer` - Extracted answer from XML tags
9. `total_reward` - Sum of all reward values
10. Individual reward columns - One column per reward function
   - `correctness_reward_func`
   - `anti_repetition_reward_func`
   - `soft_format_reward_func`
   - `strict_format_reward_func`
   - `int_reward_func`
   - `xmlcount_reward_func`
   - `topic_relevance_reward_func`

## Implementation Plan (Completed ✅)

1. **Create a new CSV Logger class** ✅
   - Added `src/logging/csv_logger.py` with thread-safe implementation
   - Implemented batch writing with queue for performance
   - Added proper error handling and flush capability

2. **Integrate with existing logging system** ✅
   - Updated `src/logging/__init__.py` to expose CSV logging functions
   - Added CSV config options to `src/logging/config/gemma_logging_config.yaml`
   - Implemented automatic initialization during logging setup

3. **Modified reward function wrappers** ✅
   - Updated reward functions to collect individual reward values
   - Implemented thread-local storage for reward collection
   - Added `reset_rewards_collection()` to clear between batches

4. **Updated OptimizedLoggingCallback** ✅
   - Modified callback to use the CSV logging system
   - Added generation of question IDs with source identifiers
   - Integration with reward collection system
   - Added CSV flushing on training end

5. **Updated test_model function** ✅
   - Added direct reward calculation for testing samples
   - Implemented CSV logging for test samples
   - Added clearly marked test question IDs with timestamps

6. **Implemented timestamped CSV files** ✅
   - Added automatic timestamp to CSV filenames
   - Modified initialization to create separate files for each run
   - Added robust directory creation to handle missing logs dir
   - Added source identification to distinguish training vs test samples

## Technical Implementation Details

### CSV Writer with Queue (Implemented ✅)
The `CSVLogger` class in `src/logging/csv_logger.py` implements a thread-safe CSV writer with:
- Background thread for non-blocking operation
- Batch processing of records for efficiency
- Automatic header detection and writing
- Configurable queue size and flush intervals

### Integration with Reward Functions (Implemented ✅)
Each reward function wrapper now:
1. Calls the original reward function
2. Logs individual rewards using the standard logging system
3. Stores rewards in a thread-local collection for CSV logging
4. Returns original rewards to maintain compatibility

### Timestamped Filenames (Implemented ✅)
Each training run now creates a separate CSV file with the format:
- Base directory from config + original filename + timestamp
- Format: `generations_and_rewards_YYYYMMDD_HHMMSS.csv`
- Preserves file extension and handles paths with or without extensions
- Automatically creates directories if they don't exist

### Source Identification (Implemented ✅)
We've added source identification to question IDs to help with analysis:
- Training samples: `training_YYYYMMDD_[hash]`
- Test samples: `test_run_YYYYMMDD_HHMMSS_sample_[index]`
- Each source has a unique format to avoid overlaps
- Makes it easy to filter by sample source in analysis

### CSV Logging Config (Implemented ✅)
```yaml
csv_logging:
  enabled: true
  path: "./logs/generations_and_rewards.csv"
  max_queue_size: 1000
  flush_interval_seconds: 10
```

## Testing and Verification

To verify the CSV logging is working properly:
1. Run training with `python src/training/optimized_gemma_training.py`
2. Check that a timestamped CSV file is created in `./logs/` directory
3. Verify that all columns are populated correctly
4. Run multiple training sessions and confirm separate files are created
5. Run validation with the test_model function and verify test samples are logged
6. Confirm that training and test samples have appropriate source identifiers

## Next Steps / Future Improvements

1. Add more flexibility in configuring which rewards are logged
2. Implement CSV compression for large datasets
3. Add option to configure timestamp format in the config
4. Create visualization tools for the CSV data
5. Add benchmarking to measure logging overhead
6. Add option to combine multiple CSV logs for analysis 
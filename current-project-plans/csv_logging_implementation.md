# CSV Logging Implementation Plan for Gemma Training

## CSV Schema

Proposed columns for the CSV logging file:

```
timestamp,global_step,question_id,question,model_output,reasoning,answer,total_reward,xmlcount_reward,soft_format_reward,strict_format_reward,int_reward,correctness_reward,anti_repetition_reward,topic_relevance_reward
```

### Column Descriptions:

1. `timestamp` - ISO format timestamp of when the generation was logged
2. `global_step` - Training step when the generation was created
3. `question_id` - Unique identifier for the question (from dataset if available, otherwise create unique reproducible hash from question + answer)
4. `question` - The actual question/prompt text
5. `ground_truth_answer` - The actual answer from the dataset
6. `model_output` - Full raw model output
7. `reasoning` - Extracted reasoning from XML tags
8. `model_answer` - Extracted answer from XML tags
9. `total_reward` - Sum of all reward values
10. Individual reward columns - One column per reward function
   - `xmlcount_reward`
   - `soft_format_reward`
   - `strict_format_reward`
   - `int_reward`
   - `correctness_reward`
   - `anti_repetition_reward`
   - `topic_relevance_reward`

## Implementation Plan

1. **Create a new CSV Logger class**
   - Add a new file `src/logging/csv_logger.py`
   - Implement thread-safe CSV writing with a queue
   - Include batch writing for performance

2. **Integrate with existing logging system**
   - Update `src/logging/__init__.py` to expose CSV logging functions
   - Add CSV config options to the logging config file

3. **Modify reward function wrappers**
   - Update `get_wrapped_reward_functions()` to capture rewards for CSV logging
   - Add a collector for reward values

4. **Update OptimizedLoggingCallback**
   - Modify callback to collect generation/reward data during training
   - Integrate with CSV logger to write at appropriate intervals

5. **Update test_model function**
   - Add CSV logging to the test_model function for evaluation runs
   - Include batch identifier to distinguish training from evaluation

## Technical Implementation Details

### CSV Writer with Queue
```python
class CSVLogger:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        self.header_written = False
        
    def log_generation(self, data_dict):
        """Add a generation record to the CSV queue"""
        self.queue.put(data_dict)
        
    def _worker(self):
        """Background thread that processes the queue and writes to CSV"""
        while True:
            # Batch process for efficiency
            batch = []
            try:
                # Get at least one item
                batch.append(self.queue.get(block=True, timeout=1.0))
                
                # Get more items if available (up to 50)
                for _ in range(49):
                    try:
                        batch.append(self.queue.get(block=False))
                    except queue.Empty:
                        break
                        
                if batch:
                    self._write_batch_to_csv(batch)
                    
                # Mark tasks as done
                for _ in range(len(batch)):
                    self.queue.task_done()
                    
            except queue.Empty:
                # No items available
                time.sleep(0.1)
                continue
                
    def _write_batch_to_csv(self, batch):
        """Write a batch of records to the CSV file"""
        if not batch:
            return
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        
        # Write header if needed
        if not self.header_written:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(batch[0].keys()))
                writer.writeheader()
                self.header_written = True
        
        # Append data
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(batch[0].keys()))
            writer.writerows(batch)
```

### Integration Points
1. Initialize CSV logger during training setup
2. Collect rewards by patching `wrapped_reward_functions`
3. Update OptimizedLoggingCallback to capture outputs and rewards
4. Add direct logging in test_model function

### Configuration Options
```yaml
csv_logging:
  enabled: true
  path: "./logs/generations_and_rewards.csv"
  max_queue_size: 1000
  flush_interval_seconds: 10
``` 
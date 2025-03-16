"""Test basic imports and project setup."""

import sys
import os

def test_imports():
    """Test that core modules can be imported."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
    
    # Import test will fail if any of these imports fail
    try:
        # Attempt to import key modules - adjust based on your actual modules
        import rewards
        import reward_config
        import reasoning_dataset
        
        # If we get here, imports succeeded
        assert True
    except ImportError as e:
        # Print the import error but convert to AssertionError to make test fail
        print(f"Import failed: {e}")
        assert False, f"Import failed: {e}" 
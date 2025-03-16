"""Pytest configuration file for gemma3-reasoning project."""

import pytest
import os
import sys

# Add src to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

@pytest.fixture
def sample_data():
    """Return sample data for testing."""
    return {"test_key": "test_value"} 
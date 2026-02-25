"""Simple test to verify pytest is working."""

import pytest

def test_simple():
    """A simple test that should always pass."""
    assert 1 + 1 == 2

def test_import():
    """Test that we can import from src."""
    try:
        # sys.path hack — commented out, use pip install -e . instead
        # import sys, os
        # sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from bionetflux.core.problem import Problem
        assert True
    except ImportError:
        pytest.fail("Could not import Problem class")

class TestSample:
    """Sample test class."""
    
    def test_method(self):
        """Test method in class."""
        assert True

# tests/__init__.py
"""
AI Lead Enhancer - Test Package

This package contains all unit tests and integration tests for the AI Lead Enhancer system.
"""

# This file can be empty or contain minimal initialization code
# It marks the tests directory as a Python package

__version__ = "1.0.0"
__author__ = "AI Lead Enhancer Team"

# Optional: Configure test settings or shared test utilities here
import os
import sys

# Add the src directory to the Python path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Optional: Define common test configurations
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
TEST_TIMEOUT = 30  # seconds

# Optional: Import commonly used test utilities
from unittest import TestCase, mock

# Make common utilities available at package level
__all__ = ['TestCase', 'mock', 'TEST_DATA_DIR', 'TEST_TIMEOUT']
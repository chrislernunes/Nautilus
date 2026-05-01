"""
tests/conftest.py — pytest configuration
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path so `python.*` imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

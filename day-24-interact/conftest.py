"""Ensures the project root is importable as `src.*` regardless of the cwd pytest is run from."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

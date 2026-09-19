# conftest.py — pytest configuration for the Aqua-Sense repo
# This file is auto-loaded by pytest before any tests run.
# It adds the repo root to sys.path so `from src.models import ...` works
# without needing to `pip install -e .`

import sys
from pathlib import Path

# Add repo root (parent of this file) to the front of sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

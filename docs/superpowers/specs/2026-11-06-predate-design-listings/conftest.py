"""Pytest bootstrap: put this folder on sys.path so the listings import as top-level modules."""
import sys
from pathlib import Path

LISTINGS_DIR = Path(__file__).resolve().parent
if str(LISTINGS_DIR) not in sys.path:
    sys.path.insert(0, str(LISTINGS_DIR))

import sys
from pathlib import Path

# Add project root to path so tests can import model, solver, etc.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

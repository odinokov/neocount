"""Make the repo-root modules and helper/ importable from tests."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for p in (_ROOT, _ROOT / "helper"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

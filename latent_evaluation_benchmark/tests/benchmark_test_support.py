"""Read-only regression fixtures, usable from the repository or module root."""
import atexit
import os
from pathlib import Path
import sys
import tempfile

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(MODULE_ROOT.parent))
from geochem_workspace import asset

if not os.environ.get("GEOCHEM_RUN_DIR"):
    os.environ["GEOCHEM_RUN_DIR"] = str(asset("latent_reference"))
if not os.environ.get("MPLCONFIGDIR"):
    _cache = tempfile.TemporaryDirectory(prefix="geochem-test-matplotlib-")
    atexit.register(_cache.cleanup)
    os.environ["MPLCONFIGDIR"] = _cache.name

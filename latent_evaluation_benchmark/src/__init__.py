"""Latent Evaluation Benchmark v0.1; bootstrap local optional dependencies first."""
import os,sys
from pathlib import Path
_root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(_root.parent))
from geochem_workspace import output_path
os.environ.setdefault('MPLCONFIGDIR',str(output_path('cache/matplotlib')))

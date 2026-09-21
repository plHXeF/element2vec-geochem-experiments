import sys,os
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent))
from geochem_workspace import require_run_dir, output_path
if __name__ == '__main__': require_run_dir()
sys.path.insert(0,str(root))
os.environ.setdefault('MPLCONFIGDIR',str(output_path('cache/matplotlib')))
from src.pipeline import main
if __name__=='__main__':main()

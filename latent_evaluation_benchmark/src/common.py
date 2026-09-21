
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from geochem_workspace import output_path, load_config_file, scientific_config
from pathlib import Path
import hashlib,json,os,sys
import numpy as np
SOURCE_ROOT=Path(__file__).resolve().parents[1]
ROOT=output_path()
sys.path.insert(0,str(ROOT/'vendor'))
NAMES=['ZVector','GeoNorm','PCAGeo','V2','RoPEz','RoPEgeo']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()
def jsonable(x):
 if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
 if isinstance(x,np.ndarray):return jsonable(x.tolist())
 if isinstance(x,np.generic):return jsonable(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 if isinstance(x,Path):return str(x)
 return x
def save_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(jsonable(x),indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(p)
def load_config(path=None):return load_config_file(path or SOURCE_ROOT/'configs/benchmark.json')
def table(frame,path):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(p.with_suffix('.csv'),index=False);frame.to_parquet(p.with_suffix('.parquet'),index=False)
 save_json(p.with_suffix('.json'),frame.to_dict('records'))
def split_for(key,seed=20260916):
 u=int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()[:8],16)/2**32
 return 'train' if u<.7 else 'val' if u<.85 else 'test'
def array_digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def config_digest(cfg):return digest(scientific_config(cfg))

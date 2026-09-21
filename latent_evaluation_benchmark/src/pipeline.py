from src.common import SOURCE_ROOT
import argparse,json
from src.common import ROOT,load_config,save_json,digest

def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','extract-real','bank','extract-bank','evaluate','render','verify','test','all']);p.add_argument('--config');p.add_argument('--models',nargs='+');p.add_argument('--limit',type=int);a=p.parse_args();cfg=load_config(a.config)
 from src.data import prepare
 records=prepare(cfg)
 if a.stage in ['extract-real','all']:
  from src.extraction import extract_real
  extract_real(cfg,records,a.models)
 if a.stage in ['bank','all']:
  from src.process_bank.episode import prepare_bank
  prepare_bank(cfg,records)
 if a.stage in ['extract-bank','all']:
  from src.verify import preflight
  preflight()
  from src.process_bank.encoding import extract_bank
  extract_bank(cfg,records,a.models,a.limit)
 if a.stage in ['evaluate','all']:
  from src.verify import preflight
  preflight()
  from src.evaluate import evaluate
  evaluate(cfg,records)
 if a.stage in ['render','all']:
  from src.visualization.report import render
  render(cfg,records)
 if a.stage in ['verify','all']:
  from src.verify import verify
  verify(cfg)
 if a.stage=='test':
  import unittest
  s=unittest.defaultTestLoader.discover(str(SOURCE_ROOT/'tests'));r=unittest.TextTestRunner(verbosity=2).run(s);raise SystemExit(not r.wasSuccessful())

import json
from threadpoolctl import threadpool_limits
from src.common import ROOT

def evaluate(cfg,records):
 from src.probes.evaluate import decodability,sufficiency
 from src.semantic.evaluate import evaluate as semantic
 from src.operators.evaluate import evaluate as process
 from src.statistics.paired import summarize
 with threadpool_limits(limits=2):
  if not (ROOT/'outputs/decodability/curve_auc.parquet').exists():decodability(cfg,records)
  if not (ROOT/'outputs/sufficiency/nuisance.parquet').exists():sufficiency(cfg,records)
  if not (ROOT/'outputs/semantic_geometry/metrics.parquet').exists():semantic(cfg,records)
  while True:
   process(cfg);extend=summarize(cfg)
   if not extend:break
   from src.process_bank.episode import prepare_bank
   from src.process_bank.encoding import extract_bank
   status=json.loads((ROOT/'outputs/tables/precision_status.json').read_text());target=status['next_targets'];print(f'Monte Carlo precision threshold: extending only {target}',flush=True);prepare_bank(cfg,records,target);extract_bank(cfg,records)

import json
import numpy as np
import pandas as pd
from src.common import ROOT,NAMES,table,save_json

def paired_delta(model,baseline,metric,seed=0,n_boot=2000,higher=True):
 merged=model[['episode_id','initial_state_id',metric]].merge(baseline[['episode_id',metric]],on='episode_id',suffixes=('_m','_b'),validate='one_to_one');assert len(merged)==len(model)==len(baseline),'Paired episode IDs must match exactly'
 d=(merged[metric+'_m']-merged[metric+'_b']).to_numpy(float)*(1 if higher else -1);valid=np.isfinite(d);d=d[valid];groups=merged.initial_state_id.to_numpy()[valid]
 if not len(d):return None
 # Cluster bootstrap respects repeated natural initial records; within every draw pairing is exact.
 unique=np.unique(groups);members=[np.flatnonzero(groups==g) for g in unique];rng=np.random.default_rng(seed);boot=[]
 if len(unique)==len(d):
  for start in range(0,n_boot,100):boot.extend(np.median(d[rng.integers(len(d),size=(min(100,n_boot-start),len(d)))],axis=1))
 else:
  for _ in range(n_boot):boot.append(np.median(d[np.concatenate([members[i] for i in rng.integers(len(unique),size=len(unique))])]))
 low,high=np.quantile(boot,[.025,.975]);return {'n':len(d),'initial_state_clusters':len(unique),'median_delta':float(np.median(d)),'ci_low':float(low),'ci_high':float(high),'ci_half_width':float((high-low)/2),'p05':float(np.quantile(d,.05)),'p95':float(np.quantile(d,.95)),'win_rate':float(np.mean(d>0)),'paired_valid_fraction':float(valid.mean()),'positive_means':'model improvement','bootstrap_resamples':n_boot}

def summarize(cfg):
 process=pd.read_parquet(ROOT/'outputs/process/trajectory_metrics.parquet');operator=pd.read_parquet(ROOT/'outputs/process/operator_metrics.parquet');composite=pd.read_parquet(ROOT/'outputs/composite/metrics.parquet');rows=[]
 sets=[(process,['tangent_coherence','simplicity','local_rank1','no_fold'],'kind'),(operator.query("split == 'test'"),['normalized_next_state_error'],'process'),(composite.query("split == 'test'"),['boundary_f1','composition_error'],'composite_type')]
 ids=json.loads((ROOT/'cache/process_bank/index.json').read_text());lookup={r['episode_id']:r['initial_state_id'] for r in ids}
 for frame,metrics,groupcol in sets:
  if 'initial_state_id' not in frame:frame=frame.assign(initial_state_id=frame.episode_id.map(lookup))
  for (process,source),group in frame.groupby([groupcol,'source_type']):
   if process=='Calibration':continue
   for metric in metrics:
    for name in NAMES[3:]:
     for base in NAMES[:3]:
      model=group[group.representation.eq(name)];baseline=group[group.representation.eq(base)];result=paired_delta(model,baseline,metric,cfg['seed'],cfg['process']['bootstrap'],metric not in ['normalized_next_state_error','composition_error'])
      if result:rows.append({'process':process,'source_type':source,'representation':name,'baseline':base,'metric':metric,**result})
 frame=pd.DataFrame(rows);table(frame,ROOT/'outputs/tables/paired_statistics')
 history_path=ROOT/'outputs/tables/convergence.csv';bank=json.loads((ROOT/'cache/process_bank/schema.json').read_text());counts=bank.get('per_process_counts',{p:bank['per_process'] for p in ['Mixing','Fractionation','PartialMelting','Alteration']});counts['Composite']=bank['composites']
 current=frame.assign(bank_per_process=bank['per_process'],bank_composites=bank['composites'],bank_total_episodes=bank['episodes']);current['episodes_in_family']=current.process.map(lambda p:counts.get(p,counts['Composite']))
 old=pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
 if len(old):
  if 'bank_total_episodes' not in old:old['bank_total_episodes']=4*old.bank_per_process+old.bank_composites+4
  if 'episodes_in_family' not in old:old['episodes_in_family']=np.where(old.process.isin(['heterogeneous','same-process']),old.bank_composites,old.bank_per_process)
  old=old[~old.bank_total_episodes.eq(bank['episodes'])]
 combined=pd.concat([old,current],ignore_index=True);table(combined,ROOT/'outputs/tables/convergence')
 failures=frame[frame.ci_half_width>cfg['process']['precision']];targets={}
 for family in failures.process.unique():
  key=family if family in counts else 'Composite'
  if counts[key]<cfg['process']['max_episodes']:targets[key]=min(2*counts[key],cfg['process']['max_episodes'])
 save_json(ROOT/'outputs/tables/precision_status.json',{'threshold':cfg['process']['precision'],'failed_comparisons':len(failures),'max_ci_half_width':frame.ci_half_width.max(),'bank_per_process':bank['per_process'],'per_process_counts':counts,'bank_composites':bank['composites'],'max_episodes':cfg['process']['max_episodes'],'next_targets':targets,'needs_extension':bool(targets)})
 return bool(targets)

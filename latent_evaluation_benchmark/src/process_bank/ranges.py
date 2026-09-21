"""Model-independent synthetic-range audit against the real training records."""
import json
import numpy as np
import pandas as pd
from src.common import ROOT,table

def audit_ranges(records):
 bank=ROOT/'cache/process_bank';idx=json.loads((bank/'index.json').read_text());fs=json.loads((bank/'schema.json').read_text())['fields'];train=records[records.split.eq('train')].reindex(columns=fs);reference=train.quantile([.01,.5,.99]);groups={};rows=[]
 for r in idx:groups.setdefault((r['kind'],r['source_type']),[]).append(r)
 for (process,source),episodes in groups.items():
  x=np.vstack([np.load(bank/r['state_file'])['states'] for r in episodes])
  for j,f in enumerate(fs):
   valid=x[:,j][np.isfinite(x[:,j])];low,median,high=reference[f].to_numpy();q=np.quantile(valid,[0,.01,.5,.99,1]) if len(valid) else [np.nan]*5
   rows.append({'process':process,'source_type':source,'field':f,'n_finite':len(valid),'real_train_n':int(train[f].notna().sum()),'real_train_p01':low,'real_train_median':median,'real_train_p99':high,'bank_min':q[0],'bank_p01':q[1],'bank_median':q[2],'bank_p99':q[3],'bank_max':q[4],'fraction_outside_train_p01_p99':float(np.mean((valid<low)|(valid>high))) if len(valid) and np.isfinite(low) else np.nan,'reference_available':bool(np.isfinite(low)),'clipping':'none'})
 table(pd.DataFrame(rows),ROOT/'outputs/process/input_range_audit')

def audit_semigroup_support():
 """An episode with no observed trace cannot count as a perfect trace-law test."""
 p=ROOT/'outputs/composite/semigroup.parquet'
 if not p.exists():return
 bank=ROOT/'cache/process_bank';idx={r['episode_id']:r for r in json.loads((bank/'index.json').read_text())};fs=json.loads((bank/'schema.json').read_text())['fields'];js=[i for i,f in enumerate(fs) if f.endswith('_ppm')];df=pd.read_parquet(p)
 df['observed_trace_fields']=[int(np.isfinite(np.asarray(idx[e]['initial_state'],float)[js]).sum()) for e in df.episode_id]
 df['status']=np.where(df.observed_trace_fields.gt(0),'ok','unavailable_no_observed_trace');df.loc[df.observed_trace_fields.eq(0),'trace_semigroup_error']=np.nan;table(df,p.with_suffix(''))

def audit_order_coverage():
 rows=pd.DataFrame(json.loads((ROOT/'cache/process_bank/index.json').read_text()))
 rows=rows[rows.kind.eq('Composite')&rows.composite_type.eq('heterogeneous')]
 counts=rows.groupby(['source_type','split']).order_available.agg(total_eligible='size',valid_reverse='sum').reset_index()
 counts['excluded_reverse']=counts.total_eligible-counts.valid_reverse;counts['coverage']=counts.valid_reverse/counts.total_eligible
 table(counts,ROOT/'outputs/composite/order_coverage')

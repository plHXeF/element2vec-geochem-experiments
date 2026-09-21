from src.common import config_digest
import json,sqlite3,sys,platform
from pathlib import Path
import numpy as np
import pandas as pd
from src.common import ROOT,sha,digest,save_json,split_for,table

def prepare(cfg):
 out=ROOT/'cache';marker=out/'data_manifest.json'
 if marker.exists():
  assert json.loads(marker.read_text())['config_hash']==config_digest(cfg),'Config changed; use a new run directory.'
  return pd.read_parquet(out/'records.parquet')
 with sqlite3.connect(f"file:{cfg['database']}?mode=ro",uri=True) as con:
  schema=[r[1] for r in con.execute(f"PRAGMA table_info({cfg['view']})")]
  cols=list(dict.fromkeys(['sample_id','rock_group','rock_origin','rock_type','canonical_ref_id','data_source']+cfg['features']+cfg['ree_measured']+cfg['isotope']+cfg['age_derived']))
  frame=pd.read_sql_query('SELECT '+','.join('"'+c+'"' for c in cols)+' FROM '+cfg['view']+' ORDER BY sample_id',con)
 assert frame.sample_id.is_unique
 rng=np.random.default_rng(cfg['seed']);selected=np.sort(rng.choice(len(frame),min(cfg['real_samples'],len(frame)),replace=False));records=frame.iloc[selected].reset_index(drop=True).copy()
 records['split']=[split_for(str(x),cfg['seed']) for x in records.sample_id]
 records.to_parquet(out/'records.parquet',index=False)
 save_json(marker,{'config_hash':config_digest(cfg),'database_sha256':sha(cfg['database']),'view_rows':len(frame),'selected_rows':len(records),'schema':schema,'sample_ids_hash':digest(records.sample_id.tolist()),'split_hash':digest(records[['sample_id','split']].values.tolist()),'split_method':'sample-ID hash 70/15/15; reference overlap allowed to measure nuisance; all records were exposed during model pretraining'})
 # Targets are computed from original records, never from masked inputs.
 y=pd.DataFrame(index=records.index);y['rock_group']=records.rock_group;y['rock_origin']=records.rock_origin;y['SiO2']=records.sio2
 mg=records.mgo/40.304;fe=records.feo/71.844;y['Mg_number']=100*mg/(mg+fe);y.loc[(mg<0)|(fe<0)|(mg+fe<=0),'Mg_number']=np.nan
 y['SigmaREE10']=records[cfg['ree_measured']].where(records[cfg['ree_measured']]>0).sum(axis=1,min_count=len(cfg['ree_measured']))
 y['total_alkali']=records.na2o+records.k2o;y['La_Yb']=records.la_ppm/records.yb_ppm;y['Sr_Y']=records.sr_ppm/records.y_ppm
 refs=cfg['baseline']['ree_reference_ppm'];y['Eu_anomaly']=(records.eu_ppm/refs['eu_ppm'])/np.sqrt((records.sm_ppm/refs['sm_ppm'])*(records.gd_ppm/refs['gd_ppm']))
 y=y.replace([np.inf,-np.inf],np.nan);y.to_parquet(out/'targets.parquet',index=False)
 maskcols=cfg['features']+cfg['isotope']+cfg['age_derived'];available=records[maskcols].notna().to_numpy();mask=np.zeros_like(available)
 for i in range(len(records)):
  js=np.flatnonzero(available[i]);chosen=rng.choice(js,max(1,int(round(.2*len(js)))),replace=False);mask[i,chosen]=True
 np.save(out/'recovery_mask.npy',mask);save_json(out/'recovery_columns.json',maskcols)
 audit=ROOT/'outputs/audit';shared=[]
 import torch
 for idx,spec in enumerate(cfg['experiments'],1):
  ck=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False,mmap=True);fields=[f['db_column'] for f in ck['config']['data']['numeric_fields']]
  item={'name':spec['name'],'checkpoint':spec['checkpoint'],'sha256':sha(spec['checkpoint']),'step':ck['step'],'config':ck['config'],'fields':fields,'sample_id':'sample_id','dataset':cfg['database'],'view':cfg['view'],'same_dataset':True,'pretraining_split':'none; full 270241 rows','evaluation_split':'new fixed sample-ID 70/15/15, not held-out pretraining','latent':'last layer contextual numeric-token arithmetic mean, 768D; all label tokens removed','missing':'omit token through attention mask; zero padding is not an observed token','synthetic_supported':True,'stats':str(Path(spec['vocab'])/'value_transform_stats.csv'),'stats_sha256':sha(Path(spec['vocab'])/'value_transform_stats.csv'),'load':'strict build_model(checkpoint.config, frozen vocab), load_state_dict; original backward-compatible branch implementation'}
  shared.append(item);save_json(audit/f'experiment_{idx}.json',item)
  (audit/f'experiment_{idx}.md').write_text('# '+spec['name']+' audit\n\n'+ '\n'.join(f'- **{k}**: `{v}`' for k,v in item.items() if k!='config')+'\n\nFull checkpoint config is in the adjacent JSON.\n')
  del ck
 save_json(audit/'compatibility.json',{'experiments':shared,'paired_samples':'all three use the same full source view and sample ID intersection','features':cfg['features'],'target_notes':{'SigmaREE10':'sum of all 10 REE available in database; never called complete 14-element total','Eu_anomaly':'external target: Sm/Eu/Gd absent from all six encoders','Mg_number':'100*molar MgO/(MgO+FeO); no unverified ferric-to-ferrous conversion'},'unsupported_input_groups':{'isotope':cfg['isotope'],'age_derived':cfg['age_derived']},'geo_difference':'negative major/CO2 and nonpositive trace may be omitted by native geo transform; no model-specific synthetic clipping'})
 return records

def task_inputs(records,cfg):
 fields={'unmasked':[],'SiO2':['sio2'],'Mg_number':['mgo','feo','fe2o3'],'SigmaREE10':cfg['ree_measured']}
 for task,mask in fields.items():
  x=records.reindex(columns=cfg['features']).copy()
  for c in mask:
   if c in x:x[c]=np.nan
  yield task,x
 mask=np.load(ROOT/'cache/recovery_mask.npy');cols=json.loads((ROOT/'cache/recovery_columns.json').read_text());x=records.reindex(columns=cfg['features']).copy()
 for j,c in enumerate(cols):
  if c in x:x.loc[mask[:,j],c]=np.nan
 yield 'recovery',x

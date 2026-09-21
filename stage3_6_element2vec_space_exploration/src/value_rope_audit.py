"""Paired V2/continuous-value RoPE audit; all generated records stay private.

Run with --help. Inputs, selected rows, perturbations, and projection pools are
persisted before model evaluation. No labels enter chemistry-only inference.
"""
from __future__ import annotations

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[2]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args
if __name__ == "__main__":
    _require_run_dir()

from geochem_workspace import asset, output_path

import argparse
import copy
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from stage3_5_element2vec_v2.src.data.dataset import load_vocab_bundle
from stage3_5_element2vec_v2.src.models.element2vec_v2 import build_model
from stage3_5_element2vec_v2.src.models.transformer_encoder import apply_rotary_embedding
from stage3_5_element2vec_v2.src.utils.io import load_config
from .extract_cache import _add_transformed_values
from .experiments_process import _close_major

STEPS = (.025, .05, .1, .25, .5, 1.)
LAYERS = (1, 3, 6, 9, 12)
MODELS = ('v2', 'rope', 'random_rope', 'random_v2')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n')


def settings():
    cfg = load_config('stage3_5_element2vec_v2/configs/v2_value_rope_z.yaml')
    vocab = load_vocab_bundle(cfg)
    return cfg, vocab, vocab.numeric_vocab['numeric_tags']


def prepare(args):
    out = args.output
    if (out/'protocol.json').exists() and not args.force:
        raise FileExistsError('This audit is already prepared. Use a new --output, or --force followed by extract --force.')
    out.mkdir(parents=True, exist_ok=True)
    cfg, vocab, tags = settings()
    meta = pd.read_csv(args.legacy / 'cache/sample_metadata.csv', low_memory=False)
    fresh = _add_transformed_values(meta, cfg, vocab)
    zcols = [f'z__{t}' for t in tags]
    np.testing.assert_allclose(meta[zcols], fresh[zcols], atol=1e-10, equal_nan=True)
    meta['background'] = np.select([meta.sio2 < 52, meta.sio2 >= 63,
                                    meta.sio2.between(52, 63, inclusive='left')],
                                   ['mafic', 'felsic', 'intermediate'], default='unknown')
    meta.to_csv(out / 'metadata.csv', index=False)
    z = meta[zcols].to_numpy(float)
    pools = asset('subspace_audit') / 'projections/fitting_pools'
    id_to_pos = pd.Series(np.arange(len(meta)), index=meta.sample_id)
    fit = id_to_pos.loc[pd.read_csv(pools / 'sample_fit_pool.csv').sample_id].to_numpy()
    ev = id_to_pos.loc[pd.read_csv(pools / 'sample_evaluation_pool.csv').sample_id].to_numpy()
    assert len(set(fit) & set(ev)) == 0 and len(fit) + len(ev) == len(meta)
    np.savez(out / 'global_inputs.npz', z=z, fit=fit, evaluation=ev)
    rng = np.random.default_rng(42)
    backgrounds = []
    for bg in ('mafic', 'intermediate', 'felsic'):
        ids = np.flatnonzero((meta.background == bg) & (meta.num_numeric_tokens >= 20))
        for i in rng.choice(ids, 10, replace=False):
            backgrounds.append((f'natural_{int(meta.iloc[i].sample_id)}', bg, z[i], int(i)))
        med = np.nanmedian(z[meta.background == bg], axis=0)
        backgrounds.append((f'median_{bg}', bg, med, -1))
    rows, arrays = [], []

    def add(values, **record):
        record['row'] = len(rows)
        rows.append(record)
        arrays.append(np.array(values, dtype=float))

    # The derivative panel preserves each natural sample's reporting mask.
    for name, bg, values, pos in backgrounds:
        for j, tag in enumerate(tags):
            if not np.isfinite(values[j]) or abs(values[j]) > 11:
                continue
            common = dict(family='derivative', case=f'{name}:{tag}', background=bg,
                          sample=name, natural=pos >= 0, element=tag, target=j)
            add(values, h=0., **common)
            for h in STEPS:
                for sign in (-1, 1):
                    v = values.copy(); v[j] += sign*h
                    add(v, h=sign*h, **common)
    sweep_tags = ['SiO2', 'MgO', 'CaO', 'Rb_ppm', 'Sr_ppm', 'Ni_ppm']
    # All three medians plus two independently chosen natural panels per group.
    sweep_bases = [b for b in backgrounds if b[3] < 0]
    for bg in ('mafic', 'intermediate', 'felsic'):
        sweep_bases += [b for b in backgrounds if b[1] == bg and b[3] >= 0][:2]
    for name, bg, values, pos in sweep_bases:
        for tag in sweep_tags:
            j = tags.index(tag)
            if not np.isfinite(values[j]):
                continue
            for scope, grid in [('local', np.linspace(-1, 1, 41) + values[j]),
                                ('wide', np.linspace(-12, 12, 97))]:
                if np.max(np.abs(grid)) > 12:
                    continue
                for t in grid:
                    v = values.copy(); v[j] = t
                    add(v, family='sweep', case=f'{name}:{tag}:{scope}', sample=name,
                        natural=pos >= 0, background=bg, element=tag, target=j,
                        scope=scope, t=t)
    groups = {'mafic': {'SiO2':-1, 'MgO':1, 'FeO':1, 'CaO':1},
              'incompatible': {'Rb_ppm':1, 'Th_ppm':1, 'U_ppm':1, 'La_ppm':1, 'Ce_ppm':1},
              'compatible': {'Ni_ppm':1, 'Cr_ppm':1, 'Co_ppm':1},
              'weathering': {'Na2O':-1, 'CaO':-1, 'Al2O3':1}}
    for name, bg, values, pos in backgrounds:
        valid = np.isfinite(values)
        # Equal L2 z displacement is essential for comparing sensitivities.
        directions = {'common_z': valid.astype(float)}
        if valid[tags.index('Rb_ppm')]:
            directions['single_Rb'] = np.eye(len(tags))[tags.index('Rb_ppm')]
        if valid[tags.index('Rb_ppm')] and valid[tags.index('Sr_ppm')]:
            directions['contrast_Rb_Sr'] = (np.eye(len(tags))[tags.index('Rb_ppm')]
                                           - np.eye(len(tags))[tags.index('Sr_ppm')])
        for kind, d in directions.items():
            d /= np.linalg.norm(d)
            for h in (0., .025, .1, .25, .5, 1., 2.):
                v = values + h*d
                add(np.clip(v, -12, 12), family='shift', case=f'{name}:{kind}',
                    sample=name, natural=pos>=0, background=bg, kind=kind, h=h,
                    clipped=int(np.sum(np.abs(v[valid]) > 12)))
        # True shared multiplicative change of measured trace concentrations.
        stats = vocab.transform_stats.set_index('db_column')
        d = np.zeros(len(tags))
        for j, field in enumerate(cfg['data']['numeric_fields']):
            if field['log_transform'] and valid[j]:
                d[j] = 1 / max(float(stats.loc[field['db_column'], 'iqr']), 1e-6)
        for h in (0., .025, .1, .25, .5):
            v = values + h*d
            add(np.clip(v, -12, 12), family='shift', case=f'{name}:trace_common_log10',
                sample=name, natural=pos>=0, background=bg, kind='trace_common_log10',
                h=h, input_norm=float(np.linalg.norm(h*d)), clipped=int(np.sum(np.abs(v[valid])>12)))
        for group, changes in groups.items():
            js = [(tags.index(t), s) for t, s in changes.items()]
            if not all(valid[j] and abs(values[j]) <= 11 for j, _ in js):
                continue
            for h in (.025, .1, .25, .5, 1.):
                common = dict(family='joint', case=f'{name}:{group}:{h}', sample=name,
                              natural=pos>=0, background=bg, group=group, h=h)
                add(values, component='base', **common)
                for j, sign in js:
                    v=values.copy(); v[j]+=sign*h
                    add(v, component=tags[j], **common)
                v=values.copy()
                for j, sign in js: v[j]+=sign*h
                add(v, component='joint', **common)
    # Replay exact persisted original compositions, including their parameter grid.
    process_specs = [('mixing', 'E1_two_endmember_mixing/synthetic_mixing_compositions.csv',
                      ['mix_pair', 'closure'], 'alpha'),
                     ('melting', 'E4_partial_melting/partial_melting_compositions.csv',
                      ['source', 'residual'], 'melt_fraction')]
    for process, path, grouping, parameter in process_specs:
        raw = pd.read_csv(args.legacy / 'results' / path)
        zz = _add_transformed_values(raw, cfg, vocab)[zcols].to_numpy()
        for i, row in raw.iterrows():
            add(zz[i], family='process', case=process+':'+':'.join(str(row[g]) for g in grouping),
                process=process, t=float(row[parameter]))
    dbcols = [f['db_column'] for f in cfg['data']['numeric_fields']]
    base = meta.loc[meta.rock_type.eq('wacke'), dbcols].median()
    for branch in ('leaching', 'k_metasomatism'):
        raw=[]
        for t in np.linspace(0, 1, 21):
            row=base.copy()
            for c, amount in [('na2o', .75), ('cao', .70), ('mgo', .45), ('sio2', .15)]: row[c]*=1-amount*t
            row['k2o'] *= 1-.25*t if branch=='leaching' else 1+.8*t
            raw.append(row)
        zz=_add_transformed_values(_close_major(pd.DataFrame(raw)), cfg, vocab)[zcols].to_numpy()
        for t, v in zip(np.linspace(0, 1, 21), zz):
            add(v, family='process', case=f'weathering:{branch}', process='weathering', t=t)
    # Continuous Rayleigh proxy is an extension; original E3 used class centroids.
    base=meta.loc[meta.rock_type.eq('subalkalic basalt'), dbcols].median()
    ds={'rb_ppm':.01, 'sr_ppm':.2, 'ba_ppm':.01, 'zr_ppm':.1, 'nb_ppm':.01,
        'y_ppm':.2, 'v_ppm':1., 'cr_ppm':3., 'ni_ppm':5., 'cu_ppm':.5,
        'zn_ppm':1., 'la_ppm':.01, 'ce_ppm':.02, 'nd_ppm':.05, 'th_ppm':.001,
        'u_ppm':.001, 'pb_ppm':.1, 'sc_ppm':1., 'co_ppm':2., 'ga_ppm':.5, 'yb_ppm':.3}
    raw=[]
    for t in np.linspace(0, .9, 37):
        row=base.copy()
        for c, d in ds.items(): row[c]*=(1-t)**(d-1)
        raw.append(row)
    zz=_add_transformed_values(pd.DataFrame(raw), cfg, vocab)[zcols].to_numpy()
    for t, v in zip(np.linspace(0, .9, 37), zz):
        add(v, family='process', case='differentiation:Rayleigh_proxy', process='differentiation_proxy', t=t)
    # Fix the queried element to exactly z=0; background alone can carry signal.
    for tag in ['Ni_ppm','Sr_ppm','Zr_ppm','Rb_ppm','K2O','CaO']:
        j=tags.index(tag)
        for bg in ('mafic','intermediate','felsic'):
            ids=np.flatnonzero((meta.background==bg) & np.isfinite(z[:,j]))
            for i in rng.choice(ids, min(100,len(ids)), replace=False):
                v=z[i].copy();v[j]=0
                add(v, family='context', case=tag, element=tag, target=j,
                    background=bg, sample=int(meta.iloc[i].sample_id), source=meta.iloc[i].canonical_ref_id)
    # Identical deletion masks in all models, both random and an analytical panel.
    for i in ev[:400]:
        for rate in (0., .1, .25, .5, -1.):
            v=z[i].copy(); valid=np.flatnonzero(np.isfinite(v))
            if rate==-1: v[13:]=np.nan
            elif rate>0:
                removed=rng.choice(valid, min(len(valid)-1, max(1,int(round(rate*len(valid))))),replace=False)
                v[removed]=np.nan
            add(v, family='dropout', case=str(int(meta.iloc[i].sample_id)), sample=int(meta.iloc[i].sample_id),
                global_pos=int(i), rate=rate)
    # This panel has no target labels or sample metadata consumed by the model.
    np.save(out/'stimulus_z.npy', np.stack(arrays).astype(np.float32))
    pd.DataFrame(rows).to_csv(out/'stimulus_index.csv', index=False)
    save_json(out/'protocol.json', dict(seed=42, global_samples=len(meta), fit=len(fit), evaluation=len(ev),
              steps=STEPS, layers=LAYERS, families=pd.Series([r['family'] for r in rows]).value_counts().to_dict(),
              sample_hash=sha(out/'metadata.csv'), input_hash=sha(out/'stimulus_z.npy'),
              checkpoints={m:sha(p) for m,p in [('v2',args.v2_checkpoint),('rope',args.rope_checkpoint)]},
              models=MODELS, random_initialization_seeds=[42], sample_backgrounds=30, synthetic_backgrounds=3))
    print(f'Prepared {len(meta)} paired samples and {len(rows)} stimuli',flush=True)


def tensor_batch(z, vocab, device, labels=None):
    b,n=z.shape
    ids=[vocab.numeric_vocab['tag_to_idx'][t] for t in vocab.numeric_vocab['numeric_tags']]
    val=torch.tensor(np.nan_to_num(z,nan=0.),dtype=torch.float32,device=device)
    valid=torch.tensor(np.isfinite(z),device=device)
    batch=dict(kind=torch.zeros(b,n,dtype=torch.long,device=device),
               numeric_tag_ids=torch.tensor(ids,device=device).expand(b,-1),
               pure_label_ids=torch.zeros(b,n,dtype=torch.long,device=device),
               values=val, mask_tag_flags=torch.zeros_like(val), mask_value_flags=torch.zeros_like(val),
               attention_mask=valid, numeric_token_mask=valid)
    if labels is not None:
        fields=['rock_group','rock_origin','rock_type']
        pureids=[vocab.pure_vocab['token_to_idx'][vocab.pure_vocab['mask_tokens'][f]] for f in fields]
        present=torch.tensor(labels[fields].notna().to_numpy(),device=device)
        for key in batch:
            if key=='kind': ext=torch.ones(b,3,dtype=torch.long,device=device)
            elif key=='pure_label_ids': ext=torch.tensor(pureids,device=device).expand(b,-1)
            elif key in ('attention_mask','mask_tag_flags'): ext=present.to(batch[key].dtype)
            else: ext=torch.zeros(b,3,dtype=batch[key].dtype,device=device)
            batch[key]=torch.cat([batch[key],ext],dim=1)
    return batch


@torch.inference_mode()
def infer(model, z, vocab, device, batch_size=128, targets=None, layers=False, labels=None):
    results={}
    for start in range(0,len(z),batch_size):
        end=min(start+batch_size,len(z))
        batch=tensor_batch(z[start:end],vocab,device,None if labels is None else labels.iloc[start:end])
        valid=batch['numeric_token_mask']
        def pool(x):
            return (x*valid.unsqueeze(-1)).sum(1)/valid.sum(1).clamp_min(1).unsqueeze(-1)
        indices=torch.tensor(np.maximum(targets[start:end],0),device=device) if targets is not None else None
        rowidx=torch.arange(end-start,device=device)
        local={}
        handles=[]
        if layers:
            def hook(level):
                def capture(module, inputs, output):
                    local[f'layer{level}']=pool(output[:,1:])
                return capture
            enc=model.transformer_encoder
            seq=enc.layers if model.architecture=='value_rope' else enc.encoder.layers
            for level in LAYERS[:-1]: handles.append(seq[level-1].register_forward_hook(hook(level)))
            if model.architecture=='value_rope':
                x=model.make_token_inputs(batch)
                x=torch.cat([model.cls_token.expand(len(x),-1,-1),x],1)
                att=seq[0].self_attn
                q,k,_=F.linear(seq[0].norm1(x),att.in_proj_weight,att.in_proj_bias).chunk(3,-1)
                pos=model.make_rope_positions(batch)
                for key,value in [('q1',q),('k1',k)]:
                    value=value.reshape(len(x),x.shape[1],att.num_heads,att.head_dim).transpose(1,2)
                    value=apply_rotary_embedding(value,pos,att.rotary_dim,base=att.rope_base,scale=att.rope_scale)
                    value=value.transpose(1,2).flatten(2)
                    local[key]=value[rowidx,indices+1] if indices is not None else pool(value[:,1:])
        try: enc=model.encode(batch)
        finally:
            for h in handles:h.remove()
        local['input']=pool(enc['projected_tokens'])
        local['mean']=pool(enc['encoded'][:,1:])
        local['cls']=enc['encoded'][:,0]
        if indices is not None:local['target']=enc['encoded'][rowidx,indices+1]
        for key,x in local.items():results.setdefault(key,[]).append(x.cpu().numpy())
        if (start//batch_size)%40==0: print(f'  inference {end}/{len(z)}',flush=True)
    return {k:np.concatenate(v) for k,v in results.items()}


def extract(args):
    cfg,vocab,tags=settings()
    meta=pd.read_csv(args.output/'metadata.csv',low_memory=False)
    zz=np.load(args.output/'global_inputs.npz')['z']
    stimuli=np.load(args.output/'stimulus_z.npy')
    index=pd.read_csv(args.output/'stimulus_index.csv',low_memory=False)
    targets=index.target.fillna(-1).to_numpy(int)
    checks={}
    for name in args.models:
        if (args.output/f'{name}_stimuli.npz').exists() and (args.output/f'{name}_global.npz').exists() and not args.force:
            saved=json.loads((args.output/f'{name}_extraction.json').read_text())
            if saved.get('protocol_sha256')!=sha(args.output/'protocol.json'):
                raise RuntimeError('Cache provenance differs or predates provenance checks; use extract --force.')
            path=args.rope_checkpoint if 'rope' in name else args.v2_checkpoint
            if saved.get('checkpoint_sha256')!=sha(path):
                raise RuntimeError('Checkpoint changed; use a new audit directory.')
            if not (args.output/f'{name}_masked.npz').exists():
                raise RuntimeError('Incomplete extraction cache; use extract --force.')
            print(f'{name}: cached',flush=True);continue
        t=time.time()
        path=args.rope_checkpoint if 'rope' in name else args.v2_checkpoint
        ckpt=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        config=copy.deepcopy(ckpt['config']);config['data']['vocab_dir']=cfg['data']['vocab_dir']
        assert ckpt['numeric_vocab']==vocab.numeric_vocab and ckpt['pure_vocab']==vocab.pure_vocab
        if 'value_transform_stats' in ckpt:
            assert ckpt['value_transform_stats']==vocab.transform_stats.to_dict(orient='records')
        torch.manual_seed(42)
        model=build_model(config,vocab)
        if not name.startswith('random'):model.load_state_dict(ckpt['model_state_dict'])
        model.to(args.device).eval()
        checks[name]=dict(step=0 if name.startswith('random') else ckpt['step'],
                          parameters=sum(p.numel() for p in model.parameters()),
                          architecture=model.architecture,config=config,
                          protocol_sha256=sha(args.output/'protocol.json'),checkpoint_sha256=sha(path))
        del ckpt
        print(f'Extracting {name} on {args.device}',flush=True)
        global_result=infer(model,zz,vocab,args.device,args.batch_size)
        if name=='v2':
            cached=np.load(args.legacy/'cache/latents_numeric_only.npz')['encoded_mean_numeric']
            checks[name]['old_cache_relative_error']=float(np.linalg.norm(global_result['mean']-cached)/np.linalg.norm(cached))
        np.savez(args.output/f'{name}_global.npz',**global_result)
        ev=np.load(args.output/'global_inputs.npz')['evaluation'][:2000]
        masked=infer(model,zz[ev],vocab,args.device,args.batch_size,labels=meta.iloc[ev])
        np.savez(args.output/f'{name}_masked.npz',indices=ev,**masked)
        result=infer(model,stimuli,vocab,args.device,args.batch_size,targets=targets,layers=True)
        np.savez(args.output/f'{name}_stimuli.npz',**result)
        # Device sensitivity: paired ±.025 perturbations need to exceed FP32 noise.
        subset=index.loc[index.family.eq('derivative')].head(26).index.to_numpy()
        model.to('cpu')
        cpu=infer(model,stimuli[subset],vocab,'cpu',8)['mean']
        checks[name]['cpu_mps_relative_error']=float(np.linalg.norm(cpu-result['mean'][subset])/np.linalg.norm(cpu))
        checks[name]['elapsed_seconds']=time.time()-t
        save_json(args.output/f'{name}_extraction.json',checks[name])
        del model,result,global_result;gc.collect()
        if args.device=='mps':torch.mps.empty_cache()
        print(f'{name} finished in {time.time()-t:.1f}s',flush=True)


@torch.inference_mode()
def final_probe(args):
    """Evaluate both final checkpoints on one regenerated fixed training probe."""
    from stage3_5_element2vec_v2.src.data.dataset import WholeRockTokenDataset
    from stage3_5_element2vec_v2.src.training.train_v2 import build_fixed_probe_batches, move_batch_to_device
    from stage3_5_element2vec_v2.src.training.losses import compute_losses
    from sklearn.metrics import r2_score
    cfg,vocab,_=settings()
    if args.sqlite_path is None:raise ValueError('--sqlite-path is required for probe')
    cfg['data']['sqlite_path']=str(args.sqlite_path)
    dataset=WholeRockTokenDataset(cfg,vocab)
    batches=build_fixed_probe_batches(dataset,cfg)
    print(f'Fixed probe: {len(dataset)} training rows, {sum(len(b["sample_ids"]) for b in batches)} probe samples',flush=True)
    output=[];records=[]
    for name,path in [('v2',args.v2_checkpoint),('rope',args.rope_checkpoint)]:
        ckpt=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        torch.manual_seed(42);model=build_model(ckpt['config'],vocab).to(args.device).eval()
        model.load_state_dict(ckpt['model_state_dict']);del ckpt
        totals={}
        for raw in batches:
            batch=move_batch_to_device(raw,torch.device(args.device));result=model(batch)
            losses=compute_losses(result,batch,cfg['data']['pure_label_fields'],cfg['training']['loss_weights'])
            for k,v in losses.items():totals[k]=totals.get(k,0.)+float(v.cpu())/len(batches)
            dec=result['decoder_outputs']
            for i,task in enumerate(raw['task_ids'].tolist()):
                rec=dict(model=name,sample_id=raw['sample_ids'][i],task=task,field=raw['pure_fields'][i])
                if task==0:
                    rec.update(target=float(raw['numeric_value_targets'][i]),prediction=float(dec['numeric_value'][i].cpu()))
                elif task==1:
                    rec.update(target=int(raw['numeric_tag_targets'][i]),prediction=int(dec['numeric_tag'][i].argmax().cpu()))
                else:
                    rec.update(target=int(raw['pure_class_targets'][i]),prediction=int(dec['pure_labels'][rec['field']][i].argmax().cpu()))
                records.append(rec)
        frame=pd.DataFrame([r for r in records if r['model']==name]);r=frame[frame.task.eq(0)]
        totals.update(value_mae=float(abs(r.prediction-r.target).mean()),value_r2=float(r2_score(r.target,r.prediction)))
        for key,g in frame[frame.task.ne(0)].groupby(['task','field'],dropna=False):
            task='element' if key[0]==1 else key[1]
            totals[f'{task}_accuracy']=float(g.target.eq(g.prediction).mean())
            totals[f'{task}_n']=len(g)
        output.append(dict(model=name,step=13200,probe_size=len(frame),**totals))
        print(name,totals,flush=True)
        del model;gc.collect()
        if args.device=='mps':torch.mps.empty_cache()
    pd.DataFrame(output).to_csv(args.output/'final_probe_summary.csv',index=False)
    pd.DataFrame(records).to_csv(args.output/'final_probe_predictions.csv',index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','extract','analyze','probe'])
    p.add_argument('--legacy',type=Path,required=True,help='Private original Stage 3.6 directory')
    p.add_argument('--v2-checkpoint',type=Path,required=True)
    p.add_argument('--rope-checkpoint',type=Path,default=asset("rope_z_checkpoint"))
    p.add_argument('--output',type=Path,default=output_path("audit"))
    p.add_argument('--device',default='mps');p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--sqlite-path',type=Path,default=asset('whole_rock'))
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS));p.add_argument('--force',action='store_true')
    args=_guard_args(p.parse_args());torch.set_num_threads(8)
    _guard_args(args)
    if args.phase=='prepare':prepare(args)
    elif args.phase=='extract':extract(args)
    elif args.phase=='probe':final_probe(args)
    else:
        from .value_rope_metrics import analyze
        analyze(args)


if __name__=='__main__':main()

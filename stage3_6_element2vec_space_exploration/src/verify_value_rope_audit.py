"""Validate private paired-audit artifacts without publishing sample records."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .value_rope_audit import MODELS, sha, save_json


def verify(out):
    protocol=json.loads((out/'protocol.json').read_text())
    assert sha(out/'metadata.csv')==protocol['sample_hash']
    assert sha(out/'stimulus_z.npy')==protocol['input_hash']
    index=pd.read_csv(out/'stimulus_index.csv',low_memory=False)
    np.testing.assert_array_equal(index.row,np.arange(len(index)))
    inputs=np.load(out/'global_inputs.npz')
    assert inputs['z'].shape==(12000,34)
    assert len(set(inputs['fit'])&set(inputs['evaluation']))==0
    assert len(inputs['fit'])==8000 and len(inputs['evaluation'])==4000
    numeric=np.load(out/'stimulus_z.npy')
    assert numeric.shape==(len(index),34)
    for _,g in index[index.family.eq('derivative')].groupby('case'):
        assert len(g)==13 and g.h.is_unique
    for _,g in index[index.family.eq('sweep')].groupby('case'):
        assert len(g)==(41 if g.scope.iloc[0]=='local' else 97)
    controlled=pd.read_csv(out/'controlled_index.csv')
    controlled_z=np.load(out/'controlled_z.npy')
    np.testing.assert_allclose(controlled_z[np.arange(len(controlled)),controlled.target],controlled.h,atol=1e-7)
    checks={}
    for name in MODELS:
        for suffix,n in [('global',12000),('masked_full',12000),('stimuli',len(index))]:
            data=np.load(out/f'{name}_{suffix}.npz')
            for key in data.files:
                a=data[key]
                assert a.shape==(n,768),(name,suffix,key,a.shape)
                assert np.isfinite(a).all(),(name,suffix,key)
        a=np.load(out/f'{name}_controlled.npy')
        assert a.shape==(len(controlled),768) and np.isfinite(a).all()
        extraction=json.loads((out/f'{name}_extraction.json').read_text())
        assert extraction['cpu_mps_relative_error']<1e-5
        if name=='v2':assert extraction['old_cache_relative_error']<1e-5
        if 'rope' in name:
            a=np.load(out/f'{name}_stimuli.npz')['input']
            for _,g in index[index.family.eq('sweep')].groupby('case'):
                np.testing.assert_allclose(a[g.index],np.broadcast_to(a[g.index[0]],(len(g),768)),atol=1e-6,rtol=1e-6)
        checks[name]=dict(finite=True,shape_check=True,device_agreement=True)
    translation=pd.read_csv(out/'translation_anchor_control.csv')
    assert (translation.all_positions_including_cls_relative_error<1e-5).all()
    assert (translation.numeric_only_shift_relative_change>1e-4).all()
    quality=pd.read_csv(out/'results/quality_modes.csv')
    assert len(quality)==4*3*45
    for _,g in quality.groupby(['model','mode']):assert g.space.nunique()==45
    summary=dict(status='passed',models=checks,global_samples=12000,fit_samples=8000,evaluation_samples=4000,
                 primary_stimuli=len(index),controlled_stimuli=len(controlled),quality_modes_rows=len(quality),
                 constant_rope_input_checked=True,translation_anchor_checked=True,
                 input_fingerprints_checked=True)
    save_json(out/'results/final_verification.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('stage3_5_element2vec_v2/outputs/value_rope_z/audit'))
    verify(p.parse_args().output)

import benchmark_test_support  # read-only local fixtures and portable imports
from src.common import SOURCE_ROOT
from src.common import ROOT  # bootstrap local pyarrow before pandas imports
import json,unittest,ast
import numpy as np
from src.common import ROOT,NAMES,load_config,digest
from src.operators.metrics import trajectory_metrics
class BankTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.cfg=load_config();cls.index=json.loads((ROOT/'cache/process_bank/index.json').read_text())
 def test_no_trajectory_or_initial_leakage(self):
  groups={}
  for r in self.index:
   key=r['initial_state_id'];self.assertEqual(groups.setdefault(key,r['split']),r['split'])
 def test_serialized_shared_bank(self):
  ids=[r['episode_id'] for r in self.index];self.assertEqual(len(ids),len(set(ids)))
  for r in self.index[::157]:
   a=np.load(ROOT/'cache/process_bank'/r['state_file']);self.assertEqual(len(a['states']),len(r['progress_parameter']));self.assertTrue(np.isfinite(a['states'][:,:11]).all())
 def test_mass_balance_and_operator_laws(self):
  for r in [v for v in self.index if v['kind']=='Fractionation'][::100]:
   a=np.load(ROOT/'cache/process_bank'/r['state_file']);states=a['states'];F=a['progress'];removed=np.array(r['process_parameters'][0]['removed_masses']);mass=states[:,:11]*F[:,None]/100;np.testing.assert_allclose(mass[1:]+removed,mass[:-1],atol=1e-9)
  from src.process_bank.composite import trace_semigroup
  errors=trace_semigroup(np.arange(1,10.),np.linspace(.1,3,9),np.linspace(-.2,.5,9),.7,.6,.3,.8);self.assertLess(max(errors),1e-12)
 def test_all_representation_episode_pairing_definition(self):
  expected={r['episode_id'] for r in self.index}
  for name in NAMES:
   directory=ROOT/'cache/process_latents'/name
   if (directory/'manifest.json').exists():
    actual={p.stem for p in directory.glob('*.npz')};self.assertEqual(expected,actual)
    manifest=json.loads((directory/'manifest.json').read_text());self.assertEqual(manifest['bank_ids_hash'],digest([r['episode_id'] for r in self.index]))
 def test_no_projection_in_geometry_metric(self):
  code=(SOURCE_ROOT/'src/operators/metrics.py').read_text();self.assertNotIn('sklearn.decomposition',code)
  z=np.c_[np.arange(33),np.zeros(33)];m=trajectory_metrics(z,self.cfg);self.assertAlmostEqual(m['simplicity'],1);self.assertAlmostEqual(m['local_rank1'],1)
 def test_checkpoint_repeatability(self):
  for name in NAMES[3:]:
   p=ROOT/'cache/real'/name/'scale.json';self.assertTrue(p.exists());self.assertLess(json.loads(p.read_text())['repeated_encode_max_abs_error'],2e-5)
 def test_composite_strict_chaining_and_mass_fraction(self):
  from src.process_bank.episode import apply
  for r in [r for r in self.index if r['kind']=='Composite'][::37]:
   saved=np.load(ROOT/'cache/process_bank'/r['state_file'])['states'];x=np.array(r['initial_state'],dtype=float);offset=0
   for process,p,g in zip(r['process_sequence'],r['process_parameters'],r['stage_grids']):
    a=apply(process,x,p,np.array(g));np.testing.assert_allclose(a[0],x,equal_nan=True,atol=1e-9);np.testing.assert_allclose(a,saved[offset:offset+len(a)],equal_nan=True,atol=1e-8)
    if process=='Fractionation':np.testing.assert_allclose(a[:,:11].sum(1),a[0,:11].sum(),atol=1e-7)
    x=a[-1];offset+=len(a)-1
 def test_composite_source_type_balance(self):
  from collections import Counter
  c=Counter((r['source_type'],r['composite_type']) for r in self.index if r['kind']=='Composite');self.assertEqual(len(c),4);self.assertEqual(len(set(c.values())),1)
 def test_legal_chemistry_and_no_source_specific_clipping(self):
  from src.process_bank.episode import legal
  fs=json.loads((ROOT/'cache/process_bank/schema.json').read_text())['fields']
  for r in self.index:
   states=np.load(ROOT/'cache/process_bank'/r['state_file'])['states'];self.assertTrue(all(legal(x,fs,self.cfg) for x in states),r['episode_id'])
 def test_source_signature_cache_agrees_with_episode_content(self):
  for name in NAMES:
   directory=ROOT/'cache/process_latents'/name;path=directory/'input_signatures.json'
   if not path.exists():continue
   signatures=json.loads(path.read_text())
   for r in self.index:
    if (directory/(r['episode_id']+'.npz')).exists():self.assertEqual(signatures[r['episode_id']],digest([r.get('states_hash'),r['process_parameters']]))
 def test_batched_rollout_matches_sequential_without_teacher_forcing(self):
  import pickle
  from src.operators.linear import rollout,rollout_batch
  for name in ['ZVector','V2','RoPEgeo']:
   for process in ['Mixing','Fractionation','Alteration']:
    art=pickle.loads((ROOT/'cache/operators'/f'{name}_{process}.pkl').read_bytes())
    rows=[r for r in self.index if r['kind']==process][:3]
    raws=[np.load(ROOT/'cache/process_bank'/r['state_file']) for r in rows]
    starts=np.stack([np.load(ROOT/'cache/process_latents'/name/(r['episode_id']+'.npz'))['z'][0] for r in rows]).astype(float)
    controls=np.stack([r['theta'] for r in raws]);grids=np.stack([r['progress'] for r in raws])
    expected=np.stack([rollout(art,z,c,g) for z,c,g in zip(starts,controls,grids)])
    got=rollout_batch(art,starts,controls,grids)
    np.testing.assert_allclose(got,expected,rtol=1e-8,atol=1e-8)
    if process=='Mixing':
     n=(controls.shape[2]-12)//3;controls[:,:,n:2*n]=12345
     np.testing.assert_allclose(rollout_batch(art,starts,controls,grids),got,rtol=1e-8,atol=1e-8)

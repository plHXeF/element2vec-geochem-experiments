import benchmark_test_support  # read-only local fixtures and portable imports
from src.common import SOURCE_ROOT
from src.common import ROOT  # bootstrap local pyarrow before pandas imports
import unittest,numpy as np,pandas as pd
from src.common import load_config,ROOT,split_for
from src.representations.baselines import ZVectorRepresentation,GeoNormRepresentation,PCAGeoRepresentation
from src.data import task_inputs
from src.process_bank.minerals import composition,molecular_mass
class Sanity(unittest.TestCase):
 def setUp(self):self.c=load_config();self.records=pd.read_parquet(ROOT/'cache/records.parquet');self.train=self.records[self.records.split.eq('train')]
 def test_z_binary_mixing_is_straight(self):
  r=ZVectorRepresentation(self.c).fit_reference(self.train);a=self.train[self.c['features']].median().to_numpy();b=a*1.4+1;t=np.linspace(0,1,33);x=pd.DataFrame((1-t[:,None])*a+t[:,None]*b,columns=self.c['features']);z=r.encode(x);np.testing.assert_allclose(z,(1-t[:,None])*z[0]+t[:,None]*z[-1],atol=2e-6)
 def test_test_data_not_used_for_statistics(self):
  r=ZVectorRepresentation(self.c).fit_reference(self.train);mean=r.mean.copy();changed=self.records.copy();changed.loc[changed.split.eq('test'),self.c['features']]=1e20;r2=ZVectorRepresentation(self.c).fit_reference(changed[changed.split.eq('train')]);np.testing.assert_array_equal(mean,r2.mean)
 def test_target_mask_before_transform(self):
  inp=dict(task_inputs(self.records,self.c));self.assertTrue(inp['SiO2'].sio2.isna().all());self.assertTrue(inp['Mg_number'][['mgo','feo','fe2o3']].isna().to_numpy().all());self.assertTrue(inp['SigmaREE10'][self.c['ree_input']].isna().to_numpy().all())
  g=GeoNormRepresentation(self.c).fit_reference(self.train);a=self.records.iloc[:4].copy();b=a.copy();b['sio2']=9999;a['sio2']=np.nan;b['sio2']=np.nan;np.testing.assert_allclose(g.encode(a),g.encode(b))
 def test_stoichiometry(self):
  self.assertEqual(composition('Quartz',self.c),{'sio2':1.});ab=composition('Ab',self.c);self.assertAlmostEqual(sum(ab.values()),1);self.assertAlmostEqual(ab['sio2'],3*molecular_mass('SiO2',self.c['atomic_weights'])/molecular_mass('NaAlSi3O8',self.c['atomic_weights']))
 def test_initial_state_split_independent_of_process(self):
  for key in ['natural:1','synthetic:3']:
   self.assertEqual(split_for(key),split_for(key))
if __name__=='__main__':unittest.main()

class PersistedIsolation(unittest.TestCase):
 def test_all_representations_share_exact_masked_records(self):
  import json
  from src.common import NAMES
  for task in ['unmasked','SiO2','Mg_number','SigmaREE10','recovery']:
   hashes=[json.loads((ROOT/'cache/real'/n/(task+'.json')).read_text())['input_hash'] for n in NAMES];self.assertEqual(len(set(hashes)),1)
 def test_operator_actual_saved_initial_ids_do_not_overlap(self):
  import pickle
  paths=list((ROOT/'cache/operators').glob('*.pkl'));self.assertTrue(paths)
  for p in paths:
   a=pickle.loads(p.read_bytes());self.assertFalse(set(a['train_initial_ids'])&set(a['test_initial_ids']))
 def test_no_PCA_in_semantic_metrics(self):
  source=(SOURCE_ROOT/'src/semantic/evaluate.py').read_text();self.assertNotIn('sklearn.decomposition',source);self.assertIn("metric='euclidean'",source)
 def test_visualization_basis_not_reused_for_metrics(self):
  for name in ['src/semantic/evaluate.py','src/operators/metrics.py','src/operators/linear.py']:
   source=(SOURCE_ROOT/name).read_text();self.assertNotIn('visualization_pca',source)

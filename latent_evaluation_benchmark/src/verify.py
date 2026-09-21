from src.common import SOURCE_ROOT
"""Artifact, provenance and numerical acceptance checks after complete evaluation."""
import json,unittest,subprocess,shutil
import numpy as np
import pandas as pd
from src.common import ROOT,NAMES,sha,save_json,digest

def verify(cfg):
 results={};suite=unittest.defaultTestLoader.discover(str(SOURCE_ROOT/'tests'));test=unittest.TextTestRunner(verbosity=2).run(suite);results['automated_tests']={'run':test.testsRun,'failures':len(test.failures),'errors':len(test.errors),'passed':test.wasSuccessful()}
 idx=json.loads((ROOT/'cache/process_bank/index.json').read_text());ids={r['episode_id'] for r in idx};bankhash=digest([r['episode_id'] for r in idx]);actual={}
 for name in NAMES:
  folder=ROOT/'cache/process_latents'/name;got={p.stem for p in folder.glob('*.npz')};assert got==ids,(name,len(got),len(ids));m=json.loads((folder/'manifest.json').read_text());assert m['bank_ids_hash']==bankhash;actual[name]=len(got)
  for r in idx[::193]:
   a=np.load(folder/(r['episode_id']+'.npz'));raw=np.load(ROOT/'cache/process_bank'/r['state_file']);assert len(a['z'])==len(raw['states']);assert np.isfinite(a['z']).all()
 results['shared_episode_ids']=actual
 original=json.loads((ROOT/'outputs/audit/compatibility.json').read_text())
 results['checkpoint_unchanged']={e['name']:sha(e['checkpoint'])==e['sha256'] for e in original['experiments']};assert all(results['checkpoint_unchanged'].values())
 d=pd.read_parquet(ROOT/'outputs/decodability/metrics.parquet');assert len(d)==5*6*4;assert d.status.eq('ok').all();results['probe_curves']=30;results['probe_points']=len(d)
 for p in ['decodability_overview.png','sufficiency_vs_leakage.png','semantic_geometry.png','process_benchmark.png','process_operator_error.png','monte_carlo_convergence.png']:assert (ROOT/'outputs/figures'/p).stat().st_size>1000
 videos=list((ROOT/'outputs/animations').glob('*.mp4'))+list((ROOT/'outputs/animations').glob('*.gif'));assert len(videos)==12;results['animations']=len(videos)
 if shutil.which('ffprobe') and shutil.which('ffmpeg') and all(p.suffix=='.mp4' for p in videos):
  selected=json.loads((ROOT/'outputs/animations/selection.json').read_text())['episodes'];lookup={r['episode_id']:r for r in idx};video_checks={}
  for choice in selected:
   name=f"{choice['kind']}_{choice['rule']}.mp4";path=ROOT/'outputs/animations'/name
   probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,width,height,codec_name','-of','json',str(path)]))['streams'][0]
   assert int(probe['nb_frames'])==len(lookup[choice['episode_id']]['progress_parameter']);assert probe['width']>1000 and probe['height']>700
   subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True,capture_output=True);video_checks[name]=probe
  results['video_decode_checks']=video_checks
 pair=pd.read_parquet(ROOT/'outputs/tables/paired_statistics.parquet');assert pair.bootstrap_resamples.eq(2000).all();results['paired_comparisons']=len(pair);precision=json.loads((ROOT/'outputs/tables/precision_status.json').read_text());assert not precision['needs_extension'];results['precision']=precision
 comp=pd.read_parquet(ROOT/'outputs/composite/semigroup.parquet');assert comp.trace_semigroup_error.max()<1e-10;results['physical_trace_semigroup_max_error']=float(comp.trace_semigroup_error.max())
 results['all_passed']=test.wasSuccessful() and all(results['checkpoint_unchanged'].values());save_json(ROOT/'outputs/verification.json',results)
 if not results['all_passed']:raise RuntimeError('Sanity verification failed')
 return results

def preflight():
 """Sanity checks runnable before the expensive formal ProcessBank inference."""
 import sys
 sys.path.insert(0,str(SOURCE_ROOT/'tests'))
 from test_sanity import Sanity,PersistedIsolation
 from test_bank import BankTests
 suite=unittest.TestSuite();suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(Sanity))
 for name in ['test_no_trajectory_or_initial_leakage','test_serialized_shared_bank','test_mass_balance_and_operator_laws','test_no_projection_in_geometry_metric','test_checkpoint_repeatability','test_composite_strict_chaining_and_mass_fraction','test_composite_source_type_balance','test_legal_chemistry_and_no_source_specific_clipping']:
  suite.addTest(BankTests(name))
 suite.addTest(PersistedIsolation('test_all_representations_share_exact_masked_records'))
 result=unittest.TextTestRunner(verbosity=1).run(suite);save_json(ROOT/'outputs/preflight.json',{'tests':result.testsRun,'passed':result.wasSuccessful(),'failures':len(result.failures),'errors':len(result.errors)})
 if not result.wasSuccessful():raise RuntimeError('Preflight sanity checks failed; formal process inference is not started')

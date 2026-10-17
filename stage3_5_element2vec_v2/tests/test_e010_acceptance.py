"""Independent E010 acceptance: synthetic artifacts only, no real optimization."""
import copy
import importlib.util
import json
from pathlib import Path
import sqlite3
from contextlib import closing
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from stage3_5_element2vec_v2.tests.test_e010 import toy_data, tiny_config
from stage3_5_element2vec_v2.src.e010 import training as tr
from stage3_5_element2vec_v2.src.e010.prepare import save_json, digest_file
from stage3_5_element2vec_v2.src.e010.evaluation import extract, evaluate, neighbors_without_self
from stage3_5_element2vec_v2.src.e010.adapter import E010CheckpointRepresentation
from stage3_5_element2vec_v2.src.e010.schema import INDEX


class IndependentAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_identical_vectors_never_retain_self_neighbor(self):
        neighbors=neighbors_without_self(np.zeros((18,8)),15)
        self.assertEqual(neighbors.shape,(18,15))
        for i,row in enumerate(neighbors):
            self.assertNotIn(i,row)
            self.assertEqual(len(set(map(int,row))),15)

    def checkpoint(self, root, n=36):
        asset=root/'asset';asset.mkdir();data=toy_data(asset,n=n)
        cfg=tiny_config();cfg['expected_rows']=n
        tr.seed_all(42);model=tr.build(cfg,data)
        path=root/'initial.pt';tr.save_checkpoint(path,model,cfg,data,device=torch.device('cpu'))
        return data,cfg,path

    def test_adapter_batching_labels_and_alias_hiding(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);d,c,path=self.checkpoint(p)
            spec={'name':'synthetic','checkpoint':str(path)}
            records=pd.DataFrame({'sio2':[42.,51.,62.], 'sr_ppm':[100.,200.,300.], 'rock_group':['a','b','c']})
            a=E010CheckpointRepresentation(spec,batch_size=1)
            b=E010CheckpointRepresentation(spec,batch_size=3)
            np.testing.assert_allclose(a.encode(records),b.encode(records),rtol=1e-5,atol=1e-6)
            changed=records.copy();changed['sio2']*=2;changed['rock_group']='changed'
            np.testing.assert_allclose(a.encode(records,hidden_tags=['Si']),a.encode(changed,hidden_tags=['Si']),rtol=1e-5,atol=1e-6)
            self.assertEqual(a.encode(records.iloc[:0]).shape,(0,8))
            a.close();b.close()

    def test_adapter_honors_feature_contract_and_censor_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);d,c,path=self.checkpoint(p)
            spec={'name':'synthetic','checkpoint':str(path)}
            records=pd.DataFrame({'sio2':[50.,60.], 'sr_ppm':[100.,200.]})
            a=E010CheckpointRepresentation(spec,features=['sio2'])
            changed=records.copy();changed['sr_ppm']*=100
            np.testing.assert_allclose(a.encode(records),a.encode(changed),rtol=0,atol=0)
            a.close()
            ck=torch.load(path,weights_only=False)
            ck['censor_rules']=[{'canonical_ref_ids':[12],'column':'sr_ppm','encoding':'negative_detection_limit','evidence_url':'https://example.invalid/policy'}]
            torch.save(ck,path)
            a=E010CheckpointRepresentation(spec)
            records=pd.DataFrame({'sio2':[50.,50.],'sr_ppm':[-2.,-2.],'canonical_ref_id':[12,13]})
            result=a.encode(records)
            self.assertFalse(np.allclose(result[0],result[1]));a.close()

    def test_fixed_probe_leaves_weights_and_rng_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);d,c,path=self.checkpoint(p)
            pd.DataFrame({'row_index':np.arange(len(d))}).to_csv(d.root/'cohort.csv',index=False)
            c['training']['probe_size']=12
            model=tr.build(c,d)
            before={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
            report=tr.fixed_probe(model,d,c,torch.ones(len(d.stats)),torch.device('cpu'))
            self.assertEqual(report['rows'],12)
            for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
            torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)

    def test_resume_guard_requires_success_after_failure_and_current_hash(self):
        source=Path(__file__).resolve().parents[2]/'scripts/experiment.py'
        module_spec=importlib.util.spec_from_file_location('e010_management_fixture',source)
        management=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(management)
        module='stage3_5_element2vec_v2.src.e010.cli'
        def step(verb,code=0,resume=False):
            return {'command':['-m',module,verb]+(['--resume'] if resume else []),'exit_code':code}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'checkpoints').mkdir();ck=p/'checkpoints/latest.pt';ck.write_bytes(b'fixture')
            save_json(p/'training_verification.json',{'passed':True,'checkpoint_sha256':digest_file(ck)})
            good={'experiment_id':'E010','commands':[step('train',1),step('train',resume=True),step('verify-training')]}
            self.assertTrue(management.verified_e010_resume(good,p))
            for bad in [dict(good,experiment_id='E009'),dict(good,commands=[step('prepare',1),step('train',resume=True),step('verify-training')]),dict(good,commands=[step('train',1),step('verify-training'),step('train',resume=True)]),dict(good,commands=[step('train',1),step('train',resume=True)])]:
                self.assertFalse(management.verified_e010_resume(bad,p))
            ck.write_bytes(b'changed');self.assertFalse(management.verified_e010_resume(good,p))

    def test_untrained_extraction_and_scientific_audit_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);d,c,path=self.checkpoint(p)
            rows=np.arange(len(d));pool=pd.DataFrame({'row_index':rows,'sample_id':d.sample_ids,'component':[f'r:{i}' for i in rows],'split':np.repeat([0,1,2],12),'stratum':'synthetic'})
            pool.to_csv(d.root/'cohort.csv',index=False)
            tokens=pd.DataFrame([(int(r),INDEX['Sr']) for r in rows],columns=['row_index','tag_index']);tokens.to_csv(d.root/'token_cohort.csv',index=False)
            meta=pd.DataFrame({'row_index':rows,'data_source':['a' if i%2 else 'b' for i in rows],'material':'rock','canonical_ref_id':rows,'observed_count':4})
            with closing(sqlite3.connect(d.root/'metadata.sqlite')) as conn, conn:meta.to_sql('samples',conn,index=False)
            out=p/'run';out.mkdir();extract(path,d,out,batch_size=5)
            manifest=json.loads((out/'representations/manifest.json').read_text())
            self.assertEqual(manifest['mode'],'chemistry_only');self.assertEqual(manifest['rows'],36)
            np.testing.assert_allclose(np.load(out/'representations/mean.npy'),np.load(out/'representations/random_mean.npy'),rtol=1e-5,atol=1e-6)
            metrics=evaluate(d,out)
            self.assertTrue(metrics);self.assertTrue(all(np.isfinite(m['value']) for m in metrics))
            self.assertTrue(json.loads((out/'evaluation/scientific_audit.json').read_text())['passed'])
            recovery=json.loads((out/'evaluation/masked_recovery.json').read_text())
            self.assertTrue(any(r['field']=='Sr' and r['status']=='evaluated' for r in recovery))


if __name__=='__main__': unittest.main()

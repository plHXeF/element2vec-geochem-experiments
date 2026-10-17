"""E010 acceptance on synthetic fixtures; never optimize real-database models."""
from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from scipy.stats import t

from stage3_5_element2vec_v2.src.e010 import schema as s
from stage3_5_element2vec_v2.src.e010.prepare import (prepare_database, PreparedData, save_json, fit_stats, coordinates)
from stage3_5_element2vec_v2.src.e010.model import E010Model, DIMENSIONS, rotate_families
from stage3_5_element2vec_v2.src.e010.config import configuration
from stage3_5_element2vec_v2.src.e010.batching import make_batch
from stage3_5_element2vec_v2.src.e010.losses import log_t4_sf, concentration_nll, per_sample_loss
from stage3_5_element2vec_v2.src.e010 import training as tr
from stage3_5_element2vec_v2.src.e010.cli import verify_data, verify_training


def frame(rows):
    return pd.DataFrame(rows).reindex(columns=s.RAW_COLUMNS+["canonical_ref_id"])


def tiny_config():
    c = configuration("native", 8)
    c["expected_rows"] = 7
    c["model"].update(layers=1, dropout=.1)
    c["training"].update(device="cpu", effective_batch=4, micro_batch=2, epochs=2,
                         retain_epochs=[1,2], checkpoint_every=1, log_every=100, warmup_steps=1, probe_size=0)
    return c


def toy_data(root=None, n=7):
    f = frame([{"sio2": 40+i, "mgo": 2+i/10, "sr_ppm": 100+i, "canonical_ref_id": i} for i in range(n)])
    v, st = s.convert_frame(f); weights=np.ones(n, dtype=np.float32)
    stats = fit_stats(v, st, weights)
    coverage = {"by_field": {tag: {name: int((st[:,j] == k).sum()) for k,name in enumerate(s.STATE_NAMES)} for j,tag in enumerate(s.FIELDS)}}
    d = SimpleNamespace(values=v.astype(np.float32), states=st, sample_ids=np.arange(n)+1,
                        labels=np.array([[i%2,i%2,i%2] for i in range(n)]), weights=weights,
                        stats=stats, label_vocab={x:["a","b"] for x in s.LABELS}, coverage=coverage,
                        fingerprint="synthetic-fixture", root=Path(root) if root else None)
    class Data(SimpleNamespace):
        def __len__(self): return len(self.values)
    d=Data(**vars(d))
    if root:
        save_json(Path(root)/"dictionary.json", s.dictionary())
        np.save(Path(root)/"sample_ids.npy",d.sample_ids)
    return d


class SchemaTests(unittest.TestCase):
    def test_complete_vocabulary_and_serialization(self):
        self.assertEqual(len(s.ELEMENTS),118); self.assertEqual(len(set(s.ELEMENTS)),118)
        self.assertEqual(len(s.FIELDS),143)
        self.assertEqual(json.loads(json.dumps(s.dictionary())),s.dictionary())

    def test_oxide_conversion_does_not_overwrite_measured(self):
        v, st = s.convert_frame(frame([{"sio2":50}, {"sio2":50,"si_ppm":10}]))
        self.assertAlmostEqual(v[0,s.INDEX["Si"]],233718.3563,places=3)
        self.assertEqual(st[0,s.INDEX["Si"]],s.DERIVED)
        self.assertEqual(v[1,s.INDEX["Si"]],10)

    def test_iron_priority_and_partial_lower_bound(self):
        v, st=s.convert_frame(frame([{"feo_tot":10,"fe2o3_tot":20,"feo":1,"fe2o3":2},
                                    {"feo":1,"fe2o3":2},{"feo":1}]))
        f1=s.oxide_factor("Fe",1,1); f2=s.oxide_factor("Fe",2,3)
        np.testing.assert_allclose(v[:,s.INDEX["Fe"]],[10*f1,f1+2*f2,f1])
        np.testing.assert_array_equal(st[:,s.INDEX["Fe"]],[s.DERIVED,s.DERIVED,s.LOWER_BOUND])

    def test_bad_values_and_negative_loi_preserved(self):
        v, st=s.convert_frame(frame([{"loi":-10,"sr_ppm":-1,"h2o_tot":7808},{"sr_ppm":0}]))
        self.assertEqual(st[0,s.INDEX["LOI"]],s.MEASURED)
        self.assertEqual(st[0,s.INDEX["Sr"]],s.UNKNOWN)
        self.assertEqual(v[0,s.INDEX["Sr"]],-1)
        self.assertEqual(st[0,s.INDEX["H2O_tot"]],s.INVALID)
        self.assertEqual(st[1,s.INDEX["Sr"]],s.REPORTED_ZERO)

    def test_censor_rule_requires_exact_evidence(self):
        f=frame([{"sr_ppm":-2,"canonical_ref_id":12},{"sr_ppm":-2,"canonical_ref_id":13}])
        rule={"canonical_ref_ids":[12],"column":"sr_ppm","encoding":"negative_detection_limit","evidence_url":"https://example.invalid/exact-source-policy"}
        v, st=s.convert_frame(f,[rule]); self.assertEqual(st[0,s.INDEX["Sr"]],s.LEFT_CENSORED)
        self.assertEqual(st[1,s.INDEX["Sr"]],s.UNKNOWN)
        self.assertEqual(v[0,s.INDEX["Sr"]],2)
        with self.assertRaises(ValueError): s.convert_frame(f,[{**rule,"evidence_url":""}])

    def test_volatile_contributions_are_not_totals(self):
        v, st=s.convert_frame(frame([{"h2o_plus":1,"h2o_minus":2,"h2o_tot":4,"co2":3,"so3":5,"loi":8}]))
        for e in ("H","C","S"): self.assertEqual(st[0,s.INDEX[e]],s.LOWER_BOUND)
        self.assertEqual(st[0,s.INDEX["O"]],s.MISSING)
        self.assertAlmostEqual(v[0,s.INDEX["H"]],4*s.oxide_factor("H",2,1))

    def test_total_iron_oxide_equivalent_above_100_is_not_rejected(self):
        v, st=s.convert_frame(frame([{"fe2o3_tot":109}]))
        self.assertEqual(st[0,s.INDEX["Fe2O3_tot"]],s.MEASURED)
        self.assertEqual(st[0,s.INDEX["Fe"]],s.DERIVED)
        self.assertLess(v[0,s.INDEX["Fe"]],1e6)

    def test_alias_closure_includes_measured_and_derived(self):
        hidden=set(s.hidden_closure(["SiO2"]))
        self.assertEqual(hidden,{s.INDEX["SiO2"],s.INDEX["Si"]})
        self.assertTrue({s.INDEX[x] for x in ("Fe","FeO","Fe2O3","FeO_tot","Fe2O3_tot")} <= set(s.hidden_closure(["Fe"])))


class MathTests(unittest.TestCase):
    def test_student_survival_matches_scipy(self):
        x=torch.tensor([-1e5,-100.,-2.,0.,2.,100.,1e5],dtype=torch.float64)
        np.testing.assert_allclose(log_t4_sf(x).numpy(),t.logsf(x.numpy(),4),rtol=1e-8,atol=1e-10)

    def test_interval_likelihood_and_gradients(self):
        mu=torch.tensor([0.,0.,0.],requires_grad=True); scale=torch.ones(3,requires_grad=True)
        target=torch.tensor([1.,1.,1.]); zero=torch.tensor([-2.,-2.,-2.])
        states=torch.tensor([s.MEASURED,s.LEFT_CENSORED,s.LOWER_BOUND])
        result=concentration_nll(mu,scale,target,states,zero,torch.zeros(3,dtype=torch.bool))
        expected=[-t.logpdf(1,4)+t.logsf(-2,4),-np.log((t.cdf(1,4)-t.cdf(-2,4))/t.sf(-2,4)),-t.logsf(1,4)+t.logsf(-2,4)]
        np.testing.assert_allclose(result.detach().numpy(),expected,rtol=1e-5)
        result.sum().backward();self.assertTrue(torch.isfinite(mu.grad).all());self.assertTrue(torch.isfinite(scale.grad).all())

    def test_tail_gradients_finite(self):
        mu=torch.tensor([-1e5,1e5],requires_grad=True)
        x=concentration_nll(mu,torch.ones(2),torch.ones(2),torch.tensor([4,5]),torch.zeros(2),torch.zeros(2,dtype=torch.bool))
        x.sum().backward();self.assertTrue(torch.isfinite(x).all());self.assertTrue(torch.isfinite(mu.grad).all())

    def test_coordinate_monotonic_finite_and_bounded(self):
        d=toy_data();v=np.repeat(d.values[:1],101,axis=0);st=np.repeat(d.states[:1],101,axis=0)
        v[:,s.INDEX["Sr"]]=np.geomspace(1e-9,1e6,101)
        z=coordinates(v,st,d.stats)[:,s.INDEX["Sr"]];theta=np.arctan(z/2)
        self.assertTrue(np.all(np.diff(theta)>=0));self.assertTrue(np.all(abs(theta)<np.pi/2))
        stat=d.stats[s.INDEX["Sr"]]
        exact=np.arctan(((np.arcsinh(v[:,s.INDEX["Sr"]].astype(float)/stat['r'])-stat['m'])/stat['scale'])/2)
        self.assertTrue(np.all(np.diff(exact)>0))
        np.testing.assert_allclose(theta,exact,atol=2e-7)

    def test_rotation_norm_and_independent_families(self):
        x=torch.randn(2,2,4,16); p=torch.randn(2,4,2)
        y=rotate_families(x,p)
        torch.testing.assert_close(x.norm(dim=-1),y.norm(dim=-1))
        p2=p.clone();p2[...,0]+=.1;z=rotate_families(x,p2)
        torch.testing.assert_close(y[...,4:8],z[...,4:8])
        self.assertFalse(torch.equal(y[...,:4],z[...,:4]))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_all_sixteen_full_architectures_meta(self):
        d=toy_data()
        for arm in ("fixed","native"):
            for width in DIMENSIONS:
                with torch.device("meta"): m=tr.build(configuration(arm,width),d)
                self.assertEqual(len(m.layers),12)
                self.assertEqual(m.d_model,768 if arm=="fixed" else width)
                self.assertEqual(m.parameter_counts()["total"],sum(p.numel() for p in m.parameters()))

    def test_all_widths_forward_backward(self):
        d=toy_data()
        for width in DIMENSIONS:
            c=tiny_config();c["model"]["embedding_dim"]=width
            m=tr.build(c,d); b=make_batch(d,[0,1],m,forced_task=0)
            out=m(b);loss,_=per_sample_loss(out,b,torch.ones(len(s.FIELDS)))
            loss.mean().backward();self.assertTrue(torch.isfinite(loss).all())
            del m

    def test_values_only_change_rotary_path(self):
        d=toy_data();m=tr.build(tiny_config(),d).eval()
        b=make_batch(d,[0],m,training=False,chemistry_only=True)
        b2={k:v.clone() for k,v in b.items()};b2["z"]+=1
        a=m.encode(b,capture=True);c=m.encode(b2,capture=True)
        torch.testing.assert_close(a["static"],c["static"]);torch.testing.assert_close(a["projected"],c["projected"])
        self.assertFalse(torch.equal(a["rotated_q"],c["rotated_q"]))

    def test_masked_target_cannot_leak_through_alias(self):
        d=toy_data();m=tr.build(tiny_config(),d).eval()
        b=make_batch(d,[0],m,forced_task=0)
        targets=b["target_tag"][b["numeric_target"]].numpy()
        d2=copy.deepcopy(d);hidden=s.hidden_closure(targets)
        d2.values[0,hidden]*=3
        b2=make_batch(d2,[0],m,forced_task=0)
        torch.testing.assert_close(m.encode(b)["encoded"],m.encode(b2)["encoded"])
        for t0 in targets:
            group=s.GROUPS[t0]
            visible=b["target_tag"][b["mask"] & b["numeric"] & ~b["numeric_target"]].numpy()
            self.assertFalse(np.any(s.GROUPS[visible]==group))

    def test_microbatch_independent_mask_plan(self):
        d=toy_data();m=tr.build(tiny_config(),d)
        all_=make_batch(d,[0,1,2],m)
        for i in range(3):
            one=make_batch(d,[i],m)
            for key in ("tag_ids","z","numeric_target","target_tag","tag_target"):
                torch.testing.assert_close(all_[key][i,all_["mask"][i]],one[key][0,one["mask"][0]])

    def test_unseen_embedding_unchanged_after_optimizer(self):
        d=toy_data();c=tiny_config();m=tr.build(c,d);opt=tr.optimizer_for(m,c)
        row=m.numeric_tag_embedding.weight[s.INDEX["Pm"]].detach().clone()
        b=make_batch(d,[0,1],m,forced_task=1);out=m(b)
        loss,_=per_sample_loss(out,b,torch.ones(len(s.FIELDS)));loss.mean().backward();opt.step()
        torch.testing.assert_close(row,m.numeric_tag_embedding.weight[s.INDEX["Pm"]],rtol=0,atol=0)

    def test_token_permutation_invariant_latent(self):
        d=toy_data();m=tr.build(tiny_config(),d).eval();b=make_batch(d,[0],m,training=False,chemistry_only=True)
        p=torch.arange(b["mask"].shape[1]-1,-1,-1);b2={k:(v[:,p] if v.ndim==2 else v) for k,v in b.items()}
        torch.testing.assert_close(m.encode(b)["mean"],m.encode(b2)["mean"],rtol=1e-5,atol=1e-6)

    def test_bad_only_record_has_no_fabricated_numeric_target(self):
        d=toy_data();d.states[0]=0;d.states[0,s.INDEX["Sr"]]=s.UNKNOWN;d.labels[0]=-1
        m=tr.build(tiny_config(),d);b=make_batch(d,[0],m,forced_task=0)
        self.assertFalse(b["numeric_target"].any());self.assertTrue(b["tag_target"].any())


class ResumeTests(unittest.TestCase):
    def test_synthetic_resume_matches_uninterrupted_and_tail_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);asset=p/"asset";asset.mkdir();d=toy_data(asset)
            a=p/"a";b=p/"b";a.mkdir();b.mkdir();cfg=tiny_config()
            with contextlib.redirect_stdout(io.StringIO()): tr.train(cfg,d,a)
            original=tr.save_checkpoint;calls=0
            def interrupt(path,*args,**kwargs):
                nonlocal calls
                original(path,*args,**kwargs)
                if Path(path).name=="latest.pt":
                    calls+=1
                    if calls==1: raise KeyboardInterrupt("fixture interruption after durable checkpoint")
            with patch.object(tr,"save_checkpoint",interrupt),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt): tr.train(cfg,d,b)
            with contextlib.redirect_stdout(io.StringIO()): tr.train(cfg,d,b,resume=True)
            one=torch.load(a/"checkpoints/latest.pt",weights_only=False);two=torch.load(b/"checkpoints/latest.pt",weights_only=False)
            self.assertEqual(one["cursor"],two["cursor"])
            for k in one["model_state"]:torch.testing.assert_close(one["model_state"][k],two["model_state"][k],rtol=0,atol=0)
            self.assertEqual(verify_training(b,d,cfg)["presentations"],14)
            changed=copy.deepcopy(cfg);changed["training"]["epochs"]=3
            with self.assertRaises(ValueError):tr.train(changed,d,b,resume=True)


def make_database(path):
    c=sqlite3.connect(path)
    c.execute("CREATE TABLE sample(sample_id INTEGER PRIMARY KEY, major_id INTEGER, trace_id INTEGER,ref_id INTEGER,rgroup_id INTEGER,comp_id INTEGER,material TEXT,latitude REAL,longitude REAL)")
    for table,cols in [("major",[x for x in s.RAW_COLUMNS if not x.endswith('_ppm')]),("trace",[x for x in s.RAW_COLUMNS if x.endswith('_ppm')])]:
        c.execute(f"CREATE TABLE {table}({table}_id INTEGER PRIMARY KEY,"+','.join('"'+x+'" REAL' for x in cols)+')')
    c.execute("CREATE TABLE canonical_reference_map(ref_id INTEGER,canonical_ref_id INTEGER)")
    c.execute("CREATE TABLE reference(ref_id INTEGER PRIMARY KEY,data_source TEXT)")
    c.execute("CREATE TABLE dedup_decisions(sample_id INTEGER PRIMARY KEY,dup_group_id TEXT,is_kept INTEGER)")
    c.execute("CREATE TABLE rockgroup(rgroup_id INTEGER PRIMARY KEY,rock_group TEXT,rock_origin TEXT)")
    c.execute("CREATE TABLE computed(comp_id INTEGER PRIMARY KEY,rock_type TEXT)")
    for i in range(1,9):
        c.execute("INSERT INTO sample VALUES(?,?,?,?,?,?,?,?,?)",(i,i,i,i,i,i,None,None,None))
        c.execute("INSERT INTO major(major_id,sio2,loi) VALUES(?,?,?)",(i,40+i if i<7 else None,-1 if i==1 else None))
        c.execute("INSERT INTO trace(trace_id,sr_ppm) VALUES(?,?)",(i,-2 if i==7 else 100+i if i<7 else None))
        c.execute("INSERT INTO reference VALUES(?,?)",(i,'source'))
        c.execute("INSERT INTO dedup_decisions VALUES(?,?,?)",(i,'dup' if i in (1,2) else None,1 if i!=2 else 0))
        c.execute("INSERT INTO rockgroup VALUES(?,?,?)",(i,'a','b'))
        c.execute("INSERT INTO computed VALUES(?,?)",(i,'c'))
    c.commit();c.close()


class PreparationTests(unittest.TestCase):
    def test_full_coverage_immutable_source_and_group_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);db=p/"source.db";make_database(db)
            from stage3_5_element2vec_v2.src.e010.prepare import digest_file
            before=digest_file(db)
            with contextlib.redirect_stdout(io.StringIO()): report=prepare_database(db,p/"prepared",expected_rows=7,chunk_size=3)
            self.assertEqual(report["rows"],7);self.assertEqual(before,digest_file(db))
            with PreparedData(p/"prepared",rehash=True) as d:
                np.testing.assert_allclose(d.weights[:2],[.5,.5]);self.assertTrue(verify_data(d)["passed"])
                self.assertEqual(d.states[-1,s.INDEX["Sr"]],s.UNKNOWN)
            self.assertIsNone(d.values)
            d.close()  # idempotent; the temporary-directory cleanup must release every file
            with self.assertRaises(FileExistsError):prepare_database(db,p/"prepared")


if __name__=="__main__":unittest.main()

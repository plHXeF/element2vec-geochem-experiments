import copy
import unittest
import numpy as np
import pandas as pd
from stage3_5_element2vec_v2.src.utils.io import load_config
from stage3_5_element2vec_v2.src.data.dataset import load_vocab_bundle
from stage3_5_element2vec_v2.src.data.geochemical_coordinates import transform_numeric_frame
from stage3_5_element2vec_v2.tests.test_geo_mask15 import fixture
from stage3_6_element2vec_space_exploration.src.geo_mask15_geometry import physical_replay, derivatives


class PhysicalReplayTests(unittest.TestCase):
    def setUp(self):
        self.cfg=load_config("stage3_5_element2vec_v2/configs/v2_value_rope_geo_mask15.yaml")
        self.old=load_config("stage3_5_element2vec_v2/configs/v2_value_rope_z.yaml")
        self.vocab=load_vocab_bundle(self.cfg)
        self.meta=fixture(self.cfg,self.vocab)
        self.meta["sample_id"]=1234
        self.cols=[f["db_column"] for f in self.cfg["data"]["numeric_fields"]]
        transformed,_=transform_numeric_frame(self.meta,self.old,self.vocab)
        self.z=transformed[self.cols].to_numpy(np.float32)
        for f in self.cfg["data"]["numeric_fields"]:
            self.meta["z__"+f["name"]]=transformed[f["db_column"]]

    def test_preserves_original_natural_values(self):
        index=pd.DataFrame([{"sample":"natural_1234","family":"derivative"}])
        actual,flags,_=physical_replay(self.z,index,self.meta,self.cfg,self.vocab)
        expected,_=transform_numeric_frame(self.meta,self.cfg,self.vocab)
        np.testing.assert_allclose(actual,expected[self.cols],rtol=1e-6,atol=1e-7)
        self.assertTrue(flags.baseline_roundtrip.all())

    def test_drop_tokens_does_not_reclose_panel(self):
        z=self.z.copy();z[0,0]=np.nan
        index=pd.DataFrame([{"sample":1234,"family":"dropout","global_pos":0}])
        actual,_,_=physical_replay(z,index,self.meta,self.cfg,self.vocab)
        expected,_=transform_numeric_frame(self.meta,self.cfg,self.vocab)
        self.assertTrue(np.isnan(actual[0,0]))
        np.testing.assert_allclose(actual[0,1:],expected[self.cols].to_numpy()[0,1:],rtol=1e-6,atol=1e-7)

    def test_derivative_excludes_closure_switch(self):
        x=np.array([[0.,0.],[1.,0.],[-1.,0.]])
        index=pd.DataFrame({"case":["a"]*3,"h":[0.,.025,-.025],"admissible":[True]*3,
                            "closed":[True,False,True],"sample":["x"]*3,"natural":[True]*3,
                            "element":["SiO2"]*3,"background":["mafic"]*3})
        self.assertTrue(derivatives(x,index,steps=(.025,)).empty)
        index["closed"]=True
        self.assertAlmostEqual(derivatives(x,index,steps=(.025,)).direction_cos.iloc[0],1.)

    def test_zero_motion_has_no_direction_result(self):
        index=pd.DataFrame({"case":["a"]*3,"h":[0.,.025,-.025],"admissible":[True]*3,
                            "closed":[True]*3,"sample":["x"]*3,"natural":[True]*3,
                            "element":["SiO2"]*3,"background":["mafic"]*3})
        self.assertTrue(derivatives(np.zeros((3,2)),index,steps=(.025,)).empty)


if __name__=="__main__":unittest.main()

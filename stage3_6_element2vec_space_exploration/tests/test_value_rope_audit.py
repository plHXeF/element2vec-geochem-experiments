import unittest

import numpy as np

from stage3_6_element2vec_space_exploration.src.value_rope_metrics import cosine, path_metrics, purity


class GeometryMetricsTests(unittest.TestCase):
    def test_nonuniform_parameter_grid_has_constant_speed_for_linear_motion(self):
        t=np.array([0., .01, .2, .7, 1.])
        x=t[:,None]*np.array([[1.,2.,-3.]])
        result=path_metrics(x,t)
        self.assertAlmostEqual(result['speed_cv'],0.,places=12)
        self.assertAlmostEqual(result['tangent_cos'],1.,places=12)
        self.assertAlmostEqual(result['uniformity_rmse'],0.,places=12)

    def test_straight_but_nonuniform_path_is_detected(self):
        t=np.linspace(0,1,101)
        result=path_metrics(np.c_[t*t,np.zeros_like(t)],t)
        self.assertAlmostEqual(result['straightness'],1.)
        self.assertGreater(result['speed_cv'],.5)
        self.assertGreater(result['uniformity_rmse'],.1)

    def test_constant_path_is_not_labeled_perfectly_smooth(self):
        result=path_metrics(np.ones((10,3)),np.arange(10))
        self.assertTrue(result['zero_motion'])
        self.assertTrue(np.isnan(result['tangent_cos']))
        self.assertTrue(np.isnan(cosine(np.zeros(3),np.ones(3))))

    def test_circular_path_detects_turning_with_constant_speed(self):
        t=np.linspace(0,np.pi,101)
        result=path_metrics(np.c_[np.cos(t),np.sin(t)],t)
        self.assertAlmostEqual(result['speed_cv'],0.,places=10)
        self.assertAlmostEqual(result['turn_rate'],1.,places=8)
        self.assertLess(result['straightness'],.65)

    def test_unknown_references_are_not_counted_as_a_pure_cluster(self):
        nn=np.array([[1,2],[0,2],[0,1]])
        actual=purity(nn,['a','nan','a'])
        np.testing.assert_allclose(actual, [1.,np.nan,1.],equal_nan=True)

    def test_unknown_neighbors_do_not_invent_a_rock_class(self):
        from stage3_6_element2vec_space_exploration.src.value_rope_supplement import vote
        actual=vote(np.array([[1,2],[0,2]]),['basalt',np.nan,np.nan])
        self.assertEqual(actual.tolist(),['[unknown]','basalt'])


if __name__=='__main__':unittest.main()

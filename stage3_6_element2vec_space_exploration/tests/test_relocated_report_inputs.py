"""Paired report evidence must come from the declared upstream run after relocation."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from stage3_6_element2vec_space_exploration.src.build_subspace_report import _attach_paired_bootstrap


class RelocatedReportTests(unittest.TestCase):
    def test_paired_full_observations_are_read_from_explicit_upstream(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            full = base / "upstream/results/F2_single_element_jacobian"
            replay = base / "different-run/subspace_audit"
            projected = replay / "results/F/F2_single_element_jacobian__PCA32"
            full.mkdir(parents=True); projected.mkdir(parents=True)
            name = "single_element_finite_differences.csv"
            column = "cross_background_direction_cosine"
            pd.DataFrame({"field": ["sio2", "mgo"], column: [.1, .2]}).to_csv(full / name, index=False)
            pd.DataFrame({"field": ["sio2", "mgo"], column: [.6, .7]}).to_csv(projected / name, index=False)
            matrix = pd.DataFrame({"experiment_id": ["F2_single_element_jacobian"], "projection_method": ["PCA32"], "stability": [""], "bootstrap_ci_low": [np.nan], "bootstrap_ci_high": [np.nan]})
            result = _attach_paired_bootstrap(matrix, replay, full.parent, repeats=20)
            self.assertAlmostEqual(result.iloc[0].bootstrap_ci_low, .5)
            self.assertAlmostEqual(result.iloc[0].bootstrap_ci_high, .5)
            self.assertIn("n=2", result.iloc[0].stability)


if __name__ == "__main__":
    unittest.main()

"""Synthetic fixtures only: coordinate and multi-target contracts."""
import copy
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from stage3_5_element2vec_v2.src.utils.io import load_config
from stage3_5_element2vec_v2.src.data.dataset import WholeRockTokenDataset, load_vocab_bundle
from stage3_5_element2vec_v2.src.data.collate import collate_token_batch
from stage3_5_element2vec_v2.src.data.geochemical_coordinates import transform_numeric_frame, coordinate_fingerprint
from stage3_5_element2vec_v2.src.models.element2vec_v2 import build_model
from stage3_5_element2vec_v2.src.training.losses import compute_losses
from stage3_5_element2vec_v2.src.training.train_v2 import save_checkpoints, train, evaluate_probe
from stage3_5_element2vec_v2.src.evaluation.extract_embeddings import validate_checkpoint_compatibility, extract


CONFIG = "stage3_5_element2vec_v2/configs/v2_value_rope_geo_mask15.yaml"


def fixture(config, vocab):
    row = {f["db_column"]: 1. for f in config["data"]["numeric_fields"]}
    # A complete oxide panel with total exactly 100 wt%.
    row.update(sio2=55., tio2=1., al2o3=15., fe2o3=2., feo=6., mgo=5.,
               cao=8., na2o=4., k2o=3., p2o5=.5, mno=.5, loi=-.5)
    row["sample_id"] = "synthetic"
    for field in config["data"]["pure_label_fields"]:
        row[field] = vocab.pure_vocab["classes"][field][0]
    return pd.DataFrame([row])


def small_config(config):
    config = copy.deepcopy(config)
    config["model"]["d_model"] = 16
    config["model"]["transformer"].update(d_model=16, num_layers=2, num_heads=4,
                                          ffn_dim=32, dropout=0., rotary_dim=4)
    return config


class GeoCoordinateTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)
        self.vocab = load_vocab_bundle(self.config)
        self.frame = fixture(self.config, self.vocab)

    def test_closure_formula_and_uniform_rescaling(self):
        first, qc = transform_numeric_frame(self.frame, self.config, self.vocab)
        self.assertEqual(qc["closure_eligible"], 1)
        eps = self.config["data"]["value_transform"]["major"]["epsilon_fraction"]
        self.assertAlmostEqual(first.sio2[0], math.sqrt(.55 + eps) - math.sqrt(eps))
        scaled = self.frame.copy()
        cols = [f["db_column"] for f in self.config["data"]["numeric_fields"][:11]]
        scaled[cols] *= .95
        second, _ = transform_numeric_frame(scaled, self.config, self.vocab)
        np.testing.assert_allclose(first[cols], second[cols], atol=1e-12)

    def test_incomplete_invalid_and_outside_total_use_unclosed_fraction(self):
        for modification, reason in [({"p2o5": np.nan}, "closure_incomplete_panel"),
                                     ({"p2o5": -1.}, "closure_invalid_oxide"),
                                     ({"sio2": 70.}, "closure_total_outside_range")]:
            raw = self.frame.assign(**modification)
            out, qc = transform_numeric_frame(raw, self.config, self.vocab)
            self.assertEqual(qc[reason], 1)
            self.assertEqual(qc["closure_eligible"], 0)
            self.assertAlmostEqual(out.sio2[0], math.sqrt(raw.sio2[0]/100 + 1e-12)-1e-6)
            if "p2o5" in modification:
                self.assertTrue(pd.isna(out.p2o5[0]))

    def test_trace_pm_units_and_invalid_values(self):
        pm = self.config["data"]["value_transform"]["trace"]["pm_ppm"]
        raw = self.frame.copy()
        for f in self.config["data"]["numeric_fields"]:
            if f["name"] in pm:
                raw[f["db_column"]] = pm[f["name"]] * 10
        out, _ = transform_numeric_frame(raw, self.config, self.vocab)
        for f in self.config["data"]["numeric_fields"]:
            if f["name"] in pm:
                self.assertAlmostEqual(out[f["db_column"]][0], 1.)
        raw = raw.assign(rb_ppm=0., sr_ppm=-1., ba_ppm=np.inf)
        out, _ = transform_numeric_frame(raw, self.config, self.vocab)
        self.assertTrue(out[["rb_ppm", "sr_ppm", "ba_ppm"]].isna().all().all())

    def test_zero_auxiliary_and_no_fitted_z_stats(self):
        raw = self.frame.assign(tio2=0., co2=-1.)
        out, _ = transform_numeric_frame(raw, self.config, self.vocab)
        self.assertEqual(out.tio2[0], 0.)
        self.assertLess(out.loi[0], 0.)
        self.assertTrue(pd.isna(out.co2[0]))
        wrong_stats = copy.deepcopy(self.vocab)
        wrong_stats.transform_stats["median"] = 99999.
        wrong_stats.transform_stats["iqr"] = 99999.
        alternative, _ = transform_numeric_frame(raw, self.config, wrong_stats)
        pd.testing.assert_frame_equal(out, alternative)

    def test_bad_reference_config_rejected(self):
        for key, value in [("family_scale", 0.), ("epsilon_ppm", -1.)]:
            config = copy.deepcopy(self.config)
            config["data"]["value_transform"]["trace"][key] = value
            with self.assertRaises(ValueError):
                transform_numeric_frame(self.frame, config, self.vocab)
        config = copy.deepcopy(self.config)
        del config["data"]["value_transform"]["trace"]["pm_ppm"]["Rb_ppm"]
        with self.assertRaises(ValueError):
            transform_numeric_frame(self.frame, config, self.vocab)


class MultiMaskTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = load_config(CONFIG)
        self.vocab = load_vocab_bundle(self.config)
        self.frame = fixture(self.config, self.vocab)
        self.config["data"]["sqlite_path"] = str(Path(self.tmp.name)/"synthetic.db")
        with sqlite3.connect(self.config["data"]["sqlite_path"]) as conn:
            self.frame.to_sql(self.config["data"]["view_name"], conn, index=False)
        self.dataset = WholeRockTokenDataset(self.config, self.vocab)

    def test_mask_fraction_shuffle_and_targets(self):
        total = 0
        by_tag = {f["name"]: self.dataset.df.iloc[0][f["db_column"]] for f in self.config["data"]["numeric_fields"]}
        for seed in range(1000):
            item = self.dataset[(0, "numeric_value", None, seed)]
            masked = item["numeric_value_targets_multi"]
            self.assertIn(len(masked), (5, 6))  # stochastic rounding of 34 * .15
            self.assertEqual(len({m["position"] for m in masked}), len(masked))
            total += len(masked)
            for target in masked:
                token = item["tokens"][target["position"]]
                self.assertEqual(token["mask_value"], 1.)
                self.assertEqual(token["value"], 0.)
                self.assertAlmostEqual(target["value"], by_tag[token["name"]])
        self.assertAlmostEqual(total / (1000*34), .15, delta=.002)
        self.dataset.df.loc[:, [f["db_column"] for f in self.config["data"]["numeric_fields"]][1:]] = np.nan
        self.assertEqual(len(self.dataset[(0, "numeric_value", None, 1)]["numeric_value_targets_multi"]), 1)

    def test_label_tasks_identical_to_single_mask_baseline(self):
        baseline = copy.deepcopy(self.dataset)
        baseline.numeric_mask = {"strategy": "single"}
        for task, field in [("numeric_tag", None), ("pure_label", "rock_type"),
                            ("pure_label", "rock_group"), ("pure_label", "rock_origin")]:
            actual = self.dataset[(0, task, field, 123)]
            self.assertEqual(actual.pop("numeric_value_targets_multi"), [])
            self.assertEqual(actual, baseline[(0, task, field, 123)])

    def test_all_masked_heads_loss_gradients_and_no_leak(self):
        items = [self.dataset[(0, "numeric_value", None, 1)],
                 self.dataset[(0, "numeric_tag", None, 2)]]
        items += [self.dataset[(0, "pure_label", field, 3)] for field in self.config["data"]["pure_label_fields"]]
        batch = collate_token_batch(items)
        config = small_config(self.config)
        model = build_model(config, self.vocab).eval()
        outputs = model(batch)
        losses = compute_losses(outputs, batch, config["data"]["pure_label_fields"], config["training"]["loss_weights"])
        mask = batch["numeric_prediction_mask"]
        self.assertEqual(tuple(outputs["numeric_value_predictions"].shape), tuple(mask.shape))
        reference = torch.nn.functional.huber_loss(outputs["numeric_value_predictions"][mask],
                                                     batch["numeric_targets_all"][mask])
        torch.testing.assert_close(reference, losses["numeric_value_loss"])
        altered = copy.deepcopy(batch)
        altered["values"][mask] = 999.
        altered["numeric_targets_all"][mask] += 5.
        altered["numeric_value_targets"] += 5.
        changed = model(altered)
        torch.testing.assert_close(outputs["encoded"], changed["encoded"])
        self.assertFalse(torch.isclose(reference, compute_losses(changed, altered, config["data"]["pure_label_fields"],
                                                        config["training"]["loss_weights"])["numeric_value_loss"]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        before = model.decoders.numeric_value_decoder.weight.detach().clone()
        losses["total_loss"].backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.decoders.numeric_value_decoder.weight))

    def test_reload_checks_coordinate_identity(self):
        config = small_config(self.config)
        model = build_model(config, self.vocab).eval()
        save_checkpoints(model, torch.optim.AdamW(model.parameters()), None, config, self.vocab, 0, Path(self.tmp.name))
        checkpoint = torch.load(Path(self.tmp.name)/"checkpoints/full_model.pt", map_location="cpu")
        validate_checkpoint_compatibility(config, checkpoint, self.vocab)
        restored = build_model(config, self.vocab).eval()
        restored.load_state_dict(checkpoint["model_state_dict"])
        batch = collate_token_batch([self.dataset[(0, "numeric_value", None, 3)]])
        torch.testing.assert_close(model(batch)["encoded"], restored(batch)["encoded"])
        for changed in ("scale", "pm", "version"):
            bad = copy.deepcopy(config)
            if changed == "scale":
                bad["data"]["value_transform"]["major"]["family_scale"] = 2.
            elif changed == "pm":
                bad["data"]["value_transform"]["trace"]["pm_ppm"]["Rb_ppm"] = .635
            else:
                bad["data"]["value_transform"] = {"name": "v2_robust_z"}
            with self.assertRaisesRegex(ValueError, "coordinate"):
                validate_checkpoint_compatibility(bad, checkpoint, self.vocab)

    def test_new_experiment_cannot_overwrite_existing_run(self):
        config = copy.deepcopy(self.config)
        config["project"]["outputs_dir"] = str(Path(self.tmp.name)/"old_rope_z")
        with self.assertRaises(ValueError):
            train(config, max_steps_override=1)
        out = Path(self.tmp.name)/"value_rope_geo_mask15"
        (out/"logs").mkdir(parents=True)
        (out/"logs/train_log.csv").touch()
        config["project"]["outputs_dir"] = str(out)
        with self.assertRaises(FileExistsError):
            train(config, max_steps_override=1)

    def test_probe_numeric_average_weights_masked_tokens(self):
        config = small_config(self.config)
        batches = [collate_token_batch([self.dataset[(0, "numeric_value", None, seed)]]) for seed in (1, 2)]
        model = build_model(config, self.vocab).eval()
        counts = [int(b["numeric_prediction_mask"].sum()) for b in batches]
        with patch("stage3_5_element2vec_v2.src.training.train_v2.compute_losses",
                   side_effect=[{"numeric_value_loss": torch.tensor(v), "total_loss": torch.tensor(v)} for v in (1., 3.)]):
            result = evaluate_probe(model, batches, torch.device("cpu"), config["data"]["pure_label_fields"], config)
        self.assertAlmostEqual(result["probe_numeric_value_loss"], (counts[0]+3*counts[1])/sum(counts))

    def test_full_config_only_changes_coordinates_and_numeric_mask(self):
        baseline = load_config("stage3_5_element2vec_v2/configs/v2_value_rope_z.yaml")
        for section in ("training", "model", "probe", "evaluation", "tokenization"):
            self.assertEqual(self.config[section], baseline[section])
        with torch.device("meta"):
            model = build_model(self.config, self.vocab)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 85255040)
        self.assertNotEqual(coordinate_fingerprint(self.config), coordinate_fingerprint(baseline))

    def test_paired_chemistry_only_extraction_keeps_metadata(self):
        config = small_config(self.config)
        out = Path(self.tmp.name)/"value_rope_geo_mask15"
        config["project"]["outputs_dir"] = str(out)
        config["training"]["device"] = "cpu"
        model = build_model(config, self.vocab)
        save_checkpoints(model, torch.optim.AdamW(model.parameters()), None, config, self.vocab, 0, out)
        ids = Path(self.tmp.name)/"ids.csv"
        pd.DataFrame({"sample_id": ["synthetic"]}).to_csv(ids, index=False)
        paths = extract(config, str(out/"checkpoints/full_model.pt"), sample_ids_path=str(ids),
                        label_mode="chemistry_only")
        result = pd.read_csv(paths["mean_numeric"])
        self.assertEqual(result.sample_id.tolist(), ["synthetic"])
        self.assertEqual(result.rock_type[0], self.vocab.pure_vocab["classes"]["rock_type"][0])
        dataset = WholeRockTokenDataset(config, self.vocab, mode="extract", mask_pure_labels=True)
        dataset.chemistry_only = True
        item = dataset[0]
        self.assertTrue(all(t["kind"] == 0 for t in item["tokens"]))
        self.assertEqual(len(item["numeric_value_targets_multi"]), 0)
        self.assertEqual(sum(t["mask_value"] for t in item["tokens"]), 0.)
        pd.DataFrame({"sample_id": ["missing"]}).to_csv(ids, index=False)
        with self.assertRaisesRegex(ValueError, "missing"):
            extract(config, str(out/"checkpoints/full_model.pt"), sample_ids_path=str(ids))


if __name__ == "__main__":
    unittest.main()

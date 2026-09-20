from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import yaml

from stage3_5_element2vec_v2.src.data.dataset import VocabBundle, load_vocab_bundle
from stage3_5_element2vec_v2.src.evaluation.extract_embeddings import validate_checkpoint_compatibility
from stage3_5_element2vec_v2.src.models.element2vec_v2 import Element2VecValueRoPE, build_model
from stage3_5_element2vec_v2.src.models.transformer_encoder import (
    ValueRoPEMultiheadAttention,
    apply_rotary_embedding,
)
from stage3_5_element2vec_v2.src.training.losses import compute_losses
from stage3_5_element2vec_v2.src.training.train_v2 import resolve_training_steps, save_checkpoints


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def tiny_vocab() -> VocabBundle:
    return VocabBundle(
        numeric_vocab={
            "numeric_tags": ["A", "B", "C"],
            "tag_to_idx": {"A": 0, "B": 1, "C": 2},
            "mask_numeric_tag": "[MASK_NUMERIC_TAG]",
            "mask_numeric_tag_idx": 3,
        },
        pure_vocab={
            "pure_tokens": [f"token:{i}" for i in range(10)],
            "classes": {
                "rock_group": ["g0", "g1"],
                "rock_origin": ["o0", "o1"],
                "rock_type": ["t0", "t1", "t2"],
            },
        },
        transform_stats=pd.DataFrame(
            [{"field": "A", "median": 0.0, "iqr": 1.0, "log_transform": False, "eps": 1e-9}]
        ),
    )


def tiny_config() -> dict:
    return {
        "model": {
            "architecture": "value_rope",
            "d_model": 16,
            "transformer": {
                "d_model": 16,
                "num_layers": 2,
                "num_heads": 4,
                "ffn_dim": 32,
                "dropout": 0.0,
                "activation": "gelu",
                "norm": "pre_ln",
                "rotary_dim": 4,
                "rope_base": 10000.0,
                "rope_scale": 1.0,
            },
        },
        "data": {"pure_label_fields": ["rock_group", "rock_origin", "rock_type"]},
        "training": {
            "batch_size": 2,
            "gradient_accumulation_steps": 2,
            "epochs": 1,
            "loss_weights": {
                "numeric_value": 1.0,
                "numeric_tag": 1.0,
                "rock_group": 1.0,
                "rock_origin": 1.0,
                "rock_type": 1.0,
            },
        },
    }


def base_batch(batch_size: int = 1) -> dict[str, torch.Tensor | list]:
    kind = torch.tensor([[0, 0, 1, 1]], dtype=torch.long).repeat(batch_size, 1)
    return {
        "kind": kind,
        "numeric_tag_ids": torch.tensor([[0, 1, 0, 0]], dtype=torch.long).repeat(batch_size, 1),
        "pure_label_ids": torch.tensor([[0, 0, 1, 2]], dtype=torch.long).repeat(batch_size, 1),
        "values": torch.tensor([[0.0, 1.5, 0.0, 0.0]], dtype=torch.float32).repeat(batch_size, 1),
        "mask_tag_flags": torch.zeros(batch_size, 4),
        "mask_value_flags": torch.zeros(batch_size, 4),
        "attention_mask": torch.ones(batch_size, 4, dtype=torch.bool),
        "numeric_token_mask": kind == 0,
        "mask_positions": torch.zeros(batch_size, dtype=torch.long),
        "task_ids": torch.zeros(batch_size, dtype=torch.long),
        "numeric_value_targets": torch.zeros(batch_size),
        "numeric_tag_targets": torch.zeros(batch_size, dtype=torch.long),
        "pure_class_targets": torch.zeros(batch_size, dtype=torch.long),
        "pure_fields": [""] * batch_size,
    }


class RotaryMathTests(unittest.TestCase):
    def test_rotary_matches_pairwise_reference_and_preserves_norm(self) -> None:
        tensor = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        positions = torch.tensor([[-2.0]])
        actual = apply_rotary_embedding(tensor, positions, rotary_dim=4, base=10000.0)
        frequencies = torch.tensor([1.0, 0.01])
        angle = -2.0 * frequencies
        pairs = tensor.reshape(1, 1, 1, 2, 2)
        expected = torch.empty_like(pairs)
        expected[..., 0] = pairs[..., 0] * angle.cos() - pairs[..., 1] * angle.sin()
        expected[..., 1] = pairs[..., 0] * angle.sin() + pairs[..., 1] * angle.cos()
        torch.testing.assert_close(actual, expected.flatten(start_dim=-2))
        torch.testing.assert_close(actual.norm(dim=-1), tensor.norm(dim=-1))

    def test_zero_position_is_identity(self) -> None:
        tensor = torch.randn(2, 3, 5, 8)
        positions = torch.zeros(2, 5)
        torch.testing.assert_close(apply_rotary_embedding(tensor, positions, 8), tensor)

    def test_zero_position_attention_matches_torch_multihead_attention(self) -> None:
        torch.manual_seed(7)
        ours = ValueRoPEMultiheadAttention(16, 4, dropout=0.0, rotary_dim=4).eval()
        reference = torch.nn.MultiheadAttention(16, 4, dropout=0.0, batch_first=True).eval()
        with torch.no_grad():
            reference.in_proj_weight.copy_(ours.in_proj_weight)
            reference.in_proj_bias.copy_(ours.in_proj_bias)
            reference.out_proj.load_state_dict(ours.out_proj.state_dict())
        tokens = torch.randn(2, 5, 16)
        valid = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
        expected, _ = reference(tokens, tokens, tokens, key_padding_mask=~valid, need_weights=False)
        actual = ours(tokens, valid, torch.zeros(2, 5))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


class ValueRoPEModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.vocab = tiny_vocab()
        self.config = tiny_config()
        self.model = build_model(self.config, self.vocab).eval()
        self.assertIsInstance(self.model, Element2VecValueRoPE)

    def test_values_only_enter_through_rope(self) -> None:
        first = base_batch()
        second = copy.deepcopy(first)
        second["values"][0, 1] = -3.0
        torch.testing.assert_close(self.model.make_token_inputs(first), self.model.make_token_inputs(second))
        with torch.no_grad():
            encoded_first = self.model.encode(first)["encoded"]
            encoded_second = self.model.encode(second)["encoded"]
        self.assertFalse(torch.allclose(encoded_first, encoded_second))

    def test_masked_value_cannot_leak_through_rope(self) -> None:
        first = base_batch()
        first["mask_value_flags"][0, 1] = 1.0
        first["values"][0, 1] = 8.0
        second = copy.deepcopy(first)
        second["values"][0, 1] = -9.0
        second["numeric_value_targets"][0] = 123.0
        with torch.no_grad():
            out_first = self.model.encode(first)
            out_second = self.model.encode(second)
        torch.testing.assert_close(out_first["rope_positions"], out_second["rope_positions"])
        torch.testing.assert_close(out_first["encoded"], out_second["encoded"])
        self.assertEqual(float(out_first["rope_positions"][0, 2]), 0.0)

    def test_real_zero_and_masked_zero_have_distinct_input_state(self) -> None:
        visible = base_batch()
        visible["values"][0, 0] = 0.0
        masked = copy.deepcopy(visible)
        masked["mask_value_flags"][0, 0] = 1.0
        visible_input = self.model.make_token_inputs(visible)
        masked_input = self.model.make_token_inputs(masked)
        self.assertFalse(torch.allclose(visible_input[:, 0], masked_input[:, 0]))
        self.assertEqual(float(self.model.make_rope_positions(visible)[0, 1]), 0.0)
        self.assertEqual(float(self.model.make_rope_positions(masked)[0, 1]), 0.0)

    def test_token_permutation_is_equivariant(self) -> None:
        batch = base_batch()
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = copy.deepcopy(batch)
        for key in (
            "kind",
            "numeric_tag_ids",
            "pure_label_ids",
            "values",
            "mask_tag_flags",
            "mask_value_flags",
            "attention_mask",
            "numeric_token_mask",
        ):
            permuted[key] = batch[key][:, permutation]
        with torch.no_grad():
            original = self.model.encode(batch)["encoded"]
            changed = self.model.encode(permuted)["encoded"]
        torch.testing.assert_close(changed[:, 0], original[:, 0], rtol=1e-5, atol=1e-6)
        inverse = torch.argsort(permutation)
        torch.testing.assert_close(changed[:, 1:][:, inverse], original[:, 1:], rtol=1e-5, atol=1e-6)
        original_mean = original[:, 1:3].mean(dim=1)
        changed_numeric = changed[:, 1:][:, torch.tensor([1, 3])].mean(dim=1)
        torch.testing.assert_close(changed_numeric, original_mean, rtol=1e-5, atol=1e-6)

    def test_padding_does_not_change_valid_outputs(self) -> None:
        batch = base_batch()
        padded = copy.deepcopy(batch)
        for key in ("kind", "numeric_tag_ids", "pure_label_ids"):
            padded[key] = torch.cat((batch[key], torch.zeros(1, 2, dtype=torch.long)), dim=1)
        for key in ("values", "mask_tag_flags", "mask_value_flags"):
            padded[key] = torch.cat((batch[key], torch.zeros(1, 2)), dim=1)
        padded["attention_mask"] = torch.cat((batch["attention_mask"], torch.zeros(1, 2, dtype=torch.bool)), dim=1)
        padded["numeric_token_mask"] = torch.cat((batch["numeric_token_mask"], torch.zeros(1, 2, dtype=torch.bool)), dim=1)
        with torch.no_grad():
            original = self.model.encode(batch)["encoded"]
            changed = self.model.encode(padded)["encoded"]
        torch.testing.assert_close(changed[:, : original.shape[1]], original, rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.isfinite(changed).all())

    def test_all_five_loss_heads_and_backbone_receive_gradients(self) -> None:
        model = build_model(self.config, self.vocab).train()
        batch = base_batch(batch_size=5)
        batch["task_ids"] = torch.tensor([0, 1, 2, 2, 2])
        batch["numeric_value_targets"] = torch.tensor([1.25, 0.0, 0.0, 0.0, 0.0])
        batch["numeric_tag_targets"] = torch.tensor([-100, 2, -100, -100, -100])
        batch["pure_class_targets"] = torch.tensor([-100, -100, 1, 1, 2])
        batch["pure_fields"] = ["", "", "rock_group", "rock_origin", "rock_type"]
        losses = compute_losses(
            model(batch), batch, self.config["data"]["pure_label_fields"], self.config["training"]["loss_weights"]
        )
        losses["total_loss"].backward()
        parameters = [
            model.numeric_tag_embedding.weight,
            model.transformer_encoder.layers[0].self_attn.in_proj_weight,
            model.decoders.numeric_value_decoder.weight,
            model.decoders.numeric_tag_decoder.weight,
            model.decoders.pure_label_decoders["rock_group"].weight,
            model.decoders.pure_label_decoders["rock_origin"].weight,
            model.decoders.pure_label_decoders["rock_type"].weight,
        ]
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_save_reload_and_compatibility_validation(self) -> None:
        model = build_model(self.config, self.vocab).eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            save_checkpoints(model, optimizer, None, self.config, self.vocab, 3, out)
            self.assertFalse((out / "checkpoints" / "token_projector.pt").exists())
            checkpoint = torch.load(out / "checkpoints" / "full_model.pt", map_location="cpu")
            validate_checkpoint_compatibility(self.config, checkpoint, self.vocab)
            reloaded = build_model(self.config, self.vocab).eval()
            reloaded.load_state_dict(checkpoint["model_state_dict"])
            batch = base_batch()
            with torch.no_grad():
                torch.testing.assert_close(model.encode(batch)["encoded"], reloaded.encode(batch)["encoded"])
            wrong_config = copy.deepcopy(self.config)
            wrong_config["model"]["architecture"] = "mlp_fusion"
            with self.assertRaisesRegex(ValueError, "architecture"):
                validate_checkpoint_compatibility(wrong_config, checkpoint, self.vocab)


class ArchitectureAndTrainingContractTests(unittest.TestCase):
    def test_full_model_parameter_counts_and_legacy_dispatch(self) -> None:
        rope_config = yaml.safe_load((PACKAGE_ROOT / "configs" / "v2_value_rope_z.yaml").read_text())
        legacy_config = yaml.safe_load((PACKAGE_ROOT / "configs" / "v2_base.yaml").read_text())
        vocab = load_vocab_bundle(rope_config)
        with torch.device("meta"):
            rope_model = build_model(rope_config, vocab)
            legacy_model = build_model(legacy_config, vocab)
        self.assertEqual(sum(p.numel() for p in rope_model.parameters()), 85_255_040)
        self.assertEqual(sum(p.numel() for p in legacy_model.parameters()), 86_798_583)
        self.assertFalse(hasattr(rope_model, "token_projector"))
        self.assertTrue(hasattr(legacy_model, "token_projector"))
        self.assertEqual(rope_model.transformer_encoder.layers[0].self_attn.head_dim, 64)
        self.assertEqual(rope_model.transformer_encoder.layers[0].self_attn.rotary_dim, 64)
        self.assertEqual(rope_model.decoders.numeric_value_decoder.out_features, 1)
        self.assertEqual(rope_model.decoders.numeric_tag_decoder.out_features, 34)
        self.assertEqual(rope_model.decoders.pure_label_decoders["rock_group"].out_features, 3)
        self.assertEqual(rope_model.decoders.pure_label_decoders["rock_origin"].out_features, 9)
        self.assertEqual(rope_model.decoders.pure_label_decoders["rock_type"].out_features, 81)

        longest = {
            "kind": torch.zeros(2, 37, dtype=torch.long, device="meta"),
            "numeric_tag_ids": torch.zeros(2, 37, dtype=torch.long, device="meta"),
            "pure_label_ids": torch.zeros(2, 37, dtype=torch.long, device="meta"),
            "values": torch.zeros(2, 37, device="meta"),
            "mask_value_flags": torch.zeros(2, 37, device="meta"),
            "attention_mask": torch.ones(2, 37, dtype=torch.bool, device="meta"),
        }
        token_inputs = rope_model.make_token_inputs(longest)
        rope_positions = rope_model.make_rope_positions(longest)
        transformer_input = torch.cat((rope_model.cls_token.expand(2, -1, -1), token_inputs), dim=1)
        self.assertEqual(tuple(token_inputs.shape), (2, 37, 768))
        self.assertEqual(tuple(rope_positions.shape), (2, 38))
        self.assertEqual(tuple(transformer_input.shape), (2, 38, 768))

    def test_training_step_count_matches_baseline_and_rejects_no_step_dataset(self) -> None:
        config = tiny_config()
        config["training"]["batch_size"] = 512
        config["training"]["gradient_accumulation_steps"] = 2
        config["training"]["epochs"] = 50
        max_steps, steps_per_epoch, target_epochs = resolve_training_steps(config, 270_241)
        self.assertEqual((max_steps, steps_per_epoch, target_epochs), (13_200, 264, 50))
        with self.assertRaisesRegex(ValueError, "fewer than gradient_accumulation_steps"):
            resolve_training_steps(config, 256, max_steps_override=2)


if __name__ == "__main__":
    unittest.main()

"""Two full-model updates plus probe, CPU reload and extraction; never a full run."""
import argparse
import copy
from pathlib import Path

import torch

from ..data.dataset import WholeRockTokenDataset, load_vocab_bundle
from ..data.collate import collate_token_batch
from ..models.element2vec_v2 import build_model
from ..evaluation.extract_embeddings import extract, validate_checkpoint_compatibility
from ..utils.io import load_config, output_dir, save_json
from .train_v2 import train, choose_device, move_batch_to_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite-path", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--outputs-dir", default="stage3_5_element2vec_v2/outputs/value_rope_geo_mask15/smoke")
    args = parser.parse_args()
    config = copy.deepcopy(load_config("stage3_5_element2vec_v2/configs/v2_value_rope_geo_mask15.yaml"))
    config["data"]["sqlite_path"] = args.sqlite_path
    config["project"]["outputs_dir"] = args.outputs_dir
    config["training"].update(device=args.device, batch_size=2)
    config["probe"].update(size=8, batch_size=2, eval_every_steps=1)
    config["evaluation"]["batch_size"] = 2
    result = train(config, max_steps_override=2, max_samples=8)
    checkpoint = torch.load(result["checkpoint"], map_location="cpu")
    vocab = load_vocab_bundle(config)
    validate_checkpoint_compatibility(config, checkpoint, vocab)
    model = build_model(config, vocab).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset = WholeRockTokenDataset(config, vocab, max_samples=8)
    batch = collate_token_batch([dataset[(i, "numeric_value", None, 42+i)] for i in range(2)])
    with torch.no_grad():
        expected = model(batch)["encoded"]
        device = choose_device(args.device)
        actual = model.to(device)(move_batch_to_device(batch, device))["encoded"].cpu()
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    parameters = sum(p.numel() for p in model.parameters())
    del model, checkpoint
    config["training"]["device"] = "cpu"
    result["extraction"] = extract(config, result["checkpoint"], max_samples=8)
    result.update(device=args.device, cpu_reload=True, parameters=parameters, samples=8,
                  micro_batch=2, gradient_accumulation=2)
    save_json(result, output_dir(config)/"logs/smoke_verification.json")
    print("[smoke] PASS: two updates, probes at 0/1/2, CPU reload and 8-sample extraction", flush=True)


if __name__ == "__main__":
    main()

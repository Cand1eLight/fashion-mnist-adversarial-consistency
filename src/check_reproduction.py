"""A fresh-directory, from-scratch repeat of the frozen proposed seed 7411."""
from dataclasses import replace
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import torch
import core
import experiment
from experiment import Config, train


def run(destination):
    reference_root = experiment.ROOT
    protocol = json.loads((reference_root/"results/frozen_protocol.json").read_text())
    reference_cfg = Config(**next(c for c in protocol["configs"] if c["tag"] == "proposed" and c["seed"] == 7411))
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination/"data").mkdir()
    for name in core.FILES:
        shutil.copy2(reference_root/"data"/name, destination/"data"/name)
    cfg = replace(reference_cfg, stage="reproduction")
    experiment.ROOT = core.ROOT = destination
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = core.load_data(device)
    record = train(cfg, data, device)
    full = core.load_data(device, include_test=True)
    checks = {}
    for version, suffix in [("raw", ""), ("average", "_average")]:
        expected = torch.load(reference_root/"checkpoints"/(reference_cfg.name+suffix+".pt"), map_location="cpu", weights_only=True)
        actual = torch.load(destination/"checkpoints"/(cfg.name+suffix+".pt"), map_location="cpu", weights_only=True)
        equal = expected.keys() == actual.keys() and all(torch.equal(expected[k], actual[k]) for k in expected)
        assert equal, "Fresh training changed the parameters"
        model = core.make_model(device)
        model.load_state_dict(actual)
        score, arr = core.evaluate(model, *full["test"], seed=9001)
        original = np.load(reference_root/"runs"/reference_cfg.name/(f"test_predictions_{version}.npz"))
        arrays_equal = {key: bool(np.array_equal(arr[key], original[key])) for key in arr}
        assert all(arrays_equal.values()), "Fresh official predictions differ"
        np.savez_compressed(destination/"runs"/cfg.name/(f"test_predictions_{version}.npz"), **arr)
        checks[version] = {"parameter_tensors_bitwise_equal": equal, "tensors": len(actual),
                           "full_test_arrays_bitwise_equal": arrays_equal, "test_metrics": score,
                           "maximum_attack_loss_difference": float(np.abs(arr["attack_loss"]-original["attack_loss"]).max())}
    result = {"reference_run": reference_cfg.name, "fresh_run": cfg.name,
              "fresh_directory": str(destination), "seed": 7411, "epochs": 30,
              "from_scratch": True, "used_cached_weights": False,
              "training_seconds": record["training_seconds"], "versions": checks,
              "environment": core.environment(), "test_configuration_changed": False}
    core.save_json(reference_root/"results/reproduction_check.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", help="An unused directory for the fresh run")
    args = parser.parse_args()
    run(args.destination)

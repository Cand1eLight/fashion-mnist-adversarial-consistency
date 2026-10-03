"""Freeze validation selection, train paired seeds, and evaluate the frozen suite."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd
import torch

import core
from experiment import Config, ROOT, train

SEEDS = [7411, 7412, 7413, 7414, 7415]
FAMILY = ["standard", "pgd", "mixed", "curriculum_mixed", "consistency", "proposed"]


def freeze():
    path = ROOT/"results/frozen_protocol.json"
    if path.exists():
        return json.loads(path.read_text())
    if not (ROOT/"results/budget_pilot.json").exists():
        raise RuntimeError("Complete the declared validation phases before freezing")
    candidates = []
    for folder in sorted((ROOT/"runs").glob("development_*")):
        record = json.loads((folder/"training.json").read_text())
        scores = json.loads((folder/"validation.json").read_text())
        cfg = dict(record["config"])
        cfg.setdefault("warmup_steps", None)
        candidates.append({"config": cfg, "run": folder.name,
                           "clean": scores["clean_correct"]/scores["n"],
                           "pgd20": scores["robust_correct"]/scores["n"],
                           "training_seconds": record["training_seconds"]})
    eligible = [c for c in candidates if c["clean"] >= .8374]
    best = max(c["pgd20"] for c in eligible)
    close = [c for c in eligible if c["pgd20"] >= best-.002-1e-12]
    chosen = min(close, key=lambda c: c["training_seconds"])
    proposed = Config(**chosen["config"])
    if proposed.objective != "hybrid" or proposed.averaging != "swa":
        raise RuntimeError("The confirmatory factorial design must match the selected family")
    base = replace(proposed, tag="proposed", seed=SEEDS[0], stage="confirmation", trace_validation=True)
    templates = [
        replace(base, tag="standard", objective="standard", ramp_epochs=0, warmup_steps=None, averaging="none"),
        replace(base, tag="pgd", objective="pgd", ramp_epochs=0, warmup_steps=None, averaging="none"),
        replace(base, tag="mixed", objective="mixed", ramp_epochs=0, warmup_steps=None),
        replace(base, tag="curriculum_mixed", objective="mixed"),
        replace(base, tag="consistency", ramp_epochs=0, warmup_steps=None),
        base,
    ]
    configs = []
    for k, seed in enumerate(SEEDS):
        # Rotate method order to spread elapsed-time drift over the comparisons.
        order = templates[k % len(templates):]+templates[:k % len(templates)]
        configs.extend(asdict(replace(c, seed=seed)) for c in order)
    protocol = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "Clean validation accuracy >= 83.74%; within 0.20 points of the highest validation PGD20 accuracy, choose the shortest measured optimization time",
        "selected_development": chosen, "all_development": candidates,
        "configs": configs, "seeds": SEEDS,
        "official_attack": core.OFFICIAL_ATTACK,
        "epochs": 30, "architecture_parameters": 235146,
        "selection_data": "55,000 training / 5,000 stratified validation; no new test evaluation",
        "revision_context": "An earlier completed study had evaluated the test set. New method selection uses validation only; the five confirmation seeds were not evaluated before this freeze.",
        "factorial_design": "Four optimization trajectories (mixed, curriculum_mixed, consistency, proposed) each provide final raw and averaged parameters: a 2x2x2 consistency/curriculum/averaging comparison. Standard and PGD baselines are additional trajectories.",
        "test_opened_before_freeze_in_this_revision": False,
    }
    core.save_json(path, protocol)
    core.save_json(ROOT/"results/environment.json", core.environment())
    print("Selected", chosen["run"], chosen["clean"], chosen["pgd20"], flush=True)
    return protocol


def configs():
    return [Config(**c) for c in freeze()["configs"]]


def train_suite(device, force=False):
    freeze()
    checks = core.check_attack(device)
    core.save_json(ROOT/"results/implementation_checks.json", checks)
    data = core.load_data(device)
    for cfg in configs():
        train(cfg, data, device, force=force)


def load_model(cfg, device, version="primary"):
    averaged = version == "average" or (version == "primary" and cfg.averaging != "none")
    suffix = "_average" if averaged else ""
    model = core.make_model(device)
    checkpoint = ROOT/"checkpoints"/(cfg.name+suffix+".pt")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model


def evaluate_suite(device, force=False):
    suite = configs()
    for cfg in suite:
        if not (ROOT/"runs"/cfg.name/"validation.json").exists():
            raise RuntimeError("Train all frozen configurations before reading test data")
    data = core.load_data(device, include_test=True)
    x, y = data["test"]
    rows = []
    for cfg in suite:
        record = json.loads((ROOT/"runs"/cfg.name/"training.json").read_text())
        versions = ["raw", "average"] if cfg.averaging != "none" else ["raw"]
        for version in versions:
            folder = ROOT/"runs"/cfg.name
            score_path = folder/(f"test_{version}.json")
            arrays_path = folder/(f"test_predictions_{version}.npz")
            if score_path.exists() and arrays_path.exists() and not force:
                score = json.loads(score_path.read_text())
            else:
                model = load_model(cfg, device, version)
                score, arrays = core.evaluate(model, x, y, seed=9001)
                core.save_json(score_path, score)
                np.savez_compressed(arrays_path, **arrays)
            rows.append({"run": cfg.name, "method": cfg.tag, "version": version,
                         "seed": cfg.seed, "epochs": cfg.epochs,
                         "training_seconds": record["training_seconds"],
                         "mean_epoch_seconds": record["mean_epoch_seconds"],
                         "mean_attack_steps": record["mean_attack_steps"],
                         "clean_accuracy": score["clean_correct"]/score["n"],
                         "pgd20_accuracy": score["robust_correct"]/score["n"],
                         "clean_correct": score["clean_correct"],
                         "robust_correct": score["robust_correct"], "n": score["n"]})
            print(cfg.name, version, score["clean_correct"], score["robust_correct"], flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT/"results/all_results.csv", index=False)
    core.save_json(ROOT/"results/all_results.json", rows)
    core.save_json(ROOT/"results/evaluation_record.json", {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_protocol_sha256": hashlib.sha256((ROOT/"results/frozen_protocol.json").read_bytes()).hexdigest(),
        "configuration_changed_after_test_evaluation": False,
        "models": len(rows), "independent_training_runs": len(suite), "test_examples_per_model": 10000})
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "check", "train", "evaluate", "audit", "analyze", "reproduce"])
    parser.add_argument("--force", action="store_true", help="Retrain or reevaluate the frozen configuration")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.action == "freeze":
        freeze()
    elif args.action == "check":
        from checks import run
        print(json.dumps(run(device), indent=2))
    elif args.action == "train":
        train_suite(device, args.force)
    elif args.action == "evaluate":
        evaluate_suite(device, args.force)
    elif args.action == "audit":
        from attack_audit import run
        run(device, force=args.force)
    elif args.action == "analyze":
        from analyze_results import run
        run()
    elif args.action == "reproduce":
        train_suite(device, args.force)
        evaluate_suite(device, args.force)


if __name__ == "__main__":
    main()

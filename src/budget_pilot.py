"""Validation-only tests of the cost of the continuation phase."""
from dataclasses import asdict
import json
import torch
import core
from experiment import Config, ROOT, train

CONFIGS = [
    Config("hybrid_b05_ramp10_warm2_swa20", objective="hybrid", beta=.5, ramp_epochs=10,
           warmup_steps=2, averaging="swa", average_start=20),
    Config("hybrid_b05_ramp10_warm3_swa20", objective="hybrid", beta=.5, ramp_epochs=10,
           warmup_steps=3, averaging="swa", average_start=20),
    Config("gated_b05_ramp10_warm2_swa20", objective="gated", beta=.5, ramp_epochs=10,
           warmup_steps=2, averaging="swa", average_start=20),
    Config("gated_b1_ramp10_warm2_swa20", objective="gated", beta=1., ramp_epochs=10,
           warmup_steps=2, averaging="swa", average_start=20),
]

def run(device):
    core.save_json(ROOT/"results"/"budget_protocol.json", {
        "configs": [asdict(c) for c in CONFIGS],
        "selection_rule": "Same validation-only rule as refinement: require clean >= 83.74%; among models within 0.20 points of the highest PGD20, select the shortest measured optimization time.",
        "test_used": False,
        "motivation": "Reduce early attack iterations while the radius is small. Later full-radius attacks still use five steps."})
    data = core.load_data(device)
    rows = []
    for cfg in CONFIGS:
        record = train(cfg, data, device)
        val = json.loads((ROOT/"runs"/cfg.name/"validation.json").read_text())
        rows.append({"tag": cfg.tag, "training_seconds": record["training_seconds"],
                     "validation_clean": val["clean_correct"]/val["n"],
                     "validation_pgd20": val["robust_correct"]/val["n"]})
        core.save_json(ROOT/"results"/"budget_pilot.json", rows)
    print(json.dumps(rows, indent=2), flush=True)

if __name__ == "__main__":
    run(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

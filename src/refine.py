"""Second validation-only phase: test explicit loss, curriculum and averaging factors."""
import json
import torch
import core
from experiment import Config, ROOT, train

CONFIGS = [
    Config("mixed5_ramp10_swa20", ramp_epochs=10, averaging="swa", average_start=20),
    Config("hybrid_b025", objective="hybrid", beta=.25),
    Config("hybrid_b025_ramp10", objective="hybrid", beta=.25, ramp_epochs=10),
    Config("hybrid_b025_swa20", objective="hybrid", beta=.25, averaging="swa", average_start=20),
    Config("hybrid_b025_ramp10_swa20", objective="hybrid", beta=.25, ramp_epochs=10, averaging="swa", average_start=20),
    Config("hybrid_b05_ramp10_swa20", objective="hybrid", beta=.5, ramp_epochs=10, averaging="swa", average_start=20),
    Config("gated_b05_ramp10_swa20", objective="gated", beta=.5, ramp_epochs=10, averaging="swa", average_start=20),
    Config("gated_b1_ramp10_swa20", objective="gated", beta=1., ramp_epochs=10, averaging="swa", average_start=20),
]


def run(device):
    core.save_json(ROOT/"results"/"refinement_protocol.json", {
        "configs": [__import__("dataclasses").asdict(c) for c in CONFIGS],
        "selection_rule": "Among candidates with validation clean accuracy at least one percentage point above the PGD baseline, maximize validation PGD-20 accuracy. If candidates are within 0.20 percentage points, select the lower training time.",
        "pgd_validation_clean_reference": .8274,
        "data_access": "Training and held-out validation only; no new test evaluation in development or refinement",
        "revision_context": "The earlier completed study had already evaluated test results; these results do not determine new configurations. Final confirmation will use five new training seeds and a frozen configuration."})
    data = core.load_data(device)
    rows = []
    for cfg in CONFIGS:
        record = train(cfg, data, device)
        scores = json.loads((ROOT/"runs"/cfg.name/"validation.json").read_text())
        rows.append({"tag": cfg.tag, "training_seconds": record["training_seconds"],
                     "validation_clean": scores["clean_correct"]/scores["n"],
                     "validation_pgd20": scores["robust_correct"]/scores["n"]})
        core.save_json(ROOT/"results"/"refinement.json", rows)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    run(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

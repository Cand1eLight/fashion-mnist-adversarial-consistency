"""Read and cross-check the saved experiment records without model weights."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def review():
    protocol_path = ROOT / "results/frozen_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    evaluation = json.loads((ROOT / "results/evaluation_record.json").read_text())
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == evaluation["frozen_protocol_sha256"]
    rows = pd.read_csv(ROOT / "results/all_results.csv")
    assert len(rows) == 50 and set(rows.seed) == {7411, 7412, 7413, 7414, 7415}
    assert rows.n.eq(10000).all() and rows.epochs.eq(30).all()
    assert np.allclose(rows.clean_accuracy, rows.clean_correct / rows.n)
    assert np.allclose(rows.pgd20_accuracy, rows.robust_correct / rows.n)
    prediction_files = 0
    for row in rows.itertuples():
        run = ROOT / "runs" / row.run
        score = json.loads((run / f"test_{row.version}.json").read_text())
        assert score["n"] == row.n
        assert score["clean_correct"] == row.clean_correct
        assert score["robust_correct"] == row.robust_correct
        assert (score["epsilon"], score["alpha"], score["steps"], score["restarts"], score["keep_best"]) == (.1, .01, 20, 1, False)
        train = json.loads((run / "training.json").read_text())
        assert train["config"]["seed"] == row.seed and train["epochs"] == 30
        assert np.isclose(train["training_seconds"], row.training_seconds)
        pred = run / f"test_predictions_{row.version}.npz"
        if pred.exists():
            with np.load(pred) as arr:
                assert len(arr["label"]) == 10000
                assert int((arr["clean_prediction"] == arr["label"]).sum()) == row.clean_correct
                assert int((arr["adversarial_prediction"] == arr["label"]).sum()) == row.robust_correct
            prediction_files += 1
    assert prediction_files in (0, 50), "Prediction evidence must be complete when supplied"
    for s in json.loads((ROOT / "results/summary.json").read_text()):
        group = rows[(rows.method == s["method"]) & (rows.version == s["version"])]
        assert len(group) == s["seeds"] == 5
        for metric in ["clean_accuracy", "pgd20_accuracy", "training_seconds", "mean_epoch_seconds", "mean_attack_steps"]:
            assert np.isclose(group[metric].mean(), s[metric + "_mean"])
            assert np.isclose(group[metric].std(ddof=1), s[metric + "_sd"])
    records = [json.loads(p.read_text()) for p in (ROOT / "runs").glob("*/training.json")]
    assert len(records) == 52 and sum(r["epochs"] for r in records) == 1560
    split = np.load(ROOT / "data/split_indices.npz")
    assert len(split["train"]) == 55000 and len(split["validation"]) == 5000
    assert not np.intersect1d(split["train"], split["validation"]).size
    return {"frozen_protocol_matches": True, "saved_model_states_checked": len(rows),
            "test_images_per_state": 10000, "optimization_runs": len(records),
            "summary_statistics_match": True, "test_counts_match_run_records": True,
            "per_image_prediction_files_checked": prediction_files,
            "source_of_displayed_results": "Saved experiment records; no fresh training in review mode"}


if __name__ == "__main__":
    print(json.dumps(review(), indent=2))

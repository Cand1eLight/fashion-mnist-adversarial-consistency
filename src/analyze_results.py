"""Example-level verification, paired effects and publication figures."""
from pathlib import Path
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import core
from experiment import ROOT
from run_suite import configs, load_model, SEEDS

BLUE, RED, GREY, GREEN = "#0F4D92", "#B64342", "#767676", "#487A3B"
LABELS = {"standard": "Standard", "pgd": "PGD-5", "mixed": "Mixed CE",
          "curriculum_mixed": "Mixed CE + curriculum", "consistency": "Consistency",
          "proposed": "Proposed"}


def save_figure(fig, name):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(ROOT/"figures"/(name+"."+ext), dpi=300, bbox_inches="tight")
    plt.close(fig)


def run():
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 9.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 1.1, "legend.frameon": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    (ROOT/"figures").mkdir(exist_ok=True)
    rows = pd.read_csv(ROOT/"results/all_results.csv")
    summaries = []
    for (method, version), group in rows.groupby(["method", "version"], sort=False):
        out = {"method": method, "version": version, "seeds": len(group)}
        for metric in ["clean_accuracy", "pgd20_accuracy", "training_seconds", "mean_epoch_seconds", "mean_attack_steps"]:
            out[metric+"_mean"], out[metric+"_sd"] = float(group[metric].mean()), float(group[metric].std(ddof=1))
        summaries.append(out)
    core.save_json(ROOT/"results/summary.json", summaries)
    pd.DataFrame(summaries).to_csv(ROOT/"results/summary.csv", index=False)
    sm = {(s["method"], s["version"]): s for s in summaries}
    main = rows[((rows.method.isin(["standard", "pgd"])) & (rows.version == "raw")) |
                ((rows.method == "proposed") & (rows.version == "average"))]
    main_keys = [("standard", "raw"), ("pgd", "raw"), ("proposed", "average")]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6), gridspec_kw={"width_ratios": [1.08, 1.08, 1]})
    for key, color in zip(main_keys, [GREY, RED, BLUE]):
        s = sm[key]
        for ax in axes[:2]:
            ax.errorbar(s["clean_accuracy_mean"]*100, s["pgd20_accuracy_mean"]*100,
                        xerr=s["clean_accuracy_sd"]*100, yerr=s["pgd20_accuracy_sd"]*100,
                        fmt="o", color=color, ms=5.5, capsize=2, label=LABELS[key[0]])
    axes[0].set(xlabel="Clean accuracy (%)", ylabel="PGD-20 accuracy (%)", title="(a) All required methods", ylim=(0, 72))
    axes[0].legend(loc="lower left", fontsize=8)
    axes[1].set(xlabel="Clean accuracy (%)", title="(b) Robust models")
    robust_clean = [sm[k]["clean_accuracy_mean"]*100 for k in main_keys[1:]]
    robust_pgd = [sm[k]["pgd20_accuracy_mean"]*100 for k in main_keys[1:]]
    axes[1].set_xlim(min(robust_clean)-.65, max(robust_clean)+.65)
    axes[1].set_ylim(min(robust_pgd)-.65, max(robust_pgd)+.65)
    for key, color in zip(main_keys[1:], [RED, BLUE]):
        s = sm[key]
        axes[1].annotate(LABELS[key[0]], (s["clean_accuracy_mean"]*100, s["pgd20_accuracy_mean"]*100),
                         xytext=(4, 8), textcoords="offset points", fontsize=8, color=color)
    bars = axes[2].barh(np.arange(3), [sm[k]["training_seconds_mean"] for k in main_keys],
                 xerr=[sm[k]["training_seconds_sd"] for k in main_keys], color=[GREY, RED, BLUE],
                 edgecolor="black", linewidth=.7, capsize=2)
    for bar, hatch in zip(bars, ["", "//", ".."]):
        bar.set_hatch(hatch)
    axes[2].set_yticks(range(3), [LABELS[k[0]] for k in main_keys])
    axes[2].invert_yaxis()
    axes[2].set(xlabel="Training time (s)", title="(c) Complete optimization")
    fig.tight_layout(pad=1.2)
    save_figure(fig, "main_tradeoff")

    predictions, classes = {}, []
    for row in rows.itertuples():
        arr = np.load(ROOT/"runs"/row.run/(f"test_predictions_{row.version}.npz"))
        label = arr["label"]
        clean, robust = arr["clean_prediction"] == label, arr["adversarial_prediction"] == label
        assert len(label) == 10000 and clean.sum() == row.clean_correct and robust.sum() == row.robust_correct
        predictions[(row.method, row.version, row.seed)] = {k: arr[k] for k in arr.files}
        for c in range(10):
            mask = label == c
            assert mask.sum() == 1000
            classes.append({"method": row.method, "version": row.version, "seed": row.seed,
                            "class_id": c, "class": core.CLASS_NAMES[c], "n": 1000,
                            "clean_accuracy": float(clean[mask].mean()), "robust_accuracy": float(robust[mask].mean())})
    class_frame = pd.DataFrame(classes)
    class_frame.to_csv(ROOT/"results/per_class.csv", index=False)
    paired = []
    for seed in SEEDS:
        a, b = predictions[("pgd", "raw", seed)], predictions[("proposed", "average", seed)]
        ca, cb = a["adversarial_prediction"] == a["label"], b["adversarial_prediction"] == b["label"]
        clean_a, clean_b = a["clean_prediction"] == a["label"], b["clean_prediction"] == b["label"]
        paired.append({"seed": seed, "gained": int((~ca & cb).sum()), "lost": int((ca & ~cb).sum()),
                       "both_correct": int((ca & cb).sum()), "both_wrong": int((~ca & ~cb).sum()),
                       "robust_difference_pp": float((cb.mean()-ca.mean())*100),
                       "clean_difference_pp": float((clean_b.mean()-clean_a.mean())*100)})
    intervals = {}
    for metric in ["robust_difference_pp", "clean_difference_pp"]:
        values = np.array([p[metric] for p in paired])
        sd = float(values.std(ddof=1))
        half = 2.776445105*sd/math.sqrt(5)
        intervals[metric] = {"mean": float(values.mean()), "sd": sd,
                             "paired_t95_lower": float(values.mean()-half), "paired_t95_upper": float(values.mean()+half),
                             "positive_seeds": int((values > 0).sum()), "seeds": 5}
    core.save_json(ROOT/"results/paired_comparisons.json", {"rows": paired, "intervals": intervals})

    fig, ax = plt.subplots(figsize=(7.2, 2.05))
    for key, color, marker in [(main_keys[1], RED, "^"), (main_keys[2], BLUE, "o")]:
        values = class_frame[(class_frame.method == key[0]) & (class_frame.version == key[1])].groupby("class_id").robust_accuracy
        ax.errorbar(range(10), values.mean()*100, yerr=values.std()*100, color=color, marker=marker,
                     capsize=2, ms=4, label=LABELS[key[0]])
    ax.set_xticks(range(10), [s.replace("T-shirt/top", "T-shirt").replace("Ankle boot", "Boot") for s in core.CLASS_NAMES], rotation=30)
    ax.set(ylabel="PGD-20 accuracy (%)", ylim=(0, 100))
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(pad=1.1)
    save_figure(fig, "class_robustness")

    traces = []
    for cfg in configs():
        path = ROOT/"runs"/cfg.name/"validation_trace.json"
        if path.exists():
            for t in json.loads(path.read_text()):
                if t["version"] == "average" and cfg.averaging == "swa" and t["epoch"] <= cfg.average_start:
                    continue
                traces.append({"method": cfg.tag, "seed": cfg.seed, **t})
    tf = pd.DataFrame(traces)
    tf.to_csv(ROOT/"results/validation_traces.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharex=True)
    trace_keys = [("pgd", "raw", RED, "-", "PGD-5"), ("mixed", "raw", GREY, "-", "Mixed CE"),
                  ("proposed", "raw", BLUE, "--", "Proposed raw"), ("proposed", "average", BLUE, "-", "Proposed average")]
    for method, version, color, ls, label in trace_keys:
        group = tf[(tf.method == method) & (tf.version == version)]
        for ax, metric in zip(axes, ["pgd_accuracy", "clean_accuracy"]):
            vals = group.groupby("epoch")[metric]
            mean, sd = vals.mean()*100, vals.std()*100
            ax.plot(mean.index, mean, color=color, ls=ls, marker="o", ms=3, label=label)
            ax.fill_between(mean.index, mean-sd, mean+sd, color=color, alpha=.12, linewidth=0)
    for ax in axes:
        ax.axvspan(0, 10, color=GREY, alpha=.09)
        ax.set(xlabel="Training epoch", xlim=(4, 31), xticks=[5, 10, 15, 20, 25, 30])
    axes[0].set(ylabel="Validation PGD-20 accuracy (%)")
    axes[1].set(ylabel="Validation clean accuracy (%)")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout(pad=1.2)
    save_figure(fig, "validation_dynamics")

    audits = json.loads((ROOT/"results/attack_audit.json").read_text())["rows"]
    pd.DataFrame(audits).to_csv(ROOT/"results/attack_audit.csv", index=False)
    sweep = pd.DataFrame(json.loads((ROOT/"results/epsilon_sweep.json").read_text()))
    fig, ax = plt.subplots(figsize=(7, 3))
    for method, color, marker in [("standard", GREY, "s"), ("pgd", RED, "^"), ("proposed", BLUE, "o")]:
        grouped = sweep[sweep.method == method].groupby("epsilon").accuracy
        ax.errorbar(grouped.mean().index, grouped.mean()*100, yerr=grouped.std()*100,
                     color=color, marker=marker, capsize=2, label=LABELS[method])
    ax.set(xlabel=r"Perturbation radius $\epsilon$", ylabel="CE-PGD-50 accuracy (%)", ylim=(0, 100), xticks=[0, .05, .1, .15, .2])
    ax.legend(fontsize=8)
    fig.tight_layout(pad=1.2)
    save_figure(fig, "epsilon_sweep")

    confusion = {}
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.8))
    for ax, key in zip(axes, [("pgd", "raw"), ("proposed", "average")]):
        matrix = np.zeros((10, 10), dtype=np.int64)
        for seed in SEEDS:
            arr = predictions[(key[0], key[1], seed)]
            np.add.at(matrix, (arr["label"], arr["adversarial_prediction"]), 1)
        percent = matrix/matrix.sum(1, keepdims=True)*100
        confusion[key[0]] = {"counts_across_five_seeds": matrix.tolist(), "row_percent": percent.tolist()}
        im = ax.imshow(percent, cmap="Blues", vmin=0, vmax=100)
        ax.set(title=LABELS[key[0]], xlabel="Attack prediction", ylabel="True class",
               xticks=range(10), yticks=range(10))
        for i in range(10):
            for j in range(10):
                if percent[i, j] >= 2:
                    ax.text(j, i, f"{percent[i,j]:.0f}", ha="center", va="center", fontsize=7,
                             color="white" if percent[i,j] > 50 else "black")
    fig.colorbar(im, ax=axes, fraction=.025, pad=.025, label="Row percentage")
    fig.subplots_adjust(wspace=.3, right=.88, bottom=.16)
    save_figure(fig, "class_confusion")
    core.save_json(ROOT/"results/class_confusion.json", confusion)

    seed = SEEDS[0]
    a, b = predictions[("pgd", "raw", seed)], predictions[("proposed", "average", seed)]
    ca, cb = a["adversarial_prediction"] == a["label"], b["adversarial_prediction"] == b["label"]
    clean_b = b["clean_prediction"] == b["label"]
    masks = [~ca & cb & clean_b, ca & ~cb & clean_b, ~ca & ~cb & clean_b, ca & cb & clean_b]
    descriptions = ["Gained", "Lost", "Shared failure", "Shared success"]
    ids = [int(np.flatnonzero(mask)[0]) for mask in masks]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = core.load_data(device, include_test=True)
    x, y = data["test"]
    cfg = next(c for c in configs() if c.tag == "proposed" and c.seed == seed)
    model = load_model(cfg, device)
    rng = torch.Generator(device=device).manual_seed(9001)
    attacked_cases = {}
    for start in range(0, (max(ids)//512+1)*512, 512):
        attacked = core.pgd(model, x[start:start+512], y[start:start+512], generator=rng)
        for idx in ids:
            if start <= idx < start+len(attacked):
                attacked_cases[idx] = attacked[idx-start].detach().cpu().numpy()
    cases = []
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 2.7))
    for col, (idx, desc) in enumerate(zip(ids, descriptions)):
        original = x[idx].cpu().numpy()
        adv = attacked_cases[idx]
        with torch.no_grad():
            pred = int(model(torch.tensor(adv, device=device)[None]).argmax(1))
        assert pred == int(b["adversarial_prediction"][idx])
        for row, image in enumerate([original, adv]):
            axes[row, col].imshow(image.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
        axes[0, col].set_title(f"{desc} #{idx}\nTrue: {core.CLASS_NAMES[int(y[idx])]}", fontsize=8.5)
        axes[1, col].set_title("Proposed: "+core.CLASS_NAMES[pred], fontsize=8)
        cases.append({"category": desc, "test_index": idx, "true": core.CLASS_NAMES[int(y[idx])],
                      "pgd_attack_prediction": core.CLASS_NAMES[int(a["adversarial_prediction"][idx])],
                      "proposed_attack_prediction": core.CLASS_NAMES[pred],
                      "linf": float(np.abs(adv-original).max())})
    for row, label in enumerate(["Clean", "PGD-20"]):
        axes[row, 0].text(-.15, .5, label, transform=axes[row, 0].transAxes, ha="right", va="center", rotation=90, fontsize=8)
    fig.tight_layout(pad=1.15)
    save_figure(fig, "failure_examples")
    core.save_json(ROOT/"results/cases.json", cases)

    # The complete validation-search ledger is kept outside the six-page body.
    candidates = json.loads((ROOT/"results/frozen_protocol.json").read_text())["all_development"]
    pd.DataFrame([{k: v for k, v in c.items() if k != "config"} for c in candidates]).to_csv(ROOT/"results/development_candidates.csv", index=False)
    records = [json.loads(p.read_text()) for p in (ROOT/"runs").glob("*/training.json")]
    development = [r for r in records if r["config"]["stage"] == "development"]
    confirmation = [r for r in records if r["config"]["stage"] == "confirmation"]
    workload = {"development_runs": len(development), "confirmation_runs": len(confirmation),
                "distinct_optimization_runs": len(records), "total_optimization_epochs": sum(r["epochs"] for r in records),
                "development_training_seconds": sum(r["training_seconds"] for r in development),
                "confirmation_training_seconds": sum(r["training_seconds"] for r in confirmation),
                "validation_trace_seconds": sum(r.get("diagnostic_validation_seconds", 0) for r in records),
                "raw_and_average_are_shared_optimization_trajectories": True}
    core.save_json(ROOT/"results/workload.json", workload)
    core.save_json(ROOT/"results/analysis_checks.json", {"full_example_counts_match": True,
        "all_test_classes_have_1000_examples": True, "displayed_attacks_match_saved_predictions": True,
        "case_selection": "First test index per declared outcome category, seed 7411",
        "confidence_interval": "Paired Student t interval over five training seeds; df=4"})
    print(pd.DataFrame(summaries).to_string(index=False))
    return summaries


if __name__ == "__main__":
    run()

"""Supplementary CE/margin attacks retain every successful intermediate iterate."""
from __future__ import annotations
import json
import numpy as np
import torch
from torch.nn import functional as F
import core
from experiment import ROOT
from run_suite import configs, load_model


def survival(model, x, y, loss="ce", epsilon=.1, alpha=.005, steps=50, restarts=3,
             seed=9002, batch=512):
    model.eval()
    rng = torch.Generator(device=x.device).manual_seed(seed)
    survivors = []
    max_delta = 0.
    for i in range(0, len(y), batch):
        xb, yb = x[i:i+batch], y[i:i+batch]
        lower, upper = (xb-epsilon).clamp(0, 1), (xb+epsilon).clamp(0, 1)
        with torch.no_grad():
            correct = model(xb).argmax(1) == yb
        for _ in range(restarts):
            noise = torch.rand(xb.shape, device=x.device, generator=rng)*2*epsilon-epsilon
            adv = (xb+noise).clamp(0, 1).detach()
            for _ in range(steps):
                adv.requires_grad_(True)
                logits = model(adv)
                correct &= logits.detach().argmax(1) == yb
                if loss == "ce":
                    losses = F.cross_entropy(logits, yb, reduction="none")
                elif loss == "margin":
                    losses = -core.margin(logits, yb)
                else:
                    raise ValueError(loss)
                grad = torch.autograd.grad(losses.sum(), adv, only_inputs=True)[0]
                adv = torch.maximum(torch.minimum(adv.detach()+alpha*grad.sign(), upper), lower).detach()
            with torch.no_grad():
                correct &= model(adv).argmax(1) == yb
                max_delta = max(max_delta, float((adv-xb).abs().max()))
        survivors.append(correct.detach().cpu().numpy())
    assert max_delta <= epsilon+1e-6
    return np.concatenate(survivors), max_delta


def run(device, force=False):
    data = core.load_data(device, include_test=True)
    x, y = data["test"]
    rows, sweep = [], []
    for cfg in configs():
        if cfg.tag not in {"standard", "pgd", "proposed"}:
            continue
        folder = ROOT/"runs"/cfg.name
        path = folder/"audit.json"
        if path.exists() and not force:
            saved = json.loads(path.read_text())
            rows.append(saved["audit"])
            sweep.extend(saved["sweep"])
            continue
        model = load_model(cfg, device)
        version = "average" if cfg.averaging != "none" else "raw"
        official = np.load(folder/(f"test_predictions_{version}.npz"))
        official_correct = official["adversarial_prediction"] == official["label"]
        ce, ce_delta = survival(model, x, y)
        rival, margin_delta = survival(model, x, y, loss="margin", seed=9003)
        combined = official_correct & ce & rival
        row = {"method": cfg.tag, "seed": cfg.seed, "n": len(y),
               "official_pgd20": float(official_correct.mean()), "ce50_3restart": float(ce.mean()),
               "margin50_3restart": float(rival.mean()), "combined": float(combined.mean()),
               "combined_correct": int(combined.sum()), "additional_failures": int((official_correct & ~combined).sum()),
               "max_linf_ce": ce_delta, "max_linf_margin": margin_delta}
        np.savez_compressed(folder/"audit_predictions.npz", label=official["label"],
                            official_correct=official_correct, ce_correct=ce,
                            margin_correct=rival, combined_correct=combined)
        local_sweep = []
        for epsilon in [.0, .05, .10, .15, .20]:
            if epsilon == .0:
                mask = official["clean_prediction"] == official["label"]
            else:
                mask, _ = survival(model, x, y, epsilon=epsilon, alpha=epsilon/20,
                                   restarts=1, seed=9100+int(epsilon*100))
            local_sweep.append({"method": cfg.tag, "seed": cfg.seed, "epsilon": epsilon,
                                "accuracy": float(mask.mean()), "correct": int(mask.sum()), "n": len(mask)})
        core.save_json(path, {"audit": row, "sweep": local_sweep})
        rows.append(row)
        sweep.extend(local_sweep)
        core.save_json(ROOT/"results/attack_audit.json", {"rows": rows,
            "protocol": "Full test, eval mode; union of official PGD20, CE-PGD50 and margin-PGD50. Supplementary attacks use alpha .005 and three independent random starts, counting failure at any intermediate iterate and on the clean image.",
            "attack_losses": ["cross_entropy", "maximum_other_logit_minus_true_logit"],
            "certified": False})
        core.save_json(ROOT/"results/epsilon_sweep.json", sweep)
        print("Audited", cfg.name, row["official_pgd20"], row["combined"], flush=True)
    core.save_json(ROOT/"results/attack_audit.json", {"rows": rows,
        "protocol": "Full test, eval mode; union of official PGD20, CE-PGD50 and margin-PGD50. Supplementary attacks use alpha .005 and three independent random starts, counting failure at any intermediate iterate and on the clean image.",
        "attack_losses": ["cross_entropy", "maximum_other_logit_minus_true_logit"], "certified": False})
    core.save_json(ROOT/"results/epsilon_sweep.json", sweep)
    return rows


if __name__ == "__main__":
    run(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

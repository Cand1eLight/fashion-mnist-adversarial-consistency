"""Validation-only development of consistency objectives and parameter averaging."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import core

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    tag: str
    objective: str = "mixed"
    seed: int = 7404
    epochs: int = 30
    batch_size: int = 256
    lr: float = .001
    weight_decay: float = .0001
    steps: int = 5
    beta: float = 2.
    clean_weight: float = .25
    ramp_epochs: int = 0
    warmup_steps: int | None = None
    averaging: str = "none"
    ema_decay: float = .995
    average_start: int = 5
    stage: str = "development"
    trace_validation: bool = False

    @property
    def name(self):
        return f"{self.stage}_{self.tag}_s{self.seed}_e{self.epochs}"


def kl_divergence(clean_logits, adversarial_logits):
    """KL(p_clean || p_adv), retaining both outer-loss gradient paths."""
    p = F.softmax(clean_logits, dim=1)
    return (p * (F.log_softmax(clean_logits, dim=1) -
                 F.log_softmax(adversarial_logits, dim=1))).sum(1)


def kl_attack(model, x, steps, epsilon, alpha, generator):
    """The clean distribution is fixed during the inner maximization."""
    lower, upper = (x-epsilon).clamp(0, 1), (x+epsilon).clamp(0, 1)
    with torch.no_grad():
        p = F.softmax(model(x), dim=1).detach()
    noise = torch.rand(x.shape, device=x.device, generator=generator)*2*epsilon-epsilon
    adv = (x+noise).clamp(0, 1).detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        objective = F.kl_div(F.log_softmax(model(adv), dim=1), p, reduction="sum")
        grad = torch.autograd.grad(objective, adv, only_inputs=True)[0]
        adv = torch.maximum(torch.minimum(adv.detach()+alpha*grad.sign(), upper), lower).detach()
    return adv


def objective(model, x, y, cfg, epsilon, rng):
    if cfg.objective == "standard":
        return F.cross_entropy(model(x), y)
    alpha = epsilon/4
    if cfg.objective == "trades":
        adv = kl_attack(model, x, cfg.steps, epsilon, alpha, rng)
        clean_logits, adv_logits = model(x), model(adv)
        return F.cross_entropy(clean_logits, y) + cfg.beta*kl_divergence(clean_logits, adv_logits).mean()
    adv = core.pgd(model, x, y, cfg.steps, epsilon, alpha, rng)
    adv_logits = model(adv)
    if cfg.objective == "pgd":
        return F.cross_entropy(adv_logits, y)
    clean_logits = model(x)
    if cfg.objective == "mixed":
        return cfg.clean_weight*F.cross_entropy(clean_logits, y) + (1-cfg.clean_weight)*F.cross_entropy(adv_logits, y)
    if cfg.objective in {"hybrid", "gated"}:
        penalty = kl_divergence(clean_logits, adv_logits)
        if cfg.objective == "gated":
            confidence = F.softmax(clean_logits, dim=1).gather(1, y[:, None]).squeeze(1).detach()
            penalty = confidence*penalty
        return (cfg.clean_weight*F.cross_entropy(clean_logits, y) +
                (1-cfg.clean_weight)*F.cross_entropy(adv_logits, y) +
                cfg.beta*penalty.mean())
    if cfg.objective == "mart":
        prob = F.softmax(adv_logits, dim=1)
        other = prob.detach().clone()
        other.scatter_(1, y[:, None], -torch.inf)
        rival = other.argmax(1)
        boosted = F.cross_entropy(adv_logits, y) - torch.log((1-prob.gather(1, rival[:, None]).squeeze(1)).clamp_min(1e-12)).mean()
        confidence = F.softmax(clean_logits, dim=1).gather(1, y[:, None]).squeeze(1)
        return boosted + cfg.beta*((1-confidence)*kl_divergence(clean_logits, adv_logits)).mean()
    raise ValueError(cfg.objective)


def train(cfg, data, device, force=False):
    folder = ROOT/"runs"/cfg.name
    checkpoint = ROOT/"checkpoints"/(cfg.name+".pt")
    average_checkpoint = ROOT/"checkpoints"/(cfg.name+"_average.pt")
    have_weights = checkpoint.exists() and (cfg.averaging == "none" or average_checkpoint.exists())
    if (folder/"training.json").exists() and (folder/"validation.json").exists() and have_weights and not force:
        record = json.loads((folder/"training.json").read_text())
        record["config"].setdefault("warmup_steps", None)
        if record["config"] != asdict(cfg):
            raise ValueError("Cached configuration differs")
        return record
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    core.seed_all(cfg.seed)
    model = core.make_model(device)
    averaged = copy.deepcopy(model) if cfg.averaging != "none" else None
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, cfg.epochs, eta_min=1e-5)
    shuffle_rng = torch.Generator(device=device).manual_seed(cfg.seed+10000)
    attack_rng = torch.Generator(device=device).manual_seed(cfg.seed+20000)
    x, y = data["train"]
    vx, vy = data["validation"]
    trace_idx = torch.cat([torch.where(vy == k)[0][:100] for k in range(10)])
    optim.zero_grad(set_to_none=True)
    F.cross_entropy(model(x[:cfg.batch_size]), y[:cfg.batch_size]).backward()
    optim.zero_grad(set_to_none=True)
    core.sync(device)
    started = time.perf_counter()
    excluded = 0.
    history, trace = [], []
    n_average = 0
    for epoch in range(cfg.epochs):
        core.sync(device)
        epoch_started = time.perf_counter()
        model.train()
        permutation = torch.randperm(len(y), device=device, generator=shuffle_rng)
        epsilon = .10*min(1., (epoch+1)/cfg.ramp_epochs) if cfg.ramp_epochs else .10
        steps = cfg.warmup_steps if cfg.warmup_steps is not None and epoch < cfg.ramp_epochs else cfg.steps
        step_cfg = replace(cfg, steps=steps)
        loss_sum = torch.zeros((), device=device)
        lr = optim.param_groups[0]["lr"]
        for i in range(0, len(y), cfg.batch_size):
            idx = permutation[i:i+cfg.batch_size]
            xb, yb = x[idx], y[idx]
            optim.zero_grad(set_to_none=True)
            loss = objective(model, xb, yb, step_cfg, epsilon, attack_rng)
            loss.backward()
            optim.step()
            if averaged is not None and cfg.averaging == "ema":
                with torch.no_grad():
                    src, dst = list(model.parameters()), list(averaged.parameters())
                    if epoch < cfg.average_start:
                        torch._foreach_copy_(dst, src)
                    else:
                        torch._foreach_mul_(dst, cfg.ema_decay)
                        torch._foreach_add_(dst, src, alpha=1-cfg.ema_decay)
            loss_sum += loss.detach()*len(yb)
        if averaged is not None and cfg.averaging == "swa" and epoch >= cfg.average_start:
            with torch.no_grad():
                src, dst = list(model.parameters()), list(averaged.parameters())
                torch._foreach_mul_(dst, n_average/(n_average+1))
                torch._foreach_add_(dst, src, alpha=1/(n_average+1))
                n_average += 1
        scheduler.step()
        core.sync(device)
        elapsed = time.perf_counter()-epoch_started
        row = {"epoch": epoch+1, "loss": float(loss_sum/len(y)), "lr": lr,
               "epsilon": epsilon, "seconds": elapsed,
               "mean_attack_steps": 0 if cfg.objective == "standard" else steps}
        history.append(row)
        print(cfg.name, f"epoch {epoch+1:02d}: loss={row['loss']:.4f}, time={elapsed:.2f}s", flush=True)
        core.save_json(folder/"progress.json", row)
        if cfg.trace_validation and (epoch+1) % 5 == 0:
            core.sync(device)
            paused = time.perf_counter()
            for version, current in [("raw", model), ("average", averaged)]:
                if current is not None:
                    if version == "average" and cfg.averaging == "swa" and epoch < cfg.average_start:
                        continue  # The late arithmetic average is not defined yet.
                    scores, _ = core.evaluate(current, vx[trace_idx], vy[trace_idx], seed=19001)
                    trace.append({"epoch": epoch+1, "version": version, **scores})
            core.sync(device)
            excluded += time.perf_counter()-paused
    core.sync(device)
    total = time.perf_counter()-started-excluded
    torch.save(model.state_dict(), checkpoint)
    if averaged is not None:
        torch.save(averaged.state_dict(), average_checkpoint)
    record = {"config": asdict(cfg), "environment": core.environment(), "history": history,
              "training_seconds": total, "epochs": cfg.epochs,
              "mean_epoch_seconds": sum(h["seconds"] for h in history)/cfg.epochs,
              "diagnostic_validation_seconds": excluded,
              "mean_attack_steps": sum(h["mean_attack_steps"] for h in history)/cfg.epochs,
              "timing": "Synchronized wall time of all optimization, attack generation, parameter averaging, epoch logs and updates; excludes preparation, warmup, diagnostic validation and checkpoint serialization",
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    core.save_json(folder/"training.json", record)
    with (folder/"history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    if trace:
        core.save_json(folder/"validation_trace.json", trace)
    scores, arrays = core.evaluate(model, vx, vy, seed=19001)
    core.save_json(folder/"validation_raw.json", scores)
    np.savez_compressed(folder/"validation_predictions_raw.npz", **arrays)
    if averaged is not None:
        scores, arrays = core.evaluate(averaged, vx, vy, seed=19001)
        core.save_json(folder/"validation_average.json", scores)
        np.savez_compressed(folder/"validation_predictions_average.npz", **arrays)
    core.save_json(folder/"validation.json", scores)
    print(cfg.name, "validation", json.dumps(scores), flush=True)
    return record


def development_configs():
    return [Config("mixed5", objective="mixed"),
            *[Config(f"trades_b{beta:g}", objective="trades", beta=beta) for beta in [1., 2., 4., 6.]],
            Config("mart_b2", objective="mart", beta=2.),
            Config("hybrid_b1", objective="hybrid", beta=1.),
            Config("mixed5_ema995", objective="mixed", averaging="ema"),
            Config("mixed5_swa20", objective="mixed", averaging="swa", average_start=20),
            Config("trades_b2_ema995", objective="trades", beta=2., averaging="ema")]


def development(device):
    data = core.load_data(device)
    rows = []
    for cfg in development_configs():
        record = train(cfg, data, device)
        val = json.loads((ROOT/"runs"/cfg.name/"validation.json").read_text())
        row = {"tag": cfg.tag, "training_seconds": record["training_seconds"],
               "validation_clean": val["clean_correct"]/val["n"],
               "validation_pgd20": val["robust_correct"]/val["n"], **asdict(cfg)}
        rows.append(row)
        core.save_json(ROOT/"results"/"development.json", rows)
    print("DEVELOPMENT SUMMARY", json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["development"])
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    development(device)

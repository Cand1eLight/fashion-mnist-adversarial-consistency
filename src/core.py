"""Shared data, fixed architecture, official PGD and correctness checks."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import random
import struct
import time
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
               "Shirt", "Sneaker", "Bag", "Ankle boot"]
OFFICIAL_ATTACK = {"epsilon": 0.10, "alpha": 0.01, "steps": 20,
                   "random_start": True, "loss": "cross_entropy", "norm": "linf",
                   "restarts": 1, "iterate": "final", "targeted": False}


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def environment():
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda": torch.version.cuda, "precision": "float32", "tf32": False,
            "deterministic_algorithms": True, "cpu_threads": 4}


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def prepare_data():
    dest = ROOT / "data"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, md5 in FILES.items():
        path = dest / name
        source = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/" + name
        if not path.exists():
            print("Downloading", name, flush=True)
            urllib.request.urlretrieve(source, path)
        blob = path.read_bytes()
        if hashlib.md5(blob).hexdigest() != md5:
            raise ValueError(f"Dataset checksum mismatch: {name}")
        manifest[name] = {"source": source, "md5": md5,
                          "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
    save_json(dest / "manifest.json", manifest)
    return manifest


def read_idx(path):
    blob = gzip.decompress(Path(path).read_bytes())
    magic, count = struct.unpack(">II", blob[:8])
    if magic == 2051:
        rows, cols = struct.unpack(">II", blob[8:16])
        return np.frombuffer(blob, dtype=np.uint8, offset=16).copy().reshape(count, rows * cols)
    if magic == 2049:
        return np.frombuffer(blob, dtype=np.uint8, offset=8).copy()
    raise ValueError("Unsupported IDX header")


def split_indices(labels):
    rng = np.random.default_rng(7404)
    train, valid = [], []
    for label in range(10):
        idx = np.flatnonzero(labels == label)
        rng.shuffle(idx)
        valid.extend(idx[:500])
        train.extend(idx[500:])
    train, valid = np.sort(train), np.sort(valid)
    assert len(train) == 55000 and len(valid) == 5000
    assert not np.intersect1d(train, valid).size
    np.savez(ROOT / "data" / "split_indices.npz", train=train, validation=valid)
    save_json(ROOT / "data" / "split_manifest.json", {
        "seed": 7404, "stratification": "500 validation examples per class",
        "training_examples": len(train), "validation_examples": len(valid),
        "train_indices_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(valid.tobytes()).hexdigest(),
        "preprocessing": "uint8 / 255, pixels in [0,1]; no augmentation or normalization"})
    return train, valid


def load_data(device, include_test=False):
    prepare_data()
    x = read_idx(ROOT / "data" / "train-images-idx3-ubyte.gz")
    y = read_idx(ROOT / "data" / "train-labels-idx1-ubyte.gz")
    train, valid = split_indices(y)
    out = {"train": (torch.tensor(x[train], device=device, dtype=torch.float32) / 255,
                     torch.tensor(y[train], device=device, dtype=torch.long)),
           "validation": (torch.tensor(x[valid], device=device, dtype=torch.float32) / 255,
                          torch.tensor(y[valid], device=device, dtype=torch.long))}
    if include_test:
        out["test"] = (torch.tensor(read_idx(ROOT / "data" / "t10k-images-idx3-ubyte.gz"),
                                    device=device, dtype=torch.float32) / 255,
                       torch.tensor(read_idx(ROOT / "data" / "t10k-labels-idx1-ubyte.gz"),
                                    device=device, dtype=torch.long))
    return out


def make_model(device):
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                         nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10)).to(device)


def margin(logits, labels):
    true = logits.gather(1, labels[:, None]).squeeze(1)
    other = logits.clone()
    other.scatter_(1, labels[:, None], -torch.inf)
    return true - other.amax(1)


def pgd(model, x, y, steps=20, epsilon=.10, alpha=.01, generator=None,
        start=None, keep_best=False):
    """Untargeted CE ascent in raw pixel coordinates; model gradients are untouched."""
    lower, upper = (x - epsilon).clamp(0, 1), (x + epsilon).clamp(0, 1)
    if start is None:
        noise = torch.rand(x.shape, device=x.device, generator=generator) * (2 * epsilon) - epsilon
        adv = (x + noise).clamp(0, 1).detach()
    else:
        adv = torch.maximum(torch.minimum(start.detach(), upper), lower)
    best = adv.clone() if keep_best else None
    best_loss = torch.full_like(y, -torch.inf, dtype=x.dtype) if keep_best else None
    for _ in range(steps):
        adv.requires_grad_(True)
        logits = model(adv)
        losses = F.cross_entropy(logits, y, reduction="none")
        if keep_best:
            improve = losses.detach() > best_loss
            best[improve] = adv.detach()[improve]
            best_loss = torch.maximum(best_loss, losses.detach())
        grad = torch.autograd.grad(losses.sum(), adv, only_inputs=True)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.maximum(torch.minimum(adv, upper), lower).detach()
    if keep_best:
        with torch.no_grad():
            last_loss = F.cross_entropy(model(adv), y, reduction="none")
            improve = last_loss > best_loss
            best[improve] = adv[improve]
        return best.detach()
    return adv


def clean_predictions(model, x, batch=512):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[i:i+batch]).argmax(1) for i in range(0, len(x), batch)])


def evaluate(model, x, y, seed=9001, steps=20, alpha=.01, restarts=1,
             keep_best=False, batch=512):
    model.eval()
    clean = clean_predictions(model, x, batch)
    rng = torch.Generator(device=x.device).manual_seed(seed)
    adv_predictions, adv_loss = [], []
    for i in range(0, len(x), batch):
        xb, yb = x[i:i+batch], y[i:i+batch]
        chosen_loss = torch.full_like(yb, -torch.inf, dtype=xb.dtype)
        chosen_pred = torch.zeros_like(yb)
        any_failure = torch.zeros_like(yb, dtype=torch.bool)
        for _ in range(restarts):
            adv = pgd(model, xb, yb, steps, .10, alpha, rng, keep_best=keep_best)
            with torch.no_grad():
                logits = model(adv)
                loss = F.cross_entropy(logits, yb, reduction="none")
                pred = logits.argmax(1)
                # Preserve any successful restart; among failures retain the largest CE.
                failed = pred != yb
                choose = (failed & ~any_failure) | ((failed == any_failure) & (loss > chosen_loss))
                chosen_pred[choose] = pred[choose]
                chosen_loss[choose] = loss[choose]
                any_failure |= failed
        adv_predictions.append(chosen_pred)
        adv_loss.append(chosen_loss)
    pred = torch.cat(adv_predictions)
    losses = torch.cat(adv_loss)
    metrics = {"n": len(y), "clean_correct": int((clean == y).sum()),
               "robust_correct": int((pred == y).sum()),
               "clean_accuracy": float((clean == y).float().mean()),
               "pgd_accuracy": float((pred == y).float().mean()),
               "attack_seed": seed, "steps": steps, "alpha": alpha,
               "epsilon": .10, "restarts": restarts, "keep_best": keep_best}
    return metrics, {"label": y.cpu().numpy(), "clean_prediction": clean.cpu().numpy(),
                     "adversarial_prediction": pred.cpu().numpy(), "attack_loss": losses.cpu().numpy()}


def check_attack(device):
    seed_all(17)
    model = make_model(device)
    x = torch.rand((32, 784), device=device)
    y = torch.randint(0, 10, (32,), device=device)
    model.eval()
    params = [p.detach().clone() for p in model.parameters()]
    rng = torch.Generator(device=device).manual_seed(71)
    adv = pgd(model, x, y, generator=rng)
    assert adv.min() >= 0 and adv.max() <= 1
    assert float((adv-x).abs().max()) <= .100001
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(a, b) for a, b in zip(params, model.parameters()))
    again = pgd(model, x, y, generator=torch.Generator(device=device).manual_seed(71))
    assert torch.equal(adv, again)
    zero = pgd(model, x, y, epsilon=0, generator=rng)
    assert torch.equal(zero, x)
    # An independent implementation checks the official final-iterate PGD numerically.
    gen = torch.Generator(device=device).manual_seed(71)
    reference = (x + torch.rand(x.shape, device=device, generator=gen)*.2-.1).clamp(0, 1)
    for _ in range(20):
        reference = reference.detach().requires_grad_(True)
        gradient = torch.autograd.grad(F.cross_entropy(model(reference), y), reference)[0]
        reference = (reference.detach()+.01*gradient.sign()).clamp(x-.1, x+.1).clamp(0, 1)
    assert torch.allclose(reference, adv, atol=1e-6)
    assert sum(p.numel() for p in model.parameters()) == 235146
    return {"pixel_bounds": True, "linf_projection": True, "parameter_unchanged": True,
            "parameter_gradients_untouched": True, "repeatability": True, "epsilon_zero": True,
            "independent_pgd_agreement": True, "parameters": 235146,
            "max_perturbation": float((adv-x).abs().max())}



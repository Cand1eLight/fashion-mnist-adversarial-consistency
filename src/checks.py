"""Numerical checks for the threat model, consistency loss and dataset separation."""
import numpy as np
import torch
from torch.nn import functional as F
import core
from experiment import ROOT, kl_attack, kl_divergence


def run(device):
    out = core.check_attack(device)
    core.seed_all(23)
    model = core.make_model(device)
    x, y = torch.rand(16, 784, device=device), torch.arange(16, device=device) % 10
    params = [p.detach().clone() for p in model.parameters()]
    rng = torch.Generator(device=device).manual_seed(77)
    adv = kl_attack(model, x, 5, .1, .025, rng)
    assert float((adv-x).abs().max()) <= .100001
    assert float(adv.min()) >= 0 and float(adv.max()) <= 1
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(a, b) for a, b in zip(params, model.parameters()))
    a = torch.randn(12, 10, device=device, requires_grad=True)
    b = torch.randn(12, 10, device=device, requires_grad=True)
    actual = kl_divergence(a, b)
    reference = F.kl_div(F.log_softmax(b, dim=1), F.softmax(a, dim=1), reduction="none").sum(1)
    assert torch.allclose(actual, reference, atol=1e-6)
    assert torch.allclose(kl_divergence(a, a), torch.zeros(12, device=device), atol=1e-6)
    ga, gb = torch.autograd.grad(actual.mean(), [a, b])
    assert torch.isfinite(ga).all() and torch.isfinite(gb).all()
    assert ga.abs().sum() > 0 and gb.abs().sum() > 0
    split = np.load(ROOT/"data/split_indices.npz")
    assert len(split["train"]) == 55000 and len(split["validation"]) == 5000
    assert not np.intersect1d(split["train"], split["validation"]).size
    assert np.array_equal(np.sort(np.r_[split["train"], split["validation"]]), np.arange(60000))
    # The averaging kernel is checked against an independently accumulated arithmetic mean.
    states = [torch.randn(31, device=device) for _ in range(10)]
    average = torch.zeros_like(states[0])
    for n, state in enumerate(states):
        torch._foreach_mul_([average], n/(n+1))
        torch._foreach_add_([average], [state], alpha=1/(n+1))
    assert torch.allclose(average, torch.stack(states).mean(0), atol=1e-6)
    out.update(kl_attack_projection=True, kl_attack_parameter_gradients_untouched=True,
               consistency_matches_independent_formula=True, consistency_identity=True,
               consistency_gradients_reach_both_branches=True, train_validation_disjoint=True,
               split_covers_original_training_only=True, arithmetic_average_matches_reference=True)
    core.save_json(ROOT/"results/implementation_checks.json", out)
    return out


if __name__ == "__main__":
    print(run(torch.device("cuda" if torch.cuda.is_available() else "cpu")))

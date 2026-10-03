# Experiment ledger

MC647418 · Fashion-MNIST · thirty-epoch fixed-network study

## Development decisions

Twenty-two configurations use training seed 7404 and the fixed 5,000-image validation split. The clean floor is 83.74%, one point above the PGD reference. Within 0.20 points of the highest eligible robustness, measured optimization time breaks the tie. The selected configuration is `development_hybrid_b05_ramp10_swa20_s7404_e30`.

| Configuration | Loss | KL coefficient | Radius ramp | Early steps | Average | Clean (%) | PGD-20 (%) | Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gated_b05_ramp10_swa20 | gated | 0.5 | 10 | 5 | swa | 84.80 | 67.66 | 48.52 |
| gated_b05_ramp10_warm2_swa20 | gated | 0.5 | 10 | 2 | swa | 84.40 | 67.32 | 66.74 |
| gated_b1_ramp10_swa20 | gated | 1.0 | 10 | 5 | swa | 83.98 | 68.08 | 70.15 |
| gated_b1_ramp10_warm2_swa20 | gated | 1.0 | 10 | 2 | swa | 84.10 | 67.42 | 63.91 |
| hybrid_b025_ramp10 | hybrid | 0.25 | 10 | 5 | none | 84.80 | 68.00 | 53.74 |
| hybrid_b025_ramp10_swa20 | hybrid | 0.25 | 10 | 5 | swa | 84.56 | 67.80 | 50.10 |
| hybrid_b025 | hybrid | 0.25 | 0 | 5 | none | 84.08 | 67.52 | 60.47 |
| hybrid_b025_swa20 | hybrid | 0.25 | 0 | 5 | swa | 84.00 | 67.42 | 53.83 |
| hybrid_b05_ramp10_swa20 | hybrid | 0.5 | 10 | 5 | swa | 84.18 | 68.18 | 48.82 |
| hybrid_b05_ramp10_warm2_swa20 | hybrid | 0.5 | 10 | 2 | swa | 84.16 | 67.50 | 66.68 |
| hybrid_b05_ramp10_warm3_swa20 | hybrid | 0.5 | 10 | 3 | swa | 84.16 | 67.72 | 70.17 |
| hybrid_b1 | hybrid | 1.0 | 0 | 5 | none | 82.96 | 67.98 | 44.20 |
| mart_b2 | mart | 2.0 | 0 | 5 | none | 81.82 | 67.24 | 50.77 |
| mixed5_ema995 | mixed | 2.0 | 0 | 5 | ema | 84.74 | 67.20 | 43.27 |
| mixed5_ramp10_swa20 | mixed | 2.0 | 10 | 5 | swa | 84.90 | 67.48 | 45.71 |
| mixed5 | mixed | 2.0 | 0 | 5 | none | 84.66 | 67.22 | 43.20 |
| mixed5_swa20 | mixed | 2.0 | 0 | 5 | swa | 84.60 | 67.28 | 42.52 |
| trades_b1 | trades | 1.0 | 0 | 5 | none | 86.98 | 63.92 | 48.13 |
| trades_b2_ema995 | trades | 2.0 | 0 | 5 | ema | 85.54 | 65.92 | 45.66 |
| trades_b2 | trades | 2.0 | 0 | 5 | none | 85.58 | 65.94 | 46.14 |
| trades_b4 | trades | 4.0 | 0 | 5 | none | 83.86 | 67.44 | 44.87 |
| trades_b6 | trades | 6.0 | 0 | 5 | none | 83.00 | 66.94 | 44.97 |

The `beta` field is the KL coefficient for TRADES/hybrid/gated objectives and MART's regularizer. It is unused by mixed CE. `swa` denotes the implemented late arithmetic average under the common cosine schedule, rather than a separate cyclical-learning-rate optimizer.

## Frozen confirmation results

Each seed shares initialization and minibatch order across methods. The order of methods rotates between seeds. Four optimized mixed-loss families each produce raw and averaged states; this avoids retraining an identical trajectory for the averaging control.

| Method | State | Seed | Time (s) | Epoch time (s) | Clean (%) | PGD-20 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| standard | raw | 7411 | 14.41 | 0.479 | 89.52 | 0.78 |
| pgd | raw | 7411 | 51.22 | 1.706 | 82.13 | 66.93 |
| mixed | raw | 7411 | 53.51 | 1.783 | 83.63 | 67.44 |
| mixed | average | 7411 | 53.51 | 1.783 | 83.52 | 67.43 |
| curriculum_mixed | raw | 7411 | 49.38 | 1.645 | 84.14 | 67.26 |
| curriculum_mixed | average | 7411 | 49.38 | 1.645 | 84.07 | 67.15 |
| consistency | raw | 7411 | 49.26 | 1.641 | 82.69 | 67.62 |
| consistency | average | 7411 | 49.26 | 1.641 | 82.72 | 67.60 |
| proposed | raw | 7411 | 45.12 | 1.503 | 83.26 | 67.88 |
| proposed | average | 7411 | 45.12 | 1.503 | 83.24 | 67.92 |
| pgd | raw | 7412 | 39.16 | 1.305 | 82.20 | 66.89 |
| mixed | raw | 7412 | 43.53 | 1.450 | 83.90 | 67.77 |
| mixed | average | 7412 | 43.53 | 1.450 | 83.78 | 67.61 |
| curriculum_mixed | raw | 7412 | 43.33 | 1.443 | 84.20 | 67.39 |
| curriculum_mixed | average | 7412 | 43.33 | 1.443 | 84.08 | 67.33 |
| consistency | raw | 7412 | 46.11 | 1.536 | 82.96 | 67.86 |
| consistency | average | 7412 | 46.11 | 1.536 | 82.97 | 67.70 |
| proposed | raw | 7412 | 46.12 | 1.537 | 83.36 | 68.31 |
| proposed | average | 7412 | 46.12 | 1.537 | 83.30 | 68.05 |
| standard | raw | 7412 | 10.00 | 0.333 | 89.44 | 0.93 |
| mixed | raw | 7413 | 44.64 | 1.487 | 83.74 | 67.73 |
| mixed | average | 7413 | 44.64 | 1.487 | 83.80 | 67.67 |
| curriculum_mixed | raw | 7413 | 43.95 | 1.464 | 84.08 | 67.53 |
| curriculum_mixed | average | 7413 | 43.95 | 1.464 | 84.15 | 67.58 |
| consistency | raw | 7413 | 45.92 | 1.530 | 82.65 | 68.26 |
| consistency | average | 7413 | 45.92 | 1.530 | 82.58 | 68.25 |
| proposed | raw | 7413 | 45.54 | 1.517 | 83.37 | 68.35 |
| proposed | average | 7413 | 45.54 | 1.517 | 83.45 | 68.18 |
| standard | raw | 7413 | 9.55 | 0.317 | 89.61 | 1.67 |
| pgd | raw | 7413 | 38.74 | 1.291 | 81.89 | 67.04 |
| curriculum_mixed | raw | 7414 | 43.41 | 1.446 | 84.06 | 67.37 |
| curriculum_mixed | average | 7414 | 43.41 | 1.446 | 84.06 | 67.23 |
| consistency | raw | 7414 | 45.17 | 1.505 | 82.58 | 67.47 |
| consistency | average | 7414 | 45.17 | 1.505 | 82.57 | 67.35 |
| proposed | raw | 7414 | 45.70 | 1.523 | 83.31 | 68.20 |
| proposed | average | 7414 | 45.70 | 1.523 | 83.24 | 68.18 |
| standard | raw | 7414 | 9.59 | 0.319 | 89.43 | 0.80 |
| pgd | raw | 7414 | 38.81 | 1.293 | 81.91 | 67.25 |
| mixed | raw | 7414 | 43.24 | 1.441 | 83.58 | 67.33 |
| mixed | average | 7414 | 43.24 | 1.441 | 83.43 | 67.40 |
| consistency | raw | 7415 | 46.09 | 1.535 | 82.55 | 67.66 |
| consistency | average | 7415 | 46.09 | 1.535 | 82.65 | 67.39 |
| proposed | raw | 7415 | 46.05 | 1.534 | 83.22 | 68.11 |
| proposed | average | 7415 | 46.05 | 1.534 | 83.08 | 67.98 |
| standard | raw | 7415 | 9.81 | 0.326 | 89.27 | 1.33 |
| pgd | raw | 7415 | 39.14 | 1.304 | 82.04 | 66.72 |
| mixed | raw | 7415 | 43.63 | 1.453 | 83.43 | 67.17 |
| mixed | average | 7415 | 43.63 | 1.453 | 83.37 | 67.05 |
| curriculum_mixed | raw | 7415 | 43.76 | 1.458 | 84.07 | 67.36 |
| curriculum_mixed | average | 7415 | 43.76 | 1.458 | 83.90 | 67.09 |

## Paired robustness and class failures

The main paired outcomes are in `results/paired_comparisons.json`. There are 10,000 paired image identities per seed, with each classifier attacked separately. Student t intervals describe the variation over five training seeds. Every test class has 1,000 images; class tables and confusion matrices retain the uneven residual robustness.

## Attack stress checks

All three required methods are audited at every confirmation seed. The union combines the official final-iterate PGD-20 attack with CE-PGD-50 and margin-PGD-50, each at epsilon 0.10, alpha 0.005 and three restarts. A failure at any intermediate iterate, or on the clean input, removes an example from the surviving set. The separate radius sweep uses one CE-PGD-50 restart, alpha = epsilon/20 and radii 0.05, 0.10, 0.15 and 0.20. These attacks do not certify robustness.

## Computational record

{
  "development_runs": 22,
  "confirmation_runs": 30,
  "distinct_optimization_runs": 52,
  "total_optimization_epochs": 1560,
  "development_training_seconds": 1152.5791602999998,
  "confirmation_training_seconds": 1173.895109799967,
  "validation_trace_seconds": 12.227657900024496,
  "raw_and_average_are_shared_optimization_trajectories": true
}

A separate fresh repeat of the proposal uses seed 7411 for thirty epochs. Its results are recorded below; its optimization is outside the development/confirmation totals.

```json
{
  "reference_run": "confirmation_proposed_s7411_e30",
  "fresh_run": "reproduction_proposed_s7411_e30",
  "fresh_directory": "D:\\Assignment\\Coursework1_MC647418_revised\\tmp\\independent_reproduction",
  "seed": 7411,
  "epochs": 30,
  "from_scratch": true,
  "used_cached_weights": false,
  "training_seconds": 58.3104332000039,
  "versions": {
    "raw": {
      "parameter_tensors_bitwise_equal": true,
      "tensors": 6,
      "full_test_arrays_bitwise_equal": {
        "label": true,
        "clean_prediction": true,
        "adversarial_prediction": true,
        "attack_loss": true
      },
      "test_metrics": {
        "n": 10000,
        "clean_correct": 8326,
        "robust_correct": 6788,
        "clean_accuracy": 0.8325999975204468,
        "pgd_accuracy": 0.6787999868392944,
        "attack_seed": 9001,
        "steps": 20,
        "alpha": 0.01,
        "epsilon": 0.1,
        "restarts": 1,
        "keep_best": false
      },
      "maximum_attack_loss_difference": 0.0
    },
    "average": {
      "parameter_tensors_bitwise_equal": true,
      "tensors": 6,
      "full_test_arrays_bitwise_equal": {
        "label": true,
        "clean_prediction": true,
        "adversarial_prediction": true,
        "attack_loss": true
      },
      "test_metrics": {
        "n": 10000,
        "clean_correct": 8324,
        "robust_correct": 6792,
        "clean_accuracy": 0.8323999643325806,
        "pgd_accuracy": 0.6791999936103821,
        "attack_seed": 9001,
        "steps": 20,
        "alpha": 0.01,
        "epsilon": 0.1,
        "restarts": 1,
        "keep_best": false
      },
      "maximum_attack_loss_difference": 0.0
    }
  },
  "environment": {
    "python": "3.12.14",
    "torch": "2.11.0+cu128",
    "numpy": "2.5.2",
    "platform": "Windows-11-10.0.26200-SP0",
    "gpu": "NVIDIA GeForce RTX 4060",
    "cuda": "12.8",
    "precision": "float32",
    "tf32": false,
    "deterministic_algorithms": true,
    "cpu_threads": 4
  },
  "test_configuration_changed": false
}
```

## Reproduction scope

The final configuration is fixed before new confirmation-seed test evaluation. Earlier test results had been observed in the preceding completed study; validation determines this revision's coefficients, schedule and final state. The tested method is an empirical result for this fixed model, dataset and epoch budget.

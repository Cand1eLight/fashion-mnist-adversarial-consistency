# Fashion-MNIST adversarial consistency experiment

CISC7404-002 · Coursework 1 · MC647418 · University of Macau

The study uses the required 784–256–128–10 MLP, trained from scratch on Fashion-MNIST. The proposed training loss combines 0.25 clean CE, 0.75 adversarial CE and 0.5 KL(clean || adversarial). Five CE-PGD steps use a radius that increases from 0.01 to 0.10 over ten epochs; alpha is radius/4. The final inference parameters average epoch-end states 21–30. All models receive thirty epochs.

## Recorded results

| Method | Optimization (s) | Clean (%) | PGD-20 (%) |
| --- | --- | --- | --- |
| standard | 10.67 ± 2.10 | 89.45 ± 0.13 | 1.10 ± 0.39 |
| pgd | 41.41 ± 5.48 | 82.03 ± 0.14 | 66.97 ± 0.20 |
| proposed | 45.71 ± 0.41 | 83.26 ± 0.13 | 68.06 ± 0.12 |

Values are mean ± sample SD across seeds 7411–7415. Official evaluation uses every test image, epsilon 0.10, alpha 0.01, twenty CE ascent steps, one random start and the final iterate in evaluation mode. Standard and PGD baselines use their final raw parameters. The proposal uses the averaged parameters fixed by validation selection.

## Package and files

This GitHub-ready source version also retains every official per-image prediction archive and every supplementary audit survival archive. The record checker verifies the saved image counts.

- `report.pdf`: the unchanged six-page report body and one reference page.
- `Coursework1_MC647418.ipynb`: complete implementation, executed in saved-record review mode.
- `src/`: all data, attack, loss, training, evaluation, development, analysis and correctness-check code.
- `data/`: the publisher license, download/checksum manifest and original fixed split indices.
- `runs/`: all 52 run records, histories and validation/test scores.
- `results/`: frozen configuration, all 50 state-level outcomes, summaries, class/paired analysis and audit records.
- `figures/`: all six PNG figures used by the report and notebook.
- `EXPERIMENT_LOG.md`: the complete development and confirmation ledger.
- `requirements.txt`: pinned package versions.

Raw Fashion-MNIST images and trained weights are omitted to keep the submission small. Training code downloads
the original data from the publisher, checks its MD5 hashes and recreates checkpoints. No training method,
evaluation setting or reported numerical result was changed during packaging. The first notebook code cell
sets `FRESH_TRAIN = False`; this mode reviews saved records without downloading images or training models.
The displayed measurements are the recorded experiments, not newly computed performance measurements.
Plots link to the supplied PNG files to avoid embedding duplicate images in the notebook.

## Environment and setup

Recorded experiment: Python 3.12.14, PyTorch 2.11.0+cu128, CUDA 12.8, NVIDIA RTX 4060 (8 GB),
float32, deterministic operations, TF32 disabled and four CPU threads.

From this directory on Windows with Python 3.12:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m notebook Coursework1_MC647418.ipynb
```

CPU execution is supported. GPU training times apply only to the recorded hardware. Internet is needed for
package installation and the first fresh data download. Saved-record review uses the supplied files.

## Review and reproduce

Review stored results and verify the attack settings, run counts, split and summary statistics:

```text
python src/review_records.py
python src/run_suite.py check
```

For fresh training and complete evaluation, set `FRESH_TRAIN = True` in the first notebook code cell and run
all cells, or use the following commands from this directory:

```text
python src/run_suite.py train --force
python src/run_suite.py evaluate --force
python src/run_suite.py audit --force
python src/run_suite.py analyze
```

These commands regenerate the thirty confirmation trajectories, fifty raw/averaged states and all test
prediction archives. The official test uses epsilon 0.10, alpha 0.01, twenty cross-entropy PGD steps,
one random start, the final iterate and evaluation mode. Supplementary audits retain failure at any iterate.

The complete validation development search is separately reproducible:

```text
python src/experiment.py development
python src/refine.py
python src/budget_pilot.py
```

The supplied frozen protocol is reused for the main reproduction. Missing development weights are retrained
by these commands. The original configuration is selected on the 5,000-image validation set. An earlier
completed study had already evaluated the same test set; new confirmation seeds and the configuration
were frozen before their test evaluation, as explained in the report.

## Cost and independent reproduction

The recorded study has 22 development runs and 30 confirmation runs, totaling 1,560 optimization epochs.
Their optimization costs are 19.21 and 19.56 minutes. Timing includes attacks, forwards/backwards, updates,
averaging and epoch logs; preparation, transfer, warmup, serialization and diagnostic validation are separate.

`results/reproduction_check.json` retains the original fresh seed-7411 repeat, with exact parameter and
full-test prediction agreement on the recorded stack. After regenerating the confirmation checkpoints and
test archives above, repeat that check in a new directory:

```text
python src/check_reproduction.py tmp/new_reproduction
```

The data source is the [publisher's Fashion-MNIST repository](https://github.com/zalandoresearch/fashion-mnist).

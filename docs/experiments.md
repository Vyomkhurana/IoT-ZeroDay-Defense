# Experiments

All experiments are configured in YAML (`configs/`) and launched with
`scripts/train.py` (one run) or `scripts/run_experiments.py` (suites).

## Experiment modes (objective O5)

| # | Config | Mode | FL strategy | DP-SGD | SecAgg | Purpose |
|---|---|---|---|---|---|---|
| 1 | `centralized.yaml` | centralized | – | – | – | Reference upper bound + IF / OC-SVM baselines |
| – | `local.yaml` | local-only | – | – | – | Value of collaboration |
| 2 | `fedavg.yaml` | federated | FedAvg | – | – | Plain FL |
| – | `fedprox.yaml` | federated | FedProx | – | – | Non-IID mitigation |
| 3 | `dp.yaml` | federated | FedAvg | ✓ | – | Cost of DP |
| 4 | `secure_aggregation.yaml` | federated | FedAvg | – | ✓ | Cost of SecAgg |
| 5 | `full_ppfl.yaml` | federated | FedAvg | ✓ | ✓ | **Proposed system** |
| 5b | `full_ppfl_fedprox.yaml` | federated | FedProx | ✓ | ✓ | Proposed system, FedProx variant |

All modes share the data partition, model initialisation, optimiser and threshold
rule for a given seed, so differences come only from the mode.

## Sweeps (objective O6) — `configs/experiments.yaml`

| Suite | Base | Varied | Values | Question |
|---|---|---|---|---|
| `privacy` | full PPFL | `privacy.noise_multiplier` | 0.4 … 2.5 | F1 / unseen recall vs ε |
| `scalability` | full PPFL | `partition.num_clients` | 3, 5, 10, 20 | Communication, compute and SecAgg overhead vs clients |
| `heterogeneity` | FedAvg | `partition.dirichlet_alpha` | 0.1, 0.3, 1, 10 | Effect of non-IID data |
| `local_epochs` | full PPFL | `federated.local_epochs` | 1, 2, 5 | Computation vs communication trade-off |

```bash
python scripts/run_experiments.py --list                  # show the plan
python scripts/run_experiments.py                         # main suite
python scripts/run_experiments.py --suite all             # everything
python scripts/run_experiments.py --suite privacy --skip-existing
python scripts/generate_report.py                         # results/summary.csv, report.md, figures/
```

Add a suite by editing `configs/experiments.yaml`: either a list of config files or
`{config, parameter, values, name}`. Every experiment can also be run on its own with
the same override:

```bash
python scripts/train.py --config configs/full_ppfl.yaml --set privacy.noise_multiplier=2.5 --name ppfl_sigma_2.5
```

## Recommended protocol for reported results

1. Prepare N-BaIoT: `python scripts/prepare_data.py --dataset nbaIoT`.
2. Inspect the partition: `python scripts/create_clients.py --config configs/fedavg.yaml`.
3. Run `--suite all` with at least three seeds, e.g. with `--set experiment.seed=1`
   and `--set experiment.output_dir=results/seed1/experiments`. Report mean ± std.
4. Report unseen-attack recall **together with** benign FPR. Recall can always be
   raised by lowering the threshold.
5. State ε together with δ, σ, C, *q*, the number of steps and the accountant, and say
   that the guarantee is example-level and per client.
6. Note that communication and time measurements come from your machine and from a
   single-process simulation.

**Runtime note.** Runs with DP-SGD are several times slower than non-private runs,
because Opacus computes a gradient per sample. Secure aggregation adds Diffie–Hellman
and Shamir work that grows quadratically with the number of clients per round. Use
`--skip-existing` to resume an interrupted suite.

## Pipeline check without data

```bash
python scripts/prepare_data.py --synthetic
python scripts/train.py --config configs/smoke_test.yaml           # ~20 s on a laptop CPU
python scripts/run_experiments.py --suite all --overlay configs/synthetic.yaml
```

The synthetic overlay writes to `results/synthetic/` so synthetic runs never mix with
real results. Every synthetic figure is watermarked. **Synthetic numbers are not
research results.**

## Outputs

```
results/
├── experiments/<name>/      config.yaml, environment.json, metrics.json, metrics.csv,
│                            per_client.csv, client_distribution.csv, privacy_ledger.csv,
│                            client_rounds.csv, scores_<model>.npz, model/, figures/, logs/
├── experiment_index.json    suite → experiment names (used to group figures)
├── summary.csv              one row per (experiment, model)
├── report.md                tables generated from the measured metrics
└── figures/                 cross-experiment comparison figures
```

### Figures

Per experiment (`results/experiments/<name>/figures/`): training loss, validation
loss, F1 / unseen recall / FPR vs round, confusion matrix (by traffic group and
binary), reconstruction-error distribution, anomaly scores by class, ROC curves,
ε vs round, communication per round, client data distribution, per-client performance.

Cross-experiment (`results/figures/`): zero-day recall comparison, FPR comparison,
F1 comparison, mode comparison (centralized vs FL vs DP-FL vs full PPFL),
communication-cost comparison, F1 vs ε, zero-day recall vs noise level, F1 vs
communication, convergence overlays (F1, validation loss, ε), and sweep plots
(clients, α, local epochs).

# IoT Zero-Day Defense (PPFL-IoT-ZeroDay)

**Privacy-preserving federated learning framework for zero-day threat detection in IoT networks using anomaly detection, differential privacy, and secure aggregation.**

> *Enable IoT gateways to collaboratively learn to recognize zero-day attacks — with a provable bound on what any single device's data reveals.*

![python](https://img.shields.io/badge/python-3.10%2B-blue) ![pytorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c) ![opacus](https://img.shields.io/badge/DP--SGD-Opacus-6f42c1) ![license](https://img.shields.io/badge/license-MIT-green)

---

## Overview

This repository is a research prototype. Simulated IoT gateways jointly train an **autoencoder** that models *normal* network traffic, and use its reconstruction error to flag attacks, including attack classes **never seen during training**. Training is federated (raw traffic never leaves a gateway). Each gateway uses **DP-SGD** so that every released update carries an (ε, δ) differential-privacy guarantee. Updates are combined with **pairwise-masking secure aggregation**, so the server only learns their sum.

Everything is configurable from YAML, reproducible from a seed, and measured: detection quality (with **unseen-attack recall** as the headline metric), privacy budget ε, communication bytes and compute time.

## Quick Start

```powershell
git clone https://github.com/Vyomkhurana/IoT-ZeroDay-Defense.git
cd IoT-ZeroDay-Defense
python -m venv .venv
.venv\Scripts\activate                      # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_data.py --synthetic  # small test dataset, no download (~10 s)
streamlit run app.py                        # opens the dashboard at http://localhost:8501
```

In the dashboard, open **Train live**, press **Start training**, then open **Detect attacks** and press **Stream traffic**. For real results, download N-BaIoT first ([step 3](#3-get-the-data)). The full instructions are in the [Operating Guide](#operating-guide).

## Problem

* **Privacy.** Centralized intrusion detection needs raw network telemetry from every site. This telemetry reveals devices, usage patterns and behaviour.
* **Scalability.** IoT deployments are distributed across many resource-constrained gateways.
* **Novel attacks.** Signature-based IDSs mainly recognise attacks they already know. New botnets and variants of known ones can evade them.

## Motivation

Anomaly detection trained on normal traffic can, in principle, flag unknown attacks. Training it well needs traffic from many sites, and pooling that traffic is exactly what privacy rules out. Federated learning removes raw-data collection but still exposes model updates, which can leak training data. Differential privacy and secure aggregation each close part of that gap. This project measures what the combination costs in detection quality, communication and computation.

## Research Question

*Can IoT gateways collaboratively train an anomaly detector that detects previously unseen attack classes while (i) keeping raw data local, (ii) hiding individual updates from the server, and (iii) bounding each record's influence with differential privacy? How much detection quality, communication and computation does each protection cost?*

## Objectives

| | Objective | Where |
|---|---|---|
| O1 | FedAvg / FedProx pipeline over simulated non-IID IoT clients | `src/ppfl/federated/` |
| O2 | Unsupervised detection with an autoencoder; Isolation Forest and One-Class SVM baselines | `src/ppfl/models/`, `src/ppfl/detection/` |
| O3 | DP-SGD (clipping, Gaussian noise, RDP accounting, reported ε) | `src/ppfl/privacy/` |
| O4 | Secure aggregation: the server only obtains the aggregate update | `src/ppfl/security/` |
| O5 | Benchmark: centralized vs FL vs FL+DP vs FL+SecAgg vs full PPFL | `configs/`, `scripts/run_experiments.py` |
| O6 | Feasibility: rounds, update size, client compute, #clients, local epochs, DP noise, SecAgg overhead | `configs/experiments.yaml` |

## System Architecture

```
                    ┌────────────────────────────────┐
                    │        Global FL Server        │
                    │  FedAvg / FedProx aggregation  │
                    │  federated threshold (histos)  │
                    └───────────────┬────────────────┘
                                    │  sees only Σ updates
                 Secure Aggregation │  (DH keys · Shamir shares · PRG masks mod 2^64)
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │ masked Δ₁                 │ masked Δ₂                 │ masked Δ_N
        ▼                           ▼                           ▼
 ┌──────────────┐           ┌──────────────┐           ┌──────────────┐
 │  IoT Client  │           │  IoT Client  │           │  IoT Client  │
 │      1       │           │      2       │    ...    │      N       │
 ├──────────────┤           ├──────────────┤           ├──────────────┤
 │ Local data   │           │ Local data   │           │ Local data   │
 │ Autoencoder  │           │ Autoencoder  │           │ Autoencoder  │
 │ DP-SGD       │           │ DP-SGD       │           │ DP-SGD       │
 │ (clip+noise) │           │ (clip+noise) │           │ (clip+noise) │
 └──────────────┘           └──────────────┘           └──────────────┘
        │                           │                           │
        └──────────── raw traffic NEVER leaves the client ──────┘
```

Per round: the server broadcasts *w_t*. Each client trains locally on benign traffic with DP-SGD (plus the FedProx term if enabled) and encodes `[n·Δ, n, n·loss]` in fixed point. It masks this vector and sends it. The server unmasks only the **sum** (recovering from dropped clients through Shamir shares) and applies the FedAvg update. After training, clients send histograms of their validation anomaly scores, aggregated the same way, and the server sets the detection threshold from them. Details: [docs/architecture.md](docs/architecture.md).

## Methodology

1. Load and clean IoT traffic. Separate benign, *known* attack and *held-out* attack classes.
2. Partition the data across *N* gateways with a Dirichlet non-IID scheme. Split each gateway's data locally into train/val/test, and fit the scaler from aggregated sufficient statistics.
3. Train the autoencoder on benign traffic: centralized, local-only, or federated (FedAvg/FedProx, ± DP-SGD, ± SecAgg).
4. Set the anomaly threshold on validation data only.
5. Evaluate on test traffic containing benign, known and **unseen** attacks.

Full description: [docs/methodology.md](docs/methodology.md).

## Zero-Day Simulation

Zero-day attacks are simulated by **holding out entire attack classes**. `data.holdout_classes` (fnmatch patterns, e.g. `["mirai_*"]`) removes those classes from every client's training and validation data. They appear only in test data. Neither the model nor the threshold selection ever sees them.

```
TRAINING / VALIDATION                TEST ONLY
  benign                               benign
  known attacks (Gafgyt: combo,        known attacks
    junk, scan, tcp, udp)              UNSEEN attacks (Mirai: ack, scan, syn, udp, udpplain)
```

**Unseen-attack recall** = held-out attack samples flagged anomalous / all held-out attack samples. It is the headline security metric and is always reported next to the benign false-positive rate.

> These are *zero-day-like* (unseen-class) attacks in a controlled experimental setting. They do not reproduce a real-world zero-day vulnerability.

## Federated Learning

* **FedAvg** (McMahan et al., 2017): sample-weighted averaging of client models; the primary baseline.
* **FedProx** (Li et al., 2020): adds `μ/2‖w − w_t‖²` to each local objective to limit client drift on heterogeneous data (`federated.mu`).
* Client sampling (`fraction_fit`), server learning rate, local epochs and rounds are configurable.
* `IoTClient` loads local data, trains, computes local metrics, produces its update, applies DP, takes part in secure aggregation and scores anomalies. `FederatedServer` initialises, selects, aggregates, updates, distributes and records.
* An optional **Flower** adapter (`src/ppfl/federated/flower_adapter.py`, `scripts/run_flower.py`) runs the same clients under Flower's FedAvg/FedProx over localhost gRPC. The native engine is used for the experiments because secure aggregation needs exact control over what the server receives.

## Differential Privacy

DP-SGD (Abadi et al., 2016) on every client, built on **Opacus**: Poisson sampling, per-sample gradients, clipping to `max_grad_norm` *C*, Gaussian noise with std σ·C, and an **RDP accountant** (PRV/GDP also available) that persists across rounds.

```yaml
privacy:
  enabled: true
  noise_multiplier: 1.0     # σ
  max_grad_norm: 1.0        # C
  delta: 1.0e-5
  target_epsilon: null      # or e.g. 3.0 -> σ calibrated per client
```

* ε is tracked **after every round**. The reported ε is the **maximum over clients**, an example-level guarantee for each client's local dataset.
* With FedProx, the data-independent proximal gradient is added *after* privatization (post-processing), so it is not clipped and costs no privacy.
* Validation-score histograms used for threshold calibration can be made DP as well (`detection.histogram_dp_epsilon`, Laplace mechanism).

## Secure Aggregation

A simulation of the protocol structure of Bonawitz et al. (CCS 2017):

* Diffie–Hellman key agreement (RFC 3526 2048-bit group). Each pair of clients derives a shared seed, which is expanded by SHAKE-256 into a mask. Client *i* adds `+M_ij` and client *j* adds `−M_ij`, so the masks cancel in the sum.
* Updates are encoded in fixed point in ℤ_{2⁶⁴}, so masking and cancellation are exact (no floating-point error from the masks).
* **Dropout recovery and double masking.** Secret keys and self-mask seeds are Shamir-shared (threshold *t*). Shares are encrypted between clients and routed through the server. Dropouts can be simulated with `secure_aggregation.dropout_rate`.
* The module is independent of FL. It aggregates arbitrary vectors and flattened state dicts, and has a seeded deterministic mode for tests.
* Tests check that `aggregate(masked(U1), masked(U2), masked(U3)) == U1 + U2 + U3`, that recovery works with dropouts, and that a federated run with SecAgg gives the same model as one without (up to 2⁻²⁴ quantisation).

> ⚠️ **Research prototype, not production cryptography.** It assumes an honest-but-curious server and has no PKI or signatures (no protection against a malicious server). There is no constant-time arithmetic, and all parties run in one process.

## Dataset

| Dataset | Status | Notes |
|---|---|---|
| **N-BaIoT** (Meidan et al., 2018) | primary | 115 features, 9 devices, Mirai + BASHLITE. Flat and per-device layouts supported. |
| CICIoT2023 | supported | generic loader, label column `label` |
| TON_IoT (network) | supported | label column `type` |
| Bot-IoT | supported | label column `category` |
| Synthetic | **testing only** | N-BaIoT-like generator, so the pipeline runs without a download |

No data is committed. Download instructions and expected layouts: [data/README.md](data/README.md).

```bash
python scripts/prepare_data.py --dataset nbaIoT          # -> data/processed/nbaiot/cleaned.parquet
python scripts/prepare_data.py --synthetic               # test data
```

Preprocessing: numeric coercion, ±inf → missing, removal of constant or mostly-missing columns, per-device/per-class caps, a stratified local train/val/test split, and a standard (or min-max) scaler fitted **only on training rows**, from federated sufficient statistics. Missing values are imputed with the training mean.

## Client Partitioning

```yaml
partition:
  num_clients: 10
  strategy: dirichlet          # dirichlet | device | iid
  dirichlet_alpha: 0.5         # lower alpha = more heterogeneous clients
  stratify_by: [device, label]
```

For every *(device, class)* stratum, client shares are drawn from `Dir(α·1)`. With small α each gateway sees a few devices and a few attack types in very different volumes ("mostly normal + attack A", "normal + attacks A and C", …). With large α the split approaches IID. Every client keeps at least `min_benign_per_client` benign rows. Heterogeneity is reported as the mean Jensen–Shannon distance to the global class mix. Inspect a partition with:

```bash
python scripts/create_clients.py --config configs/fedavg.yaml --set partition.dirichlet_alpha=0.1
```

## Model

```
x (d) → Linear(d,64) → ReLU → Linear(64,32) → ReLU → Linear(32,16)=z
z     → Linear(16,32) → ReLU → Linear(32,64) → ReLU → Linear(64,d) = x̂
```

`Autoencoder` (`src/ppfl/models/autoencoder.py`) exposes `forward()`, `encode()` and `decode()`. The input dimension is inferred from the data, and `hidden_dims`, `latent_dim`, `activation` and `dropout` are configurable. Loss is the MSE between *x* and *x̂*. There is no BatchNorm, which keeps the model DP-SGD compatible.

**Anomaly detection:** score = MSE(x, x̂); anomalous if score > τ. τ comes from validation data only: `percentile` (default, 95th percentile of benign validation errors), `mean_std`, or `validation_f1` (F1-optimal against *known* attacks). Federated runs compute τ from (securely) aggregated score histograms.

## Experimental Setup

| Mode | Config | DP | SecAgg |
|---|---|---|---|
| 1 Centralized (+ Isolation Forest, One-Class SVM) | `configs/centralized.yaml` | – | – |
| Local-only | `configs/local.yaml` | – | – |
| 2 Plain FL (FedAvg) | `configs/fedavg.yaml` | – | – |
| FedProx | `configs/fedprox.yaml` | – | – |
| 3 FL + DP | `configs/dp.yaml` | ✓ | – |
| 4 FL + SecAgg | `configs/secure_aggregation.yaml` | – | ✓ |
| **5 Full PPFL** | `configs/full_ppfl.yaml` (FedAvg), `configs/full_ppfl_fedprox.yaml` | ✓ | ✓ |

Defaults (`configs/default.yaml`): 10 clients, Dirichlet α = 0.5, 20 rounds × 2 local epochs, batch 128, Adam (lr 1e-3), AE 64-32-16, σ = 1.0, C = 1.0, δ = 1e-5, SecAgg threshold 0.6·n, 95th-percentile threshold, seed 42. Sweeps (O6) are defined in `configs/experiments.yaml`: noise multiplier, number of clients, Dirichlet α, local epochs. See [docs/experiments.md](docs/experiments.md).

**What is federated.** The autoencoder (all FL modes), the feature scaler (sufficient statistics) and the anomaly threshold (histograms). **What is not.** Isolation Forest and One-Class SVM are centralized reference baselines: tree ensembles and kernel SVMs have no meaningful parameter averaging.

## Evaluation Metrics

* **Classification:** accuracy, precision, recall, F1, false-positive rate, ROC-AUC, PR-AUC.
* **Security:** **unseen-attack recall** (headline), known-attack recall, benign FPR, per-class detection rates, anomaly-score distributions per traffic group.
* **Federated:** loss, validation loss and F1 per round, convergence, per-client performance, rounds, update size, total communication, SecAgg overhead, training time, client compute time.
* **Privacy:** ε (max and mean over clients) per round, δ, σ, C, sampling rate.
* **Trade-offs:** F1 vs ε, F1 vs communication, zero-day recall vs noise level.

## Results

Measured on **N-BaIoT** (492,641 records from 9 real IoT devices, 115 features). 10 gateways, Dirichlet α = 0.5, 20 rounds × 2 local epochs, seed 42. The whole **Mirai** botnet family (5 attack types) is held out as the zero-day attacks, and the threshold is the 95th percentile of normal validation traffic, so about 5% false alarms.

| Method | Zero-day caught | Known attacks caught | F1 | ROC-AUC | False alarms | Privacy ε |
|---|---|---|---|---|---|---|
| Plain federated (FedAvg) | **99.8%** | 99.7% | 0.993 | 0.989 | 5.2% | – |
| Full privacy, default settings (batch 128, lr 1e-3, σ = 1.0) | 67.2% | 26.6% | 0.723 | 0.937 | 5.2% | 7.5 |
| Full privacy, tuned (batch 512, lr 1e-2, ε = 8) | **85.2%** | 46.9% | 0.857 | 0.965 | 5.2% | 8.0 |

**Tuning DP-SGD at a fixed budget (ε = 8, σ calibrated automatically).** Larger batches and a higher learning rate recover most of the accuracy lost to the privacy noise:

| Batch size | Learning rate | Zero-day caught | F1 | ROC-AUC |
|---|---|---|---|---|
| 128 | 1e-3 | 70.1% | 0.756 | 0.944 |
| 512 | 3e-3 | 77.9% | 0.819 | 0.961 |
| 1024 | 5e-3 | 80.1% | 0.831 | 0.962 |
| 512 | 1e-2 | **85.2%** | **0.857** | **0.965** |

Reproduce one row:

```bash
python scripts/train.py --config configs/full_ppfl.yaml --set training.batch_size=512 --set training.learning_rate=0.01 --set privacy.target_epsilon=8 --name ppfl_tuned
```

The full suite (`python scripts/run_experiments.py --suite all`) adds centralized and local-only references, FedProx, DP-only and SecAgg-only modes, and the noise, gateway-count, heterogeneity and local-epoch sweeps. It writes:

* `results/summary.csv`: one row per (experiment, model) with every metric above.
* `results/report.md`: Markdown tables built from the measured `metrics.json` files.
* `results/figures/`: zero-day recall comparison, FPR comparison, centralized vs FL vs DP-FL vs full PPFL, communication cost, F1 vs ε, zero-day recall vs privacy level, convergence overlays and sweep plots.
* `results/experiments/<name>/figures/`: loss and F1 per round, confusion matrices, reconstruction-error distributions, scores by class, ROC curves, ε per round, communication per round, client data distribution and per-client performance.

Runs on the synthetic test data go to `results/synthetic/` and are watermarked. They only show that the pipeline works and are **not** research results.

## Operating Guide

### 1. Requirements

* Python **3.10 or newer** (tested with 3.13), Windows, macOS or Linux.
* About 2 GB of RAM for the synthetic data, about 6 GB for N-BaIoT. A GPU is optional: the models are small and everything runs on a laptop CPU.
* About 10 GB of free disk space while unpacking N-BaIoT. Only the 150 MB processed file is kept afterwards.

### 2. Install

**Windows (PowerShell):**

```powershell
git clone https://github.com/Vyomkhurana/IoT-ZeroDay-Defense.git
cd IoT-ZeroDay-Defense
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
git clone https://github.com/Vyomkhurana/IoT-ZeroDay-Defense.git
cd IoT-ZeroDay-Defense
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`pip install -e ".[dev]"` also works and installs the `ppfl-*` console commands. Optional extras:

* **GPU:** install a CUDA build of PyTorch (see pytorch.org). `experiment.device: auto` then uses it; force CPU with `--set experiment.device=cpu`.
* **Flower:** `pip install "flwr>=1.10"` for `scripts/run_flower.py`.

Check the installation:

```bash
python -m pytest            # about 1 minute; 1 test is skipped when Flower is not installed
```

### 3. Get the data

| Dataset | Command | Notes |
|---|---|---|
| Synthetic | `python scripts/prepare_data.py --synthetic` | Generated in seconds. For testing the pipeline only. |
| **N-BaIoT** (real) | see below | 1.8 GB download from UCI; used for all reported results. |

N-BaIoT, using 7-Zip on Windows (`7z` must be on the PATH) or `unrar` on macOS/Linux:

```bash
curl -L -o nbaiot.zip "https://archive.ics.uci.edu/static/public/442/detection+of+iot+botnet+attacks+n+baiot.zip"
7z x nbaiot.zip -oraw
for r in raw/*/*.rar; do 7z x "$r" -o"${r%.rar}"; done     # Git Bash / macOS / Linux shell
python scripts/prepare_data.py --dataset nbaiot --raw-dir raw
```

This writes `data/processed/nbaiot/cleaned.parquet`. You can then delete `nbaiot.zip` and `raw/`. More detail and the other supported datasets (CICIoT2023, TON_IoT, Bot-IoT) are in [data/README.md](data/README.md).

### 4. Run the dashboard

```bash
streamlit run app.py
```

Your browser opens at **http://localhost:8501** (use `--server.port 8502` if that port is taken). Stop the dashboard with **Ctrl + C** in the terminal. The dashboard runs the real `ppfl` code: training happens when you press the button, and detection scores records with the model you just trained.

| Page | What it does |
|---|---|
| **Overview** | The problem, the approach, a diagram of one training round and the headline numbers from the finished experiment suite. |
| **Train live** | Choose the dataset, privacy mode (full privacy, FL + secure aggregation, FL + DP, plain FedAvg, centralized), number of gateways and rounds, plus noise level σ, data heterogeneity α and seed under *Advanced settings*. **Start training** shows each gateway's data, then F1, zero-day recall, false alarms and ε after every round. The result view shows score distributions with the alert threshold, the detection rate per attack type and per-gateway results. |
| **Detect attacks** | Streams held-out test records through a trained model in real time. Each record is reconstructed and scored on the spot, and the page counts alerts, zero-day attacks caught and false alarms. **Why was it flagged?** shows which traffic features the model could not reconstruct. |
| **Compare results** | All methods side by side (zero-day recall, F1, ROC-AUC, false alarms, ε, data sent), the privacy/accuracy trade-off across noise levels, and the gateway, heterogeneity and local-epoch sweeps. |

**Suggested 5-minute demo**

1. **Overview:** explain the problem and the diagram.
2. **Train live:** dataset *N-BaIoT*, mode *Plain federated*, 10 rounds, then **Start training**. Zero-day recall climbs to about 99% in under a minute.
3. Same page, mode *Full privacy*, then **Start training**. Detection is lower and ε rises every round. This is the cost of privacy.
4. **Detect attacks:** **Stream traffic**, then pick an alert under **Why was it flagged?**
5. **Compare results:** every method side by side (after the experiment suite has been run).

Dashboard runs are written to `results/ui/` (git-ignored).

### 5. Command line

| Task | Command |
|---|---|
| Quick end-to-end check (synthetic, about 20 s) | `python scripts/train.py --config configs/smoke_test.yaml` |
| Train one mode | `python scripts/train.py --config configs/<mode>.yaml` |
| Train on synthetic data instead | add `--overlay configs/synthetic.yaml` |
| Change any setting | add `--set key=value` (repeatable), e.g. `--set federated.rounds=40` |
| Name the run | add `--name my_run` |
| Re-evaluate a finished run | `python scripts/evaluate.py --experiment full_ppfl [--threshold-strategy validation_f1]` |
| Inspect the client partition | `python scripts/create_clients.py --config configs/fedavg.yaml [--export]` |
| Run the main comparison | `python scripts/run_experiments.py` |
| Run everything (main + all sweeps) | `python scripts/run_experiments.py --suite all` |
| List what a suite would run | `python scripts/run_experiments.py --suite all --list` |
| Rebuild report and figures | `python scripts/generate_report.py` |
| Flower adapter (optional) | `python scripts/run_flower.py --config configs/fedavg.yaml` |

Modes (`configs/`): `centralized`, `local`, `fedavg`, `fedprox`, `dp`, `secure_aggregation`, `full_ppfl`, `full_ppfl_fedprox`. All defaults live in `configs/default.yaml`. Settings you will change most often:

| Setting | Meaning | Default |
|---|---|---|
| `federated.rounds` | training rounds | 20 |
| `federated.local_epochs` | passes over local data per round | 2 |
| `partition.num_clients` | number of gateways | 10 |
| `partition.dirichlet_alpha` | data heterogeneity (lower = more different gateways) | 0.5 |
| `privacy.noise_multiplier` | DP noise σ (higher = more private) | 1.0 |
| `privacy.target_epsilon` | set a privacy budget instead of σ | null |
| `training.batch_size` / `training.learning_rate` | optimiser settings | 128 / 1e-3 |
| `data.holdout_classes` | attack classes treated as zero-day | `["mirai_*"]` |
| `secure_aggregation.dropout_rate` | simulated gateway dropouts | 0.0 |

Each run writes to `results/experiments/<name>/`: `metrics.json` (all final metrics), `metrics.csv` (per round), `config.yaml`, `environment.json`, the trained model in `model/`, plots in `figures/` and logs in `logs/`. `ppfl-prepare`, `ppfl-train`, `ppfl-evaluate`, `ppfl-run-experiments`, `ppfl-report` and `ppfl-clients` are the same commands after `pip install -e .`.

### 6. Run the tests

```bash
python -m pytest                         # whole suite
python -m pytest tests/test_app.py       # dashboard only
python -m pytest -k secure_aggregation   # one area
```

The suite covers data loading and partitioning, the model, FedAvg/FedProx, DP-SGD clipping, noise and accounting, secure aggregation (exact sums and dropout recovery), the zero-day split, end-to-end runs of every mode, reproducibility and the dashboard.

### 7. Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit` / `python` is not recognized | Activate the virtual environment first (`.venv\Scripts\activate`), or run `python -m streamlit run app.py`. |
| `File does not exist: app.py` | Run the command from the `IoT-ZeroDay-Defense` folder (`cd IoT-ZeroDay-Defense`). |
| Streamlit asks for an email on first start | Press Enter to skip. |
| `Port 8501 is already in use` | `streamlit run app.py --server.port 8502` |
| `No N-BaIoT CSV files found` / dashboard shows no dataset | Prepare the data first (step 3). |
| `OMP: Error #15` (Windows + Anaconda) | Use a clean virtual environment (step 2). The package imports scikit-learn first to avoid the clash. |
| Full-privacy training is slow | DP-SGD computes a gradient per record. Use fewer rounds, or `--set training.batch_size=512` (also more accurate, see Results). |

## Project Structure

```
IoT-ZeroDay-Defense/
├── README.md  LICENSE  .gitignore  requirements.txt  pyproject.toml  train.py
├── app.py              Streamlit dashboard (streamlit run app.py)
├── configs/            default, centralized, local, fedavg, fedprox, dp, secure_aggregation,
│                       full_ppfl, full_ppfl_fedprox, smoke_test, synthetic (overlay), experiments
├── data/               raw/ processed/ (git-ignored) + README.md (download instructions)
├── src/ppfl/
│   ├── data/           loader.py preprocessing.py partition.py zero_day.py synthetic.py federated_dataset.py
│   ├── models/         autoencoder.py isolation_forest.py one_class_svm.py base.py
│   ├── training/       trainer.py                (shared plain / FedProx / DP-SGD loop)
│   ├── federated/      client.py server.py fedavg.py fedprox.py flower_adapter.py
│   ├── privacy/        differential_privacy.py accountant.py
│   ├── security/       secure_aggregation.py
│   ├── detection/      anomaly_score.py threshold.py
│   ├── evaluation/     metrics.py zero_day_metrics.py evaluator.py comparison.py
│   ├── visualization/  plots.py
│   ├── experiments/    centralized.py federated.py local.py runner.py tracking.py common.py
│   ├── cli/            prepare_data create_clients train evaluate run_experiments generate_report
│   └── utils/          config.py logging.py seed.py serialization.py environment.py
├── scripts/            prepare_data.py create_clients.py train.py evaluate.py
│                       run_experiments.py generate_report.py run_flower.py
├── tests/              data, model, federated, dp, secure_aggregation, zero_day,
│                       config_and_pipeline, flower_adapter, app
├── notebooks/          01_data_exploration 02_client_distribution 03_results_analysis
├── docs/               architecture.md methodology.md experiments.md
└── results/            experiments/ figures/ (generated)
```

The code lives in an installable `src/ppfl` package (src layout) instead of a bare `src` package. This avoids import-path problems, and `scripts/` are thin wrappers around `ppfl.cli`.

## Reproducibility

* One `experiment.seed` drives everything through **derived per-component seeds**: partitioning, splits, model initialisation, shuffling, DP noise and sampling, client selection, dropout simulation and (optionally) SecAgg keys. Python, NumPy and PyTorch global generators are seeded and deterministic algorithms are enabled.
* Each run stores the resolved `config.yaml`, a config fingerprint, `environment.json` (Python, package versions, OS, CPU/GPU, git commit) and structured logs (`logs/train.log`, `logs/events.jsonl`).
* Data partitions are a deterministic function of config and seed. They are rebuilt rather than stored, and `create_clients.py --export` writes them out if needed.
* A test checks that two runs with the same seed give identical metrics.
* `privacy.secure_mode: true` (cryptographically secure DP noise) and the default OS-random SecAgg keys trade bit-level reproducibility of noise and keys for security. SecAgg keys do not affect results, because the protocol output is the exact sum.

## Security Considerations

* **FL alone is not private.** Model updates can leak information about training data. This project never claims otherwise.
* **SecAgg alone is not DP.** It hides individual updates from the server, but the aggregate and the final model can still leak. DP-SGD provides the formal bound.
* **DP alone does not hide updates.** Without SecAgg the server sees each (noised) update. The guarantee still holds, but SecAgg adds defence in depth and keeps sample counts and losses private.
* The DP guarantee is **example-level, per client** (one traffic record). It is not user-level or device-level DP across clients. ε is reported as the worst case over clients. It covers model training. Threshold histograms are covered only if `histogram_dp_epsilon` is set.
* Secure aggregation assumes an **honest-but-curious** server, and the implementation is a simulation, not hardened cryptography.
* No secrets or API keys are used or stored. Data is git-ignored.

## Limitations

1. Zero-day attacks are **simulated by holding out attack classes**. They are not real zero-day exploits, and held-out classes may still resemble known ones.
2. IoT clients are **simulated** in one process, not physical devices or gateways.
3. Secure aggregation is a **research prototype**: honest-but-curious server, no authentication or PKI, not constant-time.
4. **Dataset quality** bounds detection performance. N-BaIoT is a lab capture of nine devices, and its traffic statistics may not represent production networks.
5. **DP costs utility.** Small local datasets yield large ε for a given noise level.
6. **Non-IID data** slows federated convergence. The local optimiser state is reset every round.
7. Autoencoder detection **depends on threshold selection**. Recall and FPR move together and must be read together.
8. Communication and computation numbers are **measured in the experimental environment** (serialized message sizes, single-machine wall-clock time). They are not on-device measurements.
9. Classical baselines are centralized only.

## Future Work

* User/device-level DP (DP-FedAvg with client-level clipping) and distributed DP noise under SecAgg.
* Malicious-server secure aggregation (signatures, consistency checks) and SecAgg+ for large client counts.
* Byzantine-robust aggregation against poisoned updates.
* Personalised FL (per-gateway fine-tuning or thresholds) for strongly non-IID deployments.
* Sequence and temporal models (LSTM/TCN autoencoders) and online/continual learning.
* Cross-dataset generalisation (train on N-BaIoT, test on CICIoT2023) and evaluation on real gateway hardware.

## References

Conceptual references from the project proposal:

1. H. B. McMahan, E. Moore, D. Ramage, S. Hampson, B. Agüera y Arcas. *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS 2017. (FedAvg)
2. M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, L. Zhang. *Deep Learning with Differential Privacy.* ACM CCS 2016. (DP-SGD)
3. K. Bonawitz, V. Ivanov, B. Kreuter, A. Marcedone, H. B. McMahan, S. Patel, D. Ramage, A. Segal, K. Seth. *Practical Secure Aggregation for Privacy-Preserving Machine Learning.* ACM CCS 2017. (Secure aggregation)
4. Y. Meidan, M. Bohadana, Y. Mathov, Y. Mirsky, A. Shabtai, D. Breitenbacher, Y. Elovici. *N-BaIoT — Network-Based Detection of IoT Botnet Attacks Using Deep Autoencoders.* IEEE Pervasive Computing 17(3), 2018.
5. E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, A. A. Ghorbani. *CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment.* Sensors 23(13), 2023.

Methods and tools used in the implementation:

* T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, V. Smith. *Federated Optimization in Heterogeneous Networks.* MLSys 2020. (FedProx)
* I. Mironov. *Rényi Differential Privacy.* IEEE CSF 2017. (RDP accountant)
* A. Yousefpour et al. *Opacus: User-Friendly Differential Privacy Library in PyTorch.* arXiv:2109.12298, 2021.
* A. Shamir. *How to Share a Secret.* Communications of the ACM 22(11), 1979.
* T. Kivinen, M. Kojo. *More Modular Exponential (MODP) Diffie-Hellman groups for IKE.* RFC 3526, 2003.
* T.-M. H. Hsu, H. Qi, M. Brown. *Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification.* arXiv:1909.06335, 2019. (Dirichlet partitioning)
* D. J. Beutel et al. *Flower: A Friendly Federated Learning Research Framework.* arXiv:2007.14390, 2020.

The full literature review will be expanded separately.

## License

MIT — see [LICENSE](LICENSE).

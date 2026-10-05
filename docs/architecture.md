# Architecture

## Package layout

```
src/ppfl/
├── utils/          config (dataclass schema + YAML inheritance), logging, seeding,
│                   serialization, environment capture
├── data/           loader (N-BaIoT / CICIoT2023 / TON_IoT / Bot-IoT / synthetic),
│                   preprocessing (cleaning, federated scaler), zero_day (holdout),
│                   partition (Dirichlet / device / IID), federated_dataset (pipeline)
├── models/         autoencoder (primary), isolation_forest, one_class_svm
├── training/       trainer: ONE training loop for every mode (plain / FedProx / DP-SGD)
├── privacy/        differential_privacy (Opacus DP-SGD engine), accountant (RDP/PRV/GDP)
├── security/       secure_aggregation (DH + Shamir + PRG masking, dropout recovery)
├── federated/      client (IoTClient), server (FederatedServer), fedavg, fedprox,
│                   flower_adapter (optional)
├── detection/      anomaly_score (reconstruction error), threshold (exact + histogram)
├── evaluation/     metrics, zero_day_metrics, evaluator, comparison (report)
├── visualization/  plots (per-experiment + cross-experiment figures)
├── experiments/    centralized / federated / local modes, tracking, runner
└── cli/            entry points wrapped by scripts/*.py
```

Dependencies point downwards only: `experiments` → `federated` → `training` →
`privacy` / `models`; `security` depends only on `utils`. The secure-aggregation
module knows nothing about models or FL and works on arbitrary vectors (and therefore
any flattened state dict).

## One federated round

```
FederatedServer                                 IoTClient k (one per gateway)
───────────────                                 ─────────────────────────────
select clients (seeded)
broadcast w_t + {proximal_mu} ───────────────▶  build model from w_t
                                                train_autoencoder(benign local data,
                                                   FedProx term, DP-SGD engine)
                                                outbox ← [n_k·Δ_k, n_k, n_k·loss_k]
          ┌─ plain FL ─────────────────────────  plain_message()  (float32 vector)
          │
aggregate ┤
          │  secure aggregation (SecureAggregator carries the messages):
          └─ keys ◀── advertise_keys()
             roster ──▶ share_keys() ──▶ encrypted Shamir shares (routed, opaque)
             masked ◀── masked_input()        y_k = x_k + PRG(b_k) ± Σ PRG(s_kj)  mod 2^64
             unmask ◀── unmasking_response()  shares of b_k (survivors) / s_k^SK (dropped)
             → Σ_k x_k only
decode: Δ̄ = Σ n_kΔ_k / Σ n_k
w_{t+1} = w_t + η_s · Δ̄
```

The client vector already contains the sample count and loss, so the server code is
identical for plain and secure aggregation; with secure aggregation even `n_k` and the
loss are only revealed in aggregate. `IoTClient.plain_message()` raises
`PermissionError` when secure aggregation is enabled, so an unmasked update can't be
sent by mistake.

## Threshold calibration (federated)

```
server ── w_T ──▶ client k: scores of local validation rows
                  hist_k = [benign-score histogram | known-attack histogram]   (4096 log bins)
                  (+ optional Laplace noise, detection.histogram_dp_epsilon)
server ◀── Σ_k hist_k  (plain sum or secure aggregation)
threshold = percentile / mean+k·std / F1-optimal edge of the aggregated histogram
```

No per-sample anomaly score leaves a client. The centralized and local-only modes use
the exact score arrays instead.

## Separation of protocol and measurement

The simulation runs every party in one process. Two kinds of code are kept apart:

* **Protocol code** (`FederatedServer`, `SecAggServer`): consumes only the messages
  a real server would receive.
* **Experimenter code** (`EvalContext`, `ClientRoundStats`, per-client timing,
  per-round monitoring on the pooled test set): measures the system. It never feeds
  back into training, threshold selection or model selection. The final model is
  always the last round's model.

## Reproducibility

`utils.seed.derive_seed(seed, *keys)` gives every stochastic component its own
generator: `("partition", "known")`, `("split", k)`, `("client-shuffle", k, round)`,
`("dp-noise", k)`, `("selection", round)`, `("dropout", purpose, round)`,
`("model-init",)`, … Changing one component (e.g. the number of rounds) therefore does
not perturb another (e.g. the data partition). Every experiment directory stores the
resolved `config.yaml`, `environment.json` (package versions, hardware, git commit)
and the config fingerprint.

Secure-aggregation key material comes from the OS CSPRNG (`secrets`) unless
`secure_aggregation.deterministic: true`. The protocol's output (the sum) is the same
either way, so results are reproducible in both modes.

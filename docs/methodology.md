# Methodology

## 1. Threat and privacy model

* **Clients** are IoT gateways. Each observes the traffic of the devices behind it
  and is honest. They follow the protocol and want a good shared detector.
* **The server** is *honest-but-curious*. It runs the protocol correctly but may try
  to learn about individual gateways' traffic from what it receives.
* **Goal:** learn a detector of normal behaviour that flags attacks never seen during
  training, while (i) raw traffic never leaves a gateway, (ii) the server does not see
  any individual gateway's model update, and (iii) the trained model and the updates
  come with a formal bound on what they reveal about any single traffic record.

The three privacy techniques address different parts of this goal:

| Technique | Protects against | Does **not** provide |
|---|---|---|
| Federated learning | Collection of raw traffic by the server | Protection of the updates themselves (they can leak training data) |
| Secure aggregation | Server inspecting an individual client's update | Any guarantee about what the *aggregate* or the final model reveals |
| Differential privacy (DP-SGD) | Inference about a single record from anything the client releases | Confidentiality of the update (without SecAgg the server still sees it, noised) |

## 2. Zero-day simulation

Let *C* be the set of attack classes. A subset *U ⊂ C* (`data.holdout_classes`) is
removed from **every** client's training *and* validation data and appears only in
test data. *K = C \ U* are the *known* attacks.

* The autoencoder trains on benign traffic only (optionally with a fraction
  `train_contamination` of unlabelled known-attack rows, to mimic imperfectly clean
  training traffic).
* Known-attack validation rows may be used by the `validation_f1` threshold strategy.
  Unseen classes are never used for any decision.
* **Unseen-attack recall** = (# held-out attack samples with score > threshold) /
  (# held-out attack samples) is the headline security metric.

Default for N-BaIoT: *U* = all Mirai attacks (`mirai_*`), *K* = all BASHLITE/Gafgyt
attacks. The detector never sees the Mirai botnet. These are *zero-day-like*
attacks in an experimental sense, not real zero-day vulnerabilities.

## 3. Data pipeline

1. **Load** raw CSVs. Labels are normalised; the device is kept.
2. **Clean** (schema-level only): coerce to numeric, ±inf → missing, drop columns
   > 50 % missing or constant, drop rows > 50 % missing.
3. **Zero-day split** (above).
4. **Non-IID partition** of known traffic and, separately, of unseen traffic.
5. **Local split** of each client's known traffic into train / val / test
   (stratified by class; defaults 60 / 15 / 25 %).
6. **Federated scaling.** Each client computes per-feature count, sum, sum of squares,
   min and max of its *training* rows. The server merges them (equivalent to fitting
   on the pooled training data) and broadcasts mean/std. Missing values are imputed
   with the training mean.

### Non-IID partitioning

For each stratum *s* (default: every *(device, class)* pair) client proportions
*p ~ Dir(α·1_K)* are drawn and the stratum's rows are split accordingly.

* α → ∞: every client gets the same mixture (IID).
* α ≈ 0.5 (default): clients differ in devices, attack exposure and data volume.
* α ≤ 0.1: most strata belong to one or two clients (extreme heterogeneity).

Every client is guaranteed `min_benign_per_client` benign rows, since a gateway with
no normal traffic cannot learn "normal". Heterogeneity is reported as the mean
Jensen–Shannon distance between each client's class mix and the global mix.
Alternatives: `device` (each gateway serves whole devices) and `iid`.

## 4. Model and anomaly score

Symmetric MLP autoencoder `d → h1 → … → z → … → h1 → d` with ReLU (configurable),
linear bottleneck and output, no BatchNorm (needed for per-sample gradients). It
minimises the mean squared reconstruction error on benign traffic. The anomaly score of
*x* is `MSE(x, x̂)`, and *x* is flagged when the score exceeds the threshold *τ*.

### Threshold *τ* (validation data only)

* `percentile` (default): the q-th percentile (q = 95) of benign validation scores.
  This targets roughly a 5 % false-positive rate.
* `mean_std`: mean + k·std of benign validation scores.
* `validation_f1`: maximises F1 between benign and *known-attack* validation scores.

In federated modes *τ* comes from aggregated histograms (4096 log-spaced bins from
1e-8 to 1e6, < 1 % relative bin width). The tests check that it agrees with the exact
computation within 3 %.

## 5. Federated optimisation

* **FedAvg**: `w_{t+1} = w_t + η_s Σ_k n_k (w_k − w_t) / Σ_k n_k` (η_s = 1 → weighted
  model average).
* **FedProx**: each client minimises `F_k(w) + μ/2‖w − w_t‖²`. Aggregation is unchanged.
* Each client re-initialises its local optimiser state every round (stateless
  clients), which is standard for cross-device FL.
* Client sampling: `fraction_fit` of clients per round (seeded).

## 6. Differential privacy (DP-SGD)

Per local step (Abadi et al., 2016), implemented with Opacus primitives:

1. Poisson sampling with rate *q = 1/⌈n_k/B⌉*;
2. per-sample gradients;
3. clipping each to L2 norm ≤ *C* (`max_grad_norm`);
4. sum + Gaussian noise *N(0, σ²C²I)* (`noise_multiplier` σ);
5. divide by the expected batch size, optimiser step;
6. RDP accountant records one Sampled-Gaussian-Mechanism step.

The accountant lives as long as the client, so ε accumulates over **all** rounds the
client takes part in. The reported ε is the **maximum over clients**: every record of
every client is covered by (ε, δ). The guarantee is *example-level* DP with respect to
each client's local dataset. Everything released afterwards (model deltas, masked
vectors, the global model) is post-processing.

* `target_epsilon` mode: σ is calibrated per client so that, if the client takes part
  in every round, it ends at exactly the target ε.
* **FedProx + DP.** The proximal gradient μ(w − w_t) does not depend on data. It is
  added *after* clipping and noising, which is post-processing. Putting it inside the
  per-sample loss would wrongly clip it.
* **Threshold histograms** are computed on validation data, which is disjoint from the
  DP-protected training data. They are not covered by the DP-SGD guarantee. Optional
  Laplace noise (`detection.histogram_dp_epsilon`, sensitivity 1 per record) makes
  this release ε_h-DP as well.
* δ should be < 1/n_k. A warning is logged otherwise.

## 7. Secure aggregation

Follows the protocol structure of Bonawitz et al. (2017), for an honest-but-curious
server:

1. **AdvertiseKeys.** Two Diffie–Hellman key pairs per client, over the RFC 3526
   2048-bit MODP group.
2. **ShareKeys.** Each client Shamir-shares its mask secret key *s^SK* and a self-mask
   seed *b* (threshold *t* = ⌈0.6·n⌉, field GF(2^521 − 1)). Shares are encrypted for
   each recipient (SHAKE-256 keystream + HMAC-SHA256) and routed through the server.
3. **MaskedInput.** The update is encoded in fixed point (2⁻²⁴) in ℤ_{2⁶⁴} and sent as
   *y_k = x_k + PRG(b_k) + Σ_{j≠k} ±PRG(KDF(DH(s_k, s_j)))*, using SHAKE-256 as the PRG.
4. **Unmasking.** Survivors reveal shares of *b* for survivors and of *s^SK* for
   dropped clients, never both for the same client. The server removes all masks and
   recovers Σ x_k over survivors.

Pairwise masks cancel in the sum. Self-masks protect a slow client whose pairwise
masks were reconstructed. Dropouts are simulated with
`secure_aggregation.dropout_rate`, capped so that at least *t* clients survive.
Correctness tests check exact cancellation, recovery with dropouts, state-dict
aggregation, and that federated training gives the same model with and without
SecAgg (up to the 2⁻²⁴ quantisation).

**Limitations.** No PKI or signatures, so there is no protection against an actively
malicious server (for example one that lies about dropouts or impersonates clients).
There is also no constant-time arithmetic, and every party runs in one process. This
is a faithful protocol *simulation* for research, not production cryptography.

## 8. Baselines

* **Centralized autoencoder**: same architecture, trained on the pooled training
  rows; exact threshold. Reference upper bound that ignores privacy.
* **Local-only autoencoders**: no collaboration. Shows the value of federation.
* **Isolation Forest** and **One-Class SVM**: centralized, trained on pooled benign
  training data (OC-SVM on a subsample). They are not federated, because tree
  ensembles and kernel SVMs have no meaningful FedAvg-style parameter averaging.

## 9. Metrics

* **Detection** (attack = positive): accuracy, precision, recall, F1, FPR, ROC-AUC,
  PR-AUC, on all test traffic and separately on benign + known and benign + unseen.
* **Zero-day:** unseen-attack recall (headline), known-attack recall, benign FPR,
  per-class detection rates, score statistics per group.
* **Federated:** training/validation loss per round, F1 and unseen-recall per round
  (monitoring), per-client performance on client-local test data.
* **Cost:** bytes up/down per round (serialized message sizes), model-update payload,
  SecAgg overhead, total communication; centralized "communication" is the raw data
  upload. Also wall-clock training time and per-client compute time per round.
* **Privacy:** ε (max and mean over clients) after every round, δ, σ, C, sampling rates.

Timing and byte counts are measurements on the machine that ran the experiment
(recorded in `environment.json`). They are not device-level IoT measurements.

# Data

Datasets are **never committed** to the repository (`data/raw/*` and `data/processed/*`
are git-ignored). Download them yourself, place them under `data/raw/<dataset>/`
(or pass `--raw-dir`), then run the preparation script.

```
data/
├── raw/          # original files, one sub-directory per dataset
│   ├── nbaiot/
│   ├── ciciot2023/
│   ├── ton_iot/
│   ├── bot_iot/
│   └── synthetic/        # written by --synthetic (generated, test only)
└── processed/    # cleaned.parquet + metadata.json per dataset (generated)
```

## N-BaIoT (primary dataset)

Meidan et al., *N-BaIoT — Network-Based Detection of IoT Botnet Attacks Using Deep
Autoencoders*, IEEE Pervasive Computing, 2018. 115 statistical traffic features
computed over five damped time windows, captured from 9 commercial IoT devices
infected with the Mirai and BASHLITE (Gafgyt) botnets.

* UCI Machine Learning Repository: "detection_of_IoT_botnet_attacks_N_BaIoT"
  (dataset id 442), <https://archive.ics.uci.edu/dataset/442>
* A flat-file mirror (`1.benign.csv`, `1.gafgyt.combo.csv`, `1.mirai.ack.csv`, …) is
  also commonly distributed on Kaggle.

Both layouts are recognised automatically:

```
data/raw/nbaiot/1.benign.csv                              # flat layout
data/raw/nbaiot/1.gafgyt.scan.csv
data/raw/nbaiot/1.mirai.udpplain.csv
...
data/raw/nbaiot/Danmini_Doorbell/benign_traffic.csv       # per-device layout (UCI)
data/raw/nbaiot/Danmini_Doorbell/gafgyt_attacks/scan.csv
data/raw/nbaiot/Danmini_Doorbell/mirai_attacks/udpplain.csv
```

The UCI archive ships the per-device attack folders as `.rar` files — extract them
first. `features.csv`, `data_summary.csv` and `device_info.csv` are ignored.

Class labels become `benign`, `gafgyt_<attack>` and `mirai_<attack>`; the device name
is kept for device-aware non-IID partitioning. Two devices (Ennio doorbell and
Samsung webcam) were not infected by Mirai in the original capture.

```bash
python scripts/prepare_data.py --dataset nbaIoT
```

By default at most 20 000 benign rows and 4 000 rows per attack class are kept per
device (`data.max_benign_samples_per_group`, `data.max_samples_per_group`) so the
pipeline fits comfortably in laptop memory. Set them to `null` to use everything.

## Optional datasets

| Dataset | Source | Label column used | Expected location |
|---|---|---|---|
| CICIoT2023 | Canadian Institute for Cybersecurity, <https://www.unb.ca/cic/datasets/iotdataset-2023.html> | `label` (`BenignTraffic` → benign) | `data/raw/ciciot2023/*.csv` |
| TON_IoT (network) | UNSW Canberra, <https://research.unsw.edu.au/projects/toniot-datasets> | `type` (`normal` → benign) | `data/raw/ton_iot/*.csv` |
| Bot-IoT | UNSW Canberra, <https://research.unsw.edu.au/projects/bot-iot-dataset> | `category` (`Normal` → benign) | `data/raw/bot_iot/*.csv` |

These have no per-device column, so use `partition.stratify_by: [label]`. Identifier
columns (IPs, ports, timestamps, sequence ids) are dropped and only numeric features
are kept. Column names differ between releases; override the defaults with
`data.loader_options` (keys: `label_column`, `benign_values`, `device_column`,
`drop_columns`, `file_glob`, `max_files`). Choose `data.holdout_classes` to match the
class names printed by `prepare_data.py`.

```bash
python scripts/prepare_data.py --dataset ciciot2023 --set data.holdout_classes=[ddos_*]
```

## Synthetic data (testing only)

```bash
python scripts/prepare_data.py --synthetic
```

Generates an N-BaIoT-*like* table (6 device types, 6 attack families, damped-window
style features, plus injected NaN/inf cells and a constant column to exercise the
cleaning stage). It exists only so the pipeline can be run and tested without a
download. **Results obtained on it are not research results.**

"""Interactive dashboard: train the detector live, stream test traffic through it, compare methods.

    streamlit run app.py

Every number shown is computed by the ``ppfl`` package at run time or read from finished
experiment folders; nothing is hard-coded.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import ppfl  # noqa: E402,F401  (imports scikit-learn before torch; see ppfl/__init__.py)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402

from ppfl.data.federated_dataset import build_federated_dataset  # noqa: E402
from ppfl.detection.anomaly_score import reconstruction_errors  # noqa: E402
from ppfl.experiments.runner import run_experiment  # noqa: E402
from ppfl.models.autoencoder import Autoencoder  # noqa: E402
from ppfl.utils.config import Config, load_config  # noqa: E402
from ppfl.utils.serialization import load_json  # noqa: E402

CONFIGS = ROOT / "configs"
UI_RUNS = "results/ui"

MODES = {  # label -> (config file, uses differential privacy, short explanation)
    "Full privacy (DP + secure aggregation)": ("full_ppfl.yaml", True,
        "Raw data stays on each gateway, every update is noised (differential privacy) and the server only sees the sum (secure aggregation)."),
    "Federated + secure aggregation": ("secure_aggregation.yaml", False,
        "Raw data stays local and the server only sees the sum of the updates, but no noise is added."),
    "Federated + differential privacy": ("dp.yaml", True,
        "Raw data stays local and every update is noised, but the server sees each gateway's update."),
    "Plain federated (FedAvg)": ("fedavg.yaml", False,
        "Raw data stays local; the server sees each gateway's model update."),
    "Centralized (all data pooled)": ("centralized.yaml", False,
        "Reference point with no privacy: all gateways' traffic is copied to one server. Also trains Isolation Forest and One-Class SVM."),
}
DATASETS = {"nbaiot": "N-BaIoT (real IoT botnet traffic)", "synthetic": "Synthetic (pipeline test data)"}
GROUPS = {"benign": "Normal traffic", "known_attack": "Known attacks", "unseen_attack": "Zero-day attacks"}
COLORS = {"benign": "#2a78d6", "known_attack": "#eb6834", "unseen_attack": "#1baf7a"}
ALERT = "#d03b3b"

# ----------------------------------------------------------------------------- helpers

def pct(v: float | None) -> str:
    return "–" if v is None or pd.isna(v) else f"{100 * v:.1f}%"


def rgba(hex_: str, alpha: float) -> str:
    r, g, b = (int(hex_[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def style(fig: go.Figure, height: int = 320, **layout) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=78, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff",
        font=dict(family="Source Sans Pro, Segoe UI, sans-serif", size=13, color="#0d1117"),
        title=dict(y=0.97, yanchor="top", x=0.01, font=dict(size=15)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0), hoverlabel=dict(font_size=13),
    )
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor="#e9ebef", zeroline=False, linecolor="#c4c7ce")
    fig.update_yaxes(gridcolor="#e9ebef", zeroline=False, linecolor="#c4c7ce")
    return fig


def available_datasets() -> list[str]:
    return [d for d in DATASETS if (ROOT / "data" / "processed" / d / "cleaned.parquet").exists()]


def build_config(s: dict) -> Config:
    file = MODES[s["mode"]][0]
    overlays = [CONFIGS / "synthetic.yaml"] if s["dataset"] == "synthetic" else []
    if s["dataset"] == "nbaiot" and (CONFIGS / "nbaiot.yaml").exists():
        overlays.append(CONFIGS / "nbaiot.yaml")
    sets = [
        f"experiment.name=ui_{Path(file).stem}", f"experiment.output_dir={UI_RUNS}/{s['dataset']}",
        f"experiment.seed={s['seed']}", f"partition.num_clients={s['clients']}", f"partition.dirichlet_alpha={s['alpha']}",
        f"federated.rounds={s['rounds']}", f"training.epochs={s['rounds']}", f"federated.local_epochs={s['local_epochs']}",
        "evaluation.make_figures=false",
    ]
    if MODES[s["mode"]][1]:
        sets.append(f"privacy.noise_multiplier={s['sigma']}")
    return load_config(CONFIGS / file, sets, overlays)


def data_key(cfg: Config) -> str:
    d = cfg.to_dict()
    return json.dumps([d["data"], d["partition"], cfg.experiment.seed], sort_keys=True, default=str)


@st.cache_resource(show_spinner=False, max_entries=3)
def get_dataset(key: str, _cfg: Config):
    return build_federated_dataset(_cfg)


@st.cache_resource(show_spinner=False, max_entries=6)
def load_run(run_dir: str, stamp: float):
    """Trained model, threshold, config and dataset of a finished run (``stamp`` busts the cache)."""
    d = Path(run_dir)
    cfg = load_config(d / "config.yaml")
    info = load_json(d / "model" / "model.json")
    arch = {k: v for k, v in info["architecture"].items() if k != "num_parameters"}
    model = Autoencoder(**arch, activation=cfg.model.activation)
    model.load_state_dict(torch.load(d / "model" / "autoencoder.pt", map_location="cpu"))
    model.eval()
    return model, float(info["threshold"]["value"]), cfg, get_dataset(data_key(cfg), cfg)


def finished_runs() -> list[Path]:
    runs = [p.parent for p in (ROOT / UI_RUNS).glob("*/*/metrics.json") if (p.parent / "model" / "autoencoder.pt").exists()]
    return sorted(runs, key=lambda p: (p / "metrics.json").stat().st_mtime, reverse=True)


def run_label(d: Path) -> str:
    m = load_json(d / "metrics.json")
    return f"{m['experiment']['label']} · {DATASETS.get(m['setup']['dataset'], m['setup']['dataset']).split(' (')[0]} · {m['experiment']['finished_utc'][:16].replace('T', ' ')} UTC"


# ----------------------------------------------------------------------------- pages

def page_overview() -> None:
    st.markdown('<div class="eyebrow">Privacy-preserving federated learning · IoT intrusion detection</div>', unsafe_allow_html=True)
    st.title("IoT Zero-Day Defense")
    st.markdown(
        "IoT gateways in different homes and offices **train one shared attack detector together without sharing their network traffic**. "
        "The detector is an autoencoder that learns what *normal* traffic looks like, so it can flag attacks it has **never seen before** (zero-day attacks)."
    )
    c = st.columns(3)
    c[0].markdown('<div class="step"><b>The problem</b>A good intrusion detector needs traffic from many sites, but that traffic reveals which devices people own and how they use them. Signature-based tools also miss new attacks.</div>', unsafe_allow_html=True)
    c[1].markdown('<div class="step"><b>The approach</b>Federated learning keeps raw traffic on each gateway. Differential privacy adds calibrated noise to every update. Secure aggregation lets the server see only the sum of the updates.</div>', unsafe_allow_html=True)
    c[2].markdown('<div class="step"><b>The test</b>Whole attack families are hidden from training and appear only at test time. The headline metric is how many of these zero-day attacks are caught at a 5% false-alarm budget.</div>', unsafe_allow_html=True)

    st.subheader("How one training round works")
    st.graphviz_chart(
        """
digraph { rankdir=TB; bgcolor="transparent"; node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 color="#c4c7ce" fillcolor="#ffffff" margin="0.2,0.1"];
edge [color="#80858f" fontname="Helvetica" fontsize=10 fontcolor="#4a505a"];
S [label="Aggregation server\\nholds the shared model\\nsees only the SUM of updates" fillcolor="#eaf2fc" color="#2a78d6"];
subgraph cluster_g { style=invis; G1 [label="Gateway 1\\nraw traffic stays here"]; G2 [label="Gateway 2\\nraw traffic stays here"]; G3 [label="Gateway N\\nraw traffic stays here"]; }
S -> G1 [label=" 1. model"]; S -> G2; S -> G3;
G1 -> S [label=" 2. train · 3. clip + noise · 4. mask" style=dashed color="#eb6834"]; G2 -> S [style=dashed color="#eb6834"]; G3 -> S [style=dashed color="#eb6834"];
}
""",
        use_container_width=True,
    )

    summary = best_summary()
    if summary is not None:
        df, source = summary
        st.subheader("Zero-day attacks caught")
        st.caption(f"Share of never-seen attack traffic flagged, at about 5% false alarms. From the finished experiment suite ({source}).")
        row = lambda e: df[(df.experiment == e) & (df.model == "autoencoder")].iloc[0] if ((df.experiment == e) & (df.model == "autoencoder")).any() else None  # noqa: E731
        cols = st.columns(4)
        for col, (exp, name) in zip(cols, [("centralized", "All data pooled"), ("fedavg", "Federated"), ("secagg_fl", "Federated + SecAgg"), ("full_ppfl", "Full privacy")]):
            r = row(exp)
            if r is not None:
                col.metric(name, pct(r.unseen_recall), help=f"F1 {r.f1:.3f} · false alarms {pct(r.fpr)} · AUC {r.roc_auc:.3f}")

    st.subheader("Use this app")
    c = st.columns(3)
    c[0].markdown('<div class="step"><b>1 · Train live</b>Pick a privacy mode, press Start and watch each training round update the detection score and privacy budget.</div>', unsafe_allow_html=True)
    c[1].markdown('<div class="step"><b>2 · Detect attacks</b>Stream real test traffic through the model you just trained and see every alert, including zero-day attacks.</div>', unsafe_allow_html=True)
    c[2].markdown('<div class="step"><b>3 · Compare results</b>See every method side by side and the cost of privacy in detection quality.</div>', unsafe_allow_html=True)


def page_train() -> None:
    st.markdown('<div class="eyebrow">Step 1</div>', unsafe_allow_html=True)
    st.title("Train live")
    datasets = available_datasets()
    if not datasets:
        st.error("No prepared dataset found. Run `python scripts/prepare_data.py --synthetic` (or `--dataset nbaIoT`) first.")
        return

    with st.container(border=True):
        c = st.columns([1.2, 1.6, 1, 1])
        dataset = c[0].selectbox("Dataset", datasets, format_func=DATASETS.get)
        mode = c[1].selectbox("Privacy mode", list(MODES))
        clients = c[2].slider("Gateways", 3, 20, 6 if dataset == "synthetic" else 10, disabled=mode.startswith("Centralized"))
        rounds = c[3].slider("Epochs" if mode.startswith("Centralized") else "Rounds", 3, 60, 20)
        st.caption(MODES[mode][2])
        with st.expander("Advanced settings"):
            a = st.columns(4)
            local_epochs = a[0].slider("Local epochs per round", 1, 5, 2)
            sigma = a[1].select_slider("Noise level σ", [0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.5], 1.0, disabled=not MODES[mode][1],
                                       help="More noise gives stronger privacy (smaller ε) and a weaker detector.")
            alpha = a[2].select_slider("Data similarity α", [0.1, 0.3, 0.5, 1.0, 10.0], 0.5,
                                       help="Dirichlet α: low = every gateway sees very different traffic, high = similar traffic.")
            seed = a[3].number_input("Seed", 0, 9999, 42)
        start = st.button("Start training", type="primary", icon=":material/play_arrow:")

    if start:
        s = dict(dataset=dataset, mode=mode, clients=clients, rounds=rounds, local_epochs=local_epochs, sigma=sigma, alpha=alpha, seed=int(seed))
        train_live(build_config(s))
    elif st.session_state.get("last_run"):
        show_results(Path(st.session_state["last_run"]))
    elif finished_runs():
        st.info("Showing your most recent run. Press **Start training** for a new one.")
        show_results(finished_runs()[0])


def train_live(cfg: Config) -> None:
    dp = cfg.privacy.enabled and cfg.training.mode == "federated"
    total = cfg.federated.rounds if cfg.training.mode == "federated" else cfg.training.epochs
    with st.status("Preparing gateways and splitting data…", expanded=True) as status:
        t0 = time.perf_counter()
        dataset = get_dataset(data_key(cfg), cfg)
        st.write(f"{len(dataset.clients)} gateways ready · {dataset.input_dim} traffic features · "
                 f"zero-day attacks held out: {', '.join(dataset.zero_day.unseen_attack_classes)} ({time.perf_counter() - t0:.1f}s)")
        st.plotly_chart(partition_figure(dataset.partition_table, dataset), use_container_width=True, key="partition_live")
        status.update(label="Training…", state="running")
        bar = st.progress(0.0, text="Starting")
        kpi = st.columns(4)
        slots = [k.empty() for k in kpi]
        charts = st.columns(2)
        chart_q, chart_p = charts[0].empty(), charts[1].empty()
        rows: list[dict] = []

        def on_round(row: dict) -> None:
            rows.append(row)
            n = len(rows)
            prev = rows[-2] if n > 1 else {}
            bar.progress(min(n / total, 1.0), text=f"{'Round' if cfg.training.mode == 'federated' else 'Epoch'} {n} of {total}")
            delta = lambda k: None if k not in prev or k not in row else f"{100 * (row[k] - prev[k]):+.1f} pts"  # noqa: E731
            slots[0].metric("Detection score (F1)", f"{row.get('f1', float('nan')):.3f}", None if "f1" not in prev else f"{row['f1'] - prev['f1']:+.3f}")
            slots[1].metric("Zero-day attacks caught", pct(row.get("unseen_recall")), delta("unseen_recall"))
            slots[2].metric("False alarms", pct(row.get("fpr")), delta("fpr"), delta_color="inverse")
            if dp:
                slots[3].metric("Privacy budget ε", f"{row['epsilon_max']:.2f}", help="Lower is more private. It grows with every round.")
            else:
                slots[3].metric("Reconstruction error", f"{row.get('val_loss', float('nan')):.4f}")
            df = pd.DataFrame(rows)
            chart_q.plotly_chart(quality_figure(df, total), use_container_width=True, key=f"q{n}")
            chart_p.plotly_chart(privacy_figure(df, total) if dp else loss_figure(df, total), use_container_width=True, key=f"p{n}")

        metrics = run_experiment(cfg, dataset=dataset, on_round=on_round)
        status.update(label=f"Finished in {metrics['timing']['total_time_s']:.0f}s", state="complete", expanded=False)
    st.session_state["last_run"] = str(cfg.experiment_dir)
    show_results(cfg.experiment_dir)


def show_results(run_dir: Path) -> None:
    m = load_json(run_dir / "metrics.json")
    ae = m["models"]["autoencoder"]
    s, rep = ae["summary"], ae["report"]
    st.divider()
    st.markdown(f'<div class="eyebrow">Result · {m["experiment"]["label"]} · {DATASETS.get(m["setup"]["dataset"], "")}</div>', unsafe_allow_html=True)
    if m["experiment"].get("synthetic_data"):
        st.caption("Synthetic data: this shows the pipeline working, not research results.")
    k = st.columns(6)
    k[0].metric("Zero-day caught", pct(s["unseen_recall"]), help="Attack types never seen during training that were flagged.")
    k[1].metric("Known caught", pct(s["known_recall"]))
    k[2].metric("False alarms", pct(s["fpr"]), help="Normal traffic wrongly flagged. The threshold targets 5%.")
    k[3].metric("F1 score", f"{s['f1']:.3f}")
    k[4].metric("ROC-AUC", f"{s['roc_auc']:.3f}", help="0.5 = guessing, 1.0 = perfect ranking.")
    eps = (m.get("privacy") or {}).get("epsilon")
    comm = (m.get("communication") or {}).get("total_bytes")
    k[5].metric("Privacy ε" if eps is not None else "Data sent", f"{eps:.2f}" if eps is not None else (f"{comm / 2**20:.1f} MB" if comm else "all raw data"),
                help="Example-level (ε, δ=1e-5) differential privacy, worst gateway." if eps is not None else None)

    c = st.columns([1.1, 1])
    with c[0]:
        st.plotly_chart(score_figure(run_dir, s["threshold"]), use_container_width=True, key=f"scores_{run_dir.name}")
    with c[1]:
        pc = pd.DataFrame([{"class": k_, **v} for k_, v in rep["per_class"].items()]).sort_values(["group", "flagged_rate"])
        fig = go.Figure([go.Bar(
            x=d.flagged_rate, y=d["class"], orientation="h", marker_color=COLORS[g], name=GROUPS[g],
            text=[pct(v) for v in d.flagged_rate], textposition="outside", cliponaxis=False, customdata=d.n,
            hovertemplate="<b>%{y}</b><br>flagged %{x:.1%} of %{customdata} records<extra></extra>",
        ) for g in GROUPS if len(d := pc[pc.group == g])])
        st.plotly_chart(style(fig, 360, title="Flagged as anomalous, by traffic type", xaxis=dict(range=[0, 1.12], tickformat=".0%")),
                        use_container_width=True, key=f"perclass_{run_dir.name}")

    hist = run_dir / "metrics.csv"
    if hist.exists() and m["experiment"]["mode"] != "local":
        df = pd.read_csv(hist)
        total = len(df)
        cc = st.columns(2)
        cc[0].plotly_chart(quality_figure(df, total), use_container_width=True, key=f"hq_{run_dir.name}")
        cc[1].plotly_chart(privacy_figure(df, total) if "epsilon_max" in df else loss_figure(df, total), use_container_width=True, key=f"hp_{run_dir.name}")
    if (run_dir / "per_client.csv").exists():
        with st.expander("Per-gateway results"):
            pcl = pd.read_csv(run_dir / "per_client.csv")
            show = pd.DataFrame({"Gateway": pcl.client_id + 1, "Test records": pcl.n_test, "Zero-day caught": pcl.unseen_recall.map(pct),
                                 "Known caught": pcl.known_recall.map(pct), "False alarms": pcl.fpr.map(pct), "F1": pcl.f1.round(3)})
            st.dataframe(show, hide_index=True, use_container_width=True)
    if "isolation_forest" in m["models"]:
        with st.expander("Classical baselines trained on the same pooled data"):
            st.dataframe(pd.DataFrame([{"Model": n.replace("_", " ").title(), "Zero-day caught": pct(v["summary"]["unseen_recall"]),
                                        "False alarms": pct(v["summary"]["fpr"]), "F1": round(v["summary"]["f1"], 3), "ROC-AUC": round(v["summary"]["roc_auc"], 3)}
                                       for n, v in m["models"].items()]), hide_index=True, use_container_width=True)


def page_detect() -> None:
    st.markdown('<div class="eyebrow">Step 2</div>', unsafe_allow_html=True)
    st.title("Detect attacks")
    runs = finished_runs()
    if not runs:
        st.info("Train a model on the **Train live** page first.")
        return
    last = st.session_state.get("last_run")
    default = next((i for i, r in enumerate(runs) if str(r) == last), 0)
    c = st.columns([2, 1, 1])
    run_dir = c[0].selectbox("Trained model", runs, index=default, format_func=run_label)
    n = c[1].select_slider("Records to stream", [60, 120, 240, 480], 240)
    speed = c[2].select_slider("Speed", ["slow", "normal", "fast"], "normal")
    with st.spinner("Loading the trained model and its test traffic…"):
        model, thr, cfg, dataset = load_run(str(run_dir), (run_dir / "metrics.json").stat().st_mtime)
    test, owners = dataset.global_test()
    groups = dataset.zero_day.group_of(test.labels)
    st.caption(f"The model scores each record **now**: it rebuilds the record and measures the error. Error above **{thr:.4g}** raises an alert. "
               f"The pool holds {len(test):,} unseen test records from {len(dataset.clients)} gateways.")

    go_ = st.button("Stream traffic", type="primary", icon=":material/sensors:")
    k = st.columns(4)
    view = dict(slots=[x.empty() for x in k])
    c = st.columns([1.5, 1])
    view.update(chart=c[0].empty(), feed=c[1].empty())
    state = st.session_state.get("stream")
    if go_:
        rng = np.random.default_rng()
        # equal share per group so zero-day attacks are visible even though they are rare in the pool
        idx = np.concatenate([rng.choice(np.flatnonzero(groups == g), n // 3, replace=False) for g in GROUPS if (groups == g).sum() >= n // 3])
        rng.shuffle(idx)
        delay = {"slow": 0.25, "normal": 0.09, "fast": 0.02}[speed]
        scores = np.zeros(0)
        for start in range(0, len(idx), 6):
            scores = np.concatenate([scores, reconstruction_errors(model, test.X[idx[start : start + 6]])])  # real inference, batch by batch
            render_stream(view, scores, idx, test, owners, groups, thr, f"s{start}")
            time.sleep(delay)
        state = st.session_state["stream"] = {"run": str(run_dir), "idx": idx, "scores": scores}
    elif state and state["run"] == str(run_dir):
        render_stream(view, state["scores"], state["idx"], test, owners, groups, thr, "s_final")
    if state and state["run"] == str(run_dir):
        explain(model, thr, dataset, test, state)


def render_stream(view, scores, idx, test, owners, groups, thr, key) -> None:
    seen = idx[: len(scores)]
    g, flag = groups[seen], scores > thr
    z, b = g == "unseen_attack", g == "benign"
    view["slots"][0].metric("Records analysed", f"{len(scores)}")
    view["slots"][1].metric("Alerts raised", f"{flag.sum()}")
    view["slots"][2].metric("Zero-day attacks caught", f"{(flag & z).sum()} / {z.sum()}")
    view["slots"][3].metric("False alarms", f"{(flag & b).sum()} / {b.sum()}")
    view["chart"].plotly_chart(stream_figure(scores, g, test.labels[seen], thr), use_container_width=True, key=key)
    recent = pd.DataFrame({"Verdict": np.where(flag, "ALERT", "ok"), "Traffic": test.labels[seen], "Type": [GROUPS[x] for x in g],
                           "Gateway": owners[seen] + 1, "Score": scores})[::-1].head(12)
    styled = recent.style.format({"Score": "{:.4g}"}).map(lambda v: f"color: {ALERT}; font-weight: 600" if v == "ALERT" else "color: #006300", subset=["Verdict"])
    view["feed"].dataframe(styled, hide_index=True, use_container_width=True, height=460)


def explain(model, thr, dataset, test, state) -> None:
    idx, scores = state["idx"], state["scores"]
    flagged = [int(i) for i, s in zip(idx, scores) if s > thr]
    if not flagged:
        return
    st.subheader("Why was it flagged?")
    flagged = flagged[:60]
    names = [f"#{n + 1}  {test.labels[i]} · score {float(scores[list(idx).index(i)]):.4g}" for n, i in enumerate(flagged)]
    pick = flagged[names.index(st.selectbox("Alert to explain", names))]
    x = torch.from_numpy(test.X[pick : pick + 1])
    with torch.no_grad():
        err = ((model(x) - x) ** 2).numpy()[0]
    top = np.argsort(err)[::-1][:10]
    fig = go.Figure(go.Bar(x=err[top][::-1], y=[dataset.feature_names[i] for i in top][::-1], orientation="h", marker_color=ALERT,
                           hovertemplate="%{y}<br>squared error %{x:.3g}<extra></extra>"))
    st.plotly_chart(style(fig, 360, title="Traffic features the model could not rebuild (largest error first)"), use_container_width=True, key="explain")
    st.caption("The autoencoder only learned normal traffic. Features it rebuilds badly are the ones that look least like normal behaviour.")


def page_compare() -> None:
    st.markdown('<div class="eyebrow">Step 3</div>', unsafe_allow_html=True)
    st.title("Compare results")
    summary = best_summary(all_sources=True)
    if summary is None:
        st.info("No experiment suite found. Run `python scripts/run_experiments.py --suite all` first.")
        return
    df, source = summary
    st.caption(f"Source: {source}. Every method ran on the same data split and seed.")
    names = {"centralized": "All data pooled", "local_only": "Each gateway alone", "fedavg": "Federated (FedAvg)", "fedprox": "Federated (FedProx)",
             "secagg_fl": "Federated + secure aggregation", "dp_fl": "Federated + differential privacy", "full_ppfl": "Full privacy (FedAvg)",
             "full_ppfl_fedprox": "Full privacy (FedProx)"}
    main = df[df.experiment.isin(names)].copy()
    main["Method"] = [names[e] if m == "autoencoder" else f"{m.replace('_', ' ').title()} (pooled)" for e, m in zip(main.experiment, main.model)]
    order = list(names.values())
    main["_o"] = [order.index(x) if x in order else -1 for x in main.Method]
    main = main.sort_values("_o")
    metric = st.segmented_control("Metric", ["Zero-day caught", "F1", "ROC-AUC", "False alarms"], default="Zero-day caught")
    col = {"Zero-day caught": "unseen_recall", "F1": "f1", "ROC-AUC": "roc_auc", "False alarms": "fpr"}[metric or "Zero-day caught"]
    priv = main.dp.fillna(False).astype(bool) | main.secagg.fillna(False).astype(bool)
    fig = go.Figure(go.Bar(
        x=main[col], y=main.Method, orientation="h", marker_color=np.where(priv, "#eb6834", "#2a78d6"),
        text=[f"{v:.3f}" if col in ("f1", "roc_auc") else pct(v) for v in main[col]], textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x:.3f}<extra></extra>"))
    style(fig, 420, title=f"{metric or 'Zero-day caught'} · orange = privacy protection on", yaxis=dict(autorange="reversed"),
          xaxis=dict(tickformat=".0%" if col in ("unseen_recall", "fpr") else ".2f"))
    st.plotly_chart(fig, use_container_width=True, key="cmp")

    table = pd.DataFrame({"Method": main.Method, "Zero-day caught": main.unseen_recall.map(pct), "Known caught": main.known_recall.map(pct),
                          "False alarms": main.fpr.map(pct), "F1": main.f1.round(3), "ROC-AUC": main.roc_auc.round(3),
                          "Privacy ε": main.epsilon.map(lambda v: "–" if pd.isna(v) else f"{v:.1f}"),
                          "Data sent": main.total_mb.map(lambda v: "–" if pd.isna(v) else f"{v:.1f} MB")})
    st.dataframe(table, hide_index=True, use_container_width=True)

    st.subheader("The cost of privacy")
    sweep = df[df.experiment.str.startswith("ppfl_sigma_")].sort_values("epsilon")
    if len(sweep):
        ref = df[(df.experiment == "fedavg") & (df.model == "autoencoder")]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sweep.epsilon, y=sweep.unseen_recall, mode="lines+markers+text", name="Zero-day caught",
                                 text=[f"σ {v:g}" for v in sweep.noise_multiplier], textposition="bottom center",
                                 line=dict(color="#1baf7a", width=2), marker=dict(size=9)))
        fig.add_trace(go.Scatter(x=sweep.epsilon, y=sweep.f1, mode="lines+markers", name="F1", line=dict(color="#2a78d6", width=2), marker=dict(size=9)))
        if len(ref):
            fig.add_hline(y=float(ref.unseen_recall.iloc[0]), line_dash="dot", line_color="#80858f",
                          annotation_text=f"no noise: {pct(float(ref.unseen_recall.iloc[0]))} zero-day caught", annotation_position="top left")
        style(fig, 380, title="Full privacy at different noise levels σ (left = more private)",
              xaxis=dict(type="log", tickvals=[2, 5, 10, 20, 50, 100], title="privacy budget ε (log scale)"), yaxis=dict(rangemode="tozero"))
        st.plotly_chart(fig, use_container_width=True, key="tradeoff")
    sweeps = {"ppfl_clients_": ("Number of gateways", "num_clients"), "fedavg_alpha_": ("Data similarity α", "dirichlet_alpha"),
              "ppfl_local_epochs_": ("Local epochs per round", "local_epochs")}
    cols = st.columns(3)
    for col_, (prefix, (title, x)) in zip(cols, sweeps.items()):
        sw = df[df.experiment.str.startswith(prefix)].sort_values(x)
        if len(sw):
            f = go.Figure(go.Scatter(x=sw[x].astype(str), y=sw.unseen_recall, mode="lines+markers", line=dict(color="#1baf7a", width=2), marker=dict(size=8),
                                     hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
            col_.plotly_chart(style(f, 260, title=f"Zero-day caught vs {title.lower()}", yaxis=dict(tickformat=".0%", rangemode="tozero"),
                                    xaxis=dict(type="category")), use_container_width=True, key=f"sw_{prefix}")


# ----------------------------------------------------------------------------- figures

def best_summary(all_sources: bool = False):
    sources = [(ROOT / "results" / "summary.csv", "N-BaIoT, real IoT traffic"), (ROOT / "results" / "synthetic" / "summary.csv", "synthetic test data")]
    found = [(pd.read_csv(p), label) for p, label in sources if p.exists()]
    if not found:
        return None
    if all_sources and len(found) > 1:
        label = st.radio("Results", [f[1] for f in found], horizontal=True)
        return next(f for f in found if f[1] == label)
    return found[0]


def partition_figure(table: pd.DataFrame, dataset) -> go.Figure:
    t = table.copy()
    groups = dataset.zero_day.group_of(np.array(t.columns))
    fig = go.Figure()
    for g in GROUPS:
        cols = [c for c, gg in zip(t.columns, groups) if gg == g]
        if cols:
            fig.add_trace(go.Bar(name=GROUPS[g], x=[f"G{i + 1}" for i in range(len(t))], y=t[cols].sum(axis=1), marker_color=COLORS[g]))
    return style(fig, 260, barmode="stack", title="Traffic records on each gateway (they never leave it)", bargap=0.25)


def rounds_axis(total: int) -> dict:
    return dict(range=[0.5, total + 0.5], dtick=max(1, round(total / 10)), title="round")


def quality_figure(df: pd.DataFrame, total: int) -> go.Figure:
    fig = go.Figure()
    for k, name, color in [("f1", "F1", "#2a78d6"), ("unseen_recall", "Zero-day caught", "#1baf7a"), ("fpr", "False alarms", ALERT)]:
        if k in df:
            fig.add_trace(go.Scatter(x=df["round"], y=df[k], name=name, mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=6)))
    return style(fig, 300, title="Detection quality per round", xaxis=rounds_axis(total), yaxis=dict(range=[0, 1], tickformat=".0%"))


def privacy_figure(df: pd.DataFrame, total: int) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df["round"], y=df["epsilon_max"], mode="lines+markers", name="ε", line=dict(color="#eb6834", width=2),
                               fill="tozeroy", fillcolor="rgba(235,104,52,.08)"))
    return style(fig, 300, title="Privacy budget ε spent (lower = more private)", xaxis=rounds_axis(total), yaxis=dict(rangemode="tozero"))


def loss_figure(df: pd.DataFrame, total: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["round"], y=df["train_loss"], name="training", mode="lines+markers", line=dict(color="#2a78d6", width=2)))
    if "val_loss" in df:
        fig.add_trace(go.Scatter(x=df["round"], y=df["val_loss"], name="validation", mode="lines+markers", line=dict(color="#80858f", width=2, dash="dot")))
    return style(fig, 300, title="Reconstruction error on normal traffic", xaxis=rounds_axis(total), yaxis=dict(rangemode="tozero"))


def score_figure(run_dir: Path, thr: float) -> go.Figure:
    z = np.load(run_dir / "scores_autoencoder.npz", allow_pickle=True)
    s = np.clip(z["scores"], 1e-6, None)
    lo, hi = np.log10(np.percentile(s, 0.5)), np.log10(np.percentile(s, 99.5))
    bins = np.logspace(lo, hi, 60)
    fig = go.Figure()
    for g in GROUPS:
        v = s[z["groups"] == g]
        if len(v):
            h, _ = np.histogram(v, bins)
            fig.add_trace(go.Scatter(x=np.sqrt(bins[:-1] * bins[1:]), y=h / h.sum(), name=GROUPS[g], mode="lines", line=dict(color=COLORS[g], width=2),
                                     fill="tozeroy", fillcolor=rgba(COLORS[g], 0.13), hovertemplate="score %{x:.3g}<br>%{y:.1%} of group<extra></extra>"))
    fig.add_vline(x=thr, line_color=ALERT, line_width=2, annotation_text="alert threshold", annotation_font_color=ALERT)
    return style(fig, 360, title="Anomaly scores of test traffic (right of the line = alert)", xaxis=dict(type="log", dtick=1, title="reconstruction error (log scale)"),
                 yaxis=dict(tickformat=".0%", title="share of group"))


def stream_figure(scores, groups, labels, thr) -> go.Figure:
    fig = go.Figure()
    x = np.arange(1, len(scores) + 1)
    for g in GROUPS:
        m = groups == g
        if m.any():
            fig.add_trace(go.Scatter(x=x[m], y=scores[m], mode="markers", name=GROUPS[g], customdata=labels[m],
                                     marker=dict(color=COLORS[g], size=8, line=dict(color=np.where(scores[m] > thr, ALERT, "#ffffff"), width=2)),
                                     hovertemplate="%{customdata}<br>score %{y:.3g}<extra></extra>"))
    fig.add_hline(y=thr, line_color=ALERT, line_width=2, annotation_text="alert threshold", annotation_font_color=ALERT)
    return style(fig, 460, title="Each dot is one traffic record (red ring = alert)", xaxis=dict(title="record"), yaxis=dict(type="log", dtick=1, title="anomaly score (log scale)"))


# ----------------------------------------------------------------------------- navigation

def setup_page() -> None:
    st.set_page_config(page_title="IoT Zero-Day Defense", page_icon=":material/shield:", layout="wide")
    st.markdown(
        """
    <style>
    .block-container { padding-top: 3.2rem; max-width: 1200px; }
    [data-testid="stMetric"] { background: #ffffff; border: 1px solid rgba(13,17,23,.08); border-radius: 12px; padding: 14px 16px; }
    [data-testid="stMetricLabel"] p { font-size: .85rem; color: #4a505a; }
    [data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 650; }
    .stAppDeployButton, footer { display: none !important; }
    .eyebrow { font: 600 .75rem/1 ui-monospace, Consolas, monospace; letter-spacing: .08em; text-transform: uppercase; color: #80858f; margin-bottom: .35rem; }
    .step { background: #fff; border: 1px solid rgba(13,17,23,.08); border-radius: 12px; padding: 16px; height: 100%; }
    .step b { display: block; margin-bottom: 4px; }
    .pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .8rem; font-weight: 600; }
    .pill.alert { background: #fde8e8; color: #b42323; } .pill.ok { background: #e7f6ec; color: #006300; }
    </style>
    """,
        unsafe_allow_html=True,
    )


def main() -> None:
    setup_page()
    pg = st.navigation([
        st.Page(page_overview, title="Overview", icon=":material/home:", default=True),
        st.Page(page_train, title="Train live", icon=":material/model_training:", url_path="train"),
        st.Page(page_detect, title="Detect attacks", icon=":material/radar:", url_path="detect"),
        st.Page(page_compare, title="Compare results", icon=":material/bar_chart:", url_path="compare"),
    ])
    with st.sidebar:
        st.markdown("### IoT Zero-Day Defense")
        st.caption("Federated learning · differential privacy · secure aggregation")
    pg.run()


if __name__ == "__main__":
    main()

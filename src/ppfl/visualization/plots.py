"""Publication-style figures (matplotlib + seaborn).

* :func:`experiment_figures` — figures for one experiment directory.
* :func:`comparison_figures` — cross-experiment figures from the summary table.

All figures are written as PNG (200 dpi) and every plotting function is a no-op
(returns ``None``) when the data it needs is absent, so partial experiments never
break report generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogFormatterSciNotation  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from ppfl.utils.logging import get_logger  # noqa: E402
from ppfl.utils.serialization import load_json  # noqa: E402

LOGGER = get_logger(__name__)

GROUP_COLORS = {"benign": "#1f77b4", "known_attack": "#ff7f0e", "unseen_attack": "#d62728"}
GROUP_NAMES = {"benign": "Benign", "known_attack": "Known attack", "unseen_attack": "Unseen (zero-day-like) attack"}
DPI = 200


def set_style() -> None:
    sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=1.25)
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "legend.frameon": True,
        }
    )


def _save(fig: plt.Figure, out: Path, name: str, synthetic: bool = False) -> Path:
    if synthetic:
        fig.text(0.99, -0.01, "SYNTHETIC DATA – pipeline test only", ha="right", va="top", fontsize=8, color="0.45", style="italic")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _has(df: pd.DataFrame | None, *cols: str) -> bool:
    return df is not None and not df.empty and all(c in df.columns and df[c].notna().any() for c in cols)


# ======================================================================================
# Per-experiment figures
# ======================================================================================


def plot_metric_vs_round(history: pd.DataFrame, columns: dict[str, str], title: str, ylabel: str, out: Path, name: str, xlabel: str, synthetic: bool, logy: bool = False) -> Path | None:
    cols = {c: lbl for c, lbl in columns.items() if _has(history, c)}
    if not cols:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for col, lbl in cols.items():
        d = history[["round", col]].dropna()
        ax.plot(d["round"], d[col], marker="o", markersize=3.5, linewidth=1.8, label=lbl)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if logy:
        ax.set_yscale("log")
    ax.legend()
    return _save(fig, out, name, synthetic)


def plot_group_confusion(report: dict[str, Any], out: Path, synthetic: bool) -> Path | None:
    """Rows: true traffic group; columns: predicted normal / anomaly."""
    pc = report.get("per_class", {})
    if not pc:
        return None
    groups = ["benign", "known_attack", "unseen_attack"]
    mat = np.zeros((3, 2))
    for info in pc.values():
        g = groups.index(info["group"])
        flagged = info["flagged_rate"] * info["n"]
        mat[g] += [info["n"] - flagged, flagged]
    keep = mat.sum(axis=1) > 0
    mat, labels = mat[keep], [GROUP_NAMES[g] for g, k in zip(groups, keep) if k]
    pct = mat / mat.sum(axis=1, keepdims=True)
    annot = np.array([[f"{int(round(c)):,}\n({p:.1%})" for c, p in zip(rc, rp)] for rc, rp in zip(mat, pct)])
    fig, ax = plt.subplots(figsize=(6.4, 1.4 + 1.1 * len(labels)))
    sns.heatmap(pct, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar_kws={"label": "Row fraction"},
                xticklabels=["Predicted normal", "Predicted anomaly"], yticklabels=labels, ax=ax)
    ax.set_title(f"Confusion matrix by traffic group (threshold = {report['threshold']:.4g})")
    ax.set_ylabel("True traffic group")
    return _save(fig, out, "confusion_matrix", synthetic)


def plot_binary_confusion(report: dict[str, Any], out: Path, synthetic: bool) -> Path | None:
    o = report.get("overall")
    if not o:
        return None
    mat = np.array([[o["tn"], o["fp"]], [o["fn"], o["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4.2))
    pct = mat / mat.sum(axis=1, keepdims=True)
    annot = np.array([[f"{int(c):,}\n({p:.1%})" for c, p in zip(rc, rp)] for rc, rp in zip(mat, pct)])
    sns.heatmap(pct, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False,
                xticklabels=["Normal", "Anomaly"], yticklabels=["Benign", "Attack"], ax=ax)
    ax.set(title="Confusion matrix (all test traffic)", xlabel="Predicted", ylabel="True")
    return _save(fig, out, "confusion_matrix_binary", synthetic)


def plot_score_distribution(scores: np.ndarray, groups: np.ndarray, threshold: float, out: Path, synthetic: bool, model: str = "autoencoder") -> Path | None:
    if len(scores) == 0:
        return None
    positive = scores[scores > 0]
    lo = np.percentile(positive, 0.1) if len(positive) else 1e-6
    hi = max(np.percentile(scores, 99.9), threshold * 1.5)
    use_log = model == "autoencoder" and lo > 0
    bins = np.logspace(np.log10(lo), np.log10(hi), 80) if use_log else np.linspace(scores.min(), hi, 80)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for g in ("benign", "known_attack", "unseen_attack"):
        s = scores[groups == g]
        if len(s):
            ax.hist(np.clip(s, bins[0], bins[-1]), bins=bins, alpha=0.5, density=True, color=GROUP_COLORS[g], label=f"{GROUP_NAMES[g]} (n={len(s):,})")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold:.3g}")
    if use_log:
        ax.set_xscale("log")
        if hi / lo < 1e3:  # few decades: label the minor ticks too
            ax.xaxis.set_minor_formatter(LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(3, 1)))
            ax.tick_params(axis="x", which="minor", labelsize=8)
    xl = "Reconstruction error (MSE)" if model == "autoencoder" else "Anomaly score"
    ax.set(title=f"Anomaly score distribution — {model.replace('_', ' ')}", xlabel=xl, ylabel="Density")
    ax.legend(loc="upper right")
    return _save(fig, out, f"score_distribution_{model}", synthetic)


def plot_scores_by_class(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray, threshold: float, out: Path, synthetic: bool) -> Path | None:
    if len(scores) == 0:
        return None
    df = pd.DataFrame({"score": np.maximum(scores, 1e-12), "class": labels, "group": groups})
    order = df.groupby("class")["group"].first().map({"benign": 0, "known_attack": 1, "unseen_attack": 2}).sort_values().index.tolist()
    palette = {c: GROUP_COLORS[df.loc[df["class"] == c, "group"].iloc[0]] for c in order}
    fig, ax = plt.subplots(figsize=(max(7, 0.65 * len(order) + 3), 4.6))
    sns.boxplot(data=df, x="class", y="score", hue="class", order=order, palette=palette, showfliers=False, legend=False, ax=ax)
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.3)
    ax.set_yscale("log")
    ax.set(title="Normal vs attack anomaly scores by class", xlabel="Traffic class", ylabel="Reconstruction error (log)")
    ax.tick_params(axis="x", rotation=40)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in GROUP_COLORS] + [plt.Line2D([], [], color="black", linestyle="--")]
    ax.legend(handles, [GROUP_NAMES[g] for g in GROUP_COLORS] + ["Threshold"], loc="upper left", fontsize=9)
    return _save(fig, out, "scores_by_class", synthetic)


def plot_roc(scores: np.ndarray, groups: np.ndarray, out: Path, synthetic: bool) -> Path | None:
    benign = groups == "benign"
    if benign.all() or not benign.any():
        return None
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for name, mask in (("All attacks", ~benign), ("Known attacks", groups == "known_attack"), ("Unseen attacks", groups == "unseen_attack")):
        if mask.any():
            sel = benign | mask
            fpr, tpr, _ = roc_curve(mask[sel].astype(int), scores[sel])
            ax.plot(fpr, tpr, linewidth=1.8, label=name)
    ax.plot([0, 1], [0, 1], color="0.6", linestyle=":")
    ax.set(title="ROC curves", xlabel="False positive rate", ylabel="True positive rate (recall)", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right")
    return _save(fig, out, "roc_curves", synthetic)


def plot_epsilon(history: pd.DataFrame, ledger: pd.DataFrame | None, out: Path, synthetic: bool) -> Path | None:
    if not _has(history, "epsilon_max"):
        return None
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if _has(ledger, "epsilon"):
        for i, (cid, g) in enumerate(ledger.groupby("client_id")):
            ax.plot(g["round"], g["epsilon"], color="0.7", linewidth=0.8, label="Individual clients" if i == 0 else None)
    ax.plot(history["round"], history["epsilon_max"], marker="o", markersize=3.5, linewidth=2, color="#d62728", label="Max over clients (reported ε)")
    ax.plot(history["round"], history["epsilon_mean"], linewidth=1.8, linestyle="--", color="#1f77b4", label="Mean over clients")
    ax.set(title="Privacy budget ε vs federated round", xlabel="Federated round", ylabel="Cumulative ε")
    ax.legend()
    return _save(fig, out, "epsilon_vs_round", synthetic)


def plot_client_distribution(table: pd.DataFrame, out: Path, synthetic: bool, unseen: list[str] | None = None) -> Path | None:
    if table is None or table.empty:
        return None
    t = table.set_index(table.columns[0]) if not np.issubdtype(table.iloc[:, 0].dtype, np.number) else table
    frac = t.div(t.sum(axis=1), axis=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    colors = sns.color_palette("tab20", n_colors=len(t.columns))
    frac.plot(kind="bar", stacked=True, ax=ax1, color=colors, width=0.85, legend=False)
    ax1.set(title="Class mix per client (non-IID)", xlabel="Client", ylabel="Fraction of local traffic")
    ax1.tick_params(axis="x", rotation=0)
    t.sum(axis=1).plot(kind="bar", ax=ax2, color="#4c72b0", width=0.8)
    ax2.set(title="Local data volume per client", xlabel="Client", ylabel="Rows")
    ax2.tick_params(axis="x", rotation=0)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    labels = [f"{c} (unseen)" if unseen and c in unseen else c for c in t.columns]
    fig.legend(ax1.get_legend_handles_labels()[0], labels, loc="center left", bbox_to_anchor=(1.0, 0.5), title="Class")
    for ax in (ax1, ax2):
        ax.set_xticklabels([lbl.get_text().replace("client_", "") for lbl in ax.get_xticklabels()])
    fig.tight_layout()
    return _save(fig, out, "client_data_distribution", synthetic)


def plot_per_client(per_client: pd.DataFrame, out: Path, synthetic: bool) -> Path | None:
    cols = [c for c in ("f1", "unseen_recall", "known_recall", "fpr") if _has(per_client, c)]
    if not cols:
        return None
    df = per_client.melt(id_vars="client_id", value_vars=cols, var_name="metric", value_name="value")
    df["metric"] = df["metric"].map({"f1": "F1", "unseen_recall": "Unseen-attack recall", "known_recall": "Known-attack recall", "fpr": "False positive rate"})
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * per_client["client_id"].nunique() + 3), 4.4))
    sns.barplot(data=df, x="client_id", y="value", hue="metric", ax=ax)
    ax.set(title="Per-client performance of the final model (client-local test data)", xlabel="Client", ylabel="Value", ylim=(0, 1.05))
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    return _save(fig, out, "per_client_performance", synthetic)


def plot_communication(history: pd.DataFrame, out: Path, synthetic: bool) -> Path | None:
    if not _has(history, "upload_bytes"):
        return None
    fig, ax = plt.subplots(figsize=(7, 4.2))
    kb = 1024.0
    ax.bar(history["round"], history["upload_bytes"] / kb, label="Upload (clients → server)", color="#4c72b0")
    ax.bar(history["round"], history["download_bytes"] / kb, bottom=history["upload_bytes"] / kb, label="Download (server → clients)", color="#55a868")
    if _has(history, "secagg_overhead_bytes") and history["secagg_overhead_bytes"].sum() > 0:
        ax.plot(history["round"], history["secagg_overhead_bytes"] / kb, color="#c44e52", marker="o", markersize=3, label="SecAgg protocol overhead")
    ax.set(title="Communication per federated round", xlabel="Federated round", ylabel="KiB")
    ax.legend()
    return _save(fig, out, "communication_per_round", synthetic)


def experiment_figures(exp_dir: Path) -> list[Path]:
    """Generate every per-experiment figure available for ``exp_dir``."""
    set_style()
    exp_dir = Path(exp_dir)
    out = exp_dir / "figures"
    metrics = load_json(exp_dir / "metrics.json")
    synthetic = bool(metrics["experiment"].get("synthetic_data"))
    mode = metrics["experiment"]["mode"]
    history = pd.read_csv(exp_dir / "metrics.csv") if (exp_dir / "metrics.csv").exists() else pd.DataFrame()
    xlabel = "Epoch" if mode == "centralized" else "Federated round"
    made: list[Path | None] = []
    if mode != "local":
        made.append(plot_metric_vs_round(history, {"train_loss": "Training loss (benign reconstruction MSE)"}, "Training loss", "MSE", out, "training_loss", xlabel, synthetic, logy=True))
        made.append(plot_metric_vs_round(history, {"val_loss": "Validation loss (benign)"}, "Validation reconstruction loss", "MSE", out, "validation_loss", xlabel, synthetic, logy=True))
        made.append(plot_metric_vs_round(history, {"f1": "F1 (all attacks)", "unseen_recall": "Unseen-attack recall", "known_recall": "Known-attack recall", "fpr": "False positive rate"},
                                         "Detection performance during training", "Value", out, "f1_vs_round", xlabel, synthetic))
    for name, model in metrics.get("models", {}).items():
        if model.get("report"):
            if name == "autoencoder":
                made.append(plot_group_confusion(model["report"], out, synthetic))
                made.append(plot_binary_confusion(model["report"], out, synthetic))
        npz = exp_dir / f"scores_{name}.npz"
        if npz.exists():
            d = np.load(npz, allow_pickle=False)
            made.append(plot_score_distribution(d["scores"], d["groups"], float(d["threshold"]), out, synthetic, name))
            if name == "autoencoder":
                made.append(plot_scores_by_class(d["scores"], d["labels"], d["groups"], float(d["threshold"]), out, synthetic))
                made.append(plot_roc(d["scores"], d["groups"], out, synthetic))
    ledger = pd.read_csv(exp_dir / "privacy_ledger.csv") if (exp_dir / "privacy_ledger.csv").exists() else None
    made.append(plot_epsilon(history, ledger, out, synthetic))
    made.append(plot_communication(history, out, synthetic))
    dist = exp_dir / "client_distribution.csv"
    if dist.exists():
        made.append(plot_client_distribution(pd.read_csv(dist), out, synthetic, metrics.get("setup", {}).get("unseen_attack_classes")))
    if (exp_dir / "per_client.csv").exists():
        made.append(plot_per_client(pd.read_csv(exp_dir / "per_client.csv"), out, synthetic))
    paths = [p for p in made if p is not None]
    LOGGER.info("Wrote %d figures to %s", len(paths), out)
    return paths


# ======================================================================================
# Cross-experiment figures
# ======================================================================================


def _bar(df: pd.DataFrame, value: str, title: str, ylabel: str, out: Path, name: str, synthetic: bool, highlight: bool = False, ylim: tuple[float, float] | None = (0, 1.05)) -> Path | None:
    if not _has(df, value):
        return None
    d = df.dropna(subset=[value]).sort_values(value, ascending=False)
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(d) + 3), 4.6))
    colors = ["#d62728" if highlight else "#4c72b0"] * len(d)
    bars = ax.bar(d["display"], d[value], color=colors, edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set(title=title, ylabel=ylabel, xlabel="")
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="x", rotation=35)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    return _save(fig, out, name, synthetic)


def plot_mode_comparison(df: pd.DataFrame, out: Path, synthetic: bool) -> Path | None:
    metrics = {"f1": "F1", "precision": "Precision", "recall": "Recall", "unseen_recall": "Unseen-attack recall", "roc_auc": "ROC-AUC"}
    cols = [m for m in metrics if _has(df, m)]
    if not cols or df.empty:
        return None
    long = df.melt(id_vars="display", value_vars=cols, var_name="metric", value_name="value")
    long["metric"] = long["metric"].map(metrics)
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * df["display"].nunique() + 3), 4.8))
    sns.barplot(data=long, x="display", y="value", hue="metric", order=_mode_order(df), ax=ax, edgecolor="black", linewidth=0.4)
    ax.set(title="Centralized vs federated vs privacy-preserving FL", xlabel="", ylabel="Value", ylim=(0, 1.05))
    ax.tick_params(axis="x", rotation=25)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=5, fontsize=9, frameon=False)
    return _save(fig, out, "mode_comparison", synthetic)


_MODE_RANK = ("Centralized", "Local-only", "Plain FL", "FedProx", "FL + DP", "FedProx + DP", "FL + SecAgg", "FedProx + SecAgg", "Full PPFL (FedAvg)", "Full PPFL (FedProx)")


def _mode_order(df: pd.DataFrame) -> list[str]:
    """Display order: centralized -> local -> plain FL -> +DP -> +SecAgg -> full PPFL."""
    def rank(label: str) -> int:
        return next((i for i, p in enumerate(_MODE_RANK) if label.startswith(p)), len(_MODE_RANK))
    return sorted(df["display"].unique(), key=lambda d: (rank(d), d))


def plot_f1_vs_epsilon(df: pd.DataFrame, out: Path, synthetic: bool) -> Path | None:
    dp = df[df["epsilon"].notna()] if "epsilon" in df else pd.DataFrame()
    if len(dp) < 2:
        return None
    dp = dp.sort_values("epsilon")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(dp["epsilon"], dp["f1"], marker="o", linewidth=1.8, label="F1 (DP-FL)")
    ax.plot(dp["epsilon"], dp["unseen_recall"], marker="s", linewidth=1.8, label="Unseen-attack recall (DP-FL)")
    nodp = df[(df["mode"] == "federated") & df["epsilon"].isna() & ~df["secagg"].fillna(False).astype(bool)]
    if not nodp.empty:
        ax.axhline(nodp["f1"].iloc[0], color="0.4", linestyle="--", label="F1 without DP (plain FL)")
    ax.set_xscale("log")
    ax.set(title="Privacy / utility trade-off", xlabel="ε (log scale, δ fixed)", ylabel="Value", ylim=(0, 1.05))
    ax.legend()
    return _save(fig, out, "f1_vs_epsilon", synthetic)


def plot_recall_vs_noise(df: pd.DataFrame, out: Path, synthetic: bool) -> Path | None:
    dp = df[df["noise_multiplier"].notna()] if "noise_multiplier" in df else pd.DataFrame()
    if dp["noise_multiplier"].nunique() < 2 if len(dp) else True:
        return None
    dp = dp.sort_values("noise_multiplier")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(dp["noise_multiplier"], dp["unseen_recall"], marker="o", linewidth=1.8, color="#d62728", label="Unseen-attack recall")
    ax.plot(dp["noise_multiplier"], dp["fpr"], marker="s", linewidth=1.8, color="0.4", label="False positive rate")
    ax2 = ax.twinx()
    ax2.plot(dp["noise_multiplier"], dp["epsilon"], marker="^", linewidth=1.2, linestyle=":", color="#1f77b4", label="ε")
    ax2.set_ylabel("ε", color="#1f77b4")
    ax2.grid(False)
    ax.set(title="Zero-day recall vs privacy level", xlabel="Noise multiplier σ", ylabel="Rate", ylim=(0, 1.05))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")
    return _save(fig, out, "zero_day_recall_vs_privacy", synthetic)


def plot_communication_comparison(df: pd.DataFrame, out: Path, synthetic: bool) -> Path | None:
    if not _has(df, "total_mb"):
        return None
    d = df[df["total_mb"] > 0].sort_values("total_mb")
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(d) + 3), 4.6))
    colors = ["#8172b3" if m == "centralized" else "#4c72b0" for m in d["mode"]]
    bars = ax.bar(d["display"], d["total_mb"], color=colors, edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_yscale("log")
    ax.set(title="Total communication (centralized = raw data upload)", ylabel="MiB (log scale)")
    ax.tick_params(axis="x", rotation=35)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    return _save(fig, out, "communication_comparison", synthetic)


def plot_f1_vs_communication(df: pd.DataFrame, out: Path, synthetic: bool) -> Path | None:
    d = df[(df["mode"] == "federated") & df["total_mb"].notna()] if "total_mb" in df else pd.DataFrame()
    if len(d) < 2:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.6))
    sns.scatterplot(data=d, x="total_mb", y="f1", hue="display", s=80, ax=ax, edgecolor="black")
    ax.set(title="Detection quality vs communication cost", xlabel="Total communication (MiB)", ylabel="F1")
    ax.legend(fontsize=8, loc="best")
    return _save(fig, out, "f1_vs_communication", synthetic)


def plot_convergence_overlay(histories: dict[str, pd.DataFrame], column: str, ylabel: str, title: str, out: Path, name: str, synthetic: bool, logy: bool = False) -> Path | None:
    curves = {k: h for k, h in histories.items() if _has(h, column)}
    if not curves:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, h in curves.items():
        d = h[["round", column]].dropna()
        ax.plot(d["round"], d[column], marker="o", markersize=3, linewidth=1.6, label=label)
    if logy:
        ax.set_yscale("log")
    ax.set(title=title, xlabel="Round / epoch", ylabel=ylabel)
    ax.legend(fontsize=8)
    return _save(fig, out, name, synthetic)


def plot_sweep(df: pd.DataFrame, x: str, xlabel: str, out: Path, name: str, title: str, synthetic: bool, logx: bool = False) -> Path | None:
    d = df.dropna(subset=[x]) if x in df else pd.DataFrame()
    if d.empty or d[x].nunique() < 2:
        return None
    d = d.sort_values(x)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for col, lbl in (("f1", "F1"), ("unseen_recall", "Unseen-attack recall"), ("fpr", "FPR")):
        ax.plot(d[x], d[col], marker="o", linewidth=1.8, label=lbl)
    if logx:
        ax.set_xscale("log")
    ax.set(title=title, xlabel=xlabel, ylabel="Value", ylim=(0, 1.05))
    if "train_time_s" in d and d["train_time_s"].notna().any():
        ax2 = ax.twinx()
        ax2.plot(d[x], d["train_time_s"], color="0.5", linestyle=":", marker="^", label="Training time (s)")
        ax2.set_ylabel("Training time (s)")
        ax2.grid(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="best")
    else:
        ax.legend()
    return _save(fig, out, name, synthetic)


def comparison_figures(summary: pd.DataFrame, histories: dict[str, pd.DataFrame], out: Path, groups: dict[str, list[str]] | None = None) -> list[Path]:
    """Cross-experiment figures. ``groups`` maps suite name -> experiment names (optional)."""
    set_style()
    if summary.empty:
        return []
    synthetic = bool(summary["synthetic"].any())
    ae = summary[summary["model"] == "autoencoder"].copy()
    main_names = (groups or {}).get("main") or ae["experiment"].tolist()
    main = summary[summary["experiment"].isin(main_names)].copy()
    for frame in (main,):
        dup = frame["label"].duplicated(keep=False)
        frame["display"] = frame["label"].where(~dup, frame["display"])
    main_ae = main[main["model"] == "autoencoder"]
    privacy_names = (groups or {}).get("privacy")
    dp_sweep = ae[ae["experiment"].isin(privacy_names)] if privacy_names else ae
    made = [
        _bar(main, "unseen_recall", "Zero-day (unseen attack) recall", "Unseen-attack recall", out, "zero_day_recall_comparison", synthetic, highlight=True),
        _bar(main, "fpr", "False positive rate on benign test traffic", "FPR", out, "fpr_comparison", synthetic, ylim=None),
        _bar(main, "f1", "F1 score (all test traffic)", "F1", out, "f1_comparison", synthetic),
        plot_mode_comparison(main_ae, out, synthetic),
        plot_communication_comparison(main_ae, out, synthetic),
        plot_f1_vs_epsilon(pd.concat([dp_sweep, main_ae[(main_ae["mode"] == "federated") & ~main_ae["dp"].astype(bool)]]), out, synthetic),
        plot_recall_vs_noise(dp_sweep, out, synthetic),
        plot_f1_vs_communication(ae, out, synthetic),
        plot_convergence_overlay({main_ae.set_index("experiment").loc[n, "display"]: histories[n] for n in main_ae["experiment"] if n in histories},
                                 "f1", "F1", "F1 vs round / epoch", out, "f1_convergence_overlay", synthetic),
        plot_convergence_overlay({main_ae.set_index("experiment").loc[n, "display"]: histories[n] for n in main_ae["experiment"] if n in histories},
                                 "val_loss", "Validation MSE", "Validation reconstruction loss vs round / epoch", out, "val_loss_overlay", synthetic, logy=True),
        plot_convergence_overlay({ae.set_index("experiment").loc[n, "display"]: histories[n] for n in dp_sweep["experiment"] if n in histories},
                                 "epsilon_max", "ε (max over clients)", "Privacy budget ε vs round", out, "epsilon_overlay", synthetic),
    ]
    for suite, x, xlabel, title, logx in (
        ("scalability", "num_clients", "Number of clients", "Effect of the number of clients", False),
        ("heterogeneity", "dirichlet_alpha", "Dirichlet α (lower = more non-IID)", "Effect of client heterogeneity", True),
        ("local_epochs", "local_epochs", "Local epochs per round", "Effect of local epochs", False),
    ):
        names = (groups or {}).get(suite)
        if names:
            made.append(plot_sweep(ae[ae["experiment"].isin(names)], x, xlabel, out, f"sweep_{suite}", title, synthetic, logx))
    paths = [p for p in made if p is not None]
    LOGGER.info("Wrote %d comparison figures to %s", len(paths), out)
    return paths

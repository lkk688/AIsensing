from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def otfs_seen_avg(per_config: Dict[str, float]) -> float:
    keys = [k for k in per_config.keys() if "OTFS" in k]
    return float(np.mean([per_config[k] for k in keys]))


def save_figure(fig: plt.Figure, out_png_path: Path, save_png: bool, save_pdf: bool, dpi: int = 220, bbox_inches: str | None = None) -> List[str]:
    files: List[str] = []
    if save_png:
        fig.savefig(out_png_path, dpi=dpi, bbox_inches=bbox_inches)
        files.append(out_png_path.name)
    if save_pdf:
        out_pdf = out_png_path.with_suffix(".pdf")
        fig.savefig(out_pdf, bbox_inches=bbox_inches)
        files.append(out_pdf.name)
    return files


def non_otfs_seen_avg(per_config: Dict[str, float]) -> float:
    keys = [k for k in per_config.keys() if "OTFS" not in k]
    return float(np.mean([per_config[k] for k in keys]))


def extract_metrics(name: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    history = summary["history"]
    val_arr = np.array([e["val_seen_ber"] for e in history], dtype=float)
    best_idx = int(val_arr.argmin())
    best = history[best_idx]
    last = history[-1]
    seen = summary["seen"]
    unseen = summary["unseen"]
    return {
        "name": name,
        "epochs": len(history),
        "best_epoch": int(best["epoch"]),
        "best_val_seen_ber": float(best["val_seen_ber"]),
        "last_val_seen_ber": float(last["val_seen_ber"]),
        "best_unseen_epoch": float(best.get("unseen_ber_epoch")) if best.get("unseen_ber_epoch") is not None else None,
        "last_unseen_epoch": float(last.get("unseen_ber_epoch")) if last.get("unseen_ber_epoch") is not None else None,
        "seen_avg": float(seen["avg_ber"]),
        "unseen_avg": float(unseen["avg_ber"]),
        "seen_per_config": seen["per_config"],
        "unseen_per_config": unseen["per_config"],
        "selection_best": float(min([e.get("selection_score", 1e9) for e in history])),
        "selection_last": float(last.get("selection_score", 1e9)),
        "train_loss_start": float(history[0]["train_loss"]),
        "train_loss_end": float(history[-1]["train_loss"]),
        "train_ber_start": float(history[0]["train_ber"]),
        "train_ber_end": float(history[-1]["train_ber"]),
        "otfs_seen_avg": otfs_seen_avg(seen["per_config"]),
        "non_otfs_seen_avg": non_otfs_seen_avg(seen["per_config"]),
        "snr_sweeps": summary["snr_sweeps"],
        "history": history,
    }


def plot_seen_unseen_bar(metrics: Dict[str, Dict[str, Any]], out: Path, save_png: bool, save_pdf: bool) -> List[str]:
    names = list(metrics.keys())
    seen = [metrics[n]["seen_avg"] for n in names]
    unseen = [metrics[n]["unseen_avg"] for n in names]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, seen, width=width, label="Seen Avg BER")
    ax.bar(x + width / 2, unseen, width=width, label="Unseen Avg BER")
    ax.set_xticks(x, names)
    ax.set_ylabel("BER")
    ax.set_title("Seen vs Unseen BER Across Runs")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    files = save_figure(fig, out, save_png=save_png, save_pdf=save_pdf, dpi=220)
    plt.close(fig)
    return files


def plot_otfs_nonotfs_bar(metrics: Dict[str, Dict[str, Any]], out: Path, save_png: bool, save_pdf: bool) -> List[str]:
    names = list(metrics.keys())
    otfs = [metrics[n]["otfs_seen_avg"] for n in names]
    non_otfs = [metrics[n]["non_otfs_seen_avg"] for n in names]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, otfs, width=width, label="Seen OTFS Avg BER")
    ax.bar(x + width / 2, non_otfs, width=width, label="Seen Non-OTFS Avg BER")
    ax.set_xticks(x, names)
    ax.set_ylabel("BER")
    ax.set_title("OTFS vs Non-OTFS Seen BER")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    files = save_figure(fig, out, save_png=save_png, save_pdf=save_pdf, dpi=220)
    plt.close(fig)
    return files


def plot_pareto(metrics: Dict[str, Dict[str, Any]], out: Path, save_png: bool, save_pdf: bool) -> List[str]:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for name, m in metrics.items():
        x = m["seen_avg"]
        y = m["unseen_avg"]
        ax.scatter(x, y, s=85)
        ax.text(x + 0.00012, y + 0.00012, name, fontsize=9)
    ax.set_xlabel("Seen Avg BER")
    ax.set_ylabel("Unseen Avg BER")
    ax.set_title("Seen-Unseen Tradeoff (Lower is Better)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    files = save_figure(fig, out, save_png=save_png, save_pdf=save_pdf, dpi=220)
    plt.close(fig)
    return files


def plot_training_curves(metrics: Dict[str, Dict[str, Any]], out: Path, save_png: bool, save_pdf: bool) -> List[str]:
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    for name, m in metrics.items():
        hist = m["history"]
        epochs = [e["epoch"] for e in hist]
        val = [e["val_seen_ber"] for e in hist]
        sel = [e.get("selection_score", np.nan) for e in hist]
        axs[0].plot(epochs, val, label=name, linewidth=1.8)
        axs[1].plot(epochs, sel, label=name, linewidth=1.8)
    axs[0].set_title("Validation Seen BER by Epoch")
    axs[1].set_title("Selection Score by Epoch")
    axs[0].set_xlabel("Epoch")
    axs[1].set_xlabel("Epoch")
    axs[0].set_ylabel("BER")
    axs[1].set_ylabel("Score")
    axs[0].grid(alpha=0.25)
    axs[1].grid(alpha=0.25)
    axs[0].legend(fontsize=8)
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    files = save_figure(fig, out, save_png=save_png, save_pdf=save_pdf, dpi=220)
    plt.close(fig)
    return files


def plot_per_config_heatmaps(metrics: Dict[str, Dict[str, Any]], out_prefix: Path, save_png: bool, save_pdf: bool) -> List[str]:
    run_names = list(metrics.keys())
    seen_cfgs = list(next(iter(metrics.values()))["seen_per_config"].keys())
    unseen_cfgs = list(next(iter(metrics.values()))["unseen_per_config"].keys())

    seen_mat = np.array([[metrics[r]["seen_per_config"][c] for c in seen_cfgs] for r in run_names], dtype=float)
    unseen_mat = np.array([[metrics[r]["unseen_per_config"][c] for c in unseen_cfgs] for r in run_names], dtype=float)

    def draw(mat: np.ndarray, xlabels: List[str], ylabels: List[str], title: str, out_path: Path) -> List[str]:
        fig, ax = plt.subplots(figsize=(10.5, 3.8))
        im = ax.imshow(mat, aspect="auto")
        ax.set_xticks(np.arange(len(xlabels)), xlabels, rotation=20, ha="right")
        ax.set_yticks(np.arange(len(ylabels)), ylabels)
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8, color="white" if mat[i, j] > np.mean(mat) else "black")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("BER")
        fig.tight_layout()
        files = save_figure(fig, out_path, save_png=save_png, save_pdf=save_pdf, dpi=220)
        plt.close(fig)
        return files

    files = []
    files.extend(draw(seen_mat, seen_cfgs, run_names, "Seen BER Per Config", out_prefix.with_name(out_prefix.name + "_seen.png")))
    files.extend(draw(unseen_mat, unseen_cfgs, run_names, "Unseen BER Per Config", out_prefix.with_name(out_prefix.name + "_unseen.png")))
    return files


def plot_snr_realistic_key_configs(metrics: Dict[str, Dict[str, Any]], out: Path, save_png: bool, save_pdf: bool) -> List[str]:
    configs = ["AUTOMOTIVE_OTFS_ISAC", "CN0566_OTFS_ISAC", "5G_ISAC_HighBandwidth", "AUTOMOTIVE_TRADITIONAL"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    for idx, cfg in enumerate(configs):
        ax = axs[idx]
        for run, m in metrics.items():
            rec = m["snr_sweeps"][f"{cfg}_realistic"]
            ax.plot(rec["snr"], rec["dl_ber"], marker="o", linewidth=1.8, markersize=3.5, label=run)
        ax.set_title(cfg)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("DL BER")
        ax.grid(alpha=0.25)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=len(metrics), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    files = save_figure(fig, out, save_png=save_png, save_pdf=save_pdf, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return files


def plot_snr_gain_vs_baseline(metrics: Dict[str, Dict[str, Any]], baseline: str, out: Path, save_png: bool, save_pdf: bool) -> List[str]:
    runs = [r for r in metrics.keys() if r != baseline]
    configs = ["AUTOMOTIVE_OTFS_ISAC", "CN0566_OTFS_ISAC", "5G_ISAC_HighBandwidth", "AUTOMOTIVE_TRADITIONAL"]
    x = np.arange(len(configs))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, run in enumerate(runs):
        gains = []
        for c in configs:
            b = metrics[baseline]["snr_sweeps"][f"{c}_realistic"]["dl_ber"][-1]
            cur = metrics[run]["snr_sweeps"][f"{c}_realistic"]["dl_ber"][-1]
            gains.append(cur - b)
        ax.bar(x + (i - (len(runs) - 1) / 2) * width, gains, width=width, label=f"{run} - {baseline}")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x, configs, rotation=20, ha="right")
    ax.set_ylabel("Δ BER at SNR=30dB")
    ax.set_title("High-SNR BER Change vs Baseline")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    files = save_figure(fig, out, save_png=save_png, save_pdf=save_pdf, dpi=220)
    plt.close(fig)
    return files


def plot_radar_seen_f1_bar(radar_summary: Dict[str, Any], out: Path, save_png: bool, save_pdf: bool) -> List[str]:
    per_cfg = radar_summary["seen"]["per_config"]
    cfgs = list(per_cfg.keys())
    dl = [per_cfg[c]["f1"] for c in cfgs]
    cfar = []
    for c in cfgs:
        rec = radar_summary["snr_sweeps"].get(c)
        if rec and "cfar_f1" in rec and len(rec["cfar_f1"]) > 0:
            cfar.append(float(np.mean(rec["cfar_f1"])))
        else:
            cfar.append(np.nan)
    x = np.arange(len(cfgs))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - width / 2, dl, width=width, label="DL F1 (seen eval)")
    ax.bar(x + width / 2, cfar, width=width, label="CFAR F1 (mean SNR sweep)")
    ax.set_xticks(x, cfgs, rotation=20, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Radar Per-Config F1: DL vs CFAR")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    files = save_figure(fig, out, save_png=save_png, save_pdf=save_pdf, dpi=220)
    plt.close(fig)
    return files


def plot_radar_sweep_all_configs(radar_summary: Dict[str, Any], sweep_key: str, out: Path, save_png: bool, save_pdf: bool) -> List[str]:
    sweeps = radar_summary[sweep_key]
    cfgs = list(sweeps.keys())
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    for i, cfg in enumerate(cfgs):
        ax = axs[i]
        rec = sweeps[cfg]
        x = rec["x"]
        dl = np.array(rec["dl_f1"], dtype=float)
        cfar = np.array(rec["cfar_f1"], dtype=float)
        ax.plot(
            x,
            dl,
            marker="o",
            linewidth=1.9,
            markersize=4.0,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label="DL",
            zorder=3,
        )
        ax.plot(
            x,
            cfar,
            marker="s",
            linewidth=1.7,
            linestyle="--",
            markersize=3.8,
            label="CFAR",
            zorder=2,
        )
        ax.set_title(cfg)
        ax.set_xlabel(sweep_key.replace("_sweeps", "").upper())
        ax.set_ylabel("F1")
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.25)
        if np.allclose(dl, 0.0):
            ax.text(0.02, 0.03, "DL=0 across sweep", transform=ax.transAxes, fontsize=8, color="crimson")
        elif np.allclose(dl, cfar):
            ax.text(0.02, 0.03, "DL overlaps CFAR", transform=ax.transAxes, fontsize=8, color="dimgray")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    files = save_figure(fig, out, save_png=save_png, save_pdf=save_pdf, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return files


def plot_radar_delta_heatmap(radar_summary: Dict[str, Any], out: Path, save_png: bool, save_pdf: bool) -> List[str]:
    cfgs = list(radar_summary["snr_sweeps"].keys())
    sweep_names = ["snr_sweeps", "cnr_sweeps", "rcs_sweeps"]
    mat = np.zeros((len(cfgs), len(sweep_names)), dtype=float)
    for i, cfg in enumerate(cfgs):
        for j, sk in enumerate(sweep_names):
            rec = radar_summary[sk][cfg]
            dl = np.array(rec["dl_f1"], dtype=float)
            cf = np.array(rec["cfar_f1"], dtype=float)
            mat[i, j] = float((dl - cf).mean())
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(np.arange(len(sweep_names)), ["SNR", "CNR", "RCS"])
    ax.set_yticks(np.arange(len(cfgs)), cfgs)
    ax.set_title("Radar Mean ΔF1 (DL - CFAR)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.3f}", ha="center", va="center", fontsize=9, color="white" if abs(mat[i, j]) > np.mean(np.abs(mat)) else "black")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean ΔF1")
    fig.tight_layout()
    files = save_figure(fig, out, save_png=save_png, save_pdf=save_pdf, dpi=220)
    plt.close(fig)
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="/Developer/AIsensing/data/comm_paper_figs")
    parser.add_argument("--runs", nargs="+", default=[
        "g6_comm2=/Developer/AIsensing/data/g6_comm2/comm_debug/summary.json",
        "g6_comm3=/Developer/AIsensing/data/g6_comm3/comm_debug/summary.json",
        "g6_comm4=/Developer/AIsensing/data/g6_comm4/comm_debug/summary.json",
        "g6_comm5=/Developer/AIsensing/data/g6_comm5/comm_debug/summary.json",
    ])
    parser.add_argument("--baseline", default="g6_comm2")
    parser.add_argument("--radar_summary", default="/Developer/AIsensing/data/g6_comm3/radar/summary.json")
    parser.add_argument("--include_radar", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_pdf", action="store_true")
    parser.add_argument("--save_png", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    summaries: Dict[str, Dict[str, Any]] = {}
    for item in args.runs:
        name, path = item.split("=", 1)
        summaries[name] = load_summary(Path(path))

    metrics = {name: extract_metrics(name, s) for name, s in summaries.items()}

    files: List[str] = []
    files.extend(plot_seen_unseen_bar(metrics, out_dir / "fig_seen_unseen_bar.png", save_png=args.save_png, save_pdf=args.save_pdf))
    files.extend(plot_otfs_nonotfs_bar(metrics, out_dir / "fig_otfs_nonotfs_bar.png", save_png=args.save_png, save_pdf=args.save_pdf))
    files.extend(plot_pareto(metrics, out_dir / "fig_seen_unseen_pareto.png", save_png=args.save_png, save_pdf=args.save_pdf))
    files.extend(plot_training_curves(metrics, out_dir / "fig_training_curves.png", save_png=args.save_png, save_pdf=args.save_pdf))
    files.extend(plot_per_config_heatmaps(metrics, out_dir / "fig_per_config_heatmap", save_png=args.save_png, save_pdf=args.save_pdf))
    files.extend(plot_snr_realistic_key_configs(metrics, out_dir / "fig_snr_realistic_key_configs.png", save_png=args.save_png, save_pdf=args.save_pdf))
    files.extend(plot_snr_gain_vs_baseline(metrics, args.baseline, out_dir / "fig_snr30_delta_vs_baseline.png", save_png=args.save_png, save_pdf=args.save_pdf))

    radar_manifest: Dict[str, Any] = {"included": False}
    if args.include_radar:
        radar_path = Path(args.radar_summary)
        if radar_path.exists():
            radar_summary = load_summary(radar_path)
            files.extend(plot_radar_seen_f1_bar(radar_summary, out_dir / "fig_radar_seen_f1_bar.png", save_png=args.save_png, save_pdf=args.save_pdf))
            files.extend(plot_radar_sweep_all_configs(radar_summary, "snr_sweeps", out_dir / "fig_radar_snr_sweeps.png", save_png=args.save_png, save_pdf=args.save_pdf))
            files.extend(plot_radar_sweep_all_configs(radar_summary, "cnr_sweeps", out_dir / "fig_radar_cnr_sweeps.png", save_png=args.save_png, save_pdf=args.save_pdf))
            files.extend(plot_radar_sweep_all_configs(radar_summary, "rcs_sweeps", out_dir / "fig_radar_rcs_sweeps.png", save_png=args.save_png, save_pdf=args.save_pdf))
            files.extend(plot_radar_delta_heatmap(radar_summary, out_dir / "fig_radar_delta_heatmap.png", save_png=args.save_png, save_pdf=args.save_pdf))
            radar_manifest = {
                "included": True,
                "summary_path": str(radar_path),
                "seen_global": radar_summary.get("seen", {}).get("global", {}),
                "unseen_global": radar_summary.get("unseen", {}).get("global", {}),
            }
        else:
            radar_manifest = {"included": False, "summary_path": str(radar_path), "error": "radar_summary_not_found"}

    manifest = {
        "runs": {k: {"seen_avg": v["seen_avg"], "unseen_avg": v["unseen_avg"], "best_epoch": v["best_epoch"]} for k, v in metrics.items()},
        "baseline": args.baseline,
        "radar": radar_manifest,
        "save_png": args.save_png,
        "save_pdf": args.save_pdf,
        "figures": files,
    }
    with (out_dir / "metrics_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

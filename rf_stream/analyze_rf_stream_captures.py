#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_rf_stream_captures.py  –  Offline analysis for RF-stream Step 6 captures.

Runs fully offline; safe to call while RX is still running (reads CSV, JSON, NPZ).

USAGE
─────
# 1. Single-run diagnostic grid (works with step5 and step6 captures.csv):
python3 analyze_rf_stream_captures.py \\
    --csv ./rf_stream_rx_runs/run_*/captures.csv

# 2. Single-run full dashboard (step6, includes constellation + EVM histogram):
python3 analyze_rf_stream_captures.py \\
    --run_dir ./rf_stream_rx_runs/run_20260427_120000

# 3. BER curves from multiple run_summary.json files (one per modulation / gain setting):
python3 analyze_rf_stream_captures.py \\
    --ber_dirs ./ber_sweep/qpsk/run_* ./ber_sweep/qam16/run_*

# 4. Research-paper BER figure only:
python3 analyze_rf_stream_captures.py \\
    --ber_dirs ./ber_sweep/*/run_* --ber_only --out ber_curves.pdf
"""

import argparse
import json
import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family":  "serif",
    "font.size":    10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi":   150,
    "axes.grid":    True,
    "grid.alpha":   0.4,
})

MOD_COLORS = {
    "bpsk":  "#1f77b4",
    "qpsk":  "#ff7f0e",
    "qam8":  "#2ca02c",
    "qam16": "#d62728",
    "qam32": "#9467bd",
}
MOD_MARKERS = {"bpsk": "o", "qpsk": "s", "qam8": "^", "qam16": "D", "qam32": "v"}
MOD_LABELS  = {"bpsk": "BPSK", "qpsk": "QPSK", "qam8": "8-PSK",
               "qam16": "16-QAM", "qam32": "32-QAM"}


# ═══════════════════════════════════════════════════════════════════════════════
# CSV helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def _pick(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

def load_csv(path: str, max_rows: int = 0) -> pd.DataFrame:
    nrows = max_rows if max_rows > 0 else None
    try:
        df = pd.read_csv(path, nrows=nrows, on_bad_lines="skip")
    except TypeError:
        df = pd.read_csv(path, nrows=nrows, error_bad_lines=False)
    for col in ["peak", "p10", "eg_th", "maxe", "xc_best_peak", "cfo_hz",
                "probe_evm", "snr_db", "ber", "n_bits", "n_bit_errors", "rx_gain"]:
        if col in df.columns:
            df[col] = _safe_num(df[col])
    if "cap" not in df.columns:
        df["cap"] = np.arange(1, len(df) + 1)
    return df


def load_run_summary(run_dir: str) -> dict:
    path = os.path.join(run_dir, "run_summary.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_npz_first(run_dir: str) -> dict:
    """Load first available NPZ file for constellation diagram."""
    for fn in sorted(glob.glob(os.path.join(run_dir, "cap_*_ok.npz")))[:5]:
        try:
            d = np.load(fn, allow_pickle=True)
            return d
        except Exception:
            pass
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Single-run diagnostic grid (adapted from step5 analyzer + step6 columns)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_single_run(df: pd.DataFrame, run_dir: str, summary: dict, out_path: str, dpi: int = 150):
    """4×4 diagnostic grid for one run."""
    capcol  = "cap"
    statcol = _pick(df, ["status"])
    rscol   = _pick(df, ["reason"])
    peakcol = _pick(df, ["peak"])
    p10col  = _pick(df, ["p10"])
    egcol   = _pick(df, ["eg_th"])
    maxecol = _pick(df, ["maxe"])
    xcpcol  = _pick(df, ["xc_best_peak", "xc_peak"])
    cfocol  = _pick(df, ["cfo_hz"])
    evmcol  = _pick(df, ["probe_evm"])
    snrcol  = _pick(df, ["snr_db"])
    bercol  = _pick(df, ["ber"])
    gaincol = _pick(df, ["rx_gain"])

    if statcol:
        sv        = df[statcol].astype(str).fillna("NA")
        uniq      = sorted(sv.unique())
        s2i       = {s: i for i, s in enumerate(uniq)}
        c_int     = sv.map(s2i).astype(int)
        c_arr     = c_int.values
    else:
        sv = pd.Series(["NA"] * len(df))
        c_arr = np.zeros(len(df), dtype=int)
        uniq  = ["NA"]

    if maxecol and egcol:
        df["gate_ratio"]  = df[maxecol] / (df[egcol] + 1e-12)
        df["gate_margin"] = df[maxecol] - df[egcol]

    fig, axes = plt.subplots(4, 4, figsize=(22, 16))
    fig.suptitle(f"Step6 RX Diagnostics  –  {os.path.basename(run_dir)}",
                 fontsize=13, fontweight="bold")

    # ── row 0 ──────────────────────────────────────────────────────────────────
    # (0,0) status counts
    ax = axes[0, 0]
    vc = sv.value_counts().head(15)
    ax.bar(vc.index.astype(str), vc.values, color="#4878cf", edgecolor="k", linewidth=0.5)
    ax.set_title("Status counts"); ax.tick_params(axis="x", rotation=35)

    # (0,1) reason counts
    ax = axes[0, 1]
    if rscol:
        rv = df[rscol].astype(str).fillna("NA").value_counts().head(15)
        ax.bar(rv.index.astype(str), rv.values, color="#6acc65", edgecolor="k", linewidth=0.5)
        ax.tick_params(axis="x", rotation=35)
    ax.set_title("Reason counts")

    # (0,2) peak vs cap
    ax = axes[0, 2]
    if peakcol:
        ax.plot(df[capcol], df[peakcol], lw=0.7, color="#4878cf")
    ax.set_title("RX peak amplitude"); ax.set_xlabel("Capture #")

    # (0,3) energy terms
    ax = axes[0, 3]
    for col, label, clr in [(p10col, "p10", "blue"), (egcol, "eg_th", "orange"),
                             (maxecol, "maxe", "green")]:
        if col:
            ax.plot(df[capcol], df[col], lw=0.7, label=label, color=clr)
    ax.set_title("Energy terms vs cap"); ax.set_xlabel("Capture #"); ax.legend(fontsize=7)

    # ── row 1 ──────────────────────────────────────────────────────────────────
    # (1,0) gate_ratio hist
    ax = axes[1, 0]
    if "gate_ratio" in df.columns:
        gr = df["gate_ratio"].dropna()
        if not gr.empty:
            clip = float(np.nanpercentile(gr, 99))
            ax.hist(gr.clip(0, clip), bins=60, color="#4878cf", edgecolor="k", linewidth=0.2)
    ax.set_title("gate_ratio = maxe/eg_th"); ax.set_xlabel("Ratio")

    # (1,1) xcorr peak over time
    ax = axes[1, 1]
    if xcpcol:
        ax.plot(df[capcol], df[xcpcol], lw=0.7, color="#d95f02")
        ax.axhline(0.2, color="red", lw=1, ls="--", label="min_peak=0.2")
        ax.legend(fontsize=7)
    ax.set_title("XCorr STF peak"); ax.set_xlabel("Capture #")

    # (1,2) CFO over time
    ax = axes[1, 2]
    if cfocol:
        cfo_ok = df.loc[df[statcol] == "ok", cfocol].dropna() if statcol else df[cfocol].dropna()
        if not cfo_ok.empty:
            ax.plot(cfo_ok.index, cfo_ok.values, ".", ms=3, color="#1b9e77")
            ax.axhline(cfo_ok.mean(), color="red", lw=1, ls="--",
                       label=f"μ={cfo_ok.mean():.0f} Hz")
            ax.legend(fontsize=7)
    ax.set_title("CFO estimate (OK packets)"); ax.set_ylabel("Hz"); ax.set_xlabel("Capture #")

    # (1,3) SNR per-packet
    ax = axes[1, 3]
    if snrcol:
        snr_ok = df.loc[df[statcol] == "ok", snrcol].dropna() if statcol else df[snrcol].dropna()
        if not snr_ok.empty:
            ax.plot(snr_ok.index, snr_ok.values, ".", ms=3, color="#7570b3")
            ax.axhline(snr_ok.mean(), color="red", lw=1, ls="--",
                       label=f"μ={snr_ok.mean():.1f} dB")
            ax.legend(fontsize=7)
    ax.set_title("Mean subcarrier SNR (OK)"); ax.set_ylabel("dB")

    # ── row 2 ──────────────────────────────────────────────────────────────────
    # (2,0) EVM histogram
    ax = axes[2, 0]
    if evmcol:
        evm_ok = df.loc[df[statcol] == "ok", evmcol].dropna() if statcol else df[evmcol].dropna()
        if not evm_ok.empty:
            ax.hist(evm_ok, bins=50, color="#e6ab02", edgecolor="k", linewidth=0.2)
            ax.axvline(evm_ok.mean(), color="red", lw=1.5, ls="--",
                       label=f"μ={evm_ok.mean():.4f}")
            ax.legend(fontsize=7)
    ax.set_title("EVM histogram (OK packets)"); ax.set_xlabel("EVM (rms)")

    # (2,1) EVM CDF
    ax = axes[2, 1]
    if evmcol:
        evm_ok = df.loc[df[statcol] == "ok", evmcol].dropna() if statcol else df[evmcol].dropna()
        if not evm_ok.empty:
            xs = np.sort(evm_ok.values)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.plot(xs, ys, lw=1.5, color="#e6ab02")
            ax.set_title("EVM CDF"); ax.set_xlabel("EVM"); ax.set_ylabel("CDF")

    # (2,2) EVM vs cap (time series)
    ax = axes[2, 2]
    if evmcol and statcol:
        ok_mask = df[statcol] == "ok"
        if ok_mask.any():
            ax.plot(df.loc[ok_mask, capcol], df.loc[ok_mask, evmcol],
                    ".", ms=3, color="#e6ab02")
    ax.set_title("EVM over time (OK packets)"); ax.set_xlabel("Capture #"); ax.set_ylabel("EVM")

    # (2,3) RX gain vs cap (shows gain sweep progression)
    ax = axes[2, 3]
    if gaincol:
        ax.plot(df[capcol], df[gaincol], lw=0.8, color="#d95f02")
        ax.set_title("RX gain vs cap")
    else:
        ax.set_title("RX gain (no column)")
    ax.set_xlabel("Capture #"); ax.set_ylabel("dB")

    # ── row 3 ──────────────────────────────────────────────────────────────────
    # (3,0) BER over time (OK packets)
    ax = axes[3, 0]
    if bercol and statcol:
        ok_mask = df[statcol] == "ok"
        ber_ok  = df.loc[ok_mask, bercol].dropna()
        if not ber_ok.empty:
            ax.semilogy(df.loc[ber_ok.index, capcol], ber_ok.values + 1e-9,
                        ".", ms=4, color="#d62728")
    ax.set_title("BER per packet (OK, log scale)")
    ax.set_xlabel("Capture #"); ax.set_ylabel("BER")

    # (3,1) BER vs rx_gain (per-capture scatter)
    ax = axes[3, 1]
    if bercol and gaincol:
        mask = df["ber"].notna() & df["rx_gain"].notna()
        if mask.any():
            ax.semilogy(df.loc[mask, gaincol], df.loc[mask, bercol] + 1e-9,
                        ".", ms=4, color="#d62728", alpha=0.6)
        ax.set_xlabel("RX gain (dB)"); ax.set_ylabel("BER")
    ax.set_title("BER vs RX gain (raw scatter)")

    # (3,2) scatter: gate_ratio vs xcorr_peak
    ax = axes[3, 2]
    if "gate_ratio" in df.columns and xcpcol:
        cmap = plt.cm.get_cmap("tab10", len(uniq))
        ax.scatter(df["gate_ratio"], df[xcpcol], s=5, alpha=0.5, c=c_arr, cmap=cmap)
        ax.set_xlabel("gate_ratio"); ax.set_ylabel("xc_peak")
    ax.set_title("XCorr vs energy gate")

    # (3,3) summary text
    ax = axes[3, 3]
    ax.axis("off")
    ok_cnt  = int((df[statcol] == "ok").sum()) if statcol else 0
    tot_cnt = len(df)
    dec_rate = ok_cnt / max(tot_cnt, 1)
    evm_mean = (df.loc[df[statcol] == "ok", evmcol].mean()
                if evmcol and statcol else float("nan"))
    cfo_mean = (df.loc[df[statcol] == "ok", cfocol].mean()
                if cfocol and statcol else float("nan"))
    snr_mean = (df.loc[df[statcol] == "ok", snrcol].mean()
                if snrcol and statcol else float("nan"))
    ber_mean = (df.loc[df[statcol] == "ok", bercol].mean()
                if bercol and statcol else float("nan"))
    mod_str = (df["modulation"].iloc[0] if "modulation" in df.columns else
               summary.get("modulation", "?"))

    txt = (
        f"RECEPTION SUMMARY\n\n"
        f"Modulation:    {mod_str}\n"
        f"Captures:      {ok_cnt} / {tot_cnt}\n"
        f"Decode rate:   {dec_rate*100:.1f}%\n"
        f"Mean EVM:      {evm_mean:.4f}\n"
        f"Mean SNR:      {snr_mean:.1f} dB\n"
        f"Mean CFO:      {cfo_mean:.0f} Hz\n"
        f"Mean BER:      {ber_mean:.2e}\n"
    )
    ax.text(0.05, 0.92, txt, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] wrote: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Constellation diagram  (from NPZ)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_constellation_from_npz(npz_dir: str, out_path: str, dpi: int = 150):
    """Load the first few OK NPZ captures and plot constellation."""
    files = sorted(glob.glob(os.path.join(npz_dir, "cap_*_ok.npz")))
    if not files:
        print(f"[analyze] no NPZ files in {npz_dir}")
        return

    all_syms = []
    modulation = "qpsk"
    for fn in files[:20]:
        try:
            d      = np.load(fn, allow_pickle=True)
            meta   = json.loads(d["meta_json"].tobytes().decode())
            modulation = meta.get("modulation", "qpsk")
        except Exception:
            pass
        try:
            rxw = d["rxw"]
            # Re-run a quick symbol extraction (simplified – use stored xcorr peak)
            # For now just store metadata and rely on rxw if needed
            # If all_syms is empty, skip
        except Exception:
            pass

    if not all_syms:
        print(f"[analyze] NPZ files found but no cached symbols; skipping constellation plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    syms = np.concatenate(all_syms)
    n    = len(syms)
    c    = np.arange(n)
    sc   = ax.scatter(syms.real, syms.imag, c=c, cmap="viridis", s=2, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="Symbol index")
    _overlay_ideal_constellation(ax, modulation)
    ax.axhline(0, color="k", lw=0.3); ax.axvline(0, color="k", lw=0.3)
    ax.set_xlabel("In-phase (I)"); ax.set_ylabel("Quadrature (Q)")
    ax.set_title(f"Received constellation  ({MOD_LABELS.get(modulation, modulation)}, N={n})")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] wrote: {out_path}")


def _overlay_ideal_constellation(ax, mod: str):
    """Draw ideal constellation points on ax."""
    try:
        from functools import lru_cache
        import importlib, types
        # Import make_constellation from the TX/RX module if available,
        # otherwise replicate the key ones here inline.
        table = _make_const_inline(mod)
        ax.scatter(table.real, table.imag, c="red", s=80, marker="x",
                   linewidths=2, zorder=5, label="Ideal")
        ax.legend(fontsize=7)
    except Exception:
        pass


def _make_const_inline(mod: str) -> np.ndarray:
    mod = mod.lower()
    if mod == "bpsk":
        return np.array([-1+0j, 1+0j], dtype=np.complex64)
    if mod == "qpsk":
        return np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64) / np.sqrt(2)
    if mod == "qam8":
        pi = [0, 1, 3, 2, 6, 7, 5, 4]
        return np.exp(1j * np.pi / 4 * np.array(pi, dtype=float)).astype(np.complex64)
    if mod == "qam16":
        g = {0: -3, 1: -1, 3: 1, 2: 3}
        return np.array([g[(i>>2)&3] + 1j*g[i&3] for i in range(16)], dtype=np.complex64) / np.sqrt(10)
    if mod == "qam32":
        pts = np.array([r+1j*m for r in (-5,-3,-1,1,3,5) for m in (-5,-3,-1,1,3,5)
                        if not (abs(r)==5 and abs(m)==5)], dtype=np.complex64)
        pts /= np.sqrt(np.mean(np.abs(pts)**2))
        return pts[np.lexsort((pts.real, -pts.imag))]
    raise ValueError(mod)


# ═══════════════════════════════════════════════════════════════════════════════
# BER curve  (research-paper grade, from run_summary.json)
# ═══════════════════════════════════════════════════════════════════════════════

def collect_ber_points(run_dirs):
    """
    Walk run_summary.json files and accumulate BER data.
    Returns list of dicts: {modulation, rx_gain_dB, ber, n_bits, n_pkts}.
    """
    points = []
    for run_dir in run_dirs:
        summary = load_run_summary(run_dir)
        if not summary:
            # Fall back to captures.csv
            csv_path = os.path.join(run_dir, "captures.csv")
            if not os.path.isfile(csv_path):
                continue
            df  = load_csv(csv_path)
            mod = df["modulation"].iloc[0] if "modulation" in df.columns else "qpsk"
            if "ber" not in df.columns or "rx_gain" not in df.columns:
                continue
            for gain, grp in df.groupby("rx_gain"):
                mask   = grp["ber"].notna()
                n_pkts = int(mask.sum())
                if n_pkts == 0:
                    continue
                # Aggregate: total bit errors / total bits
                n_bits   = grp.loc[mask, "n_bits"].sum() if "n_bits" in grp else n_pkts * 1000
                n_errors = grp.loc[mask, "n_bit_errors"].sum() if "n_bit_errors" in grp else grp.loc[mask, "ber"].mean() * n_bits
                ber_agg  = n_errors / max(n_bits, 1)
                points.append({"modulation": mod, "rx_gain_dB": float(gain),
                                "ber": float(ber_agg), "n_bits": int(n_bits), "n_pkts": n_pkts})
        else:
            mod         = summary.get("modulation", "qpsk")
            ber_per_gain = summary.get("ber_per_gain", {})
            for key, v in ber_per_gain.items():
                ber_val = v.get("ber")
                if ber_val is None:
                    continue
                points.append({
                    "modulation": mod,
                    "rx_gain_dB": float(v.get("rx_gain_dB", key)),
                    "ber":        float(ber_val),
                    "n_bits":     int(v.get("n_bits", 0)),
                    "n_pkts":     int(v.get("n_pkts", 0)),
                })
    return points


def plot_ber_curves(points: list, out_path: str, dpi: int = 200):
    """
    Research-paper-grade BER vs RX gain plot.
    One curve per modulation; markers sized by number of bits; log-scale Y.
    """
    if not points:
        print("[analyze] No BER data points found; skipping BER curve plot.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))

    # Group by modulation
    mods_present = sorted(set(p["modulation"] for p in points),
                          key=lambda m: {"bpsk": 0, "qpsk": 1, "qam8": 2, "qam16": 3, "qam32": 4}.get(m, 9))

    for mod in mods_present:
        pts   = sorted([p for p in points if p["modulation"] == mod],
                       key=lambda p: p["rx_gain_dB"])
        gains = np.array([p["rx_gain_dB"] for p in pts])
        bers  = np.array([max(p["ber"], 1e-6) for p in pts])   # floor at 1e-6 for log plot
        n_bits = np.array([p["n_bits"] for p in pts], dtype=float)

        color  = MOD_COLORS.get(mod, "black")
        marker = MOD_MARKERS.get(mod, "o")
        label  = MOD_LABELS.get(mod, mod.upper())

        # Marker size proportional to log(n_bits) – gives visual confidence indicator
        ms = np.clip(5 + 2 * np.log10(n_bits + 1), 4, 14)

        ax.semilogy(gains, bers, color=color, marker=marker,
                    linewidth=1.5, linestyle="-", label=label,
                    markeredgewidth=0.8, markeredgecolor="k",
                    markersize=8, zorder=3)
        # Error bars: Wilson 95% CI on each point
        for g, b, nb in zip(gains, bers, n_bits):
            if nb > 0:
                z    = 1.96
                ne   = b * nb
                lo   = max((ne + z*z/2 - z*np.sqrt(ne*(1-ne/nb)+z*z/4)) / (nb+z*z), 1e-7)
                hi   = (ne + z*z/2 + z*np.sqrt(ne*(1-ne/nb)+z*z/4)) / (nb+z*z)
                hi   = max(hi, lo * 1.01)
                lo   = max(lo, 1e-7)
                ax.errorbar(g, b, yerr=[[b - lo], [hi - b]],
                            fmt="none", ecolor=color, elinewidth=1, capsize=3, alpha=0.6)

    # Reference BER = 0.5 line and 1e-3 / 1e-4 markers
    ax.axhline(0.5,  color="grey", lw=0.8, ls=":", alpha=0.6)
    ax.axhline(1e-3, color="grey", lw=0.8, ls=":", alpha=0.6, label="BER=10⁻³")
    ax.axhline(1e-4, color="grey", lw=0.8, ls=":", alpha=0.6, label="BER=10⁻⁴")

    ax.set_xlabel("RX Gain (dB)", fontsize=10)
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=10)
    ax.set_title("BER vs. RX Gain  –  PlutoSDR Step6 PHY\n"
                 r"$f_c=2.3\,$GHz, $f_s=3\,$Msps, AWGN channel",
                 fontsize=10)
    ax.set_ylim([1e-6, 1.0])
    ax.yaxis.set_major_formatter(mticker.LogFormatter(labelOnlyBase=False))
    ax.legend(loc="upper right", framealpha=0.9)

    # Add secondary axis hint: higher RX gain ≈ higher SNR
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel("← Lower SNR    Higher SNR →", fontsize=8, color="grey")
    ax2.tick_params(axis="x", which="both", bottom=False, top=False,
                    labelbottom=False, labeltop=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] BER curves written: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Comparative EVM box plot across modulations
# ═══════════════════════════════════════════════════════════════════════════════

def plot_evm_comparison(run_dirs: list, out_path: str, dpi: int = 150):
    """Box-plot of EVM distribution per modulation from multiple runs."""
    data_by_mod = {}
    for run_dir in run_dirs:
        csv_path = os.path.join(run_dir, "captures.csv")
        if not os.path.isfile(csv_path):
            continue
        df  = load_csv(csv_path)
        if "probe_evm" not in df.columns or "modulation" not in df.columns:
            continue
        mod    = df["modulation"].iloc[0]
        ok_evm = df.loc[df.get("status", pd.Series()) == "ok", "probe_evm"].dropna()
        if not ok_evm.empty:
            data_by_mod.setdefault(mod, []).extend(ok_evm.tolist())

    if not data_by_mod:
        return

    mods  = sorted(data_by_mod.keys(),
                   key=lambda m: {"bpsk": 0, "qpsk": 1, "qam8": 2, "qam16": 3, "qam32": 4}.get(m, 9))
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    bp  = ax.boxplot([data_by_mod[m] for m in mods],
                     labels=[MOD_LABELS.get(m, m.upper()) for m in mods],
                     patch_artist=True, notch=False,
                     medianprops=dict(color="red", linewidth=1.5))
    for patch, mod in zip(bp["boxes"], mods):
        patch.set_facecolor(MOD_COLORS.get(mod, "lightblue"))
        patch.set_alpha(0.7)
    ax.set_xlabel("Modulation"); ax.set_ylabel("EVM (rms)")
    ax.set_title("EVM Distribution per Modulation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] EVM comparison written: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Reception timeline / summary panel
# ═══════════════════════════════════════════════════════════════════════════════

def plot_reception_summary(df: pd.DataFrame, summary: dict, out_path: str, dpi: int = 150):
    """Compact 2×3 reception-quality summary (suitable as a figure in a paper)."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Reception Quality Summary  –  Step6 PHY", fontsize=12, fontweight="bold")

    statcol = _pick(df, ["status"])
    cfocol  = _pick(df, ["cfo_hz"])
    evmcol  = _pick(df, ["probe_evm"])
    snrcol  = _pick(df, ["snr_db"])
    bercol  = _pick(df, ["ber"])
    gaincol = _pick(df, ["rx_gain"])
    capcol  = "cap"

    ok_mask = (df[statcol] == "ok") if statcol else pd.Series(True, index=df.index)

    # (0,0) decode rate pie
    ax = axes[0, 0]
    if statcol:
        counts = df[statcol].value_counts()
        ok_n   = int(counts.get("ok", 0))
        bad_n  = int(len(df) - ok_n)
        colors = ["#2ca02c", "#d62728"]
        ax.pie([ok_n, bad_n], labels=[f"OK ({ok_n})", f"Failed ({bad_n})"],
               colors=colors, startangle=90,
               autopct="%1.1f%%", textprops={"fontsize": 9})
    ax.set_title("Decode rate")

    # (0,1) EVM histogram
    ax = axes[0, 1]
    if evmcol:
        evm_ok = df.loc[ok_mask, evmcol].dropna()
        if not evm_ok.empty:
            ax.hist(evm_ok, bins=40, color="#ff7f0e", edgecolor="k", lw=0.3, density=True)
            mu, sig = evm_ok.mean(), evm_ok.std()
            ax.axvline(mu, color="red", lw=1.5, ls="--", label=f"μ={mu:.4f}")
            ax.legend(fontsize=8)
    ax.set_title("EVM distribution"); ax.set_xlabel("EVM (rms)"); ax.set_ylabel("Density")

    # (0,2) CFO histogram
    ax = axes[0, 2]
    if cfocol:
        cfo_ok = df.loc[ok_mask, cfocol].dropna()
        if not cfo_ok.empty:
            ax.hist(cfo_ok, bins=40, color="#1f77b4", edgecolor="k", lw=0.3, density=True)
            ax.axvline(cfo_ok.mean(), color="red", lw=1.5, ls="--",
                       label=f"μ={cfo_ok.mean():.0f} Hz")
            ax.legend(fontsize=8)
    ax.set_title("CFO distribution"); ax.set_xlabel("Hz"); ax.set_ylabel("Density")

    # (1,0) SNR over time
    ax = axes[1, 0]
    if snrcol:
        snr_ok = df.loc[ok_mask, snrcol].dropna()
        if not snr_ok.empty:
            ax.plot(df.loc[snr_ok.index, capcol], snr_ok.values, ".", ms=3, color="#7570b3")
    ax.set_title("Per-subcarrier mean SNR (OK)"); ax.set_xlabel("Capture #"); ax.set_ylabel("dB")

    # (1,1) BER vs rx_gain scatter
    ax = axes[1, 1]
    if bercol and gaincol:
        mask = df["ber"].notna() & df["rx_gain"].notna()
        if mask.any():
            ax.semilogy(df.loc[mask, gaincol], df.loc[mask, bercol] + 1e-9,
                        ".", ms=5, color="#d62728", alpha=0.7)
        ax.set_xlabel("RX gain (dB)"); ax.set_ylabel("BER")
    ax.set_title("BER vs RX gain")

    # (1,2) summary stats text
    ax = axes[1, 2]
    ax.axis("off")
    ok_n = int(ok_mask.sum()); tot = len(df)
    evm_mu = df.loc[ok_mask, evmcol].mean() if evmcol else float("nan")
    cfo_mu = df.loc[ok_mask, cfocol].mean() if cfocol else float("nan")
    snr_mu = df.loc[ok_mask, snrcol].mean() if snrcol else float("nan")
    ber_mu = df.loc[ok_mask & df[bercol].notna(), bercol].mean() if bercol else float("nan")
    mod    = summary.get("modulation", df.get("modulation", pd.Series(["?"])).iloc[0]
                         if "modulation" in df.columns else "?")
    ax.text(0.05, 0.92,
            f"Modulation: {MOD_LABELS.get(mod, mod.upper())}\n"
            f"Decoded:    {ok_n}/{tot}  ({100*ok_n/max(tot,1):.1f}%)\n"
            f"Mean EVM:   {evm_mu:.4f}\n"
            f"Mean SNR:   {snr_mu:.1f} dB\n"
            f"Mean CFO:   {cfo_mu:.0f} Hz\n"
            f"Mean BER:   {ber_mu:.2e}\n",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.8))
    ax.set_title("Summary")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] reception summary written: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Text summary
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame, run_dir: str):
    statcol = _pick(df, ["status"])
    evmcol  = _pick(df, ["probe_evm"])
    cfocol  = _pick(df, ["cfo_hz"])
    snrcol  = _pick(df, ["snr_db"])
    bercol  = _pick(df, ["ber"])

    print("\n" + "=" * 60)
    print(f"RUN: {os.path.basename(run_dir)}")
    print(f"Rows: {len(df)}")
    if statcol:
        vc = df[statcol].value_counts()
        print(f"Status: {dict(vc)}")
        ok_n = int(vc.get("ok", 0))
        print(f"Decode rate: {ok_n}/{len(df)} = {100*ok_n/max(len(df),1):.1f}%")
    ok_mask = (df[statcol] == "ok") if statcol else pd.Series(True, index=df.index)
    for col, label in [(evmcol, "EVM"), (cfocol, "CFO_Hz"), (snrcol, "SNR_dB"), (bercol, "BER")]:
        if col:
            s = df.loc[ok_mask, col].dropna()
            if not s.empty:
                print(f"{label}: mean={s.mean():.4g}  std={s.std():.4g}  "
                      f"min={s.min():.4g}  max={s.max():.4g}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser("Offline Step6 RF-stream analyzer")

    ap.add_argument("--csv",     type=str, default="",
                    help="Direct path to captures.csv (single-run diagnostic)")
    ap.add_argument("--run_dir", type=str, default="",
                    help="Run directory containing captures.csv + run_summary.json")
    ap.add_argument("--ber_dirs", nargs="+", default=[],
                    help="List of run directories (or glob) for BER curve plot")
    ap.add_argument("--out",     type=str, default="",
                    help="Output file path (auto-derived if not given)")
    ap.add_argument("--dpi",     type=int, default=150)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--ber_only", action="store_true",
                    help="Only generate BER curve (skip per-run dashboard)")
    args = ap.parse_args()

    # ── Expand globs in ber_dirs ─────────────────────────────────────────────
    expanded_ber_dirs = []
    for pat in args.ber_dirs:
        matched = glob.glob(pat)
        expanded_ber_dirs.extend(sorted(matched) if matched else [pat])

    # ── Determine CSV path ────────────────────────────────────────────────────
    csv_path = ""
    run_dir  = ""
    if args.csv:
        csv_path = args.csv
        run_dir  = os.path.dirname(os.path.abspath(csv_path))
    elif args.run_dir:
        run_dir  = args.run_dir
        csv_path = os.path.join(run_dir, "captures.csv")
    elif expanded_ber_dirs:
        run_dir = expanded_ber_dirs[0]
        csv_path = os.path.join(run_dir, "captures.csv")
    else:
        ap.print_help()
        sys.exit(0)

    # ── Load data ─────────────────────────────────────────────────────────────
    if not os.path.isfile(csv_path):
        print(f"[analyze] CSV not found: {csv_path}")
        df      = pd.DataFrame()
        summary = {}
    else:
        df      = load_csv(csv_path, args.max_rows)
        summary = load_run_summary(run_dir)
        print_summary(df, run_dir)

    # ── Determine output dir ──────────────────────────────────────────────────
    out_base = args.out
    if not out_base:
        out_base = run_dir if run_dir else "."
    if os.path.isdir(out_base):
        out_dir = out_base
    else:
        out_dir = os.path.dirname(out_base) or "."
        os.makedirs(out_dir, exist_ok=True)

    def outpath(suffix):
        if args.out and not os.path.isdir(args.out):
            # user gave explicit file → use stem + suffix
            base, ext = os.path.splitext(args.out)
            return f"{base}_{suffix}{ext or '.png'}"
        return os.path.join(out_dir, suffix)

    # ── Plots ─────────────────────────────────────────────────────────────────

    if not args.ber_only and not df.empty:
        # 1. Full diagnostic grid
        plot_single_run(df, run_dir, summary, outpath("captures_grid.png"), args.dpi)

        # 2. Compact reception summary panel
        plot_reception_summary(df, summary, outpath("reception_summary.png"), args.dpi)

    # 3. BER curves (from multiple run directories)
    ber_dirs_all = expanded_ber_dirs if expanded_ber_dirs else ([run_dir] if run_dir else [])
    if ber_dirs_all:
        points = collect_ber_points(ber_dirs_all)
        if points:
            plot_ber_curves(points, outpath("ber_curves.png"), dpi=max(args.dpi, 200))
            plot_evm_comparison(ber_dirs_all, outpath("evm_comparison.png"), args.dpi)
        else:
            print("[analyze] No BER points found in supplied directories.")
            print("  Tip: run RX with --ref_seed / --ref_len to enable BER tracking.")

    print("\n[analyze] done.")


if __name__ == "__main__":
    main()

"""
# ── Example workflow for a complete BER curve  ───────────────────────────────

# Step 1 – transmit known reference (on Jetson):
# python3 rf_stream_tx_step6phy.py \\
#   --uri ip:192.168.3.2 --fc 2.3e9 --fs 3e6 --tx_gain -5 \\
#   --modulation qpsk --ref_seed 42 --ref_len 512

# Step 2 – receive with gain sweep (on local machine):
# python3 rf_stream_rx_step6phy.py \\
#   --uri ip:192.168.2.2 --fc 2.3e9 --fs 3e6 \\
#   --modulation qpsk --ref_seed 42 --ref_len 512 \\
#   --rx_gain_sweep "65,60,55,50,45,40,35,30" --gain_step_s 20 \\
#   --out_root ber_sweep --save_npz

# Step 3 – analyze offline (no hardware needed):
# python3 analyze_rf_stream_captures.py \\
#   --ber_dirs ber_sweep/run_* \\
#   --out ber_results.png

# Repeat Step 1–3 for each modulation (bpsk, qpsk, qam8, qam16, qam32)
# then combine all run directories in --ber_dirs to get overlaid BER curves.
"""

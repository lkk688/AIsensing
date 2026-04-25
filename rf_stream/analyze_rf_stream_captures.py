#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze streaming RX captures.csv (Step5 PHY) and generate a grid plot to diagnose gating / xcorr / demod flow.

Usage:
  python3 analyze_rf_stream_captures.py --csv ./rf_stream_rx_runs/run_*/captures.csv --out out.png

What it does:
- Loads captures.csv robustly (handles missing columns / weird rows)
- Prints key summaries (status/reason distributions, basic stats)
- Computes derived signals: gate_margin=maxe-eg_th, gate_ratio=maxe/eg_th
- Produces a multi-panel grid figure:
  * status counts
  * reason counts
  * time series: peak, p10, eg_th, maxe
  * histograms and scatter plots: eg_th vs maxe, xcorr peaks, etc.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- helpers ----------
def _safe_numeric(s):
    """Convert to numeric, coerce errors to NaN."""
    return pd.to_numeric(s, errors="coerce")


def _pick_col(df, candidates):
    """Return the first existing column name from candidates, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ensure_columns(df):
    """
    Normalize likely column names across different versions of the logger.
    Returns a dict mapping canonical name -> actual column name (or None).
    """
    colmap = {}
    colmap["cap"] = _pick_col(df, ["cap", "capture", "i", "idx"])
    colmap["status"] = _pick_col(df, ["status"])
    colmap["reason"] = _pick_col(df, ["reason", "why"])
    colmap["peak"] = _pick_col(df, ["peak", "rx_peak"])
    colmap["p10"] = _pick_col(df, ["p10", "energy_p10", "noise_p10"])
    colmap["eg_th"] = _pick_col(df, ["eg_th", "energy_th", "eg_threshold"])
    colmap["maxe"] = _pick_col(df, ["maxe", "energy_max", "max_energy"])
    colmap["tone_idx"] = _pick_col(df, ["tone_idx"])
    colmap["tone_ratio"] = _pick_col(df, ["tone_ratio", "tone_snr", "tone_peak_ratio"])
    colmap["xc_idx"] = _pick_col(df, ["xc_idx", "xcorr_idx"])
    colmap["xc_peak"] = _pick_col(df, ["xc_peak", "xcorr_peak"])
    colmap["stf_idx"] = _pick_col(df, ["stf_idx"])
    colmap["cfo_hz"] = _pick_col(df, ["cfo_hz", "cfo", "cfo_est_hz"])
    colmap["probe_evm"] = _pick_col(df, ["probe_mean_evm", "probe_evm", "evm", "mean_evm"])
    colmap["parse_status"] = _pick_col(df, ["parse_status", "parse", "pkt_status"])
    return colmap


def _top_counts(series, n=12):
    vc = series.value_counts(dropna=False)
    if len(vc) > n:
        return vc.iloc[:n]
    return vc


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to captures.csv")
    ap.add_argument("--out", default="", help="Output image path (.png). Default: beside csv")
    ap.add_argument("--max_rows", type=int, default=0, help="Optional max rows to read (0=all)")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        print(f"[ERR] file not found: {csv_path}")
        sys.exit(1)

    # Robust read: skip bad lines rather than crash
    nrows = args.max_rows if args.max_rows and args.max_rows > 0 else None
    try:
        df = pd.read_csv(csv_path, nrows=nrows, on_bad_lines="skip")
    except TypeError:
        # older pandas compatibility
        df = pd.read_csv(csv_path, nrows=nrows, error_bad_lines=False)

    if df.empty:
        print("[ERR] CSV loaded but empty (or all lines skipped).")
        sys.exit(2)

    colmap = _ensure_columns(df)

    # If cap missing, create one
    if colmap["cap"] is None:
        df["cap"] = np.arange(1, len(df) + 1)
        colmap["cap"] = "cap"

    # Convert numerics
    for k in ["peak", "p10", "eg_th", "maxe", "tone_ratio", "xc_peak", "cfo_hz", "probe_evm", "tone_idx", "xc_idx", "stf_idx"]:
        c = colmap.get(k)
        if c is not None:
            df[c] = _safe_numeric(df[c])

    # Derived metrics if we have maxe and eg_th
    if colmap["maxe"] and colmap["eg_th"]:
        df["gate_margin"] = df[colmap["maxe"]] - df[colmap["eg_th"]]
        df["gate_ratio"] = df[colmap["maxe"]] / (df[colmap["eg_th"]] + 1e-12)
    else:
        df["gate_margin"] = np.nan
        df["gate_ratio"] = np.nan

    # Basic summaries
    print("\n====================")
    print("CSV SUMMARY")
    print("====================")
    print(f"rows: {len(df)}")
    print("columns:", list(df.columns))

    if colmap["status"]:
        print("\n-- status counts --")
        print(_top_counts(df[colmap["status"]], n=20).to_string())
    if colmap["reason"]:
        print("\n-- reason counts --")
        print(_top_counts(df[colmap["reason"]], n=20).to_string())

    # How many rows have evidence of signal?
    capcol = colmap["cap"]
    statuscol = colmap["status"]
    reasoncol = colmap["reason"]
    peakcol = colmap["peak"]
    p10col = colmap["p10"]
    egcol = colmap["eg_th"]
    maxecol = colmap["maxe"]
    xcpcol = colmap["xc_peak"]
    cfocol = colmap["cfo_hz"]

    # Print key numeric stats
    def print_stats(name, series):
        s = series.dropna()
        if s.empty:
            print(f"{name}: (no data)")
            return
        q = s.quantile([0.01, 0.1, 0.5, 0.9, 0.99]).to_dict()
        print(f"{name}: min={s.min():.4g} p10={q.get(0.1, np.nan):.4g} med={q.get(0.5, np.nan):.4g} "
              f"p90={q.get(0.9, np.nan):.4g} max={s.max():.4g}")

    print("\n-- numeric quick stats --")
    if peakcol: print_stats("peak", df[peakcol])
    if p10col: print_stats("p10", df[p10col])
    if egcol: print_stats("eg_th", df[egcol])
    if maxecol: print_stats("maxe", df[maxecol])
    if xcpcol: print_stats("xc_peak", df[xcpcol])
    if cfocol: print_stats("cfo_hz", df[cfocol])
    if "gate_ratio" in df.columns: print_stats("gate_ratio", df["gate_ratio"])
    if "gate_margin" in df.columns: print_stats("gate_margin", df["gate_margin"])

    # Identify “non-trivial” rows:
    # - maxe > eg_th suggests energy gate would pass
    # - xc_peak above some small threshold suggests STF correlation sees something
    mask_energy_pass = pd.Series(False, index=df.index)
    if maxecol and egcol:
        mask_energy_pass = (df[maxecol] > df[egcol])

    mask_xc_present = pd.Series(False, index=df.index)
    if xcpcol:
        mask_xc_present = (df[xcpcol] > 0.2)

    print("\n-- derived diagnostics --")
    print(f"energy_pass (maxe>eg_th): {int(mask_energy_pass.sum())}/{len(df)} ({100*mask_energy_pass.mean():.2f}%)")
    print(f"xc_present (xc_peak>0.2): {int(mask_xc_present.sum())}/{len(df)} ({100*mask_xc_present.mean():.2f}%)")

    # ---------- plotting ----------
    out_path = args.out
    if not out_path:
        out_path = os.path.join(os.path.dirname(csv_path), "captures_grid.png")

    # Choose a compact color encoding for status
    status_vals = None
    if statuscol:
        status_vals = df[statuscol].astype(str).fillna("NA")
    else:
        status_vals = pd.Series(["NA"] * len(df))

    # map statuses to integers for scatter coloring
    uniq_status = sorted(status_vals.unique().tolist())
    status_to_int = {s: i for i, s in enumerate(uniq_status)}
    c_int = status_vals.map(status_to_int).astype(int)

    # Grid layout: 3 rows x 4 cols
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle("Streaming RX captures.csv diagnostics (grid)", fontsize=16)

    # (0,0) status counts
    ax = axes[0, 0]
    vc = status_vals.value_counts().iloc[:15]
    ax.bar(vc.index.astype(str), vc.values)
    ax.set_title("Status counts (top 15)")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y")

    # (0,1) reason counts
    ax = axes[0, 1]
    if reasoncol:
        rv = df[reasoncol].astype(str).fillna("NA").value_counts().iloc[:15]
        ax.bar(rv.index.astype(str), rv.values)
        ax.set_title("Reason counts (top 15)")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, axis="y")
    else:
        ax.set_title("Reason counts")
        ax.text(0.5, 0.5, "no reason column", ha="center", va="center")
        ax.axis("off")

    # (0,2) time series: peak
    ax = axes[0, 2]
    if peakcol:
        ax.plot(df[capcol], df[peakcol], linewidth=0.8)
        ax.set_title("peak vs cap")
        ax.set_xlabel("cap")
        ax.grid(True)
    else:
        ax.axis("off")

    # (0,3) time series: p10, eg_th, maxe (if available)
    ax = axes[0, 3]
    any_plotted = False
    if p10col:
        ax.plot(df[capcol], df[p10col], linewidth=0.8, label="p10")
        any_plotted = True
    if egcol:
        ax.plot(df[capcol], df[egcol], linewidth=0.8, label="eg_th")
        any_plotted = True
    if maxecol:
        ax.plot(df[capcol], df[maxecol], linewidth=0.8, label="maxe")
        any_plotted = True
    ax.set_title("Energy terms vs cap")
    ax.set_xlabel("cap")
    ax.grid(True)
    if any_plotted:
        ax.legend(fontsize=8)
    else:
        ax.axis("off")

    # (1,0) histogram: gate_ratio
    ax = axes[1, 0]
    gr = df["gate_ratio"].dropna()
    if not gr.empty:
        ax.hist(gr.clip(0, np.nanpercentile(gr, 99)), bins=60)
        ax.set_title("gate_ratio = maxe/eg_th (clipped p99)")
        ax.set_xlabel("gate_ratio")
        ax.grid(True, axis="y")
    else:
        ax.text(0.5, 0.5, "no gate_ratio (need maxe & eg_th)", ha="center", va="center")
        ax.axis("off")

    # (1,1) histogram: gate_margin
    ax = axes[1, 1]
    gm = df["gate_margin"].dropna()
    if not gm.empty:
        clip = np.nanpercentile(np.abs(gm), 99)
        ax.hist(gm.clip(-clip, clip), bins=60)
        ax.set_title("gate_margin = maxe - eg_th (clipped p99 abs)")
        ax.set_xlabel("gate_margin")
        ax.grid(True, axis="y")
    else:
        ax.axis("off")

    # (1,2) scatter: eg_th vs maxe (colored by status)
    ax = axes[1, 2]
    if egcol and maxecol:
        x = df[egcol]
        y = df[maxecol]
        ax.scatter(x, y, s=8, alpha=0.6, c=c_int)
        # y=x line
        mn = np.nanmin([x.min(), y.min()])
        mx = np.nanmax([x.max(), y.max()])
        ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)
        ax.set_title("maxe vs eg_th (color=status)")
        ax.set_xlabel("eg_th")
        ax.set_ylabel("maxe")
        ax.grid(True)
    else:
        ax.axis("off")

    # (1,3) xcorr peak over time
    ax = axes[1, 3]
    if xcpcol:
        ax.plot(df[capcol], df[xcpcol], linewidth=0.8)
        ax.set_title("xc_peak vs cap")
        ax.set_xlabel("cap")
        ax.grid(True)
    else:
        ax.axis("off")

    # (2,0) scatter: xc_peak vs gate_ratio
    ax = axes[2, 0]
    if xcpcol and ("gate_ratio" in df.columns):
        x = df["gate_ratio"]
        y = df[xcpcol]
        ax.scatter(x, y, s=8, alpha=0.6, c=c_int)
        ax.set_title("xc_peak vs gate_ratio (color=status)")
        ax.set_xlabel("gate_ratio")
        ax.set_ylabel("xc_peak")
        ax.grid(True)
    else:
        ax.axis("off")

    # (2,1) CFO vs cap
    ax = axes[2, 1]
    if cfocol:
        ax.plot(df[capcol], df[cfocol], linewidth=0.8)
        ax.set_title("cfo_hz vs cap")
        ax.set_xlabel("cap")
        ax.grid(True)
    else:
        ax.axis("off")

    # (2,2) probe EVM vs cap
    ax = axes[2, 2]
    if colmap["probe_evm"]:
        ax.plot(df[capcol], df[colmap["probe_evm"]], linewidth=0.8)
        ax.set_title("probe_mean_evm vs cap")
        ax.set_xlabel("cap")
        ax.grid(True)
    else:
        ax.axis("off")

    # (2,3) status timeline (as integer)
    ax = axes[2, 3]
    ax.plot(df[capcol], c_int, linewidth=0.8)
    ax.set_title("status timeline (int-coded)")
    ax.set_xlabel("cap")
    ax.set_yticks(range(len(uniq_status)))
    ax.set_yticklabels(uniq_status, fontsize=7)
    ax.grid(True)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)

    print("\n====================")
    print("WROTE FIGURE")
    print("====================")
    print(out_path)

    # Also write a small text diagnosis hint
    hint_path = os.path.join(os.path.dirname(out_path), "captures_diagnosis.txt")
    with open(hint_path, "w") as f:
        f.write("Quick diagnosis hints:\n")
        f.write(f"- rows={len(df)}\n")
        f.write(f"- energy_pass(maxe>eg_th)={int(mask_energy_pass.sum())} ({100*mask_energy_pass.mean():.2f}%)\n")
        f.write(f"- xc_present(xc_peak>0.2)={int(mask_xc_present.sum())} ({100*mask_xc_present.mean():.2f}%)\n")
        f.write("\nInterpretation:\n")
        f.write("- If energy_pass ~0%: your energy gate is too strict OR scaling is inconsistent OR TX not present.\n")
        f.write("- If energy_pass high but xc_present low: signal energy exists but STF correlation isn't matching (preamble mismatch or timing/scale issue).\n")
        f.write("- If xc_present high but no decode: likely LTF/FFT convention mismatch, symbol alignment off, or phase tracking unstable.\n")
    print("wrote:", hint_path)


if __name__ == "__main__":
    main()

"""
python3 analyze_rf_stream_captures.py \
  --csv ./rf_stream_rx_runs/run_20260425_155851/captures.csv \
  --out ./rf_stream_rx_runs/run_20260425_155851/captures_grid.png

python3 analyze_rf_stream_captures.py \
  --csv ./rf_stream_rx_runs/run_20260425_160611/captures.csv \
  --out ./rf_stream_rx_runs/run_20260425_160611/captures_grid.png
"""
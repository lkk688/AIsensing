import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy.ndimage import maximum_filter
import math

from .config import DEVICE
from .utils import plot_rd, cfar2d_ca, extract_detections, gt_rv_az, match_dets_to_gts, compute_metrics_from_pairs
from .methods.fmcw import fmcw_torch
from .methods.otfs import otfs_torch
from .methods.ofdm import ofdm_tx_rx_ber
from .methods.otfs import otfs_tx_rx_ber
from .simulator import raycast_torch
from .dataset import _rd_normalize

def viz_rd_2d_compare(path, rd_f_db, rd_o_db, gts, sp, cfar_cfg=None):
    """Visualize and compare 2D Range-Doppler maps for FMCW and OTFS."""
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ra_f, va_f = sp.fmcw_axes()
    ra_o, va_o = sp.otfs_axes()

    if cfar_cfg is None:
        cfar_cfg = dict(train=(10, 8), guard=(2, 2), pfa=1e-4,
                        min_snr_db=8.0, notch_doppler_bins=2,
                        apply_nms=True, max_peaks=60)

    # FMCW Plot
    im = plot_rd(ax[0], rd_f_db, ra_f, va_f, "FMCW Range-Doppler")
    plt.colorbar(im, ax=ax[0], label='dB')

    # FMCW CFAR
    det_f, noise_f, snr_f = cfar2d_ca(rd_f_db, **cfar_cfg, return_stats=True)
    fy, fx = np.where(det_f)
    if fy.size:
        ax[0].scatter(ra_f[fx], va_f[fy], s=60, facecolors='none', edgecolors='cyan', linewidths=1.8, label='CFAR')

    # OTFS Plot
    im2 = plot_rd(ax[1], rd_o_db, ra_o, va_o, "OTFS Delay-Doppler")
    plt.colorbar(im2, ax=ax[1], label='dB')

    # Ground Truth Overlay
    pos = np.array([0, 0, sp.H])
    for i, (ra, va, rd) in enumerate([(ra_f, va_f, rd_f_db), (ra_o, va_o, rd_o_db)]):
        for gt in gts:
            P = np.array(gt['c']) - pos
            r = np.linalg.norm(P)
            v = np.dot(P/r, gt['v'])
            if 0 <= r <= ra[-1] and va[0] <= v <= va[-1]:
                ax[i].plot(r, v, 'wx', ms=10, mew=2, label='GT' if i==0 else "")
                ax[i].text(r+1, v+0.3, f"{r:.0f} m, {v:.1f} m/s", color='white', fontsize=9, weight='bold')

    for i in range(2):
        ax[i].grid(alpha=0.25, linestyle=':')
        ax[i].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    
    return det_f, ra_f, va_f, noise_f, snr_f

def run_ber_sweep_and_plot(path_png, ebn0_db_list, ofdm_cfg, otfs_cfg, rng_seed=1234):
    """Run BER sweep for OFDM and OTFS and plot results."""
    rng = np.random.default_rng(rng_seed)
    
    ber_ofdm = []
    ber_otfs = []
    
    for eb in ebn0_db_list:
        ber_ofdm.append(ofdm_tx_rx_ber(eb, **ofdm_cfg, rng=rng))
        ber_otfs.append(otfs_tx_rx_ber(eb, **otfs_cfg, rng=rng))
        
    ber_ofdm = np.array(ber_ofdm)
    ber_otfs = np.array(ber_otfs)
    
    # Theory QPSK
    ebn0_lin = 10.0 ** (np.array(ebn0_db_list) / 10.0)
    ber_theory = np.array([0.5 * math.erfc(math.sqrt(x)) for x in ebn0_lin])
    
    plt.figure(figsize=(8, 6))
    plt.semilogy(ebn0_db_list, ber_ofdm + 1e-12, 'o-', label='FMCW-Comm (OFDM)')
    plt.semilogy(ebn0_db_list, ber_otfs + 1e-12, 's-', label='OTFS-Comm')
    plt.semilogy(ebn0_db_list, ber_theory + 1e-12, 'k--', label='Theory QPSK')
    
    plt.grid(True, which='both', linestyle=':')
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.title('BER Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()
    
    return ebn0_db_list, ber_ofdm, ber_otfs, ber_theory

@torch.no_grad()
def rd_dl_infer_to_points(logits, ra, va, thr=0.1, max_peaks=64):
    """Convert DL logits to a list of detected points."""
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    print(f"[DEBUG] Logits min/max: {logits.min():.4f}/{logits.max():.4f}")
    print(f"[DEBUG] Prob min/max: {prob.min():.4f}/{prob.max():.4f}")
    mask = prob > thr
    mask = prob > thr
    if not mask.any():
        return []
        
    mxf = maximum_filter(prob, size=3)
    peaks = (prob == mxf) & mask
    yy, xx = np.where(peaks)
    
    if len(yy) > max_peaks:
        vals = prob[yy, xx]
        idx = np.argpartition(-vals, max_peaks-1)[:max_peaks]
        yy, xx = yy[idx], xx[idx]
        
    dets = [{'r': float(ra[x]), 'v': float(va[y]), 'score': float(prob[y, x])} for y, x in zip(yy, xx)]
    return dets

def evaluate_and_visualize(out_dir, sp, radar_net, gts_eval=None):
    """End-to-end evaluation on a sample scene."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    if gts_eval is None:
        gts_eval = [{'c':[20, 0, 1], 's':[4,2,2], 'v':[12, 0, 0]}]
        
    # Simulate
    pts, its, vels = raycast_torch(sp, gts_eval)
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    
    (rd_f_db,) = fmcw_torch(pts, its, vels, sp)
    (rd_o_db,) = otfs_torch(pts, its, vels, sp)
    
    # Visualize
    viz_rd_2d_compare(out/"eval_compare.png", rd_f_db, rd_o_db, gts_eval, sp)
    
    # DL Inference
    ra_f, va_f = sp.fmcw_axes()
    rd_in = torch.from_numpy(_rd_normalize(rd_f_db))[None, None].to(DEVICE)
    
    radar_net.eval()
    with torch.no_grad():
        logits = radar_net(rd_in)
        
    dets = rd_dl_infer_to_points(logits, ra_f, va_f)
    print(f"[Eval] DL detected {len(dets)} targets.")
    
    # Metrics
    gtinfo = gt_rv_az(gts_eval, sp)
    pairs = match_dets_to_gts(dets, gtinfo)
    metrics, _, _ = compute_metrics_from_pairs(pairs, gtinfo, sp)
    
    with open(out/"metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print("[Eval] Metrics:", metrics)

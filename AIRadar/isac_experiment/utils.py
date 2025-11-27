import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.patches import Rectangle
from .config import DEVICE

def to_torch(x):
    """Convert a numpy array or list to a torch tensor on the default device."""
    return torch.tensor(x, device=DEVICE, dtype=torch.float32)

def _moving_sum_2d(a, r, c):
    """
    Compute 2D moving sum using integral images (summed-area table).
    
    Args:
        a (np.ndarray): Input 2D array.
        r (int): Radius in row dimension.
        c (int): Radius in column dimension.
    
    Returns:
        np.ndarray: Moving sum array.
    """
    if r == 0 and c == 0: return a.copy()
    # Pad to handle boundaries
    ap = np.pad(a, ((r, r), (c, c)), mode='edge')
    # Compute integral image
    S = ap.cumsum(axis=0).cumsum(axis=1)
    H, W = a.shape
    # Calculate sum of rectangular regions using the integral image
    s22 = S[2*r:2*r+H, 2*c:2*c+W]
    s02 = S[0:H,       2*c:2*c+W]
    s20 = S[2*r:2*r+H, 0:W]
    s00 = S[0:H,       0:W]
    return s22 - s02 - s20 + s00

def nms2d(arr, kernel=3):
    """
    Non-Maximum Suppression (NMS) for 2D arrays.
    
    Args:
        arr (np.ndarray): Input 2D array.
        kernel (int): Size of the local window (must be odd).
    
    Returns:
        np.ndarray: Boolean mask where True indicates a local maximum.
    """
    k = max(3, int(kernel) | 1)
    pad = k // 2
    ap = np.pad(arr, ((pad, pad), (pad, pad)), mode='edge')
    max_nb = np.full_like(arr, -np.inf)
    
    # Iterate over the kernel window
    for di in range(-pad, pad + 1):
        for dj in range(-pad, pad + 1):
            if di == 0 and dj == 0: continue
            view = ap[pad+di:pad+di+arr.shape[0], pad+dj:pad+dj+arr.shape[1]]
            max_nb = np.maximum(max_nb, view)
            
    return arr > max_nb

def cfar2d_ca(rd_db,
              train=(10, 8), guard=(2, 2),
              pfa=1e-4, min_snr_db=8.0,
              notch_doppler_bins=2,
              apply_nms=True, max_peaks=60,
              return_stats=False):
    """
    Cell-Averaging Constant False Alarm Rate (CA-CFAR) detector for 2D Range-Doppler maps.
    
    Args:
        rd_db (np.ndarray): Range-Doppler map in dB.
        train (tuple): Training cell dimensions (row, col).
        guard (tuple): Guard cell dimensions (row, col).
        pfa (float): Probability of False Alarm.
        min_snr_db (float): Minimum SNR threshold in dB.
        notch_doppler_bins (int): Number of bins to notch out around zero Doppler (clutter removal).
        apply_nms (bool): Whether to apply Non-Maximum Suppression.
        max_peaks (int): Maximum number of peaks to retain.
        return_stats (bool): If True, returns detection mask, noise level, and SNR map.
    
    Returns:
        np.ndarray or tuple: Detection mask (bool) or (mask, noise, snr).
    """
    rd_lin = 10.0 ** (rd_db / 10.0)
    H, W = rd_lin.shape
    mid = H // 2
    
    # Notch filter for zero-Doppler clutter
    if notch_doppler_bins > 0:
        k = int(notch_doppler_bins)
        rd_lin[mid - k: mid + k + 1, :] = np.minimum(
            rd_lin[mid - k: mid + k + 1, :],
            np.percentile(rd_lin, 10)
        )
        
    Tr, Tc = train
    Gr, Gc = guard
    
    # Compute moving sums for total window and guard window
    tot = _moving_sum_2d(rd_lin, Tr + Gr, Tc + Gc)
    gpl = _moving_sum_2d(rd_lin, Gr, Gc)
    
    # Estimate noise from training cells
    train_sum = tot - gpl
    n_train = (2*(Tr+Gr)+1)*(2*(Tc+Gc)+1) - (2*Gr+1)*(2*Gc+1)
    noise = np.maximum(train_sum / max(n_train, 1), 1e-12)
    
    # Calculate threshold based on Pfa
    alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)
    thresh = alpha * noise
    
    det = rd_lin > thresh
    snr_db = 10.0 * np.log10(np.maximum(rd_lin / noise, 1e-12))
    
    if min_snr_db and min_snr_db > 0:
        det &= snr_db >= min_snr_db
        
    if apply_nms:
        det &= nms2d(rd_lin, kernel=3)
        
    if max_peaks is not None and np.any(det):
        yy, xx = np.where(det)
        vals = rd_lin[yy, xx]
        if len(vals) > max_peaks:
            # Keep top-k peaks
            idx = np.argpartition(-vals, max_peaks - 1)[:max_peaks]
            keep = np.zeros_like(det, dtype=bool)
            keep[yy[idx], xx[idx]] = True
            det = keep
            
    if return_stats:
        return det, noise, snr_db
    return det

def plot_rd(ax, rd_db, ra, va, title, dynamic_db=35, percentile_clip=99.2, cmap='magma'):
    """
    Plot a Range-Doppler map on a given Matplotlib axis.
    """
    top = np.percentile(rd_db, percentile_clip)
    vmin = top - dynamic_db
    im = ax.imshow(rd_db, extent=[ra[0], ra[-1], va[0], va[-1]],
                   origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=top)
    ax.set_title(title)
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Velocity (m/s)")
    return im

def extract_detections(rd_db, det_mask, ra, va, noise_db=None, snr_db=None):
    """
    Extract detection list from a detection mask.
    
    Returns:
        list: List of dictionaries containing range, velocity, magnitude, etc.
    """
    yy, xx = np.where(det_mask)
    dets = []
    for y, x in zip(yy, xx):
        det = {'r': float(ra[x]), 'v': float(va[y]), 'mag_db': float(rd_db[y, x])}
        if snr_db is not None: det['snr_db'] = float(snr_db[y, x])
        if noise_db is not None: det['noise_db'] = float(noise_db[y, x])
        dets.append(det)
    return dets

def gt_rv_az(gts, sp):
    """
    Extract Ground Truth (GT) information in Range-Velocity-Azimuth format.
    """
    pos = np.array([0.0, 0.0, sp.H])
    out = []
    for gt in gts:
        c = np.array(gt['c'])
        v = np.array(gt['v'])
        d = c - pos
        r = np.linalg.norm(d)
        if r < 1e-6: continue
        u = d / r
        vr = float(np.dot(u, v))
        az = float(np.arctan2(c[1], c[0]))  # ground-plane azimuth
        out.append({'c': c, 'r': float(r), 'v': vr, 'az': az, 's': np.array(gt['s'])})
    return out

def match_dets_to_gts(dets, gt_rv, w_r=1.0, w_v=0.5):
    """
    Match detections to ground truth targets based on weighted distance in Range-Velocity space.
    """
    used = []
    pairs = []
    for d in dets:
        best_g = None
        best_cost = 1e9
        for gi, g in enumerate(gt_rv):
            cost = w_r*abs(d['r'] - g['r']) + w_v*abs(d['v'] - g['v'])
            if cost < best_cost:
                best_cost = cost
                best_g = gi
        if best_g is not None:
            pairs.append((d, best_g, best_cost))
            used.append(best_g)
    return pairs

def inside_cube_xy(x, y, g):
    """Check if a point (x, y) is inside the 2D footprint of a GT cube."""
    cx, cy = g['c'][0], g['c'][1]
    sx, sy = g['s'][0], g['s'][1]
    return (cx - sx/2.0 <= x <= cx + sx/2.0) and (cy - sy/2.0 <= y <= cy + sy/2.0)

def project_det_xy_using_gt_az(det, g):
    """Project a detection to XY plane using the matched GT's azimuth."""
    az = g['az']
    x = det['r'] * np.cos(az)
    y = det['r'] * np.sin(az)
    return x, y

def compute_metrics_from_pairs(pairs, gtinfo, sp):
    """
    Compute detection metrics (TP, FP, Precision, Recall, F1).
    
    A detection is a True Positive (TP) if its projection falls within the GT cube's footprint.
    """
    TP, FP = 0, 0
    per_gt_tp = {i:0 for i in range(len(gtinfo))}
    tp_pts, fp_pts = [], []

    for det, gi, _ in pairs:
        g = gtinfo[gi]
        x, y = project_det_xy_using_gt_az(det, g)
        
        # Check if within BEV limits
        if not (0 <= x <= sp.bev_r_max and -sp.bev_r_max/2 <= y <= sp.bev_r_max/2):
            FP += 1
            fp_pts.append((x,y,gi))
            continue
            
        if inside_cube_xy(x, y, g):
            TP += 1
            per_gt_tp[gi] += 1
            tp_pts.append((x,y,gi))
        else:
            FP += 1
            fp_pts.append((x,y,gi))

    detected_gts = sum(1 for k,v in per_gt_tp.items() if v > 0)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = detected_gts / max(1, len(gtinfo))
    f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0

    metrics = dict(TP=TP, FP=FP, detected_gts=detected_gts,
                   total_gts=len(gtinfo), precision=precision, recall=recall, f1=f1)
    return metrics, tp_pts, fp_pts

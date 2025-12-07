"""
Joint Radar+Comm Deep Model (no attention) that replaces CFAR + demapper.

Key points:
- Uses your existing generator (AIRadar_Comm_Dataset) to regenerate data.
- Does NOT load joint_dump.npy (only uses in-memory samples from generator).
- Deep model:
    * Radar branch: CNN + residual blocks -> heatmap (target probability).
    * Comm branch: CNN + residual blocks -> per-tone logits (symbol index).
- Modes:
    --mode train      : regenerate dataset, train model, then full evaluation + figs.
    --mode evaluate   : regenerate dataset, load ckpt, run full evaluation + figs.
    --mode inference  : regenerate dataset, load ckpt, run on a single sample + figs.

Per-sample visualizations:
- Radar 2D RD: GT targets, CFAR detections, DL detections + metrics overlay.
- Radar 3D RD: DL detections (via AIRadarLib if installed, else Matplotlib 3D).
- Comm: constellation (Tx, Rx, DL demap) + approximate eye diagram + BER/SER text.
"""

import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import math
import numpy as np
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------
# Import your previous generator code
# ----------------------------------------------------------------------
# IMPORTANT: change "your_prev_generator_file" to the actual module name
# that contains RADAR_COMM_CONFIGS and AIRadar_Comm_Dataset from your
# previous code snippet.
#
# Example, if your file is "AIradar_comm_dataset_g1b.py", use:
#   from AIradar_comm_dataset_g1b import RADAR_COMM_CONFIGS, AIRadar_Comm_Dataset

# Use local module path; no package prefix required
from AIradar_comm_dataset_g1 import RADAR_COMM_CONFIGS, AIRadar_Comm_Dataset


# Optional 3D visualization from your existing library
try:
    from AIRadarLib.visualization import plot_3d_range_doppler_map_with_ground_truth
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

C = 3e8  # speed of light

# ----------------------------------------------------------------------
# Wrapper Dataset for Deep Learning
# ----------------------------------------------------------------------
class RadarCommDeepDataset(Dataset):
    """
    Wraps AIRadar_Comm_Dataset and builds labels for deep training.

    Each AIRadar_Comm_Dataset(item) is like:
      {
        'mode': 'TRADITIONAL',
        'range_doppler_map': torch.Tensor [D,R] (dB),
        'range_axis': np.array [R],
        'velocity_axis': np.array [D],
        'target_info': {'targets': [{'range','velocity','rcs'}, ...], 'snr_db': ...},
        'cfar_detections': [...],
        'ofdm_map': np.array [N_syms, N_fft] (dB),
        'comm_info': {
            'ber', 'tx_symbols', 'rx_symbols',
            'num_data_syms', 'fft_size', 'tx_ints', 'mod_order'
        }
      }

    This wrapper returns:
      radar_input:  [1, D, R] (dB RD map)
      radar_target: [1, D, R] soft heatmap label
      comm_input:   [2, N_syms, N_fft] (I/Q of Rx symbols)
      comm_target:  [N_syms, N_fft] integers (symbol indices)
      meta:         dict (mod_order, idx, etc.)
    """

    def __init__(self, base_ds: AIRadar_Comm_Dataset, config_name: str, radar_sigma_cells: float = 1.5):
        super().__init__()
        self.base_ds = base_ds
        self.config_name = config_name
        self.cfg = RADAR_COMM_CONFIGS[config_name]
        assert self.cfg["mode"] == "TRADITIONAL", "Deep model currently supports TRADITIONAL configs only."
        self.radar_sigma_cells = radar_sigma_cells

    def __len__(self):
        return len(self.base_ds)

    @staticmethod
    def _build_radar_label(rdm, r_axis, v_axis, targets, sigma_cells: float = 1.5, radius: int = 3):
        """
        Build a Gaussian heatmap label centered on GT targets in (range,velocity) space.
        rdm: [D,R] (only shape used)
        r_axis: [R], v_axis: [D]
        """
        D_r, R_r = rdm.shape
        label = np.zeros_like(rdm, dtype=np.float32)
        sigma2 = sigma_cells ** 2

        for t in targets:
            r_m = t["range"]
            v_m = t["velocity"]
            # nearest indices on axes
            r_idx = int(np.argmin(np.abs(r_axis - r_m)))
            v_idx = int(np.argmin(np.abs(v_axis - v_m)))
            if not (0 <= r_idx < R_r and 0 <= v_idx < D_r):
                continue

            for dv in range(-radius, radius + 1):
                for dr in range(-radius, radius + 1):
                    rr = r_idx + dr
                    dd = v_idx + dv
                    if 0 <= rr < R_r and 0 <= dd < D_r:
                        dist2 = dr * dr + dv * dv
                        val = math.exp(-dist2 / (2 * sigma2))
                        label[dd, rr] = max(label[dd, rr], val)

        return label

    def __getitem__(self, idx):
        s = self.base_ds[idx]  # dict from your generator

        # Radar inputs
        rdm = s["range_doppler_map"].numpy().astype(np.float32)  # [D,R]
        r_axis = np.asarray(s["range_axis"])
        v_axis = np.asarray(s["velocity_axis"])
        targets = s["target_info"]["targets"]

        radar_label = self._build_radar_label(
            rdm, r_axis, v_axis, targets, sigma_cells=self.radar_sigma_cells
        )

        radar_input = torch.from_numpy(rdm).unsqueeze(0)            # [1,D,R]
        radar_target = torch.from_numpy(radar_label).unsqueeze(0)   # [1,D,R]

        # Comm inputs: use complex Rx symbols (I/Q) instead of dB magnitude
        comm_info = s["comm_info"]
        if comm_info is None:
            raise RuntimeError("comm_info is missing; ensure TRADITIONAL mode dataset.")

        num_syms = comm_info["num_data_syms"]
        fft_size = comm_info["fft_size"]
        mod_order = comm_info["mod_order"]

        tx_ints = np.array(comm_info["tx_ints"], dtype=np.int64)
        assert tx_ints.size == num_syms * fft_size, "Mismatch between tx_ints and OFDM grid shape."
        comm_label = tx_ints.reshape(num_syms, fft_size)            # [N_syms, N_fft]

        rx_syms = np.array(comm_info["rx_symbols"], dtype=np.complex64)
        assert rx_syms.size == num_syms * fft_size, "rx_symbols size mismatch."

        rx_grid = rx_syms.reshape(num_syms, fft_size)               # [N_syms, N_fft] complex
        real = rx_grid.real
        imag = rx_grid.imag
        comm_input = np.stack([real, imag], axis=0).astype(np.float32)  # [2,N_syms,N_fft]

        comm_input = torch.from_numpy(comm_input)                   # [2,N_syms,N_fft]
        comm_target = torch.from_numpy(comm_label)                  # [N_syms,N_fft]

        meta = {
            "mod_order": mod_order,
            "num_syms": num_syms,
            "fft_size": fft_size,
            "idx": idx,
        }

        return radar_input, radar_target, comm_input, comm_target, meta


# ----------------------------------------------------------------------
# Model: Conv + Residual (no attention)
# ----------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + x)
        return out


class ConvStem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RadarBranch(nn.Module):
    """
    Radar branch: small ResNet-like, easy to train and fairly global via dilations.
    """
    def __init__(self, in_ch=1, base_ch=32, num_blocks=4):
        super().__init__()
        self.stem = ConvStem(in_ch, base_ch)
        blocks = []
        # simple pattern of dilations to get larger receptive field
        dilations = [1, 2, 4, 1][:num_blocks]
        for d in dilations:
            blocks.append(ResidualBlock(base_ch, dilation=d))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, 1, kernel_size=1),  # logits heatmap
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


class CommBranch(nn.Module):
    """
    Comm branch: ResNet-like stack for OFDM Rx I/Q grid.
    """
    def __init__(self, in_ch=2, base_ch=32, num_blocks=4, max_mod_order=64):
        super().__init__()
        self.stem = ConvStem(in_ch, base_ch)
        blocks = []
        dilations = [1, 2, 4, 1][:num_blocks]
        for d in dilations:
            blocks.append(ResidualBlock(base_ch, dilation=d))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, max_mod_order, kernel_size=1),  # logits over constellation points
        )
        self.max_mod_order = max_mod_order

    def forward(self, x, mod_order: int):
        x = self.stem(x)
        x = self.blocks(x)
        logits_full = self.head(x)          # [B,max_M,H,W]
        return logits_full[:, :mod_order]   # [B,M,H,W]


class JointRadarCommNet(nn.Module):
    def __init__(self, max_mod_order=64, base_ch=48, num_blocks=4):
        super().__init__()
        self.radar_branch = RadarBranch(in_ch=1, base_ch=base_ch, num_blocks=num_blocks)
        self.comm_branch = CommBranch(in_ch=2, base_ch=base_ch, num_blocks=num_blocks,
                                      max_mod_order=max_mod_order)

    def forward(self, radar_input, comm_input, mod_order: int):
        """
        radar_input: [B,1,D,R]
        comm_input:  [B,2,N_syms,N_fft]
        """
        radar_logits = self.radar_branch(radar_input)
        comm_logits = self.comm_branch(comm_input, mod_order=mod_order)
        return radar_logits, comm_logits


# ----------------------------------------------------------------------
# Losses & Training
# ----------------------------------------------------------------------
def compute_losses(radar_logits, radar_target, comm_logits, comm_target, mod_order,
                   radar_pos_weight=2.0, lambda_comm=1.0):
    """
    radar_logits: [B,1,D,R], radar_target: [B,1,D,R]
    comm_logits:  [B,M,H,W],  comm_target: [B,H,W]
    """
    # Radar: BCE with moderate positive class weighting (want fewer crazy FPs)
    pos_weight = torch.tensor([radar_pos_weight], device=radar_logits.device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    radar_loss = bce(radar_logits, radar_target)

    # Comm: CrossEntropy per tone
    B, M, H, W = comm_logits.shape
    assert M == mod_order, f"Expected mod_order={mod_order}, got {M}"
    ce = nn.CrossEntropyLoss()
    logits_flat = comm_logits.permute(0, 2, 3, 1).reshape(-1, M)  # [(B*H*W),M]
    labels_flat = comm_target.reshape(-1).long()                  # [(B*H*W)]
    comm_loss = ce(logits_flat, labels_flat)

    total_loss = radar_loss + lambda_comm * comm_loss
    return total_loss, radar_loss, comm_loss


def train_one_epoch(model, loader, optimizer, device, mod_order,
                    grad_clip=1.0, lambda_comm=1.0):
    model.train()
    total_loss = total_radar = total_comm = 0.0
    n_samples = 0

    for radar_in, radar_tgt, comm_in, comm_tgt, meta in loader:
        radar_in = radar_in.to(device)
        radar_tgt = radar_tgt.to(device)
        comm_in = comm_in.to(device)
        comm_tgt = comm_tgt.to(device)
        bsz = radar_in.size(0)

        optimizer.zero_grad()
        radar_logits, comm_logits = model(radar_in, comm_in, mod_order=mod_order)
        loss, l_radar, l_comm = compute_losses(
            radar_logits, radar_tgt, comm_logits, comm_tgt,
            mod_order=mod_order, lambda_comm=lambda_comm
        )
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * bsz
        total_radar += l_radar.item() * bsz
        total_comm += l_comm.item() * bsz
        n_samples += bsz

    return total_loss / n_samples, total_radar / n_samples, total_comm / n_samples


@torch.no_grad()
def evaluate_ser(model, loader, device, mod_order):
    """Compute average symbol error rate across loader (proxy for BER)."""
    model.eval()
    total_ser = 0.0
    total_samples = 0

    for radar_in, radar_tgt, comm_in, comm_tgt, meta in loader:
        radar_in = radar_in.to(device)
        comm_in = comm_in.to(device)
        comm_tgt = comm_tgt.to(device)

        B = radar_in.size(0)
        _, comm_logits = model(radar_in, comm_in, mod_order=mod_order)
        pred = comm_logits.argmax(dim=1)  # [B,H,W]
        errors = (pred != comm_tgt).sum().item()
        ser = errors / comm_tgt.numel()
        total_ser += ser * B
        total_samples += B

    return total_ser / max(total_samples, 1)


# ----------------------------------------------------------------------
# Radar post-processing (DL heatmap -> detections)
# ----------------------------------------------------------------------
def postprocess_radar_heatmap(probs, r_axis, v_axis, cfg,
                              prob_thresh=0.7):
    """
    probs: [D,R] probability map from sigmoid.
    Returns detections list:
      [{'range_m','velocity_mps','range_idx','doppler_idx','score'}, ...]
    """
    params = cfg.get("cfar_params", {})
    nms_kernel = params.get("nms_kernel_size", 5)
    min_r = params.get("min_range_m", 0.0)
    min_v = params.get("min_speed_mps", 0.0)
    notch_k = params.get("notch_doppler_bins", 0)
    max_peaks = params.get("max_peaks", None)

    # NMS on probability map
    local_max = maximum_filter(probs, size=nms_kernel)
    detections_mask = (probs >= prob_thresh) & (probs == local_max)

    idxs = np.argwhere(detections_mask)
    center = len(v_axis) // 2
    candidates = []

    for d_idx, r_idx in idxs:
        if d_idx >= len(v_axis) or r_idx >= len(r_axis):
            continue
        range_m = r_axis[r_idx]
        vel_mps = v_axis[d_idx]

        # Similar filtering as CFAR
        if range_m < min_r or abs(vel_mps) < min_v:
            continue
        if notch_k > 0 and abs(d_idx - center) <= notch_k:
            continue

        candidates.append({
            "range_m": float(range_m),
            "velocity_mps": float(vel_mps),
            "range_idx": int(r_idx),
            "doppler_idx": int(d_idx),
            "score": float(probs[d_idx, r_idx]),
        })

    # Score-based peak limiting
    if max_peaks is not None:
        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[:max_peaks]

    # Coarser NMS on cell blocks (connected component pruning)
    pruned = []
    taken = set()
    neigh = params.get("nms_kernel_size", 5)
    for det in candidates:
        key = (det["doppler_idx"] // neigh, det["range_idx"] // neigh)
        if key in taken:
            continue
        taken.add(key)
        pruned.append(det)

    return pruned


# ----------------------------------------------------------------------
# Radar metrics (reuse AIRadar_Comm_Dataset._evaluate_metrics)
# ----------------------------------------------------------------------
def radar_metrics_from_dataset(base_ds, targets, detections):
    """
    Thin wrapper to call AIRadar_Comm_Dataset._evaluate_metrics
    so we keep exactly the same matching logic as your previous code.
    """
    metrics, matched_pairs, unmatched_targets, unmatched_detections = base_ds._evaluate_metrics(
        targets, detections
    )
    return metrics, matched_pairs, unmatched_targets, unmatched_detections


# ----------------------------------------------------------------------
# Communication visualization
# ----------------------------------------------------------------------
def generate_qam_constellation(mod_order):
    if mod_order == 4:
        pts = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    elif mod_order == 16:
        x = np.arange(-3, 4, 2)
        X, Y = np.meshgrid(x, x)
        pts = (X + 1j * Y).flatten() / np.sqrt(10)
    elif mod_order == 64:
        x = np.arange(-7, 8, 2)
        X, Y = np.meshgrid(x, x)
        pts = (X + 1j * Y).flatten() / np.sqrt(42)
    else:
        raise ValueError(f"Unsupported mod_order {mod_order}")
    return pts


def plot_eye_diagram(symbols, sps=2, save_path=None):
    """
    Approximate eye diagram using real part of equalized Rx symbols.
    Not a full oversampled eye, but gives a quick visual sanity check.
    """
    x = np.real(symbols)
    seq = np.repeat(x, sps)  # simple upsample

    seg_len = 4 * sps   # 4-symbol window
    n_seg = len(seq) // seg_len

    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(n_seg):
        seg = seq[i * seg_len:(i + 1) * seg_len]
        t = np.arange(len(seg)) / sps
        ax.plot(t, seg, alpha=0.2)

    ax.set_xlabel("Symbol time")
    ax.set_ylabel("Amplitude (I)")
    ax.set_title("Eye Diagram (approx)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        return fig, ax


def plot_comm_results(comm_info, pred_ints, mod_order, dl_ser, save_prefix):
    """
    Save:
      - constellation comparison (Tx, Rx, DL demap)
      - approximate eye diagram
      - overlay BER/SER stats
    """
    tx_syms = np.array(comm_info["tx_symbols"])
    rx_syms = np.array(comm_info["rx_symbols"])
    const_pts = generate_qam_constellation(mod_order)
    tx_ints = np.array(comm_info["tx_ints"], dtype=int)
    tx_pts = const_pts[tx_ints]
    pred_pts = const_pts[pred_ints.astype(int)]
    baseline_ber = comm_info.get("ber", 0.0)

    # Constellation + eye
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Constellation
    ax0 = ax[0]
    idx = np.arange(len(tx_syms))
    if len(idx) > 2000:
        sel = np.random.choice(idx, 2000, replace=False)
    else:
        sel = idx

    ax0.scatter(np.real(rx_syms[sel]), np.imag(rx_syms[sel]),
                s=8, alpha=0.4, label="Rx (ZF+LS)")
    ax0.scatter(np.real(tx_syms[sel]), np.imag(tx_syms[sel]),
                s=12, alpha=0.6, marker="x", label="Tx")
    ax0.scatter(np.real(pred_pts[sel]), np.imag(pred_pts[sel]),
                s=10, alpha=0.5, marker="+", label="DL Demap")

    ax0.set_title(f"{mod_order}-QAM Constellation")
    ax0.set_xlabel("I")
    ax0.set_ylabel("Q")
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=8)
    ax0.set_aspect("equal")

    # Eye: render separately then embed as image
    eye_path = save_prefix + "_eye.png"
    plot_eye_diagram(rx_syms, sps=2, save_path=eye_path)
    img = plt.imread(eye_path)

    ax1 = ax[1]
    ax1.imshow(img)
    ax1.axis("off")
    text = (
        f"Baseline BER: {baseline_ber:.3e}\n"
        f"DL SER≈BER: {dl_ser:.3e}\n"
        f"#Symbols: {len(tx_ints)}"
    )
    props = dict(boxstyle="round", facecolor="white", alpha=0.8)
    ax1.text(0.02, 0.98, text, transform=ax1.transAxes,
             fontsize=9, verticalalignment="top",
             bbox=props, family="monospace")

    plt.tight_layout()
    const_path = save_prefix + "_constellation_eye.png"
    plt.savefig(const_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Radar visualization (2D & 3D)
# ----------------------------------------------------------------------
def plot_radar_2d_comparison(rdm_db, r_axis, v_axis,
                             targets, cfar_dets, dl_dets,
                             metrics_cfar, metrics_dl,
                             save_path):
    fig, ax = plt.subplots(figsize=(12, 8))

    dr = r_axis[1] - r_axis[0] if len(r_axis) > 1 else 1.0
    dv = v_axis[1] - v_axis[0] if len(v_axis) > 1 else 1.0
    extent = [r_axis[0] - dr/2, r_axis[-1] + dr/2,
              v_axis[0] - dv/2, v_axis[-1] + dv/2]

    im = ax.imshow(rdm_db, extent=extent, origin="lower",
                   cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="Magnitude (dB)")

    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Range-Doppler Map: CFAR vs Deep Model")

    # Ground truth
    for t in targets:
        ax.scatter(t["range"], t["velocity"],
                   facecolors="none", edgecolors="lime",
                   s=150, linewidth=2, label="GT")

    # CFAR detections
    for d in cfar_dets:
        ax.scatter(d["range_m"], d["velocity_mps"], marker="x", color="cyan",
                   s=80, linewidth=2, label="CFAR")

    # DL detections
    for d in dl_dets:
        ax.scatter(d["range_m"], d["velocity_mps"], marker="+", color="red",
                   s=80, linewidth=2, label="DL")

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

    text = (
        "CFAR Metrics:\n"
        f"  TP={metrics_cfar['tp']} FP={metrics_cfar['fp']} FN={metrics_cfar['fn']}\n"
        f"  dR={metrics_cfar['mean_range_error']:.2f} m, dV={metrics_cfar['mean_velocity_error']:.2f} m/s\n"
        "\nDL Metrics:\n"
        f"  TP={metrics_dl['tp']} FP={metrics_dl['fp']} FN={metrics_dl['fn']}\n"
        f"  dR={metrics_dl['mean_range_error']:.2f} m, dV={metrics_dl['mean_velocity_error']:.2f} m/s\n"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.6)
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", bbox=props,
            family="monospace")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_radar_3d(rdm_db, r_axis, v_axis, targets, detections, save_path):
    """
    3D RD map with DL detections. If AIRadarLib is available, use it;
    otherwise fallback to Matplotlib 3D surface.
    """
    if VISUALIZATION_AVAILABLE:
        range_res = r_axis[1] - r_axis[0] if len(r_axis) > 1 else 1.0
        vel_res = v_axis[1] - v_axis[0] if len(v_axis) > 1 else 1.0

        converted_targets = []
        for t in targets:
            ct = t.copy()
            ct["distance"] = t["range"]
            converted_targets.append(ct)

        cleaned_dets = []
        for d in detections:
            d2 = d.copy()
            d2["range_idx"] = int(d2.get("range_idx", 0))
            d2["doppler_idx"] = int(d2.get("doppler_idx", 0))
            cleaned_dets.append(d2)

        plot_3d_range_doppler_map_with_ground_truth(
            rd_map=rdm_db,
            targets=converted_targets,
            range_resolution=range_res,
            velocity_resolution=vel_res,
            num_range_bins=rdm_db.shape[1],
            num_doppler_bins=rdm_db.shape[0],
            save_path=save_path,
            apply_doppler_centering=True,
            detections=cleaned_dets,
            view_range_limits=(r_axis[0], r_axis[-1]),
            view_velocity_limits=(v_axis[0], v_axis[-1]),
            is_db=True,
            stride=4,
        )
    else:
        R_grid, D_grid = np.meshgrid(r_axis, v_axis)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(R_grid, D_grid, rdm_db, cmap="viridis")
        ax.set_xlabel("Range (m)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_zlabel("Mag (dB)")
        fig.colorbar(surf, shrink=0.5, aspect=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

def make_snr_sweep_plots(per_sample_stats, out_dir):
    """
    per_sample_stats: list of dicts with keys:
      'snr_db', 'metrics_cfar', 'metrics_dl',
      'baseline_ber', 'dl_ser'
    Uses the SNR stored in target_info['snr_db'] in your generator.
    """
    snrs = np.array([s["snr_db"] for s in per_sample_stats], dtype=np.float32)

    def f1_from_m(metrics):
        tp = metrics["tp"]
        fp = metrics["fp"]
        fn = metrics["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    cfar_prec_all, cfar_rec_all, cfar_f1_all = [], [], []
    dl_prec_all, dl_rec_all, dl_f1_all = [], [], []
    baseline_ber_all, dl_ser_all = [], []

    for s in per_sample_stats:
        m_c = s["metrics_cfar"]
        m_d = s["metrics_dl"]
        p_c, r_c, f_c = f1_from_m(m_c)
        p_d, r_d, f_d = f1_from_m(m_d)
        cfar_prec_all.append(p_c)
        cfar_rec_all.append(r_c)
        cfar_f1_all.append(f_c)
        dl_prec_all.append(p_d)
        dl_rec_all.append(r_d)
        dl_f1_all.append(f_d)
        baseline_ber_all.append(s["baseline_ber"])
        dl_ser_all.append(s["dl_ser"])

    cfar_prec_all = np.array(cfar_prec_all)
    cfar_rec_all = np.array(cfar_rec_all)
    cfar_f1_all = np.array(cfar_f1_all)
    dl_prec_all = np.array(dl_prec_all)
    dl_rec_all = np.array(dl_rec_all)
    dl_f1_all = np.array(dl_f1_all)
    baseline_ber_all = np.array(baseline_ber_all)
    dl_ser_all = np.array(dl_ser_all)

    # Bin by SNR (1 dB bins)
    snr_min = math.floor(snrs.min())
    snr_max = math.ceil(snrs.max())
    bin_edges = np.arange(snr_min, snr_max + 1, 1.0)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def bin_avg(values):
        out = []
        for i in range(len(bin_edges) - 1):
            mask = (snrs >= bin_edges[i]) & (snrs < bin_edges[i+1])
            if np.any(mask):
                out.append(values[mask].mean())
            else:
                out.append(np.nan)
        return np.array(out)

    cfar_f1_b = bin_avg(cfar_f1_all)
    dl_f1_b = bin_avg(dl_f1_all)
    cfar_prec_b = bin_avg(cfar_prec_all)
    dl_prec_b = bin_avg(dl_prec_all)
    cfar_rec_b = bin_avg(cfar_rec_all)
    dl_rec_b = bin_avg(dl_rec_all)
    base_ber_b = bin_avg(baseline_ber_all)
    dl_ser_b = bin_avg(dl_ser_all)

    # 1) Radar F1 vs SNR
    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, cfar_f1_b, "o-", label="CFAR F1")
    plt.plot(bin_centers, dl_f1_b, "o-", label="Deep Radar F1")
    plt.xlabel("SNR (dB)")
    plt.ylabel("F1 Score")
    plt.title("Radar F1 vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "snr_sweep_radar_f1.png"), dpi=150)
    plt.close()

    # 2) Deep Radar Precision/Recall vs SNR
    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, dl_prec_b, "o-", label="Precision")
    plt.plot(bin_centers, dl_rec_b, "o-", label="Recall")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Metric")
    plt.title("Deep Radar Precision/Recall vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "snr_sweep_radar_prec_rec.png"), dpi=150)
    plt.close()

    # 3) Comm BER vs SNR (log scale)
    plt.figure(figsize=(8, 5))
    plt.semilogy(bin_centers, base_ber_b, "o-", label="Baseline BER")
    plt.semilogy(bin_centers, dl_ser_b, "o-", label="Deep SER≈BER")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER / SER")
    plt.title("Communication BER vs SNR")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "snr_sweep_comm_ber.png"), dpi=150)
    plt.close()

def make_radar_threshold_curves(radar_probs_all, r_axes_all, v_axes_all,
                                targets_all, base_ds, cfg, out_dir):
    """
    Build radar detection curves for the Deep model by sweeping the
    probability threshold and counting TP/FP/FN over the whole dataset.

    Generates:
      - Recall vs average FP/frame (ROC-like curve for radar).
      - Precision/Recall vs threshold.
    """
    thresholds = np.linspace(0.1, 0.9, 17)

    recalls = []
    precisions = []
    avg_fp_per_frame = []

    for T in thresholds:
        total_tp = total_fp = total_fn = total_targets = 0
        for probs, r_axis, v_axis, targets in zip(
            radar_probs_all, r_axes_all, v_axes_all, targets_all
        ):
            dl_dets = postprocess_radar_heatmap(
                probs, r_axis, v_axis, cfg, prob_thresh=float(T)
            )
            metrics, _, _, _ = base_ds._evaluate_metrics(targets, dl_dets)
            total_tp += metrics["tp"]
            total_fp += metrics["fp"]
            total_fn += metrics["fn"]
            total_targets += metrics["total_targets"]

        recall = total_tp / total_targets if total_targets > 0 else 0.0
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recalls.append(recall)
        precisions.append(precision)
        avg_fp_per_frame.append(total_fp / len(radar_probs_all))

    recalls = np.array(recalls)
    precisions = np.array(precisions)
    avg_fp_per_frame = np.array(avg_fp_per_frame)

    # 1) ROC-like: Recall vs Average FP/frame
    plt.figure(figsize=(8, 5))
    plt.plot(avg_fp_per_frame, recalls, "o-")
    for T, x, y in zip(thresholds, avg_fp_per_frame, recalls):
        plt.annotate(f"{T:.2f}", (x, y),
                     textcoords="offset points", xytext=(4, 4), fontsize=7)
    plt.xlabel("Average FP per frame")
    plt.ylabel("Recall (TPR)")
    plt.title("Deep Radar Detection Curve (Recall vs FP/frame)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "radar_detection_curve_recall_vs_fp.png"), dpi=150)
    plt.close()

    # 2) Threshold vs Precision/Recall
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions, "o-", label="Precision")
    plt.plot(thresholds, recalls, "o-", label="Recall")
    plt.xlabel("Probability Threshold")
    plt.ylabel("Metric")
    plt.title("Deep Radar Precision/Recall vs Threshold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "radar_prec_rec_vs_threshold.png"), dpi=150)
    plt.close()

    # Optional: dump raw values
    with open(os.path.join(out_dir, "radar_threshold_curve.txt"), "w") as f:
        f.write("# T  precision  recall  avg_fp_per_frame\n")
        for T, p, r, fp in zip(thresholds, precisions, recalls, avg_fp_per_frame):
            f.write(f"{T:.3f} {p:.6f} {r:.6f} {fp:.6f}\n")
# ----------------------------------------------------------------------
# Full evaluation: radar & comm + visualizations
# ----------------------------------------------------------------------
@torch.no_grad()
def run_full_evaluation(model, deep_ds, base_ds, cfg, device, out_dir,
                        mod_order, max_vis=10, prob_thresh=0.7):
    os.makedirs(out_dir, exist_ok=True)

    total_cfar = {"tp": 0, "fp": 0, "fn": 0, "range_err": [], "vel_err": [], "targets": 0}
    total_dl = {"tp": 0, "fp": 0, "fn": 0, "range_err": [], "vel_err": [], "targets": 0}
    baseline_bers = []
    dl_sers = []

    # For SNR sweep & ROC-style curves
    per_sample_stats = []
    radar_probs_all = []
    r_axes_all = []
    v_axes_all = []
    targets_all = []

    for idx in range(len(deep_ds)):
        # Base generator sample
        s = base_ds[idx]
        rdm = s["range_doppler_map"].numpy().astype(np.float32)
        r_axis = np.asarray(s["range_axis"])
        v_axis = np.asarray(s["velocity_axis"])
        targets = s["target_info"]["targets"]
        snr_db = float(s["target_info"].get("snr_db", 0.0))
        cfar_dets = s["cfar_detections"]
        comm_info = s["comm_info"]

        # --- CFAR metrics ---
        metrics_cfar, _, _, _ = radar_metrics_from_dataset(base_ds, targets, cfar_dets)
        total_cfar["tp"] += metrics_cfar["tp"]
        total_cfar["fp"] += metrics_cfar["fp"]
        total_cfar["fn"] += metrics_cfar["fn"]
        total_cfar["targets"] += metrics_cfar["total_targets"]
        total_cfar["range_err"].append(metrics_cfar["mean_range_error"])
        total_cfar["vel_err"].append(metrics_cfar["mean_velocity_error"])

        # --- Deep model forward ---
        radar_in, _, comm_in, comm_tgt, meta = deep_ds[idx]
        radar_in_b = radar_in.unsqueeze(0).to(device)
        comm_in_b = comm_in.unsqueeze(0).to(device)
        comm_tgt_b = comm_tgt.unsqueeze(0).to(device)

        model.eval()
        radar_logits, comm_logits = model(radar_in_b, comm_in_b, mod_order=mod_order)
        radar_probs = torch.sigmoid(radar_logits)[0, 0].cpu().numpy()
        dl_dets = postprocess_radar_heatmap(radar_probs, r_axis, v_axis, cfg,
                                            prob_thresh=prob_thresh)

        metrics_dl, _, _, _ = radar_metrics_from_dataset(base_ds, targets, dl_dets)
        total_dl["tp"] += metrics_dl["tp"]
        total_dl["fp"] += metrics_dl["fp"]
        total_dl["fn"] += metrics_dl["fn"]
        total_dl["targets"] += metrics_dl["total_targets"]
        total_dl["range_err"].append(metrics_dl["mean_range_error"])
        total_dl["vel_err"].append(metrics_dl["mean_velocity_error"])

        # --- Comm metrics ---
        baseline_ber = float(comm_info.get("ber", 0.0))
        baseline_bers.append(baseline_ber)

        pred_ints = comm_logits.argmax(dim=1)[0].cpu().numpy().reshape(-1)
        gt_ints = comm_tgt_b.cpu().numpy().reshape(-1)
        ser = float((pred_ints != gt_ints).mean())
        dl_sers.append(ser)

        # Store for SNR sweep & ROC-style curves
        per_sample_stats.append({
            "snr_db": snr_db,
            "metrics_cfar": metrics_cfar,
            "metrics_dl": metrics_dl,
            "baseline_ber": baseline_ber,
            "dl_ser": ser,
        })
        radar_probs_all.append(radar_probs)
        r_axes_all.append(r_axis)
        v_axes_all.append(v_axis)
        targets_all.append(targets)

        # --- Per-sample visualizations (unchanged logic) ---
        if idx < max_vis:
            prefix = os.path.join(out_dir, f"sample_{idx:03d}")
            rdm_norm = rdm - np.max(rdm)

            plot_radar_2d_comparison(
                rdm_norm, r_axis, v_axis,
                targets, cfar_dets, dl_dets,
                metrics_cfar, metrics_dl,
                save_path=prefix + "_radar_2d.png",
            )

            plot_radar_3d(
                rdm_norm, r_axis, v_axis,
                targets, dl_dets,
                save_path=prefix + "_radar_3d_dl.png",
            )

            plot_comm_results(comm_info, pred_ints, cfg["mod_order"], ser,
                              save_prefix=prefix + "_comm")

    # --- Aggregate metrics over whole dataset ---
    def agg(total):
        tp = total["tp"]
        fp = total["fp"]
        fn = total["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_range = float(np.mean(total["range_err"])) if total["range_err"] else 0.0
        mean_vel = float(np.mean(total["vel_err"])) if total["vel_err"] else 0.0
        return precision, recall, f1, mean_range, mean_vel

    p_cfar, r_cfar, f1_cfar, mr_cfar, mv_cfar = agg(total_cfar)
    p_dl, r_dl, f1_dl, mr_dl, mv_dl = agg(total_dl)
    mean_baseline_ber = float(np.mean(baseline_bers)) if baseline_bers else 0.0
    mean_dl_ser = float(np.mean(dl_sers)) if dl_sers else 0.0

    summary = []
    summary.append("=== Radar Metrics (Classical CFAR) ===")
    summary.append(f"Targets: {total_cfar['targets']}")
    summary.append(f"TP={total_cfar['tp']} FP={total_cfar['fp']} FN={total_cfar['fn']}")
    summary.append(f"Precision={p_cfar:.4f} Recall={r_cfar:.4f} F1={f1_cfar:.4f}")
    summary.append(f"Mean Range Error={mr_cfar:.3f} m")
    summary.append(f"Mean Velocity Error={mv_cfar:.3f} m/s")
    summary.append("")
    summary.append("=== Radar Metrics (Deep Model) ===")
    summary.append(f"Targets: {total_dl['targets']}")
    summary.append(f"TP={total_dl['tp']} FP={total_dl['fp']} FN={total_dl['fn']}")
    summary.append(f"Precision={p_dl:.4f} Recall={r_dl:.4f} F1={f1_dl:.4f}")
    summary.append(f"Mean Range Error={mr_dl:.3f} m")
    summary.append(f"Mean Velocity Error={mv_dl:.3f} m/s")
    summary.append("")
    summary.append("=== Communication Metrics ===")
    summary.append(f"Baseline Mean BER={mean_baseline_ber:.5e}")
    summary.append(f"Deep Model SER≈BER={mean_dl_ser:.5e}")

    txt = "\n".join(summary)
    print(txt)

    with open(os.path.join(out_dir, "evaluation_summary.txt"), "w") as f:
        f.write(txt)

    # ---- New: SNR sweep plots ----
    make_snr_sweep_plots(per_sample_stats, out_dir)

    # ---- New: Radar ROC-like curves (threshold sweep) ----
    make_radar_threshold_curves(
        radar_probs_all, r_axes_all, v_axes_all,
        targets_all, base_ds, cfg, out_dir
    )

@torch.no_grad()
def run_full_evaluationv1(model, deep_ds, base_ds, cfg, device, out_dir,
                        mod_order, max_vis=10, prob_thresh=0.7):
    os.makedirs(out_dir, exist_ok=True)

    total_cfar = {"tp": 0, "fp": 0, "fn": 0, "range_err": [], "vel_err": [], "targets": 0}
    total_dl = {"tp": 0, "fp": 0, "fn": 0, "range_err": [], "vel_err": [], "targets": 0}
    baseline_bers = []
    dl_sers = []

    for idx in range(len(deep_ds)):
        # base_ds sample (original generator)
        s = base_ds[idx]
        rdm = s["range_doppler_map"].numpy().astype(np.float32)
        r_axis = np.asarray(s["range_axis"])
        v_axis = np.asarray(s["velocity_axis"])
        targets = s["target_info"]["targets"]
        cfar_dets = s["cfar_detections"]
        comm_info = s["comm_info"]

        # CFAR metrics
        metrics_cfar, _, _, _ = radar_metrics_from_dataset(base_ds, targets, cfar_dets)
        total_cfar["tp"] += metrics_cfar["tp"]
        total_cfar["fp"] += metrics_cfar["fp"]
        total_cfar["fn"] += metrics_cfar["fn"]
        total_cfar["targets"] += metrics_cfar["total_targets"]
        total_cfar["range_err"].append(metrics_cfar["mean_range_error"])
        total_cfar["vel_err"].append(metrics_cfar["mean_velocity_error"])

        # Deep dataset tensors for this sample
        radar_in, _, comm_in, comm_tgt, meta = deep_ds[idx]
        model.eval()
        radar_in_b = radar_in.unsqueeze(0).to(device)
        comm_in_b = comm_in.unsqueeze(0).to(device)
        comm_tgt_b = comm_tgt.unsqueeze(0).to(device)

        radar_logits, comm_logits = model(radar_in_b, comm_in_b, mod_order=mod_order)
        radar_probs = torch.sigmoid(radar_logits)[0, 0].cpu().numpy()
        dl_dets = postprocess_radar_heatmap(radar_probs, r_axis, v_axis, cfg,
                                            prob_thresh=prob_thresh)

        metrics_dl, _, _, _ = radar_metrics_from_dataset(base_ds, targets, dl_dets)
        total_dl["tp"] += metrics_dl["tp"]
        total_dl["fp"] += metrics_dl["fp"]
        total_dl["fn"] += metrics_dl["fn"]
        total_dl["targets"] += metrics_dl["total_targets"]
        total_dl["range_err"].append(metrics_dl["mean_range_error"])
        total_dl["vel_err"].append(metrics_dl["mean_velocity_error"])

        # Comm metrics
        baseline_ber = comm_info.get("ber", 0.0)
        baseline_bers.append(baseline_ber)

        pred_ints = comm_logits.argmax(dim=1)[0].cpu().numpy().reshape(-1)
        gt_ints = comm_tgt_b.cpu().numpy().reshape(-1)
        ser = (pred_ints != gt_ints).mean()
        dl_sers.append(ser)

        # Visualizations for first few samples
        if idx < max_vis:
            prefix = os.path.join(out_dir, f"sample_{idx:03d}")
            rdm_norm = rdm - np.max(rdm)

            plot_radar_2d_comparison(
                rdm_norm, r_axis, v_axis,
                targets, cfar_dets, dl_dets,
                metrics_cfar, metrics_dl,
                save_path=prefix + "_radar_2d.png",
            )

            plot_radar_3d(
                rdm_norm, r_axis, v_axis,
                targets, dl_dets,
                save_path=prefix + "_radar_3d_dl.png",
            )

            plot_comm_results(comm_info, pred_ints, mod_order, ser,
                              save_prefix=prefix + "_comm")

    # Aggregate metrics
    def agg(total):
        tp = total["tp"]
        fp = total["fp"]
        fn = total["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_range = float(np.mean(total["range_err"])) if total["range_err"] else 0.0
        mean_vel = float(np.mean(total["vel_err"])) if total["vel_err"] else 0.0
        return precision, recall, f1, mean_range, mean_vel

    p_cfar, r_cfar, f1_cfar, mr_cfar, mv_cfar = agg(total_cfar)
    p_dl, r_dl, f1_dl, mr_dl, mv_dl = agg(total_dl)
    mean_baseline_ber = float(np.mean(baseline_bers)) if baseline_bers else 0.0
    mean_dl_ser = float(np.mean(dl_sers)) if dl_sers else 0.0

    summary = []
    summary.append("=== Radar Metrics (Classical CFAR) ===")
    summary.append(f"Targets: {total_cfar['targets']}")
    summary.append(f"TP={total_cfar['tp']} FP={total_cfar['fp']} FN={total_cfar['fn']}")
    summary.append(f"Precision={p_cfar:.4f} Recall={r_cfar:.4f} F1={f1_cfar:.4f}")
    summary.append(f"Mean Range Error={mr_cfar:.3f} m")
    summary.append(f"Mean Velocity Error={mv_cfar:.3f} m/s")
    summary.append("")
    summary.append("=== Radar Metrics (Deep Model) ===")
    summary.append(f"Targets: {total_dl['targets']}")
    summary.append(f"TP={total_dl['tp']} FP={total_dl['fp']} FN={total_dl['fn']}")
    summary.append(f"Precision={p_dl:.4f} Recall={r_dl:.4f} F1={f1_dl:.4f}")
    summary.append(f"Mean Range Error={mr_dl:.3f} m")
    summary.append(f"Mean Velocity Error={mv_dl:.3f} m/s")
    summary.append("")
    summary.append("=== Communication Metrics ===")
    summary.append(f"Baseline Mean BER={mean_baseline_ber:.5e}")
    summary.append(f"Deep Model SER≈BER={mean_dl_ser:.5e}")

    txt = "\n".join(summary)
    print(txt)

    with open(os.path.join(out_dir, "evaluation_summary.txt"), "w") as f:
        f.write(txt)


# ----------------------------------------------------------------------
# Utilities & main
# ----------------------------------------------------------------------
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "inference"],
                        default="train",
                        help="train: fit model; evaluate: full eval; inference: single-sample viz")
    parser.add_argument("--config_name", type=str, default="CN0566_TRADITIONAL",
                        choices=list(RADAR_COMM_CONFIGS.keys()))
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of simulated samples to generate with AIRadar_Comm_Dataset.")
    parser.add_argument("--data_root", type=str,
                        default="data/AIradar_comm_dataset_deep",
                        help="Base directory passed to AIRadar_Comm_Dataset.save_path.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_comm", type=float, default=1.0)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="data/AIradar_comm_dataset_deep")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint path for evaluate/inference modes")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="For inference mode: which sample index to visualize.")
    parser.add_argument("--prob_thresh", type=float, default=0.7,
                        help="DL radar detection probability threshold.")
    parser.add_argument("--draw_fig_gen", action="store_true",
                        help="If set, generator will also draw its own figures (slower).")
    args = parser.parse_args()

    set_seed(42)

    cfg = RADAR_COMM_CONFIGS[args.config_name]
    assert cfg["mode"] == "TRADITIONAL", "Deep model currently supports TRADITIONAL configs only."

    # ------------------------------------------------------------------
    # 1) REGENERATE DATASET using your original generator (no joint_dump load)
    # ------------------------------------------------------------------
    save_path = os.path.join(args.data_root, args.config_name)
    os.makedirs(save_path, exist_ok=True)

    print(f"[Generator] Creating AIRadar_Comm_Dataset for {args.config_name} with {args.num_samples} samples")
    base_ds = AIRadar_Comm_Dataset(
        config_name=args.config_name,
        num_samples=args.num_samples,
        save_path=save_path,
        drawfig=args.draw_fig_gen,
    )
    print(f"[Generator] Dataset generated in memory. Samples: {len(base_ds)}")

    # Wrap for deep learning
    deep_ds = RadarCommDeepDataset(base_ds, args.config_name)

    device = torch.device(args.device)
    mod_order = cfg["mod_order"]

    model = JointRadarCommNet(max_mod_order=64, base_ch=48, num_blocks=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # -------------------------- TRAIN MODE --------------------------
    if args.mode == "train":
        n_total = len(deep_ds)
        n_train = int(0.8 * n_total)
        n_val = n_total - n_train
        train_ds, val_ds = torch.utils.data.random_split(deep_ds, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)

        best_ser = float("inf")
        os.makedirs(args.out_dir, exist_ok=True)

        for epoch in range(1, args.epochs + 1):
            train_loss, train_radar, train_comm = train_one_epoch(
                model, train_loader, optimizer, device,
                mod_order=mod_order, lambda_comm=args.lambda_comm
            )
            val_ser = evaluate_ser(model, val_loader, device, mod_order)

            print(f"[Epoch {epoch:02d}] "
                  f"TrainLoss={train_loss:.4f} (Radar={train_radar:.4f}, Comm={train_comm:.4f}) "
                  f"| Val SER≈BER={val_ser:.4e}")

            if val_ser < best_ser:
                best_ser = val_ser
                ckpt_path = os.path.join(args.out_dir,
                                         f"joint_net_{args.config_name}_best.pt")
                torch.save({
                    "model_state": model.state_dict(),
                    "config_name": args.config_name,
                    "mod_order": mod_order,
                }, ckpt_path)
                print(f"  -> New best model saved to {ckpt_path}")

        # After training, evaluate on full dataset + save visualizations
        eval_dir = os.path.join(args.out_dir, "train_eval")
        run_full_evaluation(model, deep_ds, base_ds, cfg, device, eval_dir,
                            mod_order=mod_order, max_vis=10,
                            prob_thresh=args.prob_thresh)

    # ------------------------ EVALUATE MODE -------------------------
    elif args.mode == "evaluate":
        assert args.ckpt is not None, "Provide --ckpt for evaluate mode."
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)

        eval_dir = os.path.join(args.out_dir, "evaluate")
        run_full_evaluation(model, deep_ds, base_ds, cfg, device, eval_dir,
                            mod_order=mod_order, max_vis=10,
                            prob_thresh=args.prob_thresh)

    # ------------------------ INFERENCE MODE ------------------------
    elif args.mode == "inference":
        assert args.ckpt is not None, "Provide --ckpt for inference mode."
        ckpt = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()

        idx = args.sample_idx
        if idx < 0 or idx >= len(deep_ds):
            raise IndexError(f"sample_idx {idx} out of range [0, {len(deep_ds)-1}]")

        radar_in, radar_tgt, comm_in, comm_tgt, meta = deep_ds[idx]
        s = base_ds[idx]

        rdm = s["range_doppler_map"].numpy().astype(np.float32)
        r_axis = np.asarray(s["range_axis"])
        v_axis = np.asarray(s["velocity_axis"])
        targets = s["target_info"]["targets"]
        cfar_dets = s["cfar_detections"]
        comm_info = s["comm_info"]

        out_dir = os.path.join(args.out_dir, f"inference_sample_{idx:03d}")
        os.makedirs(out_dir, exist_ok=True)

        radar_in_b = radar_in.unsqueeze(0).to(device)
        comm_in_b = comm_in.unsqueeze(0).to(device)
        comm_tgt_b = comm_tgt.unsqueeze(0).to(device)

        radar_logits, comm_logits = model(radar_in_b, comm_in_b, mod_order=mod_order)
        radar_probs = torch.sigmoid(radar_logits)[0, 0].cpu().numpy()
        dl_dets = postprocess_radar_heatmap(radar_probs, r_axis, v_axis, cfg,
                                            prob_thresh=args.prob_thresh)

        metrics_cfar, _, _, _ = radar_metrics_from_dataset(base_ds, targets, cfar_dets)
        metrics_dl, _, _, _ = radar_metrics_from_dataset(base_ds, targets, dl_dets)

        rdm_norm = rdm - np.max(rdm)
        prefix = os.path.join(out_dir, f"sample_{idx:03d}")

        plot_radar_2d_comparison(
            rdm_norm, r_axis, v_axis,
            targets, cfar_dets, dl_dets,
            metrics_cfar, metrics_dl,
            save_path=prefix + "_radar_2d.png",
        )

        plot_radar_3d(
            rdm_norm, r_axis, v_axis,
            targets, dl_dets,
            save_path=prefix + "_radar_3d_dl.png",
        )

        pred_ints = comm_logits.argmax(dim=1)[0].cpu().numpy().reshape(-1)
        gt_ints = comm_tgt_b.cpu().numpy().reshape(-1)
        ser = (pred_ints != gt_ints).mean()

        plot_comm_results(comm_info, pred_ints, mod_order, ser,
                          save_prefix=prefix + "_comm")

        with open(os.path.join(out_dir, "inference_summary.txt"), "w") as f:
            f.write("Inference summary\n")
            f.write("CFAR metrics:\n")
            f.write(str(metrics_cfar) + "\n")
            f.write("Deep model metrics:\n")
            f.write(str(metrics_dl) + "\n")
            f.write(f"CFAR baseline BER={comm_info.get('ber', 0.0):.5e}\n")
            f.write(f"Deep model SER≈BER={ser:.5e}\n")


if __name__ == "__main__":
    main()
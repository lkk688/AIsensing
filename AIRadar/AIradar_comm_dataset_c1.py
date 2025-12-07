"""
AIradar_comm_dataset_multimod_otfs_cfar_fixed.py

Integrated Communication + Radar Dataset with:
  - FMCW + OFDM (BPSK/QPSK/QAM)
  - OTFS integrated comm+radar

This version:
  - Uses the SAME CFAR as your working AIRadarDataset code
    (_cfar_2d_custom + cfar_detection).
  - Uses an OTFS radar map identical in structure to simulate_otfs_signal()
    from AIRadarDataset.
  - Reuses your 2D/3D RD visualizations (_plot_2d_rdm, _plot_3d_rdm).
"""

import os
import math
import json
import random
import numpy as np
import h5py
from scipy.constants import c

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm                # noqa: F401

# ---------------------------------------------------------------------
# Optional external visualization utilities (if available)
# ---------------------------------------------------------------------
try:
    from AIRadarLib.visualization import (
        plot_signal_time_and_spectrum,
        plot_instantaneous_frequency,
        plot_3d_range_doppler_map_with_ground_truth,
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


# ---------------------------------------------------------------------
# View limits (same as previous code)
# ---------------------------------------------------------------------
VIEW_RANGE_LIMITS = (0, 100)
VIEW_VELOCITY_LIMITS = (-48, 48)


# ---------------------------------------------------------------------
# 2D / 3D RD plotting (copied from your working AIRadarDataset code)
# ---------------------------------------------------------------------
def _plot_2d_rdm(dataset_instance, rdm, sample_idx, metrics,
                 matched_pairs, unmatched_targets, unmatched_detections, save_path):
    """
    Plot 2D Range-Doppler Map with annotations (same style as old code).
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    dr = dataset_instance.range_axis[1] - dataset_instance.range_axis[0]
    dv = dataset_instance.velocity_axis[1] - dataset_instance.velocity_axis[0]
    extent = [dataset_instance.range_axis[0] - dr / 2,
              dataset_instance.range_axis[-1] + dr / 2,
              dataset_instance.velocity_axis[0] - dv / 2,
              dataset_instance.velocity_axis[-1] + dv / 2]

    im = ax.imshow(rdm, extent=extent, origin='lower',
                   cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label="Magnitude (dB)")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(f"Range-Doppler Map with CFAR Detection - Sample {sample_idx}")

    ax.set_xlim(VIEW_RANGE_LIMITS)
    ax.set_ylim(VIEW_VELOCITY_LIMITS)

    legend_elements = []

    # GT matched
    for t, d in matched_pairs:
        ax.scatter(t['range'], t['velocity'], facecolors='none',
                   edgecolors='lime', s=150, linewidth=2, label='Matched GT')
        ax.plot([t['range'], d['range_m']],
                [t['velocity'], d['velocity_mps']], 'w--', alpha=0.5)

    # Unmatched GT (FN)
    for t in unmatched_targets:
        ax.scatter(t['range'], t['velocity'], facecolors='none',
                   edgecolors='red', s=150, linewidth=2, label='Missed GT (FN)')

    # TP detections
    for t, d in matched_pairs:
        ax.scatter(d['range_m'], d['velocity_mps'], marker='x', color='cyan',
                   s=100, linewidth=2, label='True Positive (TP)')

    # FP detections
    for d in unmatched_detections:
        ax.scatter(d['range_m'], d['velocity_mps'], marker='x', color='orange',
                   s=100, linewidth=2, label='False Alarm (FP)')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend_elements.extend(by_label.values())

    metrics_text = (
        f"Evaluation Metrics:\n"
        f"-------------------\n"
        f"Targets: {metrics['num_targets']}\n"
        f"Detections: {metrics['num_detections']}\n"
        f"TP: {metrics['tp']} | FP: {metrics['fp']} | FN: {metrics['fn']}\n"
        f"Range Error (MAE): {metrics['mean_range_error']:.2f} m\n"
        f"Vel Error (MAE): {metrics['mean_velocity_error']:.2f} m/s"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_3d_rdm(dataset_instance, rdm, sample_idx, targets, detections, save_path):
    """
    3D RD plot – uses AIRadarLib if available, otherwise simple fallback.
    """
    if VISUALIZATION_AVAILABLE:
        converted_targets = []
        for t in targets:
            ct = t.copy()
            ct['distance'] = t['range']
            converted_targets.append(ct)

        range_res = dataset_instance.range_axis[1] - dataset_instance.range_axis[0]
        vel_res = dataset_instance.velocity_axis[1] - dataset_instance.velocity_axis[0]

        cleaned_detections = []
        for det in detections:
            d_copy = det.copy()
            if 'range_idx' in d_copy:
                d_copy['range_idx'] = int(d_copy['range_idx'])
            if 'doppler_idx' in d_copy:
                d_copy['doppler_idx'] = int(d_copy['doppler_idx'])
            cleaned_detections.append(d_copy)

        plot_3d_range_doppler_map_with_ground_truth(
            rd_map=rdm,
            targets=converted_targets,
            range_resolution=range_res,
            velocity_resolution=vel_res,
            num_range_bins=rdm.shape[1],
            num_doppler_bins=rdm.shape[0],
            save_path=save_path,
            apply_doppler_centering=True,
            detections=cleaned_detections,
            view_range_limits=VIEW_RANGE_LIMITS,
            view_velocity_limits=VIEW_VELOCITY_LIMITS,
            is_db=True,
            stride=8,
        )
    else:
        # Simple fallback
        D, R = rdm.shape
        ra = dataset_instance.range_axis[:R]
        va = dataset_instance.velocity_axis[:D]
        RA, VA = np.meshgrid(ra, va)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(RA, VA, rdm, cmap="viridis")
        fig.colorbar(surf, shrink=0.5, aspect=5, label="Magnitude (dB)")
        ax.set_xlabel("Range (m)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_zlabel("Magnitude (dB)")
        ax.set_title(f"3D RD Map Sample {sample_idx}")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------
# Radar+Comm Configs (updated CFAR params for OTFS configs to match config_otfs)
# ---------------------------------------------------------------------
RADAR_COMM_CONFIGS = {
    "cn0566_traditional": {
        "mode": "traditional",
        "radar": {
            "name": "CN0566_FMCW_10GHz",
            "fc": 10e9,
            "B": 500e6,
            "T_chirp": 500e-6,
            "fs": 2.0e6,
            "N_chirps": 64,
            "R_max": 100.0,
            "zero_pad_factor": 2,
            "cfar_params": {
                "num_train": 10,
                "num_guard": 4,
                "threshold_offset": 15,
                "nms_kernel_size": 5,
            },
        },
        "comm": {
            "name": "CN0566_OFDM",
            "fs": 2.0e6,
            "n_subcarriers": 128,
            "cp_len": 16,
            "n_symbols": 16,
            "mod_order": 4,      # QPSK
            "channel_type": "flat",
            "num_paths": 1,
            "max_delay_samp": 8,
            "snr_db_min": 15,
            "snr_db_max": 30,
        },
    },

    # OTFS integrated (now using CFAR params same as config_otfs)
    "otfs_integrated": {
        "mode": "otfs_integrated",
        "radar": {
            "name": "OTFS_77GHz_integrated",
            "fc": 77e9,
            "B": 1.536e9,
            "T_chirp": 40e-6,
            "fs": 51.2e6,
            "N_chirps": 128,
            "N_samples": 512,
            "R_max": 100.0,
            "zero_pad_factor": 1,
            "cfar_params": {
                "num_train": 16,
                "num_guard": 8,
                "threshold_offset": 25,
                "nms_kernel_size": 9,
            },
        },
        "comm": {
            "name": "OTFS_QPSK_grid",
            "n_delay": 512,
            "n_doppler": 128,
            "mod_order": 4,  # QPSK
            "snr_db_min": 15,
            "snr_db_max": 30,
        },
    },

    "auto77_lr_traditional": {
        "mode": "traditional",
        "radar": {
            "name": "Auto_77GHz_LongRange_FMCW",
            "fc": 77e9,
            "B": 1.5e9,
            "T_chirp": 40e-6,
            "fs": 40e6,
            "N_chirps": 128,
            "R_max": 250.0,
            "zero_pad_factor": 4,
            "cfar_params": {
                "num_train": 12,
                "num_guard": 6,
                "threshold_offset": 18,
                "nms_kernel_size": 5,
            },
        },
        "comm": {
            "name": "Auto_77GHz_OFDM_LongRange",
            "fs": 40e6,
            "n_subcarriers": 512,
            "cp_len": 64,
            "n_symbols": 32,
            "mod_order": 16,     # 16-QAM
            "channel_type": "tdl",
            "num_paths": 4,
            "max_delay_samp": 48,
            "snr_db_min": 10,
            "snr_db_max": 30,
        },
    },

    "auto77_mr_traditional": {
        "mode": "traditional",
        "radar": {
            "name": "Auto_77GHz_MidRange_FMCW",
            "fc": 77e9,
            "B": 1.0e9,
            "T_chirp": 30e-6,
            "fs": 30e6,
            "N_chirps": 96,
            "R_max": 150.0,
            "zero_pad_factor": 4,
            "cfar_params": {
                "num_train": 10,
                "num_guard": 4,
                "threshold_offset": 18,
                "nms_kernel_size": 5,
            },
        },
        "comm": {
            "name": "Auto_77GHz_OFDM_MidRange",
            "fs": 30e6,
            "n_subcarriers": 256,
            "cp_len": 48,
            "n_symbols": 24,
            "mod_order": 64,     # 64-QAM
            "channel_type": "tdl",
            "num_paths": 4,
            "max_delay_samp": 32,
            "snr_db_min": 5,
            "snr_db_max": 25,
        },
    },

    "auto77_otfs_integrated": {
        "mode": "otfs_integrated",
        "radar": {
            "name": "Auto_77GHz_OTFS_integrated",
            "fc": 77e9,
            "B": 1.536e9,
            "T_chirp": 40e-6,
            "fs": 30e6,
            "N_chirps": 64,
            "N_samples": 256,
            "R_max": 200.0,
            "zero_pad_factor": 1,
            "cfar_params": {
                "num_train": 16,
                "num_guard": 8,
                "threshold_offset": 25,
                "nms_kernel_size": 9,
            },
        },
        "comm": {
            "name": "Auto_77GHz_OTFS_QPSK_grid",
            "n_delay": 256,
            "n_doppler": 64,
            "mod_order": 4,  # QPSK
            "snr_db_min": 10,
            "snr_db_max": 30,
        },
    },
}


# ---------------------------------------------------------------------
# Generic modulation (BPSK + M-QAM including QPSK)
# ---------------------------------------------------------------------
def _bits_to_int(bits_row):
    val = 0
    for b in bits_row:
        val = (val << 1) | int(b)
    return val


def _int_to_bits(val, num_bits):
    return np.array([(val >> (num_bits - 1 - i)) & 1 for i in range(num_bits)], dtype=np.int8)


def modulate_bits_generic(bits, mod_order):
    M = mod_order
    if M == 2:
        syms = 1 - 2 * bits.astype(np.float32)
        return syms.astype(np.complex64)

    k = int(np.log2(M))
    assert len(bits) % k == 0
    num_syms = len(bits) // k
    bits_reshaped = bits.reshape(num_syms, k)

    if M == 4:
        mapping = {
            (0, 0): (1 + 1j),
            (0, 1): (1 - 1j),
            (1, 1): (-1 - 1j),
            (1, 0): (-1 + 1j),
        }
        syms = np.array([mapping[tuple(b)] for b in bits_reshaped], dtype=np.complex64) / np.sqrt(2)
        return syms

    sqrtM = int(np.sqrt(M))
    assert sqrtM * sqrtM == M
    bits_per_dim = k // 2
    m_per_dim = sqrtM
    scale = math.sqrt(3.0 / (2.0 * (M - 1)))

    syms = np.zeros(num_syms, dtype=np.complex64)
    for i in range(num_syms):
        b = bits_reshaped[i]
        bI = b[:bits_per_dim]
        bQ = b[bits_per_dim:]
        mI = _bits_to_int(bI)
        mQ = _bits_to_int(bQ)
        aI = (2 * mI - (m_per_dim - 1)) * scale
        aQ = (2 * mQ - (m_per_dim - 1)) * scale
        syms[i] = aI + 1j * aQ
    return syms


def demodulate_bits_generic(symbols, mod_order):
    M = mod_order
    if M == 2:
        bits = (symbols.real < 0).astype(np.int8)
        return bits

    k = int(np.log2(M))
    num_syms = symbols.shape[0]

    if M == 4:
        bits = np.zeros((num_syms, 2), dtype=np.int8)
        for i, s in enumerate(symbols):
            re = s.real
            im = s.imag
            b0 = 0 if re >= 0 else 1
            b1 = 0 if im >= 0 else 1
            bits[i] = np.array([b0, b1], dtype=np.int8)
        return bits.reshape(-1)

    sqrtM = int(np.sqrt(M))
    bits_per_dim = k // 2
    m_per_dim = sqrtM
    scale = math.sqrt(3.0 / (2.0 * (M - 1)))
    bits_list = []

    for s in symbols:
        aI_unscaled = s.real / scale
        aQ_unscaled = s.imag / scale
        mI = int(np.round((aI_unscaled + (m_per_dim - 1)) / 2.0))
        mQ = int(np.round((aQ_unscaled + (m_per_dim - 1)) / 2.0))
        mI = max(0, min(mI, m_per_dim - 1))
        mQ = max(0, min(mQ, m_per_dim - 1))
        bI = _int_to_bits(mI, bits_per_dim)
        bQ = _int_to_bits(mQ, bits_per_dim)
        bits_list.append(np.concatenate([bI, bQ]))

    bits_arr = np.stack(bits_list, axis=0).reshape(-1)
    return bits_arr.astype(np.int8)


# ---------------------------------------------------------------------
# OFDM utilities with pilot & channel
# ---------------------------------------------------------------------
def ofdm_modulate_with_pilot(bits, n_subcarriers, cp_len, n_symbols_total, mod_order):
    M = mod_order
    k = int(np.log2(M))
    n_data_symbols = n_symbols_total - 1
    bits_per_sym = k * n_subcarriers
    assert len(bits) == bits_per_sym * n_data_symbols

    tx_syms = np.zeros((n_symbols_total, n_subcarriers), dtype=np.complex64)
    tx_time = np.zeros((n_symbols_total, n_subcarriers + cp_len), dtype=np.complex64)

    pilot_freq = np.ones(n_subcarriers, dtype=np.complex64)
    tx_syms[0, :] = pilot_freq
    pilot_time = np.fft.ifft(pilot_freq)
    cp = pilot_time[-cp_len:]
    tx_time[0, :] = np.concatenate([cp, pilot_time])

    bits_reshaped = bits.reshape(n_data_symbols, bits_per_sym)
    for s in range(1, n_symbols_total):
        b_sym = bits_reshaped[s - 1]
        data_syms = modulate_bits_generic(b_sym, mod_order)
        tx_syms[s, :] = data_syms
        ofdm_time = np.fft.ifft(data_syms)
        cp = ofdm_time[-cp_len:]
        tx_time[s, :] = np.concatenate([cp, ofdm_time])

    return tx_time, tx_syms


def generate_ofdm_channel(channel_type, num_paths, max_delay_samp, cp_len):
    if channel_type == "awgn":
        return np.array([1 + 0j], dtype=np.complex64)
    if channel_type == "flat":
        h = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
        return np.array([h], dtype=np.complex64)

    L = max(1, num_paths)
    max_delay = max(1, min(max_delay_samp, cp_len - 1))
    delays = np.random.randint(0, max_delay, size=L)
    taps = (np.random.randn(L) + 1j * np.random.randn(L)) / np.sqrt(2 * L)
    h_len = max_delay + 1
    h = np.zeros(h_len, dtype=np.complex64)
    for d, t in zip(delays, taps):
        h[d] += t
    return h


def apply_ofdm_channel(tx_time, h):
    n_sym, ofdm_len = tx_time.shape
    rx_time = np.zeros_like(tx_time, dtype=np.complex64)
    for s in range(n_sym):
        conv = np.convolve(tx_time[s], h, mode="full")
        rx_time[s, :] = conv[:ofdm_len]
    return rx_time


# ---------------------------------------------------------------------
# CFAR (exactly your _cfar_2d_custom + cfar_detection)
# ---------------------------------------------------------------------
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter


def _cfar_2d_custom(rd_map_db, num_train=8, num_guard=4,
                    range_res=0.5, doppler_res=0.25,
                    max_range=100, max_speed=50,
                    threshold_offset=4, nms_kernel_size=3,
                    mtd=False):
    """
    Custom 2D CFAR (GO-CFAR) over dB-domain RD maps.
    """
    rows, cols = rd_map_db.shape
    k = num_guard + num_train
    window_size = 2 * k + 1
    full_kernel = np.ones((window_size, window_size), dtype=np.float32)
    guard_area = np.zeros_like(full_kernel)
    guard_area[num_train:num_train + 2 * num_guard + 1,
               num_train:num_train + 2 * num_guard + 1] = 1
    train_kernel = full_kernel - guard_area

    horiz_kernel = train_kernel.copy()
    horiz_kernel[num_train:num_train + 2 * num_guard + 1, :] = 0
    vert_kernel = train_kernel.copy()
    vert_kernel[:, num_train:num_train + 2 * num_guard + 1] = 0

    noise_h = convolve2d(rd_map_db,
                         horiz_kernel / np.sum(horiz_kernel),
                         mode='same', boundary='symm')
    noise_v = convolve2d(rd_map_db,
                         vert_kernel / np.sum(vert_kernel),
                         mode='same', boundary='symm')
    noise_est = np.maximum(noise_h, noise_v)

    threshold = noise_est + threshold_offset
    detections = rd_map_db > threshold

    if nms_kernel_size > 1:
        local_max = maximum_filter(rd_map_db, size=nms_kernel_size)
        detections &= (rd_map_db == local_max)

    doppler_idxs, range_idxs = np.where(detections)
    results = []

    num_doppler = rows
    for d_idx, r_idx in zip(doppler_idxs, range_idxs):
        range_m = r_idx * range_res
        velocity_mps = (d_idx - num_doppler // 2) * doppler_res

        if not (0.5 < range_m < max_range and abs(velocity_mps) < max_speed):
            continue

        if mtd and abs(velocity_mps) < 1.0:
            continue

        results.append({
            "range_idx": int(r_idx),
            "doppler_idx": int(d_idx),
            "range_m": float(range_m),
            "velocity_mps": float(velocity_mps),
            "angle_deg": None
        })

    return results


# ---------------------------------------------------------------------
# Main Dataset class
# ---------------------------------------------------------------------
class AIradarCommDataset(Dataset):
    """
    Integrated Communication + Radar Dataset.

    mode = 'traditional':
      - FMCW radar + CFAR (same as AIRadarDataset FMCW CFAR).
      - OFDM comm with pilot-based channel estimation.

    mode = 'otfs_integrated':
      - OTFS waveform for both radar & comm.
      - Radar CFAR identical to AIRadarDataset OTFS CFAR.
    """

    def __init__(self,
                 num_samples=100,
                 config_name="cn0566_traditional",
                 save_path="data/comm_radar_all_configs_multimod",
                 precision="float32",
                 drawfig=False):

        assert config_name in RADAR_COMM_CONFIGS, f"Unknown config: {config_name}"
        self.config_name = config_name
        self.cfg = RADAR_COMM_CONFIGS[config_name]
        self.mode = self.cfg["mode"]

        self.num_samples = num_samples
        self.save_path = save_path
        self.precision = precision
        self.drawfig = drawfig

        self._init_radar_params()
        self._init_comm_params()
        self._allocate_storage()
        self._generate_dataset()
        self._save_dataset()

    # ---------------- Radar init -----------------
    def _init_radar_params(self):
        r = self.cfg["radar"]
        self.fc = r["fc"]
        self.B = r["B"]
        self.T = r["T_chirp"]
        self.fs = r["fs"]
        self.Nc = r["N_chirps"]
        self.R_max_desired = r["R_max"]
        self.zero_pad_factor = r.get("zero_pad_factor", 1)
        self.cfar_params = r["cfar_params"]

        self.lambda_c = c / self.fc
        self.slope = self.B / self.T
        if self.mode == "otfs_integrated":
            self.Ns = r.get("N_samples", int(self.fs * self.T))
        else:
            self.Ns = int(self.fs * self.T)

        self.zero_pad = self.zero_pad_factor * self.Ns

        self.range_resolution_theor = c / (2 * self.B)
        self.velocity_resolution = self.lambda_c / (2 * self.Nc * self.T)
        self.max_unambiguous_velocity = self.lambda_c / (4 * self.T)

        # range_axis & num_range_bins
        if self.mode == "otfs_integrated":
            # same logic as AIRadarDataset OTFS config
            range_res = c / (2 * self.fs)
            full_range_axis = np.arange(int(self.R_max_desired / range_res) + 2) * range_res
            self.range_axis = full_range_axis
            self.num_range_bins = len(self.range_axis)
            self.max_unambiguous_range = self.range_axis[-1]
            self.R_max = self.range_axis[-1]
        else:
            range_res_fft = (c * self.fs) / (2 * self.slope * self.zero_pad)
            full_range_axis = np.arange(self.zero_pad // 2) * range_res_fft
            max_bin_idx = int(self.R_max_desired / range_res_fft)
            self.num_range_bins = min(self.zero_pad // 2, max_bin_idx)
            self.range_axis = full_range_axis[:self.num_range_bins]
            self.max_unambiguous_range = self.range_axis[-1]
            self.R_max = self.range_axis[-1]

        self.num_doppler_bins = self.Nc
        self.velocity_axis = np.fft.fftshift(
            np.fft.fftfreq(self.Nc, d=self.T)
        ) * self.lambda_c / 2

        print("\n=== Radar Parameters ===")
        print(f"Config:            {self.config_name}")
        print(f"Mode:              {self.mode}")
        print(f"fc:                {self.fc/1e9:.2f} GHz")
        print(f"B:                 {self.B/1e6:.1f} MHz")
        print(f"T_chirp/symbol:    {self.T*1e6:.1f} us")
        print(f"fs:                {self.fs/1e6:.2f} MHz")
        print(f"Nc (chirps/sym):   {self.Nc}")
        print(f"Ns (samples):      {self.Ns}")
        print(f"R_max desired:     {self.R_max_desired:.1f} m")
        print(f"R_max effective:   {self.R_max:.1f} m")
        print(f"Range bins:        {self.num_range_bins}")
        print(f"Doppler bins:      {self.num_doppler_bins}")
        print(f"Range res (theor): {self.range_resolution_theor:.2f} m")
        print(f"Vel res (theor):   {self.velocity_resolution:.2f} m/s")
        print("========================\n")

        # time vectors
        self.t_fast = np.arange(self.Ns) / self.fs
        self.t_slow = np.arange(self.Nc) * self.T

    # ---------------- Comm init -----------------
    def _init_comm_params(self):
        c_cfg = self.cfg["comm"]
        self.comm_mod_order = c_cfg.get("mod_order", 4)
        self.comm_snr_db_min = c_cfg.get("snr_db_min", 15)
        self.comm_snr_db_max = c_cfg.get("snr_db_max", 30)

        if self.mode == "traditional":
            self.ofdm_fs = c_cfg["fs"]
            self.ofdm_n_sc = c_cfg["n_subcarriers"]
            self.ofdm_cp_len = c_cfg["cp_len"]
            self.ofdm_n_sym_total = c_cfg["n_symbols"]
            self.ofdm_n_data_sym = self.ofdm_n_sym_total - 1
            self.comm_channel_type = c_cfg.get("channel_type", "tdl")
            self.comm_num_paths = c_cfg.get("num_paths", 3)
            self.comm_max_delay_samp = c_cfg.get("max_delay_samp", self.ofdm_cp_len - 4)

            k = int(np.log2(self.comm_mod_order))
            self.ofdm_bits_per_sym = k * self.ofdm_n_sc
            self.ofdm_bits_per_frame = self.ofdm_bits_per_sym * self.ofdm_n_data_sym
            self.ofdm_symbol_len = self.ofdm_n_sc + self.ofdm_cp_len
        else:
            self.otfs_n_delay = c_cfg["n_delay"]
            self.otfs_n_doppler = c_cfg["n_doppler"]
            k = int(np.log2(self.comm_mod_order))
            self.otfs_bits_per_frame = k * self.otfs_n_delay * self.otfs_n_doppler

    # ---------------- Storage -------------------
    def _allocate_storage(self):
        dtype = self.precision
        self.radar_time_domain = np.zeros(
            (self.num_samples, self.Nc, self.Ns, 2), dtype=dtype
        )
        self.radar_rdm_db = np.zeros(
            (self.num_samples, self.num_doppler_bins, self.num_range_bins), dtype=dtype
        )
        self.radar_mask = np.zeros(
            (self.num_samples, self.num_doppler_bins, self.num_range_bins, 1), dtype=dtype
        )
        self.radar_targets_list = []
        self.radar_cfar_list = []

        if self.mode == "traditional":
            self.comm_time_domain = np.zeros(
                (self.num_samples, self.ofdm_n_sym_total, self.ofdm_symbol_len, 2), dtype=dtype
            )
            self.comm_tx_bits = np.zeros(
                (self.num_samples, self.ofdm_bits_per_frame), dtype=np.int8
            )
            self.comm_rx_bits = np.zeros_like(self.comm_tx_bits)
            self.comm_ber = np.zeros(self.num_samples, dtype=np.float32)
        else:
            self.comm_time_domain = np.zeros(
                (self.num_samples, 1, self.Ns * self.Nc, 2), dtype=dtype
            )
            self.comm_tx_bits = np.zeros(
                (self.num_samples, self.otfs_bits_per_frame), dtype=np.int8
            )
            self.comm_rx_bits = np.zeros_like(self.comm_tx_bits)
            self.comm_ber = np.zeros(self.num_samples, dtype=np.float32)

    # ---------------- Target generation -------------------
    def _generate_targets(self, max_targets=3):
        n = random.randint(1, max_targets)
        targets = []
        for _ in range(n):
            r = np.random.uniform(10.0, self.R_max - 10.0)
            v = np.random.uniform(
                -self.max_unambiguous_velocity + 1,
                self.max_unambiguous_velocity - 1
            )
            rcs = np.random.uniform(5.0, 30.0)
            targets.append(
                {
                    "range": r,
                    "velocity": v,
                    "rcs": rcs,
                    "azimuth": np.random.uniform(-30, 30),
                    "elevation": np.random.uniform(-10, 10),
                }
            )
        return targets

    def _generate_clutter_targets(self, clutter_intensity=0.1):
        clutter_targets = []
        intensity_db = 10 * np.log10(max(clutter_intensity, 1e-6))

        num_static = random.randint(5, 15)
        for _ in range(num_static):
            clutter_targets.append(
                {
                    "range": random.uniform(5, self.R_max),
                    "velocity": np.random.normal(0.0, 0.05),
                    "rcs": random.uniform(-40, -20) + intensity_db,
                    "azimuth": random.uniform(-30, 30),
                    "elevation": 0,
                }
            )

        num_ground = random.randint(20, 50)
        for _ in range(num_ground):
            dist = random.uniform(1.0, self.R_max * 0.5)
            clutter_targets.append(
                {
                    "range": dist,
                    "velocity": np.random.normal(0.0, 0.3),
                    "rcs": random.uniform(-50, -30) + intensity_db,
                    "azimuth": random.uniform(-60, 60),
                    "elevation": random.uniform(-10, 0),
                }
            )
        return clutter_targets

    def _generate_coupling_target(self, clutter_intensity=0.1):
        intensity_db = 10 * np.log10(max(clutter_intensity, 1e-6))
        return {
            "range": 0.001,
            "velocity": 0.0,
            "rcs": 0.0 + intensity_db,
            "azimuth": 0.0,
            "elevation": 0.0,
        }

    # ---------------- FMCW simulation -------------------
    def _simulate_fmcw(self, targets, snr_db=20, clutter_intensity=0.3):
        sim_targets = list(targets)
        sim_targets.extend(self._generate_clutter_targets(clutter_intensity))
        sim_targets.append(self._generate_coupling_target(clutter_intensity))

        beat = np.zeros((self.Nc, self.Ns), dtype=np.complex128)
        ranges = np.array([t["range"] for t in sim_targets])
        velocities = np.array([t["velocity"] for t in sim_targets])
        rcs = np.array([t["rcs"] for t in sim_targets])
        rcs_lin = 10 ** (rcs / 10.0)
        amps = np.sqrt(rcs_lin)

        fb = 2 * ranges * self.slope / c
        fd = 2 * velocities / self.lambda_c

        fb_grid = fb[:, None, None]
        fd_grid = fd[:, None, None]
        amps_grid = amps[:, None, None]
        t_fast = self.t_fast[None, None, :]
        t_slow = self.t_slow[None, :, None]

        phase = 2 * np.pi * (fb_grid * t_fast + fd_grid * t_slow)
        sig = amps_grid * np.exp(1j * phase)
        beat = np.sum(sig, axis=0)

        win_r = np.hanning(self.Ns)
        win_d = np.hanning(self.Nc)
        beat *= win_r[None, :]
        beat *= win_d[:, None]

        sig_pow = np.mean(np.abs(beat) ** 2)
        if sig_pow > 0:
            snr_lin = 10 ** (snr_db / 10.0)
            noise_pow = sig_pow / snr_lin
            noise_std = np.sqrt(noise_pow / 2)
            noise = (np.random.randn(*beat.shape) + 1j * np.random.randn(*beat.shape)) * noise_std
            beat += noise

        rfft = np.fft.fft(beat, n=self.zero_pad, axis=1)
        rfft = rfft[:, : self.zero_pad // 2]
        dfft = np.fft.fftshift(np.fft.fft(rfft, axis=0), axes=0)
        rdm = 20 * np.log10(np.abs(dfft) + 1e-6)

        if rdm.shape[1] > self.num_range_bins:
            rdm = rdm[:, : self.num_range_bins]

        return beat.astype(np.complex64), rdm.astype(np.float32)

    # ---------------- OTFS mod/demod (same structure as old code) ---------------
    def _otfs_modulate(self, dd_grid):
        tf_grid = np.fft.fft(dd_grid, axis=0)
        tf_grid = np.fft.ifft(tf_grid, axis=1)
        time_domain_grid = np.fft.ifft(tf_grid, axis=0)
        tx_signal = time_domain_grid.flatten(order='F')
        return tx_signal

    def _otfs_demodulate(self, rx_signal):
        time_domain_grid = rx_signal.reshape((self.Ns, self.Nc), order='F')
        tf_grid = np.fft.fft(time_domain_grid, axis=0)
        dd_grid = np.fft.fft(tf_grid, axis=1)
        dd_grid = np.fft.ifft(dd_grid, axis=0)
        return dd_grid

    def _simulate_otfs_integrated(self, targets, snr_db=20):
        """
        OTFS radar+comm simulation using the same radar map pipeline
        as simulate_otfs_signal() from AIRadarDataset, extended to output bits.
        """
        # QPSK (mod_order=4) in DD grid
        N_delay = self.Ns
        N_dopp = self.Nc
        num_syms = N_delay * N_dopp

        # symbol indices 0..3
        sym_idx = np.random.randint(0, 4, num_syms)
        mod_map = np.array([
            (1 + 1j) / np.sqrt(2),
            (1 - 1j) / np.sqrt(2),
            (-1 + 1j) / np.sqrt(2),
            (-1 - 1j) / np.sqrt(2),
        ], dtype=np.complex64)
        tx_syms = mod_map[sym_idx]
        tx_dd_grid = tx_syms.reshape((N_delay, N_dopp))

        # bits: index -> 2 bits
        tx_bits_list = []
        for idx in sym_idx:
            b0 = (idx >> 1) & 1
            b1 = idx & 1
            tx_bits_list.extend([b0, b1])
        tx_bits = np.array(tx_bits_list, dtype=np.int8)

        # OTFS modulation
        tx_signal = self._otfs_modulate(tx_dd_grid)

        n_samples = tx_signal.size
        rx_signal = np.zeros(n_samples, dtype=complex)
        time_vector = np.arange(n_samples) / self.fs

        for target in targets:
            range_m = target['range']
            velocity_mps = target['velocity']
            rcs = target['rcs']

            amplitude = np.sqrt(10 ** (rcs / 10))
            delay_sec = 2 * range_m / c
            delay_samples = int(round(delay_sec * self.fs))

            if delay_samples < n_samples:
                delayed_signal = np.roll(tx_signal, delay_samples)
                delayed_signal[:delay_samples] = 0
                doppler_hz = 2 * velocity_mps * self.fc / c
                doppler_shift = np.exp(1j * 2 * np.pi * doppler_hz * time_vector)
                rx_signal += amplitude * delayed_signal * doppler_shift

        # AWGN
        signal_power = np.mean(np.abs(rx_signal) ** 2)
        if signal_power > 0:
            snr_linear = 10 ** (snr_db / 10)
            noise_power = signal_power / snr_linear
            noise = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) * np.sqrt(noise_power / 2)
            rx_signal += noise

        # OTFS demod
        rx_dd_grid = self._otfs_demodulate(rx_signal)

        # DD-domain channel estimate (same as your code)
        rx_dd_fft = np.fft.fft2(rx_dd_grid)
        tx_dd_fft = np.fft.fft2(tx_dd_grid)
        epsilon = 1e-6
        ddm_fft = rx_dd_fft / (tx_dd_fft + epsilon)
        ddm_complex = np.fft.ifft2(ddm_fft)

        ddm_transposed = ddm_complex.T       # [Nc, Ns]
        ddm_shifted = np.fft.fftshift(ddm_transposed, axes=0)
        ddm_mag = np.abs(ddm_shifted)
        ddm_db = 20 * np.log10(ddm_mag + 1e-6)

        # crop to num_range_bins
        if ddm_db.shape[1] > self.num_range_bins:
            ddm_db = ddm_db[:, :self.num_range_bins]

        rdm_db = ddm_db  # [Nc, num_range_bins]

        # Comms equalization: rx_dd / channel
        H_dd = ddm_complex
        eq_syms = rx_dd_grid / (H_dd + epsilon)
        eq_vec = eq_syms.reshape(-1)

        # Demap QPSK
        rx_sym_idx = np.zeros(eq_vec.shape[0], dtype=np.int64)
        for i, s in enumerate(eq_vec):
            dists = np.abs(s - mod_map)
            rx_sym_idx[i] = np.argmin(dists)

        rx_bits_list = []
        for idx in rx_sym_idx:
            b0 = (idx >> 1) & 1
            b1 = idx & 1
            rx_bits_list.extend([b0, b1])
        rx_bits = np.array(rx_bits_list, dtype=np.int8)

        rx_time = rx_signal.reshape((self.Ns, self.Nc), order='F').T  # [Nc,Ns]
        return rx_time.astype(np.complex64), rdm_db.astype(np.float32), tx_bits, rx_bits

    # ---------------- Masks & metrics -------------------
    def _create_radar_mask(self, targets):
        mask = np.zeros((self.num_doppler_bins, self.num_range_bins), dtype=np.float32)
        for t in targets:
            r_idx = int(np.argmin(np.abs(self.range_axis - t["range"])))
            v_idx = int(np.argmin(np.abs(self.velocity_axis - t["velocity"])))
            for dr in range(-1, 2):
                for dv in range(-1, 2):
                    rr = r_idx + dr
                    dd = v_idx + dv
                    if 0 <= rr < self.num_range_bins and 0 <= dd < self.num_doppler_bins:
                        mask[dd, rr] = 1.0
        return mask

    def _evaluate_metrics(self, targets, detections, match_dist_thresh=3.0):
        tp = 0
        range_errors = []
        vel_errors = []

        unmatched_targets = targets.copy()
        unmatched_detections = detections.copy()
        matched_pairs = []

        for target in targets:
            best_det = None
            best_dist = float("inf")
            best_det_idx = -1

            for i, det in enumerate(unmatched_detections):
                d_r = target["range"] - det["range_m"]
                d_v = target["velocity"] - det["velocity_mps"]
                dist = np.sqrt(d_r**2 + d_v**2)
                if dist < match_dist_thresh and dist < best_dist:
                    best_dist = dist
                    best_det = det
                    best_det_idx = i

            if best_det:
                tp += 1
                range_errors.append(abs(target["range"] - best_det["range_m"]))
                vel_errors.append(abs(target["velocity"] - best_det["velocity_mps"]))
                matched_pairs.append((target, best_det))
                unmatched_targets.remove(target)
                unmatched_detections.pop(best_det_idx)

        fp = len(unmatched_detections)
        fn = len(unmatched_targets)

        metrics = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "mean_range_error": float(np.mean(range_errors)) if range_errors else 0.0,
            "mean_velocity_error": float(np.mean(vel_errors)) if vel_errors else 0.0,
            "num_targets": len(targets),
            "num_detections": len(detections),
        }
        return metrics, matched_pairs, unmatched_targets, unmatched_detections

    def cfar_detection(self, rd_map):
        """
        CFAR detection using EXACT same logic as AIRadarDataset.
        """
        range_res = self.range_axis[1] - self.range_axis[0]
        vel_res = self.velocity_axis[1] - self.velocity_axis[0]

        mtd_enabled = True
        cfr = _cfar_2d_custom(
            rd_map,
            num_train=self.cfar_params.get("num_train", 10),
            num_guard=self.cfar_params.get("num_guard", 4),
            range_res=range_res,
            doppler_res=vel_res,
            max_range=self.R_max,
            max_speed=self.max_unambiguous_velocity,
            threshold_offset=self.cfar_params.get("threshold_offset", 15),
            nms_kernel_size=self.cfar_params.get("nms_kernel_size", 5),
            mtd=mtd_enabled,
        )

        for det in cfr:
            d_idx = det["doppler_idx"]
            r_idx = det["range_idx"]
            if 0 <= d_idx < rd_map.shape[0] and 0 <= r_idx < rd_map.shape[1]:
                det["magnitude"] = rd_map[d_idx, r_idx]

        return cfr

    # ---------------- Dataset generation -------------------
    def _generate_dataset(self):
        print(f"Generating {self.num_samples} samples for {self.config_name} ({self.mode})...")
        os.makedirs(self.save_path, exist_ok=True)
        vis_dir = os.path.join(self.save_path, "visualizations")
        if self.drawfig and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        for i in range(self.num_samples):
            targets = self._generate_targets(max_targets=3)
            radar_snr = random.uniform(20, 40)
            comm_snr = random.uniform(self.comm_snr_db_min, self.comm_snr_db_max)

            if self.mode == "traditional":
                beat, rdm_db = self._simulate_fmcw(targets, radar_snr, clutter_intensity=0.3)
                mask = self._create_radar_mask(targets)
                cfar_det = self.cfar_detection(rdm_db)

                # Communication: OFDM
                tx_bits = np.random.randint(0, 2, self.ofdm_bits_per_frame).astype(np.int8)
                tx_time, tx_syms = ofdm_modulate_with_pilot(
                    tx_bits,
                    n_subcarriers=self.ofdm_n_sc,
                    cp_len=self.ofdm_cp_len,
                    n_symbols_total=self.ofdm_n_sym_total,
                    mod_order=self.comm_mod_order,
                )
                h = generate_ofdm_channel(
                    channel_type=self.comm_channel_type,
                    num_paths=self.comm_num_paths,
                    max_delay_samp=self.comm_max_delay_samp,
                    cp_len=self.ofdm_cp_len,
                )
                rx_time = apply_ofdm_channel(tx_time, h)
                sig_pow = np.mean(np.abs(rx_time) ** 2)
                if sig_pow > 0:
                    snr_lin = 10 ** (comm_snr / 10.0)
                    noise_pow = sig_pow / snr_lin
                    noise_std = np.sqrt(noise_pow / 2)
                    noise = (np.random.randn(*rx_time.shape) + 1j * np.random.randn(*rx_time.shape)) * noise_std
                    rx_time += noise

                N_sc = self.ofdm_n_sc
                cp = self.ofdm_cp_len
                eps = 1e-6
                y_pilot_t = rx_time[0]
                y_pilot_no_cp = y_pilot_t[cp:cp + N_sc]
                Y_pilot = np.fft.fft(y_pilot_no_cp)
                X_pilot = tx_syms[0]
                H_est = Y_pilot / (X_pilot + eps)

                data_syms_eq = []
                for s in range(1, self.ofdm_n_sym_total):
                    y = rx_time[s]
                    y_no_cp = y[cp:cp + N_sc]
                    Y = np.fft.fft(y_no_cp)
                    X_hat = Y / (H_est + eps)
                    data_syms_eq.append(X_hat)
                data_syms_eq = np.stack(data_syms_eq, axis=0)
                eq_vec = data_syms_eq.reshape(-1)
                rx_bits = demodulate_bits_generic(eq_vec, self.comm_mod_order)
                ber = float(np.mean(tx_bits != rx_bits))

                self.radar_time_domain[i, :, :, 0] = beat.real.astype(self.precision)
                self.radar_time_domain[i, :, :, 1] = beat.imag.astype(self.precision)
                self.radar_rdm_db[i] = rdm_db.astype(self.precision)
                self.radar_mask[i, :, :, 0] = mask.astype(self.precision)
                self.radar_targets_list.append(targets)
                self.radar_cfar_list.append(cfar_det)

                self.comm_time_domain[i, :, :, 0] = rx_time.real.astype(self.precision)
                self.comm_time_domain[i, :, :, 1] = rx_time.imag.astype(self.precision)
                self.comm_tx_bits[i] = tx_bits
                self.comm_rx_bits[i] = rx_bits
                self.comm_ber[i] = ber

                if self.drawfig and i < 3:
                    rdm_norm = rdm_db - np.max(rdm_db)
                    metrics, mp, ut, ud = self._evaluate_metrics(targets, cfar_det)
                    _plot_2d_rdm(
                        self, rdm_norm, i, metrics, mp, ut, ud,
                        os.path.join(vis_dir, f"{self.config_name}_radar_sample_{i}_2d.png"),
                    )
                    _plot_3d_rdm(
                        self, rdm_norm, i, targets, cfar_det,
                        os.path.join(vis_dir, f"{self.config_name}_radar_sample_{i}_3d.png"),
                    )

            else:
                # OTFS integrated
                beat, rdm_db, tx_bits, rx_bits = self._simulate_otfs_integrated(targets, comm_snr)
                mask = self._create_radar_mask(targets)
                cfar_det = self.cfar_detection(rdm_db)
                ber = float(np.mean(tx_bits != rx_bits))

                self.radar_time_domain[i, :, :, 0] = beat.real.astype(self.precision)
                self.radar_time_domain[i, :, :, 1] = beat.imag.astype(self.precision)
                self.radar_rdm_db[i] = rdm_db.astype(self.precision)
                self.radar_mask[i, :, :, 0] = mask.astype(self.precision)
                self.radar_targets_list.append(targets)
                self.radar_cfar_list.append(cfar_det)

                rx_signal = beat.reshape(1, -1)
                self.comm_time_domain[i, 0, :, 0] = rx_signal.real.astype(self.precision)
                self.comm_time_domain[i, 0, :, 1] = rx_signal.imag.astype(self.precision)
                self.comm_tx_bits[i] = tx_bits
                self.comm_rx_bits[i] = rx_bits
                self.comm_ber[i] = ber

                if self.drawfig and i < 3:
                    rdm_norm = rdm_db - np.max(rdm_db)
                    metrics, mp, ut, ud = self._evaluate_metrics(targets, cfar_det)
                    _plot_2d_rdm(
                        self, rdm_norm, i, metrics, mp, ut, ud,
                        os.path.join(vis_dir, f"{self.config_name}_radar_sample_{i}_2d.png"),
                    )
                    _plot_3d_rdm(
                        self, rdm_norm, i, targets, cfar_det,
                        os.path.join(vis_dir, f"{self.config_name}_radar_sample_{i}_3d.png"),
                    )

        print(f"Generation for {self.config_name} complete.")

    # ---------------- Save -------------------
    def _save_dataset(self):
        fname = os.path.join(self.save_path, "radar_comm_dataset.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset("radar_time_domain", data=self.radar_time_domain, compression="gzip")
            f.create_dataset("radar_rdm_db", data=self.radar_rdm_db, compression="gzip")
            f.create_dataset("radar_mask", data=self.radar_mask, compression="gzip")
            f.create_dataset("comm_time_domain", data=self.comm_time_domain, compression="gzip")
            f.create_dataset("comm_tx_bits", data=self.comm_tx_bits)
            f.create_dataset("comm_rx_bits", data=self.comm_rx_bits)
            f.create_dataset("comm_ber", data=self.comm_ber)

            f.create_dataset("range_axis", data=self.range_axis)
            f.create_dataset("velocity_axis", data=self.velocity_axis)

            tinfo = [json.dumps(t, default=str) for t in self.radar_targets_list]
            f.create_dataset("radar_targets", data=tinfo, dtype=h5py.string_dtype())

            cfar_info = [json.dumps(d, default=str) for d in self.radar_cfar_list]
            f.create_dataset("radar_cfar", data=cfar_info, dtype=h5py.string_dtype())

            f.attrs["mode"] = self.mode
            f.attrs["config_name"] = self.config_name
            f.attrs["R_max_effective"] = float(self.R_max)

        print(f"Dataset saved to {fname}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "mode": self.mode,
            "radar_time_domain": torch.from_numpy(self.radar_time_domain[idx]).float(),
            "radar_rdm_db": torch.from_numpy(self.radar_rdm_db[idx]).float(),
            "radar_mask": torch.from_numpy(self.radar_mask[idx]).float(),
            "radar_targets": self.radar_targets_list[idx],
            "radar_cfar": self.radar_cfar_list[idx],
            "comm_time_domain": torch.from_numpy(self.comm_time_domain[idx]).float(),
            "comm_tx_bits": torch.from_numpy(self.comm_tx_bits[idx].astype(np.int8)),
            "comm_rx_bits": torch.from_numpy(self.comm_rx_bits[idx].astype(np.int8)),
            "comm_ber": torch.tensor(self.comm_ber[idx], dtype=torch.float32),
            "range_axis": self.range_axis,
            "velocity_axis": self.velocity_axis,
        }


# ---------------------------------------------------------------------
# Evaluation helpers (similar to your evaluate_dataset_metrics)
# ---------------------------------------------------------------------
def evaluate_radar_cfar_comm_dataset(dataset: AIradarCommDataset, name: str, save_dir=None):
    print(f"\nEvaluating Radar CFAR Metrics for {name}...")
    all_tp = all_fp = all_fn = 0
    all_re = []
    all_ve = []

    for i in range(len(dataset)):
        s = dataset[i]
        targets = s["radar_targets"]
        detections = s["radar_cfar"]
        metrics, _, _, _ = dataset._evaluate_metrics(targets, detections)
        all_tp += metrics["tp"]
        all_fp += metrics["fp"]
        all_fn += metrics["fn"]
        if metrics["mean_range_error"] > 0:
            all_re.append(metrics["mean_range_error"])
        if metrics["mean_velocity_error"] > 0:
            all_ve.append(metrics["mean_velocity_error"])

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_re = float(np.mean(all_re)) if all_re else 0.0
    mean_ve = float(np.mean(all_ve)) if all_ve else 0.0

    lines = [
        f"--- Radar CFAR ({name}) ---",
        f"Samples: {len(dataset)}",
        f"Precision: {precision:.4f}",
        f"Recall:    {recall:.4f}",
        f"F1 Score:  {f1:.4f}",
        f"Mean Range Error: {mean_re:.4f} m",
        f"Mean Vel Error:   {mean_ve:.4f} m/s",
        "-" * 40,
    ]
    print("\n".join(lines))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f"{name}_radar_eval.txt")
        with open(fname, "w") as f:
            f.write("\n".join(lines))
        print(f"Radar CFAR evaluation saved to {fname}")


def evaluate_comm_dataset(dataset: AIradarCommDataset, name: str, save_dir=None):
    print(f"\nEvaluating Communication Metrics for {name}...")
    ber_vals = np.array([float(dataset[i]["comm_ber"]) for i in range(len(dataset))])
    mean_ber = ber_vals.mean()
    med_ber = np.median(ber_vals)
    min_ber = ber_vals.min()
    max_ber = ber_vals.max()

    lines = [
        f"--- Comm BER ({name}) ---",
        f"Samples: {len(dataset)}",
        f"Mean BER:   {mean_ber:.6f}",
        f"Median BER: {med_ber:.6f}",
        f"Min BER:    {min_ber:.6f}",
        f"Max BER:    {max_ber:.6f}",
        "-" * 40,
    ]
    print("\n".join(lines))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f"{name}_comm_eval.txt")
        with open(fname, "w") as f:
            f.write("\n".join(lines))
        print(f"Comm evaluation saved to {fname}")


# ---------------------------------------------------------------------
# Standalone test for all configs
# ---------------------------------------------------------------------
if __name__ == "__main__":
    base_out_dir = "data/AIradar_comm_dataset_c1"
    num_samples_per_config = 100

    for cfg_name in RADAR_COMM_CONFIGS.keys():
        print("\n" + "=" * 70)
        print(f"Running test for config: {cfg_name}")
        print("=" * 70)

        cfg_out_dir = os.path.join(base_out_dir, cfg_name)

        dataset = AIradarCommDataset(
            num_samples=num_samples_per_config,
            config_name=cfg_name,
            save_path=cfg_out_dir,
            drawfig=True,
        )

        evaluate_radar_cfar_comm_dataset(
            dataset,
            name=cfg_name,
            save_dir=cfg_out_dir,
        )

        evaluate_comm_dataset(
            dataset,
            name=cfg_name,
            save_dir=cfg_out_dir,
        )

    print("\nAll configs processed. Visualizations and evaluations saved under:")
    print(f"  {base_out_dir}")
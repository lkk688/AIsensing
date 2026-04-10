#!/usr/bin/env python3
"""
radar_comm_config_dataset_refactor.py

Refactored configuration system and dataset generator for multi-configuration
ISAC experiments spanning FMCW/OFDM and OTFS.

Goals of this rewrite
---------------------
1. Unify config schema across waveform families.
2. Make OFDM and OTFS communication outputs structurally consistent.
3. Remove hidden assumptions like OTFS modulation hard-coded to QPSK.
4. Produce communication tensors that are directly usable by a unified model,
   including ISACFoundationModel.forward_comm(...).
5. Keep backward-compatible adapters where practical.

This file focuses on:
- normalized config registry
- config conversion helpers
- dataset generator with consistent outputs
- utilities to build model-ready tensors and IDs

Notes
-----
- This version intentionally does NOT auto-generate data inside __init__.
- Save/load is done in one shot rather than rewriting dump files every sample.
- Baseline communication receiver is clearly named as ZF/MMSE-style hard demap.
- OTFS simulation now honors self.mod_order rather than being fixed to 4.
"""

from __future__ import annotations

import os
import math
import json
from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter
from tqdm import tqdm


# =============================================================================
# ID definitions shared with unified model
# =============================================================================

WAVEFORM_IDS = {
    "FMCW": 0,
    "OFDM": 1,
    "OTFS": 2,
}

TASK_IDS = {
    "RADAR": 0,
    "COMM": 1,
}

MODULATION_IDS = {
    4: 1,
    8: 2,
    16: 3,
    64: 4,
}


# =============================================================================
# Unified config schema
# =============================================================================


def _default_cfar() -> Dict[str, Any]:
    return {
        "num_train": 12,
        "num_guard": 4,
        "threshold_offset": 25,
        "nms_kernel_size": 7,
        "min_range_m": 0.0,
        "min_speed_mps": 0.0,
        "notch_doppler_bins": 0,
        "global_percentile": None,
        "max_peaks": None,
    }


def _default_clutter() -> Dict[str, Any]:
    return {
        "ground_clutter": True,
        "ground_intensity": 0.005,
        "k_shape": 2.0,
        "range_exponent": 2.5,
        "weather_clutter": False,
        "weather_intensity": 0.0,
        "doppler_spread": 3.0,
    }


def make_traditional_config(
    name: str,
    fc: float,
    mod_order: int,
    radar_bandwidth: float,
    radar_chirp_duration: float,
    radar_fs: float,
    comm_bandwidth: float,
    comm_fs: float,
    comm_fft_size: int,
    comm_cp_len: int,
    max_range: float,
    num_rx: int,
    channel_model: str = "multipath",
    csi_error: float = 0.1,
    adaptive_cfar: bool = True,
    cfar: Optional[Dict[str, Any]] = None,
    clutter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = {
        "meta": {
            "name": name,
            "mode": "TRADITIONAL",
            "radar_waveform": "FMCW",
            "comm_waveform": "OFDM",
        },
        "rf": {
            "fc": fc,
            "num_rx": num_rx,
        },
        "radar": {
            "waveform": "FMCW",
            "bandwidth": radar_bandwidth,
            "chirp_duration": radar_chirp_duration,
            "fs": radar_fs,
            "max_range": max_range,
            "num_chirps": 64,
        },
        "comm": {
            "waveform": "OFDM",
            "mod_order": mod_order,
            "bandwidth": comm_bandwidth,
            "fs": comm_fs,
            "fft_size": comm_fft_size,
            "cp_len": comm_cp_len,
            "num_symbols": 14,
        },
        "channel": {
            "model": channel_model,
            "csi_error": csi_error,
        },
        "cfar": {**_default_cfar(), **(cfar or {})},
        "clutter": {**_default_clutter(), **(clutter or {})},
        "adaptive_cfar": adaptive_cfar,
    }
    return cfg



def make_otfs_config(
    name: str,
    fc: float,
    mod_order: int,
    bandwidth: float,
    fs: float,
    n_doppler: int,
    n_delay: int,
    symbol_duration: float,
    max_range: float,
    num_rx: int,
    channel_model: str = "multipath",
    csi_error: float = 0.1,
    adaptive_cfar: bool = True,
    cfar: Optional[Dict[str, Any]] = None,
    clutter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = {
        "meta": {
            "name": name,
            "mode": "OTFS",
            "radar_waveform": "OTFS",
            "comm_waveform": "OTFS",
        },
        "rf": {
            "fc": fc,
            "num_rx": num_rx,
        },
        "radar": {
            "waveform": "OTFS",
            "bandwidth": bandwidth,
            "symbol_duration": symbol_duration,
            "fs": fs,
            "max_range": max_range,
            "n_doppler": n_doppler,
            "n_delay": n_delay,
        },
        "comm": {
            "waveform": "OTFS",
            "mod_order": mod_order,
            "bandwidth": bandwidth,
            "fs": fs,
            "n_doppler": n_doppler,
            "n_delay": n_delay,
            "symbol_duration": symbol_duration,
        },
        "channel": {
            "model": channel_model,
            "csi_error": csi_error,
        },
        "cfar": {**_default_cfar(), **(cfar or {})},
        "clutter": {**_default_clutter(), **(clutter or {})},
        "adaptive_cfar": adaptive_cfar,
    }
    return cfg


UNIFIED_CONFIGS: Dict[str, Dict[str, Any]] = {
    "CN0566_TRADITIONAL": make_traditional_config(
        name="CN0566_TRADITIONAL",
        fc=10.25e9,
        mod_order=16,
        radar_bandwidth=500e6,
        radar_chirp_duration=5e-4,
        radar_fs=2e6,
        comm_bandwidth=40e6,
        comm_fs=61.44e6,
        comm_fft_size=64,
        comm_cp_len=16,
        max_range=150.0,
        num_rx=1,
        csi_error=0.1,
        cfar={"threshold_offset": 25, "nms_kernel_size": 7},
        clutter={
            "ground_clutter": True,
            "ground_intensity": 0.005,
            "k_shape": 2.0,
            "range_exponent": 2.5,
            "weather_clutter": False,
            "weather_intensity": 0.02,
            "doppler_spread": 3.0,
        },
    ),
    "CN0566_OTFS_ISAC": make_otfs_config(
        name="CN0566_OTFS_ISAC",
        fc=10.25e9,
        mod_order=4,
        bandwidth=40e6,
        fs=40e6,
        n_doppler=64,
        n_delay=512,
        symbol_duration=1.28e-5,
        max_range=100.0,
        num_rx=1,
        csi_error=0.15,
        cfar={
            "num_train": 4,
            "num_guard": 2,
            "threshold_offset": 25,
            "nms_kernel_size": 5,
            "min_range_m": 2.0,
        },
        clutter={
            "ground_clutter": True,
            "ground_intensity": 0.03,
            "k_shape": 1.5,
            "range_exponent": 2.0,
            "weather_clutter": False,
            "weather_intensity": 0.01,
            "doppler_spread": 5.0,
        },
    ),
    "Automotive_77GHz_LongRange": make_traditional_config(
        name="Automotive_77GHz_LongRange",
        fc=77e9,
        mod_order=4,
        radar_bandwidth=1.5e9,
        radar_chirp_duration=4e-5,
        radar_fs=51.2e6,
        comm_bandwidth=400e6,
        comm_fs=512e6,
        comm_fft_size=1024,
        comm_cp_len=72,
        max_range=100.0,
        num_rx=1,
        csi_error=0.05,
        clutter={
            "ground_clutter": True,
            "ground_intensity": 0.008,
            "k_shape": 3.0,
            "range_exponent": 3.0,
            "weather_clutter": True,
            "weather_intensity": 0.03,
            "doppler_spread": 2.0,
        },
    ),
    "8QAM_MediumRange": make_traditional_config(
        name="8QAM_MediumRange",
        fc=28e9,
        mod_order=8,
        radar_bandwidth=800e6,
        radar_chirp_duration=1e-4,
        radar_fs=40e6,
        comm_bandwidth=100e6,
        comm_fs=122.88e6,
        comm_fft_size=256,
        comm_cp_len=32,
        max_range=80.0,
        num_rx=1,
        csi_error=0.08,
        clutter={
            "ground_clutter": True,
            "ground_intensity": 0.007,
            "k_shape": 2.5,
            "range_exponent": 2.5,
            "weather_clutter": True,
            "weather_intensity": 0.025,
            "doppler_spread": 2.5,
        },
    ),
    "XBand_10GHz_MediumRange": make_traditional_config(
        name="XBand_10GHz_MediumRange",
        fc=10e9,
        mod_order=16,
        radar_bandwidth=1e9,
        radar_chirp_duration=1.6e-4,
        radar_fs=40e6,
        comm_bandwidth=40e6,
        comm_fs=40e6,
        comm_fft_size=64,
        comm_cp_len=16,
        max_range=100.0,
        num_rx=1,
        csi_error=0.1,
        cfar={"num_train": 24, "num_guard": 8, "nms_kernel_size": 9},
        clutter={
            "ground_clutter": True,
            "ground_intensity": 0.006,
            "k_shape": 2.5,
            "range_exponent": 2.5,
            "weather_clutter": True,
            "weather_intensity": 0.04,
            "doppler_spread": 4.0,
        },
    ),
    "AUTOMOTIVE_TRADITIONAL": make_traditional_config(
        name="AUTOMOTIVE_TRADITIONAL",
        fc=77e9,
        mod_order=16,
        radar_bandwidth=1.5e9,
        radar_chirp_duration=6e-5,
        radar_fs=50e6,
        comm_bandwidth=400e6,
        comm_fs=512e6,
        comm_fft_size=1024,
        comm_cp_len=72,
        max_range=250.0,
        num_rx=4,
        csi_error=0.01,
        cfar={"num_train": 16, "num_guard": 4, "nms_kernel_size": 9},
        clutter={
            "ground_clutter": True,
            "ground_intensity": 0.01,
            "k_shape": 2.0,
            "range_exponent": 2.0,
            "weather_clutter": True,
            "weather_intensity": 0.05,
            "doppler_spread": 3.0,
        },
    ),
    "AUTOMOTIVE_OTFS_ISAC": make_otfs_config(
        name="AUTOMOTIVE_OTFS_ISAC",
        fc=77e9,
        mod_order=4,
        bandwidth=1.536e9,
        fs=51.2e6,
        n_doppler=128,
        n_delay=512,
        symbol_duration=4e-5,
        max_range=100.0,
        num_rx=1,
        csi_error=0.1,
        cfar={
            "num_train": 16,
            "num_guard": 8,
            "threshold_offset": 22,
            "nms_kernel_size": 11,
            "min_range_m": 2.0,
        },
        clutter={
            "ground_clutter": True,
            "ground_intensity": 0.03,
            "k_shape": 2.5,
            "range_exponent": 2.5,
            "weather_clutter": True,
            "weather_intensity": 0.04,
            "doppler_spread": 2.5,
        },
    ),
    "5G_ISAC_HighBandwidth": make_traditional_config(
        name="5G_ISAC_HighBandwidth",
        fc=28e9,
        mod_order=64,
        radar_bandwidth=2e9,
        radar_chirp_duration=2e-5,
        radar_fs=100e6,
        comm_bandwidth=800e6,
        comm_fs=1e9,
        comm_fft_size=2048,
        comm_cp_len=144,
        max_range=60.0,
        num_rx=1,
        csi_error=0.02,
        cfar={"num_train": 16, "num_guard": 4, "threshold_offset": 22},
        clutter={
            "ground_clutter": True,
            "ground_intensity": 0.005,
            "k_shape": 2.0,
            "range_exponent": 2.0,
            "weather_clutter": False,
            "weather_intensity": 0.0,
            "doppler_spread": 2.0,
        },
    ),
    "OTFS_HighMobility_Wideband": make_otfs_config(
        name="OTFS_HighMobility_Wideband",
        fc=39e9,
        mod_order=16,
        bandwidth=2e9,
        fs=100e6,
        n_doppler=128,
        n_delay=1024,
        symbol_duration=1e-5,
        max_range=80.0,
        num_rx=1,
        csi_error=0.08,
        cfar={
            "num_train": 16,
            "num_guard": 8,
            "threshold_offset": 25,
            "nms_kernel_size": 9,
            "min_range_m": 2.0,
        },
        clutter={
            "ground_clutter": True,
            "ground_intensity": 0.02,
            "k_shape": 1.5,
            "range_exponent": 2.0,
            "weather_clutter": True,
            "weather_intensity": 0.05,
            "doppler_spread": 8.0,
        },
    ),
}


def export_legacy_configs(unified_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Temporary compatibility bridge for old code paths.
    Prefer using UNIFIED_CONFIGS directly in new code.
    """
    legacy = {}
    for name, cfg in unified_configs.items():
        meta = cfg["meta"]
        rf = cfg["rf"]
        radar = cfg["radar"]
        comm = cfg["comm"]
        channel = cfg["channel"]
        item = {
            "mode": meta["mode"],
            "fc": rf["fc"],
            "mod_order": comm["mod_order"],
            "R_max": radar["max_range"],
            "num_rx": rf["num_rx"],
            "cfar_params": deepcopy(cfg["cfar"]),
            "adaptive_cfar": cfg.get("adaptive_cfar", True),
            "csi_error": channel["csi_error"],
            "channel_model": channel["model"],
            "clutter_params": deepcopy(cfg["clutter"]),
        }
        if meta["mode"] == "TRADITIONAL":
            item.update({
                "radar_B": radar["bandwidth"],
                "radar_T": radar["chirp_duration"],
                "radar_fs": radar["fs"],
                "comm_B": comm["bandwidth"],
                "comm_fs": comm["fs"],
                "comm_fft_size": comm["fft_size"],
                "comm_cp_len": comm["cp_len"],
            })
        else:
            item.update({
                "B": comm["bandwidth"],
                "fs": comm["fs"],
                "N_doppler": comm["n_doppler"],
                "N_delay": comm["n_delay"],
                "T_symbol": comm["symbol_duration"],
            })
        legacy[name] = item
    return legacy


RADAR_COMM_CONFIGS_G2 = export_legacy_configs(UNIFIED_CONFIGS)
CONFIG_ID_MAP = {name: idx for idx, name in enumerate(UNIFIED_CONFIGS.keys())}


# =============================================================================
# Utility helpers
# =============================================================================


def build_config_tensor(cfg: Dict[str, Any], snr_db: float = 20.0) -> torch.Tensor:
    rf = cfg["rf"]
    radar = cfg["radar"]
    comm = cfg["comm"]
    meta = cfg["meta"]

    comm_bw = float(comm.get("bandwidth", 0.0))
    radar_bw = float(radar.get("bandwidth", 0.0))
    waveform_scalar = 0.0 if meta["comm_waveform"] == "OFDM" else 1.0
    size_scalar = float(comm.get("fft_size", comm.get("n_delay", 64))) / 1024.0

    return torch.tensor([
        rf["fc"] / 1e9,
        radar_bw / 1e9,
        comm_bw / 1e9,
        size_scalar,
        comm["mod_order"] / 64.0,
        snr_db / 35.0,
        radar["max_range"] / 300.0,
        waveform_scalar,
    ], dtype=torch.float32)



def get_waveform_id(cfg: Dict[str, Any], task: str = "COMM") -> int:
    if task == "RADAR":
        return WAVEFORM_IDS[cfg["meta"]["radar_waveform"]]
    return WAVEFORM_IDS[cfg["meta"]["comm_waveform"]]



def get_mod_id(cfg: Dict[str, Any]) -> int:
    return MODULATION_IDS.get(cfg["comm"]["mod_order"], 0)


# =============================================================================
# Dataset generator
# =============================================================================


class AIRadar_Comm_Dataset_G3(Dataset):
    """
    Refactored multi-configuration radar/communication dataset.

    Key differences from old version
    --------------------------------
    - uses unified config schema
    - does NOT auto-generate in __init__ unless requested
    - OFDM and OTFS comm_info follow the same structural contract
    - OTFS honors self.mod_order
    - one-shot dump save instead of per-sample rewrite
    - communication tensors can be built consistently for unified models
    """

    TDL_MODELS = {
        "TDL-A": {
            "delays_s": np.array([0, 30, 70, 90, 110, 190, 410]) * 1e-9,
            "powers_dB": np.array([0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8]),
        },
        "TDL-B": {
            "delays_s": np.array([0, 10, 20, 30, 60, 90, 130]) * 1e-9,
            "powers_dB": np.array([0, -2.2, -4.0, -3.2, -9.8, -13.0, -15.0]),
        },
        "TDL-D": {
            "delays_s": np.array([0, 30, 100, 200, 230, 500]) * 1e-9,
            "powers_dB": np.array([-0.2, -13.5, -18.8, -21.0, -22.8, -17.9]),
        },
        "TDL-E": {
            "delays_s": np.array([0, 50, 120, 200, 230, 500]) * 1e-9,
            "powers_dB": np.array([-0.03, -22.0, -17.0, -20.0, -21.0, -22.0]),
        },
    }

    def __init__(
        self,
        config_name: str = "CN0566_TRADITIONAL",
        num_samples: int = 100,
        save_path: str = "data/radar_comm_dataset_g3",
        drawfig: bool = False,
        clutter_intensity: float = 0.1,
        fixed_snr: Optional[float] = None,
        enable_clutter: bool = True,
        enable_imperfect_csi: bool = True,
        enable_rf_impairments: bool = True,
        target_rcs_range: Optional[Tuple[float, float]] = (10, 30),
        auto_generate: bool = True,
    ):
        self.config_name = config_name
        self.config = deepcopy(UNIFIED_CONFIGS[config_name])
        self.num_samples = num_samples
        self.save_path = save_path
        self.drawfig = drawfig
        self.clutter_intensity = 0.1 if clutter_intensity is None else clutter_intensity
        self.fixed_snr = fixed_snr
        self.enable_clutter = enable_clutter
        self.enable_imperfect_csi = enable_imperfect_csi
        self.enable_rf_impairments = enable_rf_impairments
        self.target_rcs_range = (10, 30) if target_rcs_range is None else target_rcs_range

        self.meta = self.config["meta"]
        self.rf = self.config["rf"]
        self.radar_cfg = self.config["radar"]
        self.comm_cfg = self.config["comm"]
        self.channel_cfg = self.config["channel"]
        self.cfar_params = deepcopy(self.config["cfar"])
        self.clutter_params = deepcopy(self.config["clutter"])

        self.mode = self.meta["mode"]
        self.mod_order = int(self.comm_cfg["mod_order"])
        self.channel_model_type = self.channel_cfg.get("model", "awgn")
        self.csi_error = self.channel_cfg.get("csi_error", 0.0) if enable_imperfect_csi else 0.0
        self.adaptive_cfar = self.config.get("adaptive_cfar", True)

        self.fc = self.rf["fc"]
        self.c = 3e8
        self.lambda_c = self.c / self.fc
        self.data_samples: List[Dict[str, Any]] = []

        if self.mode == "TRADITIONAL":
            self.radar_B = self.radar_cfg["bandwidth"]
            self.radar_T = self.radar_cfg["chirp_duration"]
            self.radar_fs = self.radar_cfg["fs"]
            self.radar_slope = self.radar_B / self.radar_T
            self.radar_Ns = int(self.radar_fs * self.radar_T)
            self.radar_Nc = int(self.radar_cfg.get("num_chirps", 64))
            self.comm_B = self.comm_cfg["bandwidth"]
            self.comm_fs = self.comm_cfg["fs"]
            self.comm_fft = int(self.comm_cfg["fft_size"])
            self.comm_cp = int(self.comm_cfg["cp_len"])
            self.comm_num_symbols = int(self.comm_cfg.get("num_symbols", 14))
        else:
            self.B = self.comm_cfg["bandwidth"]
            self.fs = self.comm_cfg["fs"]
            self.Nd = int(self.comm_cfg["n_doppler"])
            self.Nt = int(self.comm_cfg["n_delay"])
            self.T_symbol = self.comm_cfg["symbol_duration"]

        os.makedirs(self.save_path, exist_ok=True)
        if auto_generate:
            self.generate_dataset()

    # -------------------------------------------------------------------------
    # Small helpers
    # -------------------------------------------------------------------------

    def _estimate_snr(self, rdm_db: np.ndarray) -> float:
        noise_floor = np.percentile(rdm_db, 25)
        peak_power = np.max(rdm_db)
        return float(peak_power - noise_floor)

    def _compute_adaptive_threshold(self, rdm_db: np.ndarray, base_threshold: float) -> float:
        estimated_snr = self._estimate_snr(rdm_db)
        if estimated_snr > 35:
            return base_threshold - 5
        if estimated_snr > 30:
            return base_threshold - 3
        if estimated_snr > 20:
            return base_threshold
        if estimated_snr > 15:
            return base_threshold + 3
        return base_threshold + 5

    def _generate_ground_clutter(self, rdm_shape: Tuple[int, int], r_axis: np.ndarray) -> np.ndarray:
        if not self.enable_clutter or not self.clutter_params.get("ground_clutter", False):
            return np.zeros(rdm_shape, dtype=np.float32)
        intensity = self.clutter_params.get("ground_intensity", 0.05)
        k_shape = self.clutter_params.get("k_shape", 2.0)
        range_exp = self.clutter_params.get("range_exponent", 2.5)
        range_profile = (r_axis + 1.0) ** (-range_exp)
        range_profile = range_profile / (np.max(range_profile) + 1e-8)
        gamma_samples = np.random.gamma(k_shape, 1.0 / max(k_shape, 1e-8), rdm_shape)
        rayleigh_samples = np.abs(np.random.randn(*rdm_shape) + 1j * np.random.randn(*rdm_shape))
        clutter = gamma_samples * rayleigh_samples
        if len(r_axis) == rdm_shape[1]:
            clutter = clutter * range_profile[None, :]
        return (intensity * clutter).astype(np.float32)

    def _generate_weather_clutter(self, rdm_shape: Tuple[int, int], v_axis: np.ndarray) -> np.ndarray:
        if not self.enable_clutter or not self.clutter_params.get("weather_clutter", False):
            return np.zeros(rdm_shape, dtype=np.float32)
        intensity = self.clutter_params.get("weather_intensity", 0.03)
        doppler_spread = self.clutter_params.get("doppler_spread", 3.0)
        doppler_profile = np.exp(-(v_axis ** 2) / (2.0 * doppler_spread ** 2))
        doppler_profile = doppler_profile / (np.max(doppler_profile) + 1e-8)
        weather = np.abs(np.random.randn(*rdm_shape) + 1j * np.random.randn(*rdm_shape))
        if len(v_axis) == rdm_shape[0]:
            weather = weather * doppler_profile[:, None]
        return (intensity * weather).astype(np.float32)

    def _add_clutter_to_rdm(self, rdm_db: np.ndarray, r_axis: np.ndarray, v_axis: np.ndarray) -> np.ndarray:
        rdm_linear = 10 ** (rdm_db / 20.0)
        signal_peak = np.percentile(rdm_linear, 99)
        clutter_scale = self.clutter_intensity * signal_peak
        ground = self._generate_ground_clutter(rdm_linear.shape, r_axis) * clutter_scale
        weather = self._generate_weather_clutter(rdm_linear.shape, v_axis) * clutter_scale * 0.5
        out = rdm_linear + ground + weather
        return 20.0 * np.log10(out + 1e-9)

    def _apply_imperfect_csi(self, h_true: np.ndarray) -> np.ndarray:
        if self.csi_error <= 0:
            return h_true
        err = self.csi_error * (np.random.randn(*h_true.shape) + 1j * np.random.randn(*h_true.shape)) * np.sqrt(0.5)
        return h_true + err * np.abs(h_true)

    def _generate_qam_symbols(self, num_symbols: int, mod_order: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if mod_order == 4:
            pts = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        elif mod_order == 8:
            pts = np.array([1+1j, 1-1j, -1+1j, -1-1j, 3+0j, -3+0j, 0+3j, 0-3j]) / np.sqrt(6)
        elif mod_order == 16:
            x = np.arange(-3, 4, 2)
            y = np.arange(-3, 4, 2)
            X, Y = np.meshgrid(x, y)
            pts = (X + 1j * Y).flatten() / np.sqrt(10)
        elif mod_order == 64:
            x = np.arange(-7, 8, 2)
            y = np.arange(-7, 8, 2)
            X, Y = np.meshgrid(x, y)
            pts = (X + 1j * Y).flatten() / np.sqrt(42)
        else:
            raise ValueError(f"Unsupported modulation order: {mod_order}")
        ints = np.random.randint(0, mod_order, num_symbols)
        return pts[ints], ints, pts

    def _demodulate_qam(self, rx_symbols: np.ndarray, const_pts: np.ndarray) -> np.ndarray:
        dists = np.abs(rx_symbols[:, None] - const_pts[None, :])
        return np.argmin(dists, axis=1)

    def _generate_dmrs(self, num_subcarriers: int, dmrs_spacing: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        dmrs_positions = np.arange(0, num_subcarriers, dmrs_spacing)
        N_zc = len(dmrs_positions)
        u = 25
        n = np.arange(N_zc)
        dmrs_symbols = np.exp(-1j * np.pi * u * n * (n + 1) / max(N_zc, 1))
        return dmrs_positions, dmrs_symbols

    def _mmse_channel_estimation(self, rx_dmrs, tx_dmrs, snr_db, dmrs_positions, num_subcarriers):
        H_ls = rx_dmrs / (tx_dmrs + 1e-10)
        snr_linear = 10 ** (snr_db / 10)
        mmse_weight = snr_linear / (snr_linear + 1)
        H_mmse_dmrs = H_ls * mmse_weight
        H_est = np.zeros(num_subcarriers, dtype=np.complex128)
        H_est[dmrs_positions] = H_mmse_dmrs
        for i in range(len(dmrs_positions) - 1):
            s, e = dmrs_positions[i], dmrs_positions[i + 1]
            sv, ev = H_mmse_dmrs[i], H_mmse_dmrs[i + 1]
            for k in range(s + 1, e):
                alpha = (k - s) / max((e - s), 1)
                H_est[k] = (1 - alpha) * sv + alpha * ev
        if dmrs_positions[0] > 0:
            H_est[:dmrs_positions[0]] = H_mmse_dmrs[0]
        if dmrs_positions[-1] < num_subcarriers - 1:
            H_est[dmrs_positions[-1] + 1:] = H_mmse_dmrs[-1]
        return H_est

    def _apply_fading_channel(self, signal: np.ndarray, fs: float, snr_db: float, tdl_model: Optional[str] = None):
        if self.channel_model_type != "multipath":
            sig_pow = np.mean(np.abs(signal) ** 2)
            noise_pow = sig_pow / 10 ** (snr_db / 10)
            noise = (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))) * np.sqrt(noise_pow / 2)
            return signal + noise, np.array([1.0 + 0j])

        if tdl_model and tdl_model in self.TDL_MODELS:
            model = self.TDL_MODELS[tdl_model]
            delays_s = model["delays_s"]
            powers_dB = model["powers_dB"]
            num_taps = len(delays_s)
        else:
            num_taps = np.random.randint(2, 6)
            max_delay = 2e-7
            delays_s = np.sort(np.random.uniform(0, max_delay, num_taps))
            delays_s[0] = 0
            powers_dB = -np.random.uniform(2, 5, num_taps) * np.arange(num_taps)

        delays_samp = np.round(delays_s * fs).astype(int)
        delays_samp = np.unique(delays_samp)
        powers_dB = powers_dB[: len(delays_samp)]
        powers_lin = 10 ** (powers_dB / 10)
        powers_lin /= np.sum(powers_lin)
        taps = np.sqrt(powers_lin) * (np.random.randn(len(delays_samp)) + 1j * np.random.randn(len(delays_samp))) / np.sqrt(2)
        h_imp = np.zeros(delays_samp[-1] + 1 if len(delays_samp) > 0 else 1, dtype=np.complex128)
        h_imp[delays_samp] = taps
        rx_clean = np.convolve(signal, h_imp, mode="full")
        sig_pow = np.mean(np.abs(rx_clean) ** 2)
        noise_pow = sig_pow / 10 ** (snr_db / 10)
        noise = (np.random.randn(len(rx_clean)) + 1j * np.random.randn(len(rx_clean))) * np.sqrt(noise_pow / 2)
        return rx_clean + noise, h_imp

    def _apply_rf_impairments(self, signal: np.ndarray, snr_db: float, fs: float) -> np.ndarray:
        severity = np.clip(1.0 - (snr_db - 5) / 40.0, 0.15, 0.5)
        phase_noise_std = 0.01 * severity
        phase_noise = np.cumsum(np.random.randn(len(signal))) * phase_noise_std
        signal = signal * np.exp(1j * phase_noise)
        g_imb = 0.015 * severity
        phi_imb = 0.02 * severity
        I, Q = signal.real, signal.imag
        I_out = (1 + g_imb) * I
        Q_out = (1 - g_imb) * (Q * np.cos(phi_imb) + I * np.sin(phi_imb))
        signal = I_out + 1j * Q_out
        cfo_hz = np.random.uniform(-10, 10) * severity
        t = np.arange(len(signal)) / fs
        signal = signal * np.exp(1j * 2 * np.pi * cfo_hz * t)
        return signal

    # -------------------------------------------------------------------------
    # Traditional simulation
    # -------------------------------------------------------------------------

    def _simulate_traditional(self, targets: List[Dict[str, float]], snr_db: float, use_dmrs: bool = True) -> Dict[str, Any]:
        Nfft = self.comm_fft
        Ncp = self.comm_cp
        num_data_syms = self.comm_num_symbols

        _, _, const_pts = self._generate_qam_symbols(0, self.mod_order)
        if use_dmrs:
            dmrs_positions, dmrs_symbols = self._generate_dmrs(Nfft, dmrs_spacing=4)
            pilot_syms = np.zeros(Nfft, dtype=np.complex128)
            pilot_syms[dmrs_positions] = dmrs_symbols
            non_dmrs = np.setdiff1d(np.arange(Nfft), dmrs_positions)
            pilot_syms[non_dmrs], _, _ = self._generate_qam_symbols(len(non_dmrs), 4)
        else:
            pilot_syms, _, _ = self._generate_qam_symbols(Nfft, 4)

        total_data_qam = num_data_syms * Nfft
        data_syms, data_ints, _ = self._generate_qam_symbols(total_data_qam, self.mod_order)
        data_grid = data_syms.reshape(num_data_syms, Nfft)
        full_grid = np.vstack([pilot_syms[None, :], data_grid])

        ifft_out = np.fft.ifft(full_grid, axis=1)
        cp = ifft_out[:, -Ncp:]
        tx_grid_time = np.hstack([cp, ifft_out])
        tx_signal = tx_grid_time.flatten()

        rx_time_full, h_true = self._apply_fading_channel(tx_signal, self.comm_fs, snr_db)
        if self.enable_rf_impairments:
            rx_time_full = self._apply_rf_impairments(rx_time_full, snr_db, self.comm_fs)

        first_tap_idx = np.argmax(np.abs(h_true) > 0)
        rx_time = rx_time_full[first_tap_idx:first_tap_idx + len(tx_signal)]
        if len(rx_time) < len(tx_signal):
            rx_time = np.pad(rx_time, (0, len(tx_signal) - len(rx_time)))

        rx_reshaped = rx_time.reshape(num_data_syms + 1, Nfft + Ncp)
        rx_no_cp = rx_reshaped[:, Ncp:]
        rx_grid = np.fft.fft(rx_no_cp, axis=1)

        Y_pilot = rx_grid[0, :]
        if use_dmrs:
            rx_dmrs = Y_pilot[dmrs_positions]
            H_est = self._mmse_channel_estimation(rx_dmrs, dmrs_symbols, snr_db, dmrs_positions, Nfft)
        else:
            H_est = Y_pilot / (pilot_syms + 1e-10)
            H_est = self._apply_imperfect_csi(H_est)

        Y_data = rx_grid[1:, :]
        noise_var = 1.0 / (10 ** (snr_db / 10))
        H_eq = np.conj(H_est) / (np.abs(H_est) ** 2 + noise_var + 1e-10)
        X_hat_grid = Y_data * H_eq[None, :]
        rx_const = X_hat_grid.flatten()
        demod_ints = self._demodulate_qam(rx_const, const_pts)
        ber = float(np.mean(data_ints != demod_ints))

        # Radar FMCW
        Nc = self.radar_Nc
        Ns = self.radar_Ns
        fs = self.radar_fs
        slope = self.radar_slope
        t_fast = np.arange(Ns) / fs
        t_slow = np.arange(Nc) * self.radar_T
        beat_signal = np.zeros((Nc, Ns), dtype=np.complex64)
        for tgt in targets:
            fb = slope * 2 * tgt["range"] / self.c
            fd = 2 * tgt["velocity"] / self.lambda_c
            phase = 2 * np.pi * (fb * t_fast[None, :] + fd * t_slow[:, None])
            amp = np.sqrt(10 ** (tgt["rcs"] / 10))
            beat_signal += amp * np.exp(1j * phase)

        sig_pow_rad = np.mean(np.abs(beat_signal) ** 2)
        if sig_pow_rad > 0:
            noise_pow_rad = sig_pow_rad / 10 ** (snr_db / 10)
            noise_rad = (np.random.randn(Nc, Ns) + 1j * np.random.randn(Nc, Ns)) * np.sqrt(noise_pow_rad / 2)
            beat_signal += noise_rad

        win_range = np.hanning(Ns)[None, :]
        win_doppler = np.hanning(Nc)[:, None]
        beat_signal_win = beat_signal * win_range * win_doppler
        r_fft = np.fft.fft(beat_signal_win, axis=1)
        rd_map = np.fft.fftshift(np.fft.fft(r_fft, axis=0), axes=0)
        rd_map_db = 20 * np.log10(np.abs(rd_map) + 1e-9)

        r_res = self.c * fs / (2 * slope * Ns)
        v_res = self.lambda_c / (2 * Nc * self.radar_T)
        r_axis = np.arange(Ns) * r_res
        v_axis = np.arange(-Nc // 2, Nc // 2) * v_res
        rd_map_db = self._add_clutter_to_rdm(rd_map_db, r_axis, v_axis)

        comm_info = {
            "waveform": "OFDM",
            "baseline_name": "MMSE_hard_demap",
            "ber": ber,
            "tx_symbols": data_grid.flatten(),
            "rx_symbols": rx_const,
            "tx_ints": data_ints,
            "demod_ints": demod_ints,
            "mod_order": self.mod_order,
            "snr_db": snr_db,
            "grid_h": num_data_syms,
            "grid_w": Nfft,
            "num_data_syms": num_data_syms,
            "fft_size": Nfft,
            "channel_est": H_est,
            "rx_grid": X_hat_grid,
            "pilot_rx": Y_pilot,
            "raw_rx_grid": Y_data,
            "noise_var": noise_var,
        }

        return {
            "rd_map": rd_map_db,
            "r_axis": r_axis,
            "v_axis": v_axis,
            "comm_info": comm_info,
            "comm_feature_grid": X_hat_grid,
        }

    # -------------------------------------------------------------------------
    # OTFS simulation
    # -------------------------------------------------------------------------

    def _otfs_modulate(self, dd_grid: np.ndarray) -> np.ndarray:
        tf_grid = np.fft.fft(dd_grid, axis=0)
        tf_grid = np.fft.ifft(tf_grid, axis=1)
        time_domain_grid = np.fft.ifft(tf_grid, axis=0)
        return time_domain_grid.flatten(order="F")

    def _otfs_demodulate(self, rx_signal: np.ndarray, Nt: int, Nd: int) -> np.ndarray:
        time_domain_grid = rx_signal.reshape((Nt, Nd), order="F")
        tf_grid = np.fft.fft(time_domain_grid, axis=0)
        dd_grid = np.fft.fft(tf_grid, axis=1)
        dd_grid = np.fft.ifft(dd_grid, axis=0)
        return dd_grid

    def _simulate_otfs(self, targets: List[Dict[str, float]], snr_db: float) -> Dict[str, Any]:
        Ns = self.Nt
        Nc = self.Nd
        num_symbols = Ns * Nc

        tx_symbols, tx_ints, const_pts = self._generate_qam_symbols(num_symbols, self.mod_order)
        tx_dd_grid = tx_symbols.reshape((Ns, Nc))
        tx_signal = self._otfs_modulate(tx_dd_grid)

        # Radar echo generation
        n_samples = tx_signal.size
        rx_radar = np.zeros(n_samples, dtype=np.complex128)
        time_vector = np.arange(n_samples) / self.fs
        for tgt in targets:
            range_m = tgt["range"]
            velocity_mps = tgt["velocity"]
            rcs = tgt["rcs"]
            amplitude = np.sqrt(10 ** (rcs / 10))
            delay_sec = 2 * range_m / self.c
            delay_samples = int(round(delay_sec * self.fs))
            if delay_samples < n_samples:
                delayed_signal = np.roll(tx_signal, delay_samples)
                delayed_signal[:delay_samples] = 0
                doppler_hz = 2 * velocity_mps * self.fc / self.c
                doppler_shift = np.exp(1j * 2 * np.pi * doppler_hz * time_vector)
                rx_radar += amplitude * delayed_signal * doppler_shift

        sig_pow = np.mean(np.abs(rx_radar) ** 2)
        if sig_pow > 0:
            noise_pow = sig_pow / 10 ** (snr_db / 10)
            noise = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) * np.sqrt(noise_pow / 2)
            rx_radar += noise

        # Comm through channel
        rx_comm_full, h_true = self._apply_fading_channel(tx_signal, self.fs, snr_db)
        if self.enable_rf_impairments:
            rx_comm_full = self._apply_rf_impairments(rx_comm_full, snr_db, self.fs)

        first_tap_idx = np.argmax(np.abs(h_true) > 0)
        rx_comm = rx_comm_full[first_tap_idx:first_tap_idx + len(tx_signal)]
        if len(rx_comm) < len(tx_signal):
            rx_comm = np.pad(rx_comm, (0, len(tx_signal) - len(rx_comm)))

        rx_dd_comm = self._otfs_demodulate(rx_comm, Ns, Nc)

        # Simple equivalent channel estimate in delay-frequency sense
        H_freq = np.fft.fft(h_true, n=Ns)
        H_freq = self._apply_imperfect_csi(H_freq)
        H_eq = np.tile(H_freq[:, None], (1, Nc))
        noise_var = 1.0 / (10 ** (snr_db / 10))
        W = np.conj(H_eq) / (np.abs(H_eq) ** 2 + noise_var + 1e-10)
        rx_dd_eq = rx_dd_comm * W

        rx_const = rx_dd_eq.flatten()
        demod_ints = self._demodulate_qam(rx_const, const_pts)
        ber = float(np.mean(tx_ints != demod_ints))

        # OTFS radar map
        rx_time_radar = rx_radar.reshape((Ns, Nc), order="F")
        rx_tf_radar = np.fft.fft(rx_time_radar, axis=0)
        rx_dd_radar = np.fft.fft(rx_tf_radar, axis=1)
        rx_dd_radar = np.fft.ifft(rx_dd_radar, axis=0)
        rx_dd_fft = np.fft.fft2(rx_dd_radar)
        tx_dd_fft = np.fft.fft2(tx_dd_grid)
        ddm_fft = rx_dd_fft / (tx_dd_fft + 1e-6)
        ddm_complex = np.fft.ifft2(ddm_fft)
        ddm_shifted = np.fft.fftshift(ddm_complex.T, axes=0)
        ddm_mag = np.abs(ddm_shifted)
        rd_map_db_full = 20 * np.log10(ddm_mag + 1e-6)

        r_res = self.c / (2 * self.fs)
        num_range_bins = int(self.radar_cfg.get("max_range", 100.0) / r_res)
        num_range_bins = max(1, min(num_range_bins, rd_map_db_full.shape[1]))
        rd_map_db = rd_map_db_full[:, :num_range_bins]
        r_axis = np.arange(num_range_bins) * r_res
        T_actual = Ns / self.fs
        v_axis = np.fft.fftshift(np.fft.fftfreq(Nc, d=T_actual)) * self.lambda_c / 2
        rd_map_db = self._add_clutter_to_rdm(rd_map_db, r_axis, v_axis)

        comm_info = {
            "waveform": "OTFS",
            "baseline_name": "OTFS_eq_hard_demap",
            "ber": ber,
            "tx_symbols": tx_symbols,
            "rx_symbols": rx_const,
            "tx_ints": tx_ints,
            "demod_ints": demod_ints,
            "mod_order": self.mod_order,
            "snr_db": snr_db,
            "grid_h": Ns,
            "grid_w": Nc,
            "n_delay": Ns,
            "n_doppler": Nc,
            "channel_est": H_eq,
            "rx_grid": rx_dd_eq,
            "raw_rx_grid": rx_dd_comm,
            "noise_var": noise_var,
        }

        return {
            "rd_map": rd_map_db,
            "r_axis": r_axis,
            "v_axis": v_axis,
            "comm_info": comm_info,
            "comm_feature_grid": rx_dd_eq,
        }

    # -------------------------------------------------------------------------
    # Common post-processing
    # -------------------------------------------------------------------------

    def _run_cfar(self, rdm_db: np.ndarray, r_axis: np.ndarray, v_axis: np.ndarray) -> List[Dict[str, float]]:
        p = self.cfar_params
        nt, ng = p["num_train"], p["num_guard"]
        base_thresh = p["threshold_offset"]
        thresh = self._compute_adaptive_threshold(rdm_db, base_thresh) if self.adaptive_cfar else base_thresh

        norm_rdm = rdm_db.copy()
        gp = p.get("global_percentile", None)
        if gp is not None:
            pval = np.percentile(norm_rdm, gp)
            norm_rdm = np.minimum(norm_rdm, pval)

        kernel_size = 1 + 2 * (nt + ng)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
        guard_region = 1 + 2 * ng
        s, e = nt, nt + guard_region
        kernel[s:e, s:e] = 0
        kernel /= np.sum(kernel)

        noise_est = convolve2d(norm_rdm, kernel, mode="same", boundary="symm")
        detections = norm_rdm > (noise_est + thresh)
        if p["nms_kernel_size"] > 1:
            local_max = maximum_filter(norm_rdm, size=p["nms_kernel_size"])
            detections = detections & (norm_rdm == local_max)

        idxs = np.argwhere(detections)
        min_r = p.get("min_range_m", 0.0)
        min_v = p.get("min_speed_mps", 0.0)
        notch_k = p.get("notch_doppler_bins", 0)
        center = len(v_axis) // 2
        candidates = []
        for d_idx, r_idx in idxs:
            if d_idx >= len(v_axis) or r_idx >= len(r_axis):
                continue
            range_m = r_axis[r_idx]
            vel_mps = v_axis[d_idx]
            if range_m < min_r or abs(vel_mps) < min_v:
                continue
            if notch_k > 0 and abs(d_idx - center) <= notch_k:
                continue
            candidates.append({
                "range_m": float(range_m),
                "velocity_mps": float(vel_mps),
                "range_idx": int(r_idx),
                "doppler_idx": int(d_idx),
                "power": float(norm_rdm[d_idx, r_idx]),
            })
        max_peaks = p.get("max_peaks", None)
        if max_peaks is not None:
            candidates = sorted(candidates, key=lambda x: x["power"], reverse=True)[:max_peaks]
        return candidates

    def _target_range_velocity_sampler(self) -> Tuple[float, float]:
        max_range = float(self.radar_cfg["max_range"])
        fc = self.rf["fc"]
        comm_waveform = self.meta["comm_waveform"]

        if comm_waveform == "OTFS" and "HighMobility" in self.config_name:
            velocity = np.random.uniform(-45.0, 45.0)
        elif fc >= 60e9:
            velocity = np.random.uniform(-35.0, 35.0)
        else:
            velocity = np.random.uniform(-15.0, 15.0)

        if max_range > 150:
            range_m = np.random.uniform(10.0, 0.85 * max_range)
        else:
            range_m = np.random.uniform(5.0, 0.8 * max_range)
        return range_m, velocity

    # -------------------------------------------------------------------------
    # Dataset generation and model-ready tensor builder
    # -------------------------------------------------------------------------

    def generate_dataset(self, save_dump: bool = True) -> None:
        clutter_str = "ON" if self.enable_clutter else "OFF"
        csi_str = f"{self.csi_error * 100:.0f}%" if self.enable_imperfect_csi else "Perfect"
        print(f"Generating {self.num_samples} samples in {self.mode} mode...")
        print(f"Config: {self.mod_order}-QAM | Channel: {self.channel_model_type} | Clutter: {clutter_str} | CSI Error: {csi_str}")

        dump_items = []
        self.data_samples = []

        for _ in tqdm(range(self.num_samples)):
            num_t = np.random.randint(1, 4)
            targets = []
            for _ in range(num_t):
                range_m, velocity = self._target_range_velocity_sampler()
                targets.append({
                    "range": float(range_m),
                    "velocity": float(velocity),
                    "rcs": float(np.random.uniform(self.target_rcs_range[0], self.target_rcs_range[1])),
                })

            snr_db = float(self.fixed_snr if self.fixed_snr is not None else np.random.uniform(5, 35))
            out = self._simulate_traditional(targets, snr_db) if self.mode == "TRADITIONAL" else self._simulate_otfs(targets, snr_db)

            detections = self._run_cfar(out["rd_map"], out["r_axis"], out["v_axis"])
            sample = {
                "mode": self.mode,
                "config_name": self.config_name,
                "mod_order": self.mod_order,
                "comm_waveform": self.meta["comm_waveform"],
                "radar_waveform": self.meta["radar_waveform"],
                "channel_model": self.channel_model_type,
                "range_doppler_map": torch.tensor(out["rd_map"], dtype=torch.float32),
                "range_axis": out["r_axis"],
                "velocity_axis": out["v_axis"],
                "target_info": {
                    "targets": targets,
                    "snr_db": snr_db,
                },
                "comm_info": out["comm_info"],
                "cfar_detections": detections,
                "comm_feature_grid": out["comm_feature_grid"],
            }
            self.data_samples.append(sample)
            dump_items.append({
                "range_doppler_map": sample["range_doppler_map"].numpy(),
                "range_axis": sample["range_axis"],
                "velocity_axis": sample["velocity_axis"],
                "target_info": sample["target_info"],
                "comm_info": sample["comm_info"],
                "cfar_detections": sample["cfar_detections"],
            })

        if save_dump:
            dump_path = os.path.join(self.save_path, f"joint_dump_{self.config_name}_{self.num_samples}.npy")
            np.save(dump_path, np.array(dump_items, dtype=object))
            with open(os.path.join(self.save_path, f"meta_{self.config_name}.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "config_name": self.config_name,
                    "num_samples": self.num_samples,
                    "config": self.config,
                }, f, indent=2)

    def build_comm_tensor(self, sample: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build unified communication tensor for ISACFoundationModel.forward_comm().

        Returns:
            x_comm:        [C, H, W]
            config_tensor: [8]
            waveform_id:   scalar tensor
            mod_id:        scalar tensor
        """
        comm_info = sample["comm_info"]
        snr_db = float(comm_info.get("snr_db", sample["target_info"].get("snr_db", 20.0)))
        cfg = self.config

        rx_grid = np.asarray(comm_info["rx_grid"])
        H_est = np.asarray(comm_info["channel_est"])

        if rx_grid.ndim == 1:
            h = int(comm_info["grid_h"])
            w = int(comm_info["grid_w"])
            rx_grid = rx_grid.reshape(h, w)

        if H_est.ndim == 1:
            if self.meta["comm_waveform"] == "OFDM":
                H_est = np.tile(H_est[None, :], (rx_grid.shape[0], 1))
            else:
                H_est = np.tile(H_est[:, None], (1, rx_grid.shape[1]))

        eq_real = rx_grid.real.astype(np.float32)
        eq_imag = rx_grid.imag.astype(np.float32)
        H_mag = np.abs(H_est).astype(np.float32)
        H_mag = H_mag / (H_mag.max() + 1e-8)
        H_phase = (np.angle(H_est) / np.pi).astype(np.float32)
        snr_ch = np.full_like(eq_real, snr_db / 35.0, dtype=np.float32)

        x_comm = torch.tensor(np.stack([eq_real, eq_imag, H_mag, H_phase, snr_ch], axis=0), dtype=torch.float32)
        config_tensor = build_config_tensor(cfg, snr_db=snr_db)
        waveform_id = torch.tensor(get_waveform_id(cfg, task="COMM"), dtype=torch.long)
        mod_id = torch.tensor(get_mod_id(cfg), dtype=torch.long)
        return x_comm, config_tensor, waveform_id, mod_id

    def __len__(self) -> int:
        return len(self.data_samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data_samples[idx]
        x_comm, config_tensor, waveform_id, mod_id = self.build_comm_tensor(sample)
        sample = dict(sample)
        sample["isac_comm_tensor"] = x_comm
        sample["config_tensor"] = config_tensor
        sample["waveform_id"] = waveform_id
        sample["mod_id"] = mod_id
        sample["config_id"] = torch.tensor(CONFIG_ID_MAP[self.config_name], dtype=torch.long)
        return sample


# =============================================================================
# Backward-compatible model factory helper
# =============================================================================


def create_comm_model(model_name: str, device: torch.device):
    from typing import Tuple
    import torch.nn as nn

    # Lazy imports to avoid hard dependency ordering.
    from isac_foundation_model_v1 import ISACFoundationModel

    model_name = model_name.lower()
    if model_name == "isac":
        return ISACFoundationModel(comm_in_channels=5, base_ch=64, cond_dim=64, config_dim=8).to(device), False
    raise ValueError(f"Unsupported comm model in this refactor module: {model_name}")


# =============================================================================
# Example usage
# =============================================================================


def test_one():
    ds = AIRadar_Comm_Dataset_G3(
        config_name="OTFS_HighMobility_Wideband",
        num_samples=4,
        save_path="data/debug_g3_refactor",
        auto_generate=True,
    )
    item = ds[0]
    print("config_name:", item["config_name"])
    print("mode:", item["mode"])
    print("rdm:", tuple(item["range_doppler_map"].shape))
    print("comm tensor:", tuple(item["isac_comm_tensor"].shape))
    print("config_tensor:", tuple(item["config_tensor"].shape))
    print("waveform_id:", int(item["waveform_id"]))
    print("mod_id:", int(item["mod_id"]))
    print("comm waveform:", item["comm_info"]["waveform"])
    print("comm mod order:", item["comm_info"]["mod_order"])

if __name__ == "__main__":
    test_configs = [
        "CN0566_TRADITIONAL",
        "CN0566_OTFS_ISAC",
        "AUTOMOTIVE_TRADITIONAL",
        "AUTOMOTIVE_OTFS_ISAC",
        "5G_ISAC_HighBandwidth",
        "OTFS_HighMobility_Wideband",
    ]

    for cfg_name in test_configs:
        print("\\n" + "=" * 80)
        print("Testing:", cfg_name)

        ds = AIRadar_Comm_Dataset_G3(
            config_name=cfg_name,
            num_samples=2,
            save_path="data/debug_g3_refactor",
            auto_generate=True,
        )
        item = ds[0]
        print("config_name:", item["config_name"])
        print("mode:", item["mode"])
        print("rdm:", tuple(item["range_doppler_map"].shape))
        print("comm tensor:", tuple(item["isac_comm_tensor"].shape))
        print("config_tensor:", tuple(item["config_tensor"].shape))
        print("waveform_id:", int(item["waveform_id"]))
        print("mod_id:", int(item["mod_id"]))
        print("comm waveform:", item["comm_info"]["waveform"])
        print("comm mod order:", item["comm_info"]["mod_order"])

"""
python AIRadar/AIradar_comm_model_g6_commtrain.py --comm_type all --qam_type all --epochs 2 --train_samples 20 --val_samples 8 --test_samples 8
"""
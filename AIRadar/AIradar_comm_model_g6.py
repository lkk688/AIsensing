#!/usr/bin/env python3
"""
AIradar_comm_model_g3_refactor.py

Refactored standalone training and evaluation pipeline for multi-configuration
radar and communication learning.

Design goals:
1. Correctness first: reduce duplicated logic and hidden mismatch bugs.
2. Paper-ready evaluation: seen-config, unseen-config, SNR sweeps, radar/comm baselines.
3. Multi-config generalization: balanced config sampling and explicit conditioning.
4. Clear separation of responsibilities: data, model, train, eval, report.

This script intentionally keeps radar and communication models separate, but makes
both of them configuration-aware and evaluates them with a common philosophy.

Expected external dependencies from prior project code:
- AIradar_comm_model_g2c.py
- AIradar_comm_dataset_g2.py

If some imported functions/classes have slightly different signatures in your repo,
adjust those adapter calls in the small helper functions near the imports.
"""

from __future__ import annotations

import os
import json
import math
import time
import copy
import random
import pickle
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import torch
from AIradar_comm_models import ISACFoundationModel
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler


from scipy.constants import c
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter
import h5py
from tqdm import tqdm

RADAR_COMM_CONFIGS_G2 = {'CN0566_TRADITIONAL': {'mode': 'TRADITIONAL',
                        'fc': 10250000000.0,
                        'mod_order': 16,
                        'radar_B': 500000000.0,
                        'radar_T': 0.0005,
                        'radar_fs': 2000000.0,
                        'comm_B': 40000000.0,
                        'comm_fs': 61440000.0,
                        'comm_fft_size': 64,
                        'comm_cp_len': 16,
                        'channel_model': 'multipath',
                        'R_max': 150.0,
                        'num_rx': 1,
                        'cfar_params': {'num_train': 12,
                                        'num_guard': 4,
                                        'threshold_offset': 25,
                                        'nms_kernel_size': 7},
                        'adaptive_cfar': True,
                        'csi_error': 0.1,
                        'clutter_params': {'ground_clutter': True,
                                           'ground_intensity': 0.005,
                                           'k_shape': 2.0,
                                           'range_exponent': 2.5,
                                           'weather_clutter': False,
                                           'weather_intensity': 0.02,
                                           'doppler_spread': 3.0}},
 'CN0566_OTFS_ISAC': {'mode': 'OTFS',
                      'fc': 10250000000.0,
                      'mod_order': 4,
                      'B': 40000000.0,
                      'fs': 40000000.0,
                      'N_doppler': 64,
                      'N_delay': 512,
                      'T_symbol': 1.28e-05,
                      'channel_model': 'multipath',
                      'R_max': 100.0,
                      'num_rx': 1,
                      'cfar_params': {'num_train': 4,
                                      'num_guard': 2,
                                      'threshold_offset': 25,
                                      'nms_kernel_size': 5,
                                      'min_range_m': 2.0,
                                      'min_speed_mps': 0.0,
                                      'notch_doppler_bins': 0},
                      'adaptive_cfar': True,
                      'csi_error': 0.15,
                      'clutter_params': {'ground_clutter': True,
                                         'ground_intensity': 0.03,
                                         'k_shape': 1.5,
                                         'range_exponent': 2.0,
                                         'weather_clutter': False,
                                         'weather_intensity': 0.01,
                                         'doppler_spread': 5.0}},
 'Automotive_77GHz_LongRange': {'mode': 'TRADITIONAL',
                                'fc': 77000000000.0,
                                'mod_order': 4,
                                'radar_B': 1500000000.0,
                                'radar_T': 4e-05,
                                'radar_fs': 51200000.0,
                                'comm_B': 400000000.0,
                                'comm_fs': 512000000.0,
                                'comm_fft_size': 1024,
                                'comm_cp_len': 72,
                                'channel_model': 'multipath',
                                'R_max': 100.0,
                                'num_rx': 1,
                                'cfar_params': {'num_train': 10,
                                                'num_guard': 4,
                                                'threshold_offset': 25,
                                                'nms_kernel_size': 7},
                                'adaptive_cfar': True,
                                'csi_error': 0.05,
                                'clutter_params': {'ground_clutter': True,
                                                   'ground_intensity': 0.008,
                                                   'k_shape': 3.0,
                                                   'range_exponent': 3.0,
                                                   'weather_clutter': True,
                                                   'weather_intensity': 0.03,
                                                   'doppler_spread': 2.0}},
 '8QAM_MediumRange': {'mode': 'TRADITIONAL',
                      'fc': 28000000000.0,
                      'mod_order': 8,
                      'radar_B': 800000000.0,
                      'radar_T': 0.0001,
                      'radar_fs': 40000000.0,
                      'comm_B': 100000000.0,
                      'comm_fs': 122880000.0,
                      'comm_fft_size': 256,
                      'comm_cp_len': 32,
                      'channel_model': 'multipath',
                      'R_max': 80.0,
                      'num_rx': 1,
                      'cfar_params': {'num_train': 12,
                                      'num_guard': 4,
                                      'threshold_offset': 25,
                                      'nms_kernel_size': 7},
                      'adaptive_cfar': True,
                      'csi_error': 0.08,
                      'clutter_params': {'ground_clutter': True,
                                         'ground_intensity': 0.007,
                                         'k_shape': 2.5,
                                         'range_exponent': 2.5,
                                         'weather_clutter': True,
                                         'weather_intensity': 0.025,
                                         'doppler_spread': 2.5}},
 'XBand_10GHz_MediumRange': {'mode': 'TRADITIONAL',
                             'fc': 10000000000.0,
                             'mod_order': 16,
                             'radar_B': 1000000000.0,
                             'radar_T': 0.00016,
                             'radar_fs': 40000000.0,
                             'comm_B': 40000000.0,
                             'comm_fs': 40000000.0,
                             'comm_fft_size': 64,
                             'comm_cp_len': 16,
                             'channel_model': 'multipath',
                             'R_max': 100.0,
                             'num_rx': 1,
                             'cfar_params': {'num_train': 24,
                                             'num_guard': 8,
                                             'threshold_offset': 25,
                                             'nms_kernel_size': 9},
                             'adaptive_cfar': True,
                             'csi_error': 0.1,
                             'clutter_params': {'ground_clutter': True,
                                                'ground_intensity': 0.006,
                                                'k_shape': 2.5,
                                                'range_exponent': 2.5,
                                                'weather_clutter': True,
                                                'weather_intensity': 0.04,
                                                'doppler_spread': 4.0}},
 'AUTOMOTIVE_TRADITIONAL': {'mode': 'TRADITIONAL',
                            'fc': 77000000000.0,
                            'mod_order': 16,
                            'radar_B': 1500000000.0,
                            'radar_T': 6e-05,
                            'radar_fs': 50000000.0,
                            'comm_B': 400000000.0,
                            'comm_fs': 512000000.0,
                            'comm_fft_size': 1024,
                            'comm_cp_len': 72,
                            'channel_model': 'multipath',
                            'R_max': 250.0,
                            'num_rx': 4,
                            'cfar_params': {'num_train': 16,
                                            'num_guard': 4,
                                            'threshold_offset': 25,
                                            'nms_kernel_size': 9},
                            'adaptive_cfar': True,
                            'csi_error': 0.01,
                            'clutter_params': {'ground_clutter': True,
                                               'ground_intensity': 0.01,
                                               'k_shape': 2.0,
                                               'range_exponent': 2.0,
                                               'weather_clutter': True,
                                               'weather_intensity': 0.05,
                                               'doppler_spread': 3.0}},
 'AUTOMOTIVE_OTFS_ISAC': {'mode': 'OTFS',
                          'fc': 77000000000.0,
                          'mod_order': 4,
                          'B': 1536000000.0,
                          'fs': 51200000.0,
                          'N_doppler': 128,
                          'N_delay': 512,
                          'T_symbol': 4e-05,
                          'channel_model': 'multipath',
                          'R_max': 100.0,
                          'num_rx': 1,
                          'cfar_params': {'num_train': 16,
                                          'num_guard': 8,
                                          'threshold_offset': 22,
                                          'nms_kernel_size': 11,
                                          'min_range_m': 2.0,
                                          'min_speed_mps': 0.0,
                                          'notch_doppler_bins': 0},
                          'adaptive_cfar': True,
                          'csi_error': 0.1,
                          'clutter_params': {'ground_clutter': True,
                                             'ground_intensity': 0.03,
                                             'k_shape': 2.5,
                                             'range_exponent': 2.5,
                                             'weather_clutter': True,
                                             'weather_intensity': 0.04,
                                             'doppler_spread': 2.5}},
 '5G_ISAC_HighBandwidth': {'mode': 'TRADITIONAL',
                           'fc': 28000000000.0,
                           'mod_order': 64,
                           'radar_B': 2000000000.0,
                           'radar_T': 2e-05,
                           'radar_fs': 100000000.0,
                           'comm_B': 800000000.0,
                           'comm_fs': 1000000000.0,
                           'comm_fft_size': 2048,
                           'comm_cp_len': 144,
                           'channel_model': 'multipath',
                           'R_max': 60.0,
                           'num_rx': 1,
                           'cfar_params': {'num_train': 16,
                                           'num_guard': 4,
                                           'threshold_offset': 22,
                                           'nms_kernel_size': 7},
                           'adaptive_cfar': True,
                           'csi_error': 0.02,
                           'clutter_params': {'ground_clutter': True,
                                              'ground_intensity': 0.005,
                                              'k_shape': 2.0,
                                              'range_exponent': 2.0,
                                              'weather_clutter': False,
                                              'weather_intensity': 0.0,
                                              'doppler_spread': 2.0}},
 'OTFS_HighMobility_Wideband': {'mode': 'OTFS',
                                'fc': 39000000000.0,
                                'mod_order': 16,
                                'B': 2000000000.0,
                                'fs': 100000000.0,
                                'N_doppler': 128,
                                'N_delay': 1024,
                                'T_symbol': 1e-05,
                                'channel_model': 'multipath',
                                'R_max': 80.0,
                                'num_rx': 1,
                                'cfar_params': {'num_train': 16,
                                                'num_guard': 8,
                                                'threshold_offset': 25,
                                                'nms_kernel_size': 9,
                                                'min_range_m': 2.0,
                                                'min_speed_mps': 0.0,
                                                'notch_doppler_bins': 0},
                                'adaptive_cfar': True,
                                'csi_error': 0.08,
                                'clutter_params': {'ground_clutter': True,
                                                   'ground_intensity': 0.02,
                                                   'k_shape': 1.5,
                                                   'range_exponent': 2.0,
                                                   'weather_clutter': True,
                                                   'weather_intensity': 0.05,
                                                   'doppler_spread': 8.0}}}

class AIRadar_Comm_Dataset_G2(Dataset):
    """
    Enhanced Radar-Communication Dataset Generator (G2)
    
    Improvements over G1:
    - Adaptive CFAR thresholds based on SNR estimation
    - Realistic clutter modeling (ground + weather)
    - Imperfect CSI for realistic communication performance
    - Multi-SNR evaluation support
    """

    def __init__(self, config_name='CN0566_TRADITIONAL', num_samples=100, save_path='data/radar_comm_dataset_g2', drawfig=False, clutter_intensity=0.1, fixed_snr=None, enable_clutter=True, enable_imperfect_csi=True, enable_rf_impairments=True, target_rcs_range=(10, 30)):
        self.config = RADAR_COMM_CONFIGS_G2[config_name]
        self.config_name = config_name
        self.mode = self.config['mode']
        self.num_samples = num_samples
        self.save_path = save_path
        self.drawfig = drawfig
        self.clutter_intensity = clutter_intensity if clutter_intensity is not None else 0.1
        self.fixed_snr = fixed_snr
        self.enable_clutter = enable_clutter
        self.enable_imperfect_csi = enable_imperfect_csi
        self.enable_rf_impairments = enable_rf_impairments
        self.target_rcs_range = target_rcs_range if target_rcs_range is not None else (10, 30)
        self.adaptive_cfar = self.config.get('adaptive_cfar', True)
        self.csi_error = self.config.get('csi_error', 0.1) if enable_imperfect_csi else 0.0
        self.clutter_params = self.config.get('clutter_params', {})
        self.fc = self.config['fc']
        self.cfar_params = self.config['cfar_params'].copy()
        self.mod_order = self.config.get('mod_order', 4)
        self.channel_model_type = self.config.get('channel_model', 'awgn')
        if self.mode == 'TRADITIONAL':
            self.radar_B = self.config['radar_B']
            self.radar_T = self.config['radar_T']
            self.radar_fs = self.config['radar_fs']
            self.radar_slope = self.radar_B / self.radar_T
            self.radar_Ns = int(self.radar_fs * self.radar_T)
            self.radar_Nc = 64
            self.comm_B = self.config['comm_B']
            self.comm_fs = self.config['comm_fs']
            self.comm_fft = self.config['comm_fft_size']
            self.comm_cp = self.config['comm_cp_len']
        elif self.mode == 'OTFS':
            self.B = self.config['B']
            self.fs = self.config['fs']
            self.Nd = self.config['N_doppler']
            self.Nt = self.config['N_delay']
        self.c = 300000000.0
        self.lambda_c = self.c / self.fc
        self.data_samples = []
        os.makedirs(self.save_path, exist_ok=True)
        if self.drawfig:
            os.makedirs(os.path.join(self.save_path, 'vis'), exist_ok=True)
        self.generate_dataset()

    def _estimate_snr(self, rdm_db):
        """Estimate SNR from RDM statistics using noise floor estimation"""
        noise_floor = np.percentile(rdm_db, 25)
        peak_power = np.max(rdm_db)
        return peak_power - noise_floor

    def _compute_adaptive_threshold(self, rdm_db, base_threshold):
        """Compute adaptive threshold based on estimated SNR"""
        estimated_snr = self._estimate_snr(rdm_db)
        if estimated_snr > 35:
            return base_threshold - 5
        elif estimated_snr > 30:
            return base_threshold - 3
        elif estimated_snr > 20:
            return base_threshold
        elif estimated_snr > 15:
            return base_threshold + 3
        else:
            return base_threshold + 5

    def _generate_ground_clutter(self, rdm_shape, r_axis):
        """
        Generate K-distributed ground clutter.
        K-distribution models spiky clutter from rough surfaces.
        Clutter power decreases with range (R^-n law).
        """
        if not self.enable_clutter or not self.clutter_params.get('ground_clutter', False):
            return np.zeros(rdm_shape)
        intensity = self.clutter_params.get('ground_intensity', 0.05)
        k_shape = self.clutter_params.get('k_shape', 2.0)
        range_exp = self.clutter_params.get('range_exponent', 2.5)
        range_profile = (r_axis + 1) ** (-range_exp)
        range_profile = range_profile / np.max(range_profile)
        gamma_samples = np.random.gamma(k_shape, 1.0 / k_shape, rdm_shape)
        rayleigh_samples = np.abs(np.random.randn(*rdm_shape) + 1j * np.random.randn(*rdm_shape))
        clutter = gamma_samples * rayleigh_samples
        if len(r_axis) == rdm_shape[1]:
            clutter = clutter * range_profile[None, :]
        return intensity * clutter

    def _generate_weather_clutter(self, rdm_shape, v_axis):
        """
        Generate weather clutter with Doppler spread.
        Weather returns typically concentrated at low velocities.
        """
        if not self.enable_clutter or not self.clutter_params.get('weather_clutter', False):
            return np.zeros(rdm_shape)
        intensity = self.clutter_params.get('weather_intensity', 0.03)
        doppler_spread = self.clutter_params.get('doppler_spread', 3.0)
        doppler_profile = np.exp(-v_axis ** 2 / (2 * doppler_spread ** 2))
        doppler_profile = doppler_profile / np.max(doppler_profile)
        weather = np.abs(np.random.randn(*rdm_shape) + 1j * np.random.randn(*rdm_shape))
        if len(v_axis) == rdm_shape[0]:
            weather = weather * doppler_profile[:, None]
        return intensity * weather

    def _add_clutter_to_rdm(self, rdm_db, r_axis, v_axis):
        """Add combined clutter to RDM (in dB domain)
        
        G3 clutter model fix: Clutter now scales with clutter_intensity parameter
        and is relative to RDM peak (not noise floor) for realistic CNR effect.
        """
        rdm_linear = 10 ** (rdm_db / 20)
        signal_peak = np.percentile(rdm_linear, 99)
        noise_floor = np.median(rdm_linear)
        clutter_scale = self.clutter_intensity * signal_peak
        ground = self._generate_ground_clutter(rdm_linear.shape, r_axis) * clutter_scale
        weather = self._generate_weather_clutter(rdm_linear.shape, v_axis) * clutter_scale * 0.5
        rdm_with_clutter = rdm_linear + ground + weather
        return 20 * np.log10(rdm_with_clutter + 1e-09)

    def _apply_imperfect_csi(self, H_true):
        """
        Add estimation error to channel for realistic CSI.
        Error is proportional to channel magnitude.
        """
        if self.csi_error <= 0:
            return H_true
        error = self.csi_error * (np.random.randn(*H_true.shape) + 1j * np.random.randn(*H_true.shape)) * np.sqrt(0.5)
        H_estimated = H_true + error * np.abs(H_true)
        return H_estimated

    def _generate_dmrs(self, num_subcarriers, dmrs_spacing=4, dmrs_type=1):
        """
        Generate 5G NR-like DMRS (Demodulation Reference Signals).
        
        Args:
            num_subcarriers: Total number of subcarriers
            dmrs_spacing: Spacing between DMRS subcarriers (Type 1: 4, Type 2: 6)
            dmrs_type: DMRS configuration type (1 or 2)
        
        Returns:
            dmrs_positions: Subcarrier indices for DMRS
            dmrs_symbols: DMRS symbols (constant amplitude zero autocorrelation)
        """
        if dmrs_type == 1:
            dmrs_positions = np.arange(0, num_subcarriers, dmrs_spacing)
        else:
            dmrs_positions = np.sort(np.concatenate([np.arange(0, num_subcarriers, 6), np.arange(1, num_subcarriers, 6)]))
        N_zc = len(dmrs_positions)
        u = 25
        n = np.arange(N_zc)
        dmrs_symbols = np.exp(-1j * np.pi * u * n * (n + 1) / N_zc)
        return (dmrs_positions, dmrs_symbols)

    def _mmse_channel_estimation(self, rx_dmrs, tx_dmrs, snr_db, dmrs_positions, num_subcarriers):
        """
        MMSE (Minimum Mean Square Error) channel estimation.
        
        Uses DMRS pilots for estimation, then interpolates to all subcarriers.
        Significantly better than LS estimation at low SNR.
        
        Args:
            rx_dmrs: Received DMRS symbols
            tx_dmrs: Transmitted DMRS symbols
            snr_db: Signal-to-noise ratio in dB
            dmrs_positions: Subcarrier indices of DMRS
            num_subcarriers: Total number of subcarriers
        
        Returns:
            H_est: Estimated channel response for all subcarriers
        """
        H_ls = rx_dmrs / (tx_dmrs + 1e-10)
        snr_linear = 10 ** (snr_db / 10)
        mmse_weight = snr_linear / (snr_linear + 1)
        H_mmse_dmrs = H_ls * mmse_weight
        H_est = np.zeros(num_subcarriers, dtype=np.complex128)
        H_est[dmrs_positions] = H_mmse_dmrs
        for i in range(len(dmrs_positions) - 1):
            start_pos = dmrs_positions[i]
            end_pos = dmrs_positions[i + 1]
            start_val = H_mmse_dmrs[i]
            end_val = H_mmse_dmrs[i + 1]
            for k in range(start_pos + 1, end_pos):
                alpha = (k - start_pos) / (end_pos - start_pos)
                H_est[k] = (1 - alpha) * start_val + alpha * end_val
        if dmrs_positions[0] > 0:
            H_est[:dmrs_positions[0]] = H_mmse_dmrs[0]
        if dmrs_positions[-1] < num_subcarriers - 1:
            H_est[dmrs_positions[-1] + 1:] = H_mmse_dmrs[-1]
        return H_est

    def _simulate_traditional_with_dmrs(self, targets, snr_db):
        """
        Simulate TRADITIONAL mode with 5G-like DMRS channel estimation.
        This is a G3-enhanced version of _simulate_traditional.
        """
        return self._simulate_traditional(targets, snr_db, use_dmrs=True)

    def _fec_encode(self, bits, code_rate=1 / 3):
        """
        Simple repetition code encoder.
        
        Args:
            bits: Input bit array (0s and 1s)
            code_rate: 1/n where n is repetition factor (1/3 = repeat 3x)
        
        Returns:
            encoded_bits: Encoded bits (length = len(bits) / code_rate)
        """
        n = int(1 / code_rate)
        encoded = np.repeat(bits, n)
        return encoded

    def _fec_decode_hard(self, rx_bits, code_rate=1 / 3):
        """
        Hard-decision repetition code decoder (majority vote).
        
        Args:
            rx_bits: Received bits (0s and 1s)
            code_rate: 1/n where n is repetition factor
        
        Returns:
            decoded_bits: Decoded bits (length = len(rx_bits) * code_rate)
        """
        n = int(1 / code_rate)
        num_info_bits = len(rx_bits) // n
        decoded = np.zeros(num_info_bits, dtype=int)
        for i in range(num_info_bits):
            block = rx_bits[i * n:(i + 1) * n]
            decoded[i] = 1 if np.sum(block) > n / 2 else 0
        return decoded

    def _fec_decode_soft(self, llrs, code_rate=1 / 3):
        """
        Soft-decision repetition code decoder (LLR combining).
        
        Args:
            llrs: Log-likelihood ratios (positive = more likely 0)
            code_rate: 1/n where n is repetition factor
        
        Returns:
            decoded_bits: Decoded bits based on combined LLRs
        """
        n = int(1 / code_rate)
        num_info_bits = len(llrs) // n
        decoded = np.zeros(num_info_bits, dtype=int)
        for i in range(num_info_bits):
            block_llr = llrs[i * n:(i + 1) * n]
            combined_llr = np.sum(block_llr)
            decoded[i] = 0 if combined_llr > 0 else 1
        return decoded

    def _compute_llr(self, rx_symbols, H_est, noise_var, constellation):
        """
        Compute Log-Likelihood Ratios for BPSK/QPSK.
        Simplified: LLR = 2 * Re(y * conj(h)) / noise_var
        """
        y_eq = rx_symbols / (H_est + 1e-10)
        llr_real = 2 * np.real(y_eq) * np.sqrt(2) / noise_var
        llr_imag = 2 * np.imag(y_eq) * np.sqrt(2) / noise_var
        llrs = np.empty(2 * len(y_eq))
        llrs[0::2] = llr_real
        llrs[1::2] = llr_imag
        return llrs

    def _simulate_traditional_with_fec(self, targets, snr_db, code_rate=1 / 3):
        """
        Simulate TRADITIONAL mode with FEC coding.
        Uses repetition code for ~2-3 dB coding gain.
        """
        pass

    def _generate_qam_symbols(self, num_symbols, mod_order=4):
        """Generate random M-QAM symbols"""
        if mod_order == 4:
            pts = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
        elif mod_order == 8:
            pts = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j, 3 + 0j, -3 + 0j, 0 + 3j, 0 - 3j]) / np.sqrt(6)
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
            raise ValueError(f'Modulation order {mod_order} not supported yet.')
        ints = np.random.randint(0, mod_order, num_symbols)
        symbols = pts[ints]
        return (symbols, ints, pts)

    def _demodulate_qam(self, rx_symbols, mod_order=4, const_pts=None):
        """Minimum Distance Demodulation"""
        if const_pts is None:
            _, _, const_pts = self._generate_qam_symbols(0, mod_order)
        dists = np.abs(rx_symbols[:, None] - const_pts[None, :])
        demod_ints = np.argmin(dists, axis=1)
        return demod_ints
    TDL_MODELS = {'TDL-A': {'delays_ns': np.array([0, 30, 70, 90, 110, 190, 410]) * 1e-09, 'powers_dB': np.array([0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8])}, 'TDL-B': {'delays_ns': np.array([0, 10, 20, 30, 60, 90, 130]) * 1e-09, 'powers_dB': np.array([0, -2.2, -4.0, -3.2, -9.8, -13.0, -15.0])}, 'TDL-D': {'delays_ns': np.array([0, 30, 100, 200, 230, 500]) * 1e-09, 'powers_dB': np.array([-0.2, -13.5, -18.8, -21.0, -22.8, -17.9])}, 'TDL-E': {'delays_ns': np.array([0, 50, 120, 200, 230, 500]) * 1e-09, 'powers_dB': np.array([-0.03, -22.0, -17.0, -20.0, -21.0, -22.0])}}

    def _apply_fading_channel(self, signal, fs, snr_db, tdl_model=None):
        """
        Apply Multipath Fading + AWGN.
        
        Args:
            signal: Input signal
            fs: Sampling frequency
            snr_db: SNR in dB
            tdl_model: Optional TDL model name ('TDL-A', 'TDL-B', 'TDL-D', 'TDL-E')
                      If None, uses random TDL (default multipath behavior)
        """
        if self.channel_model_type != 'multipath':
            sig_pow = np.mean(np.abs(signal) ** 2)
            noise_pow = sig_pow / 10 ** (snr_db / 10)
            noise = (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))) * np.sqrt(noise_pow / 2)
            return (signal + noise, np.array([1.0]))
        if tdl_model and tdl_model in self.TDL_MODELS:
            model = self.TDL_MODELS[tdl_model]
            delays_sec = model['delays_ns']
            powers_dB = model['powers_dB']
            num_taps = len(delays_sec)
        else:
            num_taps = np.random.randint(2, 6)
            max_delay = 2e-07
            delays_sec = np.sort(np.random.uniform(0, max_delay, num_taps))
            delays_sec[0] = 0
            powers_dB = -np.random.uniform(2, 5, num_taps) * np.arange(num_taps)
        delays_samp = np.round(delays_sec * fs).astype(int)
        delays_samp = np.unique(delays_samp)
        num_taps = len(delays_samp)
        powers_dB = powers_dB[:num_taps]
        powers_lin = 10 ** (powers_dB / 10)
        powers_lin /= np.sum(powers_lin)
        taps = np.sqrt(powers_lin) * (np.random.randn(num_taps) + 1j * np.random.randn(num_taps)) / np.sqrt(2)
        max_samp = delays_samp[-1] if len(delays_samp) > 0 else 0
        h_imp = np.zeros(max_samp + 1, dtype=np.complex128)
        h_imp[delays_samp] = taps
        rx_signal_clean = np.convolve(signal, h_imp, mode='full')
        sig_pow = np.mean(np.abs(rx_signal_clean) ** 2)
        noise_pow = sig_pow / 10 ** (snr_db / 10)
        noise = (np.random.randn(len(rx_signal_clean)) + 1j * np.random.randn(len(rx_signal_clean))) * np.sqrt(noise_pow / 2)
        return (rx_signal_clean + noise, h_imp)

    def _apply_tdl_channel(self, signal, fs, snr_db, model_name='TDL-A'):
        """
        Apply specific 3GPP TDL channel model.
        Wrapper for _apply_fading_channel with TDL support.
        """
        return self._apply_fading_channel(signal, fs, snr_db, tdl_model=model_name)

    def _apply_rf_impairments(self, signal, snr_db, fs):
        """
        Apply realistic RF impairments that are difficult for traditional methods.
        
        These impairments break the assumptions of traditional MMSE/ZF equalization,
        giving DL models an advantage in realistic scenarios.
        
        Severity reduced to allow 16-QAM convergence while still challenging.
        """
        severity = np.clip(1.0 - (snr_db - 5) / 40, 0.15, 0.5)
        phase_noise_std = 0.01 * severity
        phase_noise = np.cumsum(np.random.randn(len(signal))) * phase_noise_std
        signal = signal * np.exp(1j * phase_noise)
        g_imb = 0.015 * severity
        phi_imb = 0.02 * severity
        I, Q = (signal.real, signal.imag)
        I_out = (1 + g_imb) * I
        Q_out = (1 - g_imb) * (Q * np.cos(phi_imb) + I * np.sin(phi_imb))
        signal = I_out + 1j * Q_out
        cfo_hz = np.random.uniform(-10, 10) * severity
        t = np.arange(len(signal)) / fs
        signal = signal * np.exp(1j * 2 * np.pi * cfo_hz * t)
        if np.random.random() < 0.1 * severity:
            amp = np.abs(signal)
            phase = np.angle(signal)
            p_sat = 1.5
            p = 4
            amp_out = amp / (1 + (amp / p_sat) ** (2 * p)) ** (1 / (2 * p))
            am_pm = 0.03 * severity * (amp / p_sat) ** 2
            signal = amp_out * np.exp(1j * (phase + am_pm))
        return signal

    def _simulate_traditional(self, targets, snr_db, use_dmrs=False):
        """
        Simulate TRADITIONAL mode (OFDM + FMCW).
        
        Args:
            targets: List of target dictionaries
            snr_db: Signal-to-noise ratio
            use_dmrs: If True, use 5G NR DMRS+MMSE estimation (G3)
                     If False, use LS estimation with pilot (G2)
        """
        Nfft = self.comm_fft
        Ncp = self.comm_cp
        num_data_syms = 14
        _, _, const_pts = self._generate_qam_symbols(0, self.mod_order)
        if use_dmrs:
            dmrs_positions, dmrs_symbols = self._generate_dmrs(Nfft, dmrs_spacing=4)
            pilot_syms = np.zeros(Nfft, dtype=np.complex128)
            pilot_syms[dmrs_positions] = dmrs_symbols
            non_dmrs = np.setdiff1d(np.arange(Nfft), dmrs_positions)
            pilot_syms[non_dmrs], _, _ = self._generate_qam_symbols(len(non_dmrs), mod_order=4)
        else:
            pilot_syms, _, _ = self._generate_qam_symbols(Nfft, mod_order=4)
        total_data_qam = num_data_syms * Nfft
        data_syms, data_ints, _ = self._generate_qam_symbols(total_data_qam, self.mod_order)
        data_grid = data_syms.reshape(num_data_syms, Nfft)
        full_grid = np.vstack([pilot_syms[None, :], data_grid])
        ifft_out = np.fft.ifft(full_grid, axis=1)
        cp = ifft_out[:, -Ncp:]
        ofdm_time = np.hstack([cp, ifft_out]).flatten()
        rx_time_full, h_true = self._apply_fading_channel(ofdm_time, self.comm_fs, snr_db)
        if self.enable_rf_impairments:
            rx_time_full = self._apply_rf_impairments(rx_time_full, snr_db, self.comm_fs)
        first_tap_idx = np.argmax(np.abs(h_true) > 0)
        rx_time = rx_time_full[first_tap_idx:first_tap_idx + len(ofdm_time)]
        rx_reshaped = rx_time.reshape(num_data_syms + 1, Nfft + Ncp)
        rx_no_cp = rx_reshaped[:, Ncp:]
        rx_grid = np.fft.fft(rx_no_cp, axis=1)
        Y_pilot = rx_grid[0, :]
        if use_dmrs:
            rx_dmrs = Y_pilot[dmrs_positions]
            tx_dmrs = dmrs_symbols
            H_est = self._mmse_channel_estimation(rx_dmrs, tx_dmrs, snr_db, dmrs_positions, Nfft)
        else:
            X_pilot = pilot_syms
            H_est = Y_pilot / (X_pilot + 1e-10)
        if not use_dmrs:
            H_est = self._apply_imperfect_csi(H_est)
        Y_data = rx_grid[1:, :]
        X_hat_grid = Y_data / (H_est[None, :] + 1e-10)
        rx_const = X_hat_grid.flatten()
        demod_ints = self._demodulate_qam(rx_const, self.mod_order, const_pts)
        errors = np.sum(data_ints != demod_ints)
        ber = errors / len(data_ints)
        Nc = self.radar_Nc
        Ns = self.radar_Ns
        fs = self.radar_fs
        slope = self.radar_slope
        t_fast = np.arange(Ns) / fs
        t_slow = np.arange(Nc) * self.radar_T
        beat_signal = np.zeros((Nc, Ns), dtype=np.complex64)
        for t in targets:
            fb = slope * 2 * t['range'] / self.c
            fd = 2 * t['velocity'] / self.lambda_c
            phase = 2 * np.pi * (fb * t_fast[None, :] + fd * t_slow[:, None])
            amp = np.sqrt(10 ** (t['rcs'] / 10))
            beat_signal += amp * np.exp(1j * phase)
        sig_pow_rad = np.mean(np.abs(beat_signal) ** 2)
        if sig_pow_rad > 0:
            noise_pow_rad = sig_pow_rad / 10 ** (snr_db / 10)
            noise_rad = (np.random.randn(Nc, Ns) + 1j * np.random.randn(Nc, Ns)) * np.sqrt(noise_pow_rad / 2)
            beat_signal += noise_rad
        if self.enable_clutter and self.clutter_intensity > 0.2:
            clutter_power = sig_pow_rad * self.clutter_intensity * 0.01
            clutter_time = np.zeros((Nc, Ns), dtype=np.complex64)
            num_clutter_gates = max(3, Ns // 100)
            for range_idx in range(0, Ns, Ns // num_clutter_gates):
                clutter_amp = np.sqrt(clutter_power) * np.random.uniform(0.5, 1.5)
                low_doppler = np.random.uniform(-2, 2)
                phase = 2 * np.pi * low_doppler * t_slow[:, None]
                gate_width = min(5, Ns // 50)
                clutter_time[:, max(0, range_idx - gate_width):min(Ns, range_idx + gate_width)] += clutter_amp * np.exp(1j * phase)[:, :gate_width * 2]
            beat_signal += clutter_time
        win_range = np.hanning(Ns)[None, :]
        win_doppler = np.hanning(Nc)[:, None]
        beat_signal_win = beat_signal * win_range * win_doppler
        r_fft = np.fft.fft(beat_signal_win, axis=1)
        rd_map = np.fft.fftshift(np.fft.fft(r_fft, axis=0), axes=0)
        rd_map_db = 20 * np.log10(np.abs(rd_map) + 1e-09)
        r_res = self.c * fs / (2 * slope * Ns)
        v_res = self.lambda_c / (2 * Nc * self.radar_T)
        r_axis = np.arange(Ns) * r_res
        v_axis = np.arange(-Nc // 2, Nc // 2) * v_res
        rd_map_db = self._add_clutter_to_rdm(rd_map_db, r_axis, v_axis)
        return {'rd_map': rd_map_db, 'r_axis': r_axis, 'v_axis': v_axis, 'comm_info': {'ber': ber, 'tx_symbols': data_grid.flatten(), 'rx_symbols': rx_const, 'num_data_syms': num_data_syms, 'fft_size': Nfft, 'tx_ints': data_ints, 'mod_order': self.mod_order, 'channel_est': H_est, 'rx_grid': Y_data, 'pilot_rx': Y_pilot}, 'ofdm_map': 20 * np.log10(np.abs(X_hat_grid) + 1e-09)}

    def _otfs_modulate(self, dd_grid):
        """
        Modulates a Delay-Doppler grid to a time-domain signal.
        Uses v8 reference logic with [Delay, Doppler] = [Nt, Nd] ordering.
        Tx Chain: ISFFT (DD->TF) -> Heisenberg (TF->Time)
        
        Args:
            dd_grid: [Nt, Nd] array (Delay x Doppler)
        Returns:
            tx_signal: 1D time-domain signal
        """
        tf_grid = np.fft.fft(dd_grid, axis=0)
        tf_grid = np.fft.ifft(tf_grid, axis=1)
        time_domain_grid = np.fft.ifft(tf_grid, axis=0)
        tx_signal = time_domain_grid.flatten(order='F')
        return tx_signal

    def _otfs_demodulate(self, rx_signal, Nt, Nd):
        """
        Demodulates a time-domain signal back to a Delay-Doppler grid.
        Uses v8 reference logic with [Delay, Doppler] = [Nt, Nd] ordering.
        Rx Chain: Wigner (Time->TF) -> SFFT (TF->DD)
        
        Args:
            rx_signal: 1D time-domain signal of length Nt * Nd
            Nt: Number of delay bins
            Nd: Number of Doppler bins
        Returns:
            dd_grid: [Nt, Nd] array (Delay x Doppler)
        """
        time_domain_grid = rx_signal.reshape((Nt, Nd), order='F')
        tf_grid = np.fft.fft(time_domain_grid, axis=0)
        dd_grid = np.fft.fft(tf_grid, axis=1)
        dd_grid = np.fft.ifft(dd_grid, axis=0)
        return dd_grid

    def _simulate_otfs(self, targets, snr_db):
        """
        Simulates Integrated Sensing and Communication (ISAC) using OTFS.
        Aligned exactly with AIradar_datasetv8.py for correct radar processing.
        
        Naming convention (matching v8):
        - Ns = delay bins (self.Nt in config)
        - Nc = Doppler bins (self.Nd in config)
        """
        Ns = self.Nt
        Nc = self.Nd
        num_symbols = Ns * Nc
        bits = np.random.randint(0, 4, num_symbols)
        mod_map = {0: (1 + 1j) / np.sqrt(2), 1: (1 - 1j) / np.sqrt(2), 2: (-1 + 1j) / np.sqrt(2), 3: (-1 - 1j) / np.sqrt(2)}
        tx_symbols = np.array([mod_map[b] for b in bits])
        tx_dd_grid = tx_symbols.reshape((Ns, Nc))
        tx_ints = bits
        const_pts = np.array([mod_map[i] for i in range(4)])
        tf_grid = np.fft.fft(tx_dd_grid, axis=0)
        tf_grid = np.fft.ifft(tf_grid, axis=1)
        time_domain_grid = np.fft.ifft(tf_grid, axis=0)
        tx_signal = time_domain_grid.flatten(order='F')
        n_samples = tx_signal.size
        rx_radar = np.zeros(n_samples, dtype=complex)
        time_vector = np.arange(n_samples) / self.fs
        for t in targets:
            range_m = t['range']
            velocity_mps = t['velocity']
            rcs = t['rcs']
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
            snr_linear = 10 ** (snr_db / 10)
            noise_pow = sig_pow / snr_linear
            noise = (np.random.randn(n_samples) + 1j * np.random.randn(n_samples)) * np.sqrt(noise_pow / 2)
            rx_radar += noise
        rx_comm_full, h_true = self._apply_fading_channel(tx_signal, self.fs, snr_db)
        first_tap_idx = np.argmax(np.abs(h_true) > 0)
        if first_tap_idx + len(tx_signal) <= len(rx_comm_full):
            rx_comm = rx_comm_full[first_tap_idx:first_tap_idx + len(tx_signal)]
        else:
            rx_comm = rx_comm_full[first_tap_idx:]
            rx_comm = np.pad(rx_comm, (0, len(tx_signal) - len(rx_comm)))
        rx_time_grid = rx_comm.reshape((Ns, Nc), order='F')
        rx_tf_grid = np.fft.fft(rx_time_grid, axis=0)
        H_freq = np.fft.fft(h_true, n=Ns)
        H_freq = self._apply_imperfect_csi(H_freq)
        noise_var = 1.0 / 10 ** (snr_db / 10)
        H_eq = np.conj(H_freq) / (np.abs(H_freq) ** 2 + noise_var + 1e-10)
        rx_tf_eq = rx_tf_grid * H_eq[:, None]
        rx_dd_comm = np.fft.fft(rx_tf_eq, axis=1)
        rx_dd_comm = np.fft.ifft(rx_dd_comm, axis=0)
        rx_const = rx_dd_comm.flatten()
        demod_ints = self._demodulate_qam(rx_const, 4, const_pts)
        errors = np.sum(tx_ints != demod_ints)
        ber = errors / len(tx_ints)
        rx_time_radar = rx_radar.reshape((Ns, Nc), order='F')
        rx_tf_radar = np.fft.fft(rx_time_radar, axis=0)
        rx_dd_radar = np.fft.fft(rx_tf_radar, axis=1)
        rx_dd_radar = np.fft.ifft(rx_dd_radar, axis=0)
        rx_dd_fft = np.fft.fft2(rx_dd_radar)
        tx_dd_fft = np.fft.fft2(tx_dd_grid)
        epsilon = 1e-06
        ddm_fft = rx_dd_fft / (tx_dd_fft + epsilon)
        ddm_complex = np.fft.ifft2(ddm_fft)
        ddm_transposed = ddm_complex.T
        ddm_shifted = np.fft.fftshift(ddm_transposed, axes=0)
        ddm_mag = np.abs(ddm_shifted)
        rd_map_db_full = 20 * np.log10(ddm_mag + 1e-06)
        r_res = self.c / (2 * self.fs)
        num_range_bins = int(self.config.get('R_max', 100.0) / r_res)
        num_range_bins = max(1, min(num_range_bins, rd_map_db_full.shape[1]))
        rd_map_db = rd_map_db_full[:, :num_range_bins]
        r_axis = np.arange(num_range_bins) * r_res
        T_actual = Ns / self.fs
        v_axis = np.fft.fftshift(np.fft.fftfreq(Nc, d=T_actual)) * self.lambda_c / 2
        rd_map_db = self._add_clutter_to_rdm(rd_map_db, r_axis, v_axis)
        return {'rd_map': rd_map_db, 'r_axis': r_axis, 'v_axis': v_axis, 'channel_model': self.channel_model_type, 'mod_order': self.mod_order, 'comm_info': {'ber': ber, 'tx_symbols': tx_symbols, 'rx_symbols': rx_const, 'tx_ints': tx_ints, 'mod_order': 4}, 'ofdm_map': None}

    def _run_cfar(self, rdm_db, r_axis, v_axis):
        """
        Constant False Alarm Rate (CFAR) Detector (CA-CFAR).
        G2 Enhancement: Adaptive threshold based on estimated SNR.
        """
        params = self.cfar_params
        nt = params['num_train']
        ng = params['num_guard']
        base_thresh = params['threshold_offset']
        if self.adaptive_cfar:
            thresh = self._compute_adaptive_threshold(rdm_db, base_thresh)
        else:
            thresh = base_thresh
        norm_rdm = rdm_db.copy()
        gp = params.get('global_percentile', None)
        if gp is not None:
            pval = np.percentile(norm_rdm, gp)
            norm_rdm = np.minimum(norm_rdm, pval)
        kernel_size = 1 + 2 * (nt + ng)
        kernel = np.ones((kernel_size, kernel_size))
        guard_region = 1 + 2 * ng
        start_g = nt
        end_g = nt + guard_region
        kernel[start_g:end_g, start_g:end_g] = 0
        kernel /= np.sum(kernel)
        noise_est = convolve2d(norm_rdm, kernel, mode='same', boundary='symm')
        detections = norm_rdm > noise_est + thresh
        if params['nms_kernel_size'] > 1:
            local_max = maximum_filter(norm_rdm, size=params['nms_kernel_size'])
            detections = detections & (norm_rdm == local_max)
        idxs = np.argwhere(detections)
        results = []
        min_r = params.get('min_range_m', 0.0)
        min_v = params.get('min_speed_mps', 0.0)
        notch_k = params.get('notch_doppler_bins', 0)
        center = len(v_axis) // 2
        candidates = []
        for idx in idxs:
            d_idx, r_idx = idx
            if d_idx >= len(v_axis) or r_idx >= len(r_axis):
                continue
            range_m = r_axis[r_idx]
            vel_mps = v_axis[d_idx]
            if range_m < min_r or abs(vel_mps) < min_v:
                continue
            if notch_k > 0 and abs(d_idx - center) <= notch_k:
                continue
            candidates.append({'range_m': range_m, 'velocity_mps': vel_mps, 'range_idx': r_idx, 'doppler_idx': d_idx, 'power': norm_rdm[d_idx, r_idx]})
        max_peaks = params.get('max_peaks', None)
        if max_peaks is not None:
            candidates.sort(key=lambda x: x['power'], reverse=True)
            candidates = candidates[:max_peaks]
        pruned = []
        taken = set()
        neigh = params.get('nms_kernel_size', 5)
        for det in candidates:
            key = (det['doppler_idx'] // neigh, det['range_idx'] // neigh)
            if key in taken:
                continue
            pruned.append(det)
            taken.add(key)
        return pruned

    def _evaluate_metrics(self, targets, detections, match_dist_thresh=3.0):
        tp = 0
        range_errors = []
        velocity_errors = []
        unmatched_targets = targets.copy()
        unmatched_detections = detections.copy()
        matched_pairs = []
        for target in targets:
            best_dist = float('inf')
            best_det_idx = -1
            for i, det in enumerate(unmatched_detections):
                d_r = target['range'] - det['range_m']
                d_v = target['velocity'] - det['velocity_mps']
                dist = np.sqrt(d_r ** 2 + d_v ** 2)
                if dist < match_dist_thresh and dist < best_dist:
                    best_dist = dist
                    best_det_idx = i
            if best_det_idx != -1:
                tp += 1
                det = unmatched_detections[best_det_idx]
                range_errors.append(abs(target['range'] - det['range_m']))
                velocity_errors.append(abs(target['velocity'] - det['velocity_mps']))
                matched_pairs.append((target, det))
                unmatched_detections.pop(best_det_idx)
                unmatched_targets.remove(target)
        fp = len(unmatched_detections)
        fn = len(targets) - tp
        metrics = {'tp': tp, 'fp': fp, 'fn': fn, 'mean_range_error': np.mean(range_errors) if range_errors else 0.0, 'mean_velocity_error': np.mean(velocity_errors) if velocity_errors else 0.0, 'total_targets': len(targets)}
        return (metrics, matched_pairs, unmatched_targets, unmatched_detections)

    def generate_dataset(self):
        clutter_str = 'ON' if self.enable_clutter else 'OFF'
        csi_str = f'{self.csi_error * 100:.0f}%' if self.enable_imperfect_csi else 'Perfect'
        print(f'Generating {self.num_samples} samples in {self.mode} mode...')
        print(f'Config: {self.mod_order}-QAM | Channel: {self.channel_model_type} | Clutter: {clutter_str} | CSI Error: {csi_str}')
        for i in tqdm(range(self.num_samples)):
            num_t = np.random.randint(1, 4)
            targets = []
            for _ in range(num_t):
                targets.append({'range': np.random.uniform(5, self.config['R_max'] * 0.8), 'velocity': np.random.uniform(-15, 15), 'rcs': np.random.uniform(self.target_rcs_range[0], self.target_rcs_range[1])})
            if self.fixed_snr is not None:
                snr = self.fixed_snr
            else:
                snr = np.random.uniform(5, 35)
            if self.mode == 'TRADITIONAL':
                out = self._simulate_traditional(targets, snr)
            else:
                out = self._simulate_otfs(targets, snr)
            dets = self._run_cfar(out['rd_map'], out['r_axis'], out['v_axis'])
            self.range_axis = out['r_axis']
            self.velocity_axis = out['v_axis']
            sample = {'mode': self.mode, 'mod_order': self.mod_order, 'channel_model': self.channel_model_type, 'range_doppler_map': torch.tensor(out['rd_map'], dtype=torch.float32), 'range_axis': out['r_axis'], 'velocity_axis': out['v_axis'], 'target_info': {'targets': targets, 'snr_db': snr}, 'comm_info': out['comm_info'], 'cfar_detections': dets, 'ofdm_map': out.get('ofdm_map', None)}
            errs = []
            for t in targets:
                dists = [abs(t['range'] - d['range_m']) for d in dets]
                if dists:
                    errs.append(min(dists))
            mean_err = np.mean(errs) if errs else 0.0
            sample['metrics'] = {'mean_range_error': mean_err}
            self.data_samples.append(sample)
            if self.drawfig:
                plot_combined_sample(sample, os.path.join(self.save_path, f'vis/sample_{i}_{self.mode}.png'))
                rdm = sample['range_doppler_map'].numpy()
                rdm_norm = rdm - np.max(rdm)
                metrics, matched_pairs, unmatched_targets, unmatched_detections = self._evaluate_metrics(targets, dets)
                _plot_2d_rdm(self, rdm_norm, i, metrics, matched_pairs, unmatched_targets, unmatched_detections, os.path.join(self.save_path, f'vis/rdm_sample_{i}.png'))
                _plot_3d_rdm(self, rdm_norm, i, targets, dets, os.path.join(self.save_path, f'vis/rdm_3d_sample_{i}.png'))
            dump_item = {'range_doppler_map': sample['range_doppler_map'].numpy(), 'cfar_detections': sample['cfar_detections'], 'target_info': sample['target_info'], 'ofdm_map': sample.get('ofdm_map', None), 'comm_info': sample.get('comm_info', None)}
            dump_path = os.path.join(self.save_path, 'joint_dump.npy')
            existing = []
            if os.path.exists(dump_path):
                try:
                    existing = list(np.load(dump_path, allow_pickle=True))
                except Exception:
                    existing = []
            existing.append(dump_item)
            np.save(dump_path, np.array(existing, dtype=object))

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        return self.data_samples[idx]

CONFIG_ID_MAP = {name: i for i, name in enumerate(RADAR_COMM_CONFIGS_G2.keys())}

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for config adaptation."""

    def __init__(self, feature_dim, cond_dim):
        super().__init__()
        self.gamma_fc = nn.Linear(cond_dim, feature_dim)
        self.beta_fc = nn.Linear(cond_dim, feature_dim)
        nn.init.ones_(self.gamma_fc.bias.data)
        nn.init.zeros_(self.beta_fc.bias.data)

    def forward(self, x, cond):
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta

class ConfigEncoder(nn.Module):
    """Encode radar/comm configuration to embedding vector."""
    CHANNEL_MAP = {'awgn': 0, 'multipath': 1, 'rayleigh': 2, 'none': 3}
    MODE_MAP = {'TRADITIONAL': 0, 'OTFS': 1}

    def __init__(self, embed_dim=64):
        super().__init__()
        self.fc_legacy = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, embed_dim), nn.ReLU())
        self.fc_generalized = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, embed_dim), nn.ReLU())

    def forward(self, config_tensor):
        if config_tensor.dim() == 1:
            config_tensor = config_tensor.unsqueeze(0)
        if config_tensor.shape[-1] == 5:
            return self.fc_generalized(config_tensor)
        return self.fc_legacy(config_tensor)

    @staticmethod
    def encode_config(config: dict) -> torch.Tensor:
        """Legacy: Convert config dict to normalized tensor (8-dim)."""
        channel_id = ConfigEncoder.CHANNEL_MAP.get(config.get('channel_model', 'multipath'), 1)
        mode_id = ConfigEncoder.MODE_MAP.get(config.get('mode', 'TRADITIONAL'), 0)
        return torch.tensor([config.get('fc', 10000000000.0) / 100000000000.0, config.get('radar_B', 500000000.0) / 1000000000.0, config.get('radar_Nc', 64) / 256, config.get('radar_Ns', 1000) / 4000, np.log2(config.get('mod_order', 16)) / 6, channel_id / 3, mode_id, config.get('radar_T', 5e-05) * 10000.0], dtype=torch.float32)

    @staticmethod
    def encode_config_generalized(config: dict) -> torch.Tensor:
        """
        Continuous physical parameter encoding for zero-shot generalization.
        Uses only physical parameters without discrete IDs.
        """
        return torch.tensor([config.get('fc', 10000000000.0) / 100000000000.0, config.get('radar_B', 500000000.0) / 2000000000.0, config.get('snr_db', 20) / 40, config.get('delay_spread', 1e-07) * 10000000.0, np.log2(config.get('mod_order', 4)) / 6.0], dtype=torch.float32)

class FiLMConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, cond_dim, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride, 1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.film = FiLMLayer(out_ch, cond_dim)

    def forward(self, x, cond):
        x = self.conv(x)
        x = self.norm(x)
        x = self.film(x, cond)
        return F.relu(x)

class SEBlock(nn.Module):

    def __init__(self, ch, reduction=8):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // reduction, 1)
        self.fc2 = nn.Conv2d(ch // reduction, ch, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class ComplexConvBlock(nn.Module):
    """
    Complex convolution that respects I/Q phase-amplitude coupling.
    Mathematically: (A+Bi)*(C+Di) = (AC-BD) + (AD+BC)i
    Uses GroupNorm instead of BatchNorm for stability in signal processing.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv_re = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.conv_im = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.norm_re = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm_im = nn.GroupNorm(min(8, out_ch), out_ch)

    def forward(self, x_re, x_im):
        out_re = self.conv_re(x_re) - self.conv_im(x_im)
        out_im = self.conv_re(x_im) + self.conv_im(x_re)
        out_re = F.relu(self.norm_re(out_re))
        out_im = F.relu(self.norm_im(out_im))
        return (out_re, out_im)

class GeneralizedRadarNet(nn.Module):
    """Generalized radar detection network with FiLM conditioning."""

    def __init__(self, in_ch=1, base_ch=48, cond_dim=64, target_size=(512, 512)):
        super().__init__()
        self.target_size = target_size
        self.config_encoder = ConfigEncoder(embed_dim=cond_dim)
        self.enc1 = FiLMConvBlock(in_ch, base_ch, cond_dim)
        self.enc2 = FiLMConvBlock(base_ch, base_ch * 2, cond_dim, stride=2)
        self.enc3 = FiLMConvBlock(base_ch * 2, base_ch * 4, cond_dim, stride=2)
        self.enc4 = FiLMConvBlock(base_ch * 4, base_ch * 8, cond_dim, stride=2)
        self.se = SEBlock(base_ch * 8)
        self.dec4 = FiLMConvBlock(base_ch * 8, base_ch * 4, cond_dim)
        self.dec3 = FiLMConvBlock(base_ch * 4 + base_ch * 4, base_ch * 2, cond_dim)
        self.dec2 = FiLMConvBlock(base_ch * 2 + base_ch * 2, base_ch, cond_dim)
        self.dec1 = FiLMConvBlock(base_ch + base_ch, base_ch, cond_dim)
        self.out_conv = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x, config_tensor):
        B = x.size(0)
        orig_size = x.shape[-2:]
        if x.shape[-2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        cond = self.config_encoder(config_tensor)
        e1 = self.enc1(x, cond)
        e2 = self.enc2(e1, cond)
        e3 = self.enc3(e2, cond)
        e4 = self.enc4(e3, cond)
        z = self.se(e4)
        d4 = self.dec4(z, cond)
        d4 = F.interpolate(d4, size=e3.shape[-2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d4, e3], dim=1), cond)
        d3 = F.interpolate(d3, size=e2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d3, e2], dim=1), cond)
        d2 = F.interpolate(d2, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d2, e1], dim=1), cond)
        logits = self.out_conv(d1)
        if logits.shape[-2:] != orig_size:
            logits = F.interpolate(logits, size=orig_size, mode='bilinear', align_corners=False)
        return logits

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, max(channels // reduction, 4)), nn.ReLU(), nn.Linear(max(channels // reduction, 4), channels), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w

class ResConvBlock(nn.Module):
    """Residual convolutional block with optional channel attention."""

    def __init__(self, in_ch, out_ch, use_attention=False):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.attn = ChannelAttention(out_ch) if use_attention else None

    def forward(self, x):
        out = self.conv(x)
        if self.attn is not None:
            out = self.attn(out)
        out = out + self.skip(x)
        return self.relu(out)

class RobustRadarNetG3(nn.Module):
    """
    Improved Radar Detection Network with:
    - Residual connections for better gradient flow
    - Channel attention (SE blocks) for feature selection
    - FiLM conditioning for multi-config generalization
    - Dropout for regularization against channel variations

    Input:  [B, 1, H, W] range-doppler map (normalized to [0,1])
    Config: [B, 8] radar configuration vector
    Output: [B, 1, H, W] detection heatmap (logits)
    """

    def __init__(self, base_ch=48, cond_dim=64, dropout=0.1):
        super().__init__()
        self.config_encoder = ConfigEncoder(embed_dim=cond_dim)
        self.enc1 = ResConvBlock(1, base_ch, use_attention=True)
        self.enc2 = ResConvBlock(base_ch, base_ch * 2, use_attention=True)
        self.enc3 = ResConvBlock(base_ch * 2, base_ch * 4, use_attention=True)
        self.film1 = FiLMLayer(base_ch, cond_dim)
        self.film2 = FiLMLayer(base_ch * 2, cond_dim)
        self.film3 = FiLMLayer(base_ch * 4, cond_dim)
        self.bottleneck = ResConvBlock(base_ch * 4, base_ch * 4, use_attention=True)
        self.dropout = nn.Dropout2d(dropout)
        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec3 = ResConvBlock(base_ch * 4, base_ch * 2)
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec2 = ResConvBlock(base_ch * 2, base_ch)
        self.out_conv = nn.Sequential(nn.Conv2d(base_ch, base_ch // 2, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base_ch // 2, 1, 1))

    def forward(self, x, config_tensor):
        cond = self.config_encoder(config_tensor)
        input_size = x.shape[2:]
        e1 = self.enc1(x)
        e1 = self.film1(e1, cond)
        e2 = F.max_pool2d(e1, 2)
        e2 = self.enc2(e2)
        e2 = self.film2(e2, cond)
        e3 = F.max_pool2d(e2, 2)
        e3 = self.enc3(e3)
        e3 = self.film3(e3, cond)
        b = self.bottleneck(e3)
        b = self.dropout(b)
        d3 = self.up3(b)
        if d3.shape[2:] != e2.shape[2:]:
            d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        if d2.shape[2:] != e1.shape[2:]:
            d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        out = self.out_conv(d2)
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        return out

class RadarCRNN(nn.Module):
    """
    CNN + Bidirectional GRU for radar detection.

    Architecture:
    1. 2D CNN encoder reduces spatial dimensions and extracts features
    2. FiLM conditioning from radar config
    3. Bidirectional GRU across Doppler dimension (captures cross-Doppler patterns)
    4. 2D conv decoder reconstructs detection map

    Input:  [B, 1, 64, 1000] RDM
    Output: [B, 1, 64, 1000] detection logits
    """

    def __init__(self, base_ch=64, cond_dim=64, rnn_hidden=64, rnn_layers=2, dropout=0.1):
        super().__init__()
        self.config_encoder = ConfigEncoder(embed_dim=cond_dim)
        self.encoder = nn.Sequential(nn.Conv2d(1, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True), nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Conv2d(base_ch, base_ch * 2, 3, padding=1), nn.BatchNorm2d(base_ch * 2), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.film = FiLMLayer(base_ch * 2, cond_dim)
        self.rnn_hidden = rnn_hidden
        self.gru = nn.GRU(input_size=base_ch * 2, hidden_size=rnn_hidden, num_layers=rnn_layers, batch_first=True, bidirectional=True, dropout=dropout if rnn_layers > 1 else 0)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(rnn_hidden * 2, base_ch, 2, stride=2), nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True), nn.ConvTranspose2d(base_ch, base_ch // 2, 2, stride=2), nn.BatchNorm2d(base_ch // 2), nn.ReLU(inplace=True), nn.Dropout2d(dropout), nn.Conv2d(base_ch // 2, 1, 1))

    def forward(self, x, config_tensor):
        B = x.shape[0]
        input_size = x.shape[2:]
        cond = self.config_encoder(config_tensor)
        feat = self.encoder(x)
        feat = self.film(feat, cond)
        C, D, R = (feat.shape[1], feat.shape[2], feat.shape[3])
        feat_gru = feat.permute(0, 3, 2, 1).reshape(B * R, D, C)
        gru_out, _ = self.gru(feat_gru)
        gru_out = gru_out.reshape(B, R, D, -1).permute(0, 3, 2, 1)
        out = self.decoder(gru_out)
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        return out

class RadarTransformerNet(nn.Module):
    """
    Vision Transformer for radar detection.

    Architecture:
    1. Patch embedding: (8, 50) patches -> 160 tokens from [64, 1000]
    2. Learned positional encoding + config conditioning
    3. Transformer encoder (multi-head self-attention)
    4. Patch-wise decoder + refinement conv

    Input:  [B, 1, 64, 1000] RDM
    Output: [B, 1, 64, 1000] detection logits
    """

    def __init__(self, patch_size=(8, 50), embed_dim=256, num_heads=8, depth=4, cond_dim=64, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.config_encoder = ConfigEncoder(embed_dim=cond_dim)
        patch_dim = patch_size[0] * patch_size[1]
        self.patch_embed = nn.Sequential(nn.Linear(patch_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.num_patches_d = 64 // patch_size[0]
        self.num_patches_r = 1000 // patch_size[1]
        num_patches = self.num_patches_d * self.num_patches_r
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        self.config_proj = nn.Linear(cond_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.patch_decoder = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, patch_dim))
        self.refine = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1))

    def forward(self, x, config_tensor):
        B, _, H, W = x.shape
        pH, pW = self.patch_size
        cond = self.config_encoder(config_tensor)
        nD = H // pH
        nR = W // pW
        patches = x.reshape(B, 1, nD, pH, nR, pW)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(B, nD * nR, pH * pW)
        tokens = self.patch_embed(patches) + self.pos_embed[:, :nD * nR]
        config_token = self.config_proj(cond).unsqueeze(1)
        tokens = tokens + config_token
        tokens = self.transformer(tokens)
        decoded = self.patch_decoder(tokens)
        decoded = decoded.reshape(B, nD, nR, pH, pW)
        out = decoded.permute(0, 1, 3, 2, 4).reshape(B, nD * pH, nR * pW)
        out = out.unsqueeze(1)
        out = self.refine(out)
        if out.shape[2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out

class RadarDualPathNet(nn.Module):
    """
    Dual-path processing network for radar detection.

    Architecture:
    1. Shared 2D encoder
    2. Range path: Dilated Conv1d along Range dimension
    3. Doppler path: Dilated Conv1d along Doppler dimension
    4. SE attention fusion + residual from encoder
    5. 2D decoder

    Input:  [B, 1, 64, 1000] RDM
    Output: [B, 1, 64, 1000] detection logits
    """

    def __init__(self, base_ch=64, cond_dim=64, num_dilated_blocks=3, dropout=0.1):
        super().__init__()
        self.config_encoder = ConfigEncoder(embed_dim=cond_dim)
        self.encoder = nn.Sequential(ResConvBlock(1, base_ch // 2), nn.MaxPool2d(2), ResConvBlock(base_ch // 2, base_ch))
        self.film = FiLMLayer(base_ch, cond_dim)
        range_layers = []
        for i in range(num_dilated_blocks):
            dilation = 2 ** i
            range_layers.extend([nn.Conv1d(base_ch, base_ch, 7, padding=3 * dilation, dilation=dilation), nn.BatchNorm1d(base_ch), nn.ReLU(inplace=True)])
        self.range_path = nn.Sequential(*range_layers)
        doppler_layers = []
        for i in range(num_dilated_blocks):
            dilation = 2 ** i
            doppler_layers.extend([nn.Conv1d(base_ch, base_ch, 5, padding=2 * dilation, dilation=dilation), nn.BatchNorm1d(base_ch), nn.ReLU(inplace=True)])
        self.doppler_path = nn.Sequential(*doppler_layers)
        self.fusion_attn = ChannelAttention(base_ch * 2)
        self.fusion_conv = nn.Conv2d(base_ch * 2, base_ch, 1)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(base_ch, base_ch // 2, 2, stride=2), nn.BatchNorm2d(base_ch // 2), nn.ReLU(inplace=True), nn.Dropout2d(dropout), nn.Conv2d(base_ch // 2, 1, 1))

    def forward(self, x, config_tensor):
        B = x.shape[0]
        input_size = x.shape[2:]
        cond = self.config_encoder(config_tensor)
        feat = self.encoder(x)
        feat = self.film(feat, cond)
        C, D, R = (feat.shape[1], feat.shape[2], feat.shape[3])
        range_in = feat.reshape(B * D, C, R)
        range_feat = self.range_path(range_in)
        range_feat = range_feat.reshape(B, D, C, R).permute(0, 2, 1, 3)
        doppler_in = feat.permute(0, 3, 1, 2).reshape(B * R, C, D)
        doppler_feat = self.doppler_path(doppler_in)
        doppler_feat = doppler_feat.reshape(B, R, C, D).permute(0, 2, 3, 1)
        fused = torch.cat([range_feat, doppler_feat], dim=1)
        fused = self.fusion_attn(fused)
        fused = self.fusion_conv(fused)
        fused = fused + feat
        out = self.decoder(fused)
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        return out


def create_radar_model(model_name: str, device: torch.device) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "generalized":
        return GeneralizedRadarNet(in_ch=1, base_ch=48, cond_dim=64).to(device)
    elif model_name == "robustg3":
        return RobustRadarNetG3(base_ch=48, cond_dim=64).to(device)
    elif model_name == "crnn":
        return RadarCRNN(base_ch=64, cond_dim=64).to(device)
    elif model_name == "transformer":
        return RadarTransformerNet(cond_dim=64).to(device)
    elif model_name == "dualpath":
        return RadarDualPathNet(base_ch=64, cond_dim=64).to(device)
    elif model_name == "isac":
        return ISACFoundationModel(radar_in_channels=1, base_ch=64, cond_dim=64, config_dim=8).to(device)
    else:
        raise ValueError(f"Unsupported radar model: {model_name}")


class AdaptiveCommNet(nn.Module):
    """CNN backbone with QAM-specific adapter heads.
    
    Architecture:
        Input [B, 4, H, W] → ZF Pre-Eq → [B, 6, H, W]
                          → Shared Backbone → Features [B, base_ch*2, H, W]
                          → QAM Adapter (selected by mod_order) → Logits
    
    Each adapter is a small 1x1 conv head that specializes in its QAM order's
    decision boundaries. The backbone learns general channel-aware features,
    while adapters learn QAM-specific constellation geometry.
    
    Training strategy:
        1. Train backbone + all adapters jointly
        2. Optionally freeze backbone, fine-tune specific adapter
    """
    SUPPORTED_MOD_ORDERS = [4, 16, 64]

    def __init__(self, in_ch=4, base_ch=64, cond_dim=64, max_mod_order=64):
        super().__init__()
        self.base_ch = base_ch
        self.max_mod_order = max_mod_order
        self.config_encoder = ConfigEncoder(embed_dim=cond_dim)
        actual_in_ch = 5
        self.conv1 = nn.Conv2d(actual_in_ch, base_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_ch)
        self.film1 = FiLMLayer(base_ch, cond_dim)
        self.conv2 = nn.Conv2d(base_ch, base_ch * 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_ch * 2)
        self.film2 = FiLMLayer(base_ch * 2, cond_dim)
        self.conv3 = nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_ch * 2)
        self.film3 = FiLMLayer(base_ch * 2, cond_dim)
        self.conv4 = nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(base_ch * 2)
        self.adapters = nn.ModuleDict()
        for mod in self.SUPPORTED_MOD_ORDERS:
            self.adapters[str(mod)] = self._build_adapter(base_ch * 2, mod)
        self.fallback_adapter = nn.Conv2d(base_ch * 2, max_mod_order, 1)

    def _build_adapter(self, in_ch, out_ch):
        """Build adapter head with capacity proportional to modulation order."""
        if out_ch >= 16:
            return nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True), nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True), nn.Conv2d(in_ch, out_ch, 1))
        else:
            return nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True), nn.Conv2d(in_ch, out_ch, 1))

    def freeze_backbone(self):
        """Freeze backbone for adapter-only fine-tuning."""
        for name, param in self.named_parameters():
            if 'adapter' not in name and 'fallback' not in name:
                param.requires_grad = False
        print('[AdaptiveCommNet] Backbone frozen, only adapters trainable')

    def unfreeze_backbone(self):
        """Unfreeze backbone for full training."""
        for param in self.parameters():
            param.requires_grad = True
        print('[AdaptiveCommNet] All parameters trainable')

    def forward_backbone(self, x, config_tensor):
        """Forward through shared backbone.
        
        Args:
            x: [B, 5, H, W] - (eq_real, eq_imag, H_mag, H_phase, snr)
               Already ZF-equalized and constellation-normalized from dataset
        """
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        cond = self.config_encoder(config_tensor)
        h = F.relu(self.film1(self.bn1(self.conv1(x)), cond))
        h = F.relu(self.film2(self.bn2(self.conv2(h)), cond))
        h = F.relu(self.film3(self.bn3(self.conv3(h)), cond))
        h = F.relu(self.bn4(self.conv4(h)))
        return h

    def forward(self, x, config_tensor, mod_order=None, config_id=None):
        """
        Args:
            x: [B, 5, H, W] - (eq_real, eq_imag, H_mag, H_phase, snr)
            config_tensor: [B, cond_dim] conditioning
            mod_order: modulation order (4, 16, or 64) - selects adapter
            config_id: ignored
        Returns:
            logits: [B, mod_order, H, W]
        """
        features = self.forward_backbone(x, config_tensor)
        mod_key = str(mod_order) if mod_order in self.SUPPORTED_MOD_ORDERS else None
        if mod_key and mod_key in self.adapters:
            logits = self.adapters[mod_key](features)
        else:
            logits = self.fallback_adapter(features)
            if mod_order is not None and mod_order < self.max_mod_order:
                logits = logits[:, :mod_order, :, :]
        return logits

class G2DeepDataset(Dataset):
    """Wrapper for AIRadar_Comm_Dataset_G2 for deep learning with caching."""

    def __init__(self, config_name: str, num_samples: int, save_root: str, split: str='train', target_size=(512, 512), radar_sigma=3.0, enable_rf_impairments=True):
        super().__init__()
        self.config_name = config_name
        self.config = RADAR_COMM_CONFIGS_G2[config_name]
        self.target_size = target_size
        self.radar_sigma = radar_sigma
        self.config_id = CONFIG_ID_MAP[config_name]
        self.enable_rf_impairments = enable_rf_impairments
        cache_suffix = '_rf' if enable_rf_impairments else ''
        save_path = os.path.join(save_root, split, config_name)
        os.makedirs(save_path, exist_ok=True)
        cache_file = os.path.join(save_path, f'cache_{num_samples}{cache_suffix}.pkl')
        if os.path.exists(cache_file):
            print(f'[Cache] Loading {config_name}/{split} from {cache_file}')
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            self.g2_ds = type('CachedDataset', (), {'data_samples': cached['samples'], '__len__': lambda s: len(s.data_samples), '__getitem__': lambda s, i: s.data_samples[i]})()
        else:
            print(f'[Generate] Creating {config_name}/{split} ({num_samples} samples, RF={enable_rf_impairments})')
            self.g2_ds = AIRadar_Comm_Dataset_G2(config_name=config_name, num_samples=num_samples, save_path=save_path, drawfig=False, enable_clutter=True, enable_imperfect_csi=True, enable_rf_impairments=enable_rf_impairments)
            print(f'[Cache] Saving to {cache_file}')
            with open(cache_file, 'wb') as f:
                pickle.dump({'samples': self.g2_ds.data_samples}, f)
        self.config_tensor = ConfigEncoder.encode_config(self.config)

    def __len__(self):
        return len(self.g2_ds)

    def _build_radar_label(self, rdm_shape, r_axis, v_axis, targets):
        """Build Gaussian heatmap for radar targets."""
        D, R = rdm_shape
        label = np.zeros((D, R), dtype=np.float32)
        sigma2 = self.radar_sigma ** 2
        for t in targets:
            r_m = t.get('range', 0)
            v_m = t.get('velocity', 0)
            r_idx = int(np.argmin(np.abs(r_axis - r_m)))
            v_idx = int(np.argmin(np.abs(v_axis - v_m)))
            if not (0 <= r_idx < R and 0 <= v_idx < D):
                continue
            radius = 5
            for dv in range(-radius, radius + 1):
                for dr in range(-radius, radius + 1):
                    rr = r_idx + dr
                    dd = v_idx + dv
                    if 0 <= rr < R and 0 <= dd < D:
                        dist2 = dr * dr + dv * dv
                        val = math.exp(-dist2 / (2 * sigma2))
                        label[dd, rr] = max(label[dd, rr], val)
        return label

    def _normalize_rdm(self, rdm, top_p=99.5, dyn_db=40.0):
        """Normalize RDM to [0, 1]."""
        top = np.percentile(rdm, top_p)
        rdm = np.clip(rdm, top - dyn_db, top)
        rdm = (rdm - (top - dyn_db)) / dyn_db
        return rdm.astype(np.float32)

    def __getitem__(self, idx):
        sample = self.g2_ds[idx]
        rdm = np.array(sample['range_doppler_map'])
        rdm_norm = self._normalize_rdm(rdm)
        r_axis = np.array(sample['range_axis'])
        v_axis = np.array(sample['velocity_axis'])
        targets = sample['target_info']['targets']
        snr_db = sample['target_info'].get('snr_db', 15.0)
        radar_label = self._build_radar_label(rdm.shape, r_axis, v_axis, targets)
        radar_input = torch.from_numpy(rdm_norm).unsqueeze(0).contiguous()
        radar_target = torch.from_numpy(radar_label).unsqueeze(0).contiguous()
        comm_info = sample['comm_info']
        mod_order = self.config.get('mod_order', 16)
        tx_symbols = np.array(comm_info.get('tx_symbols', []), dtype=np.complex64)
        rx_symbols = np.array(comm_info.get('rx_symbols', []), dtype=np.complex64)
        channel_est = np.array(comm_info.get('channel_est', None))
        if len(rx_symbols) == 0:
            H, W = (8, 256)
            comm_input = torch.zeros(4, H, W, dtype=torch.float32)
            comm_target = torch.zeros(H, W, dtype=torch.long)
        else:
            n_syms = comm_info.get('num_data_syms', 8)
            fft_size = comm_info.get('fft_size', len(rx_symbols) // n_syms)
            try:
                rx_grid = rx_symbols.reshape(n_syms, fft_size)
            except:
                total = len(rx_symbols)
                fft_size = min(256, total)
                n_syms = total // fft_size
                rx_grid = rx_symbols[:n_syms * fft_size].reshape(n_syms, fft_size)
            try:
                is_valid_array = channel_est is not None and isinstance(channel_est, np.ndarray) and (channel_est.ndim > 0) and (channel_est.size > 0)
                if is_valid_array:
                    if len(channel_est) != fft_size:
                        from scipy.ndimage import zoom
                        H_est_resized = zoom(channel_est.real, fft_size / len(channel_est)) + 1j * zoom(channel_est.imag, fft_size / len(channel_est))
                    else:
                        H_est_resized = channel_est
                    H_grid = np.tile(H_est_resized[None, :], (n_syms, 1))
                else:
                    H_grid = np.ones_like(rx_grid, dtype=np.complex64)
            except Exception:
                H_grid = np.ones_like(rx_grid, dtype=np.complex64)
            H_safe = np.where(np.abs(H_grid) > 1e-06, H_grid, 1e-06 + 0j)
            eq_symbols = rx_grid / H_safe
            if mod_order == 4:
                scale_factor = np.sqrt(2)
            elif mod_order == 16:
                scale_factor = np.sqrt(10)
            else:
                scale_factor = np.sqrt(42)
            eq_real = eq_symbols.real / scale_factor
            eq_imag = eq_symbols.imag / scale_factor
            eq_real = np.clip(eq_real, -3, 3)
            eq_imag = np.clip(eq_imag, -3, 3)
            H_mag = np.abs(H_grid) / (np.abs(H_grid).max() + 1e-06)
            H_phase = np.angle(H_grid) / np.pi
            snr_normalized = snr_db / 35.0
            snr_channel = np.full_like(eq_real, snr_normalized)
            comm_input = torch.tensor(np.stack([eq_real, eq_imag, H_mag, H_phase, snr_channel], axis=0), dtype=torch.float32).contiguous()
            tx_ints = np.array(comm_info.get('tx_ints', []), dtype=np.int64)
            if len(tx_ints) >= n_syms * fft_size:
                comm_target = torch.tensor(tx_ints[:n_syms * fft_size].reshape(n_syms, fft_size), dtype=torch.long).contiguous()
            else:
                comm_target = torch.zeros(n_syms, fft_size, dtype=torch.long)
        meta = {'config_id': self.config_id, 'config_name': self.config_name, 'config_tensor': self.config_tensor, 'mod_order': mod_order, 'snr_db': snr_db, 'mode': self.config.get('mode', 'TRADITIONAL'), 'targets': targets, 'cfar_detections': sample.get('cfar_detections', []), 'r_axis': r_axis, 'v_axis': v_axis, 'ber': comm_info.get('ber', 0.0)}
        return (radar_input, radar_target, comm_input, comm_target, meta)

def g2_collate_fn(batch):
    """Custom collate function for G2 dataset with variable-size meta."""
    radar_inputs = torch.stack([b[0] for b in batch])
    radar_targets = torch.stack([b[1] for b in batch])
    comm_inputs = torch.stack([b[2] for b in batch])
    comm_targets = torch.stack([b[3] for b in batch])
    meta = {'config_id': torch.tensor([b[4]['config_id'] for b in batch]), 'config_name': [b[4]['config_name'] for b in batch], 'config_tensor': torch.stack([b[4]['config_tensor'] for b in batch]), 'mod_order': torch.tensor([b[4]['mod_order'] for b in batch]), 'snr_db': torch.tensor([b[4]['snr_db'] for b in batch], dtype=torch.float32), 'mode': [b[4]['mode'] for b in batch], 'targets': [b[4]['targets'] for b in batch], 'cfar_detections': [b[4]['cfar_detections'] for b in batch], 'r_axis': [b[4]['r_axis'] for b in batch], 'v_axis': [b[4]['v_axis'] for b in batch], 'ber': torch.tensor([b[4]['ber'] for b in batch], dtype=torch.float32)}
    return (radar_inputs, radar_targets, comm_inputs, comm_targets, meta)

def symbol_to_bits(symbols, mod_order):
    """Convert symbol indices to bit representation.
    
    Args:
        symbols: [B, H, W] symbol indices (0 to mod_order-1)
        mod_order: Modulation order (4, 16, 64)
    Returns:
        bits: [B, num_bits, H, W] binary representation
    """
    num_bits = int(np.log2(mod_order))
    B, H, W = symbols.shape
    bits = torch.zeros(B, num_bits, H, W, device=symbols.device, dtype=torch.float32)
    for i in range(num_bits):
        bits[:, i] = (symbols >> i & 1).float()
    return bits

def compute_llr_loss(llr_logits, comm_target, mod_order, lambda_comm=1.0):
    """Compute LLR-based communication loss with masked bits.
    
    Args:
        llr_logits: [B, max_bits, H, W] LLR outputs from UniversalCommNet
        comm_target: [B, H, W] symbol indices
        mod_order: Modulation order (4, 16, 64)
        lambda_comm: Weight for comm loss
    Returns:
        comm_loss: Bit-wise BCE loss on active bits only
    """
    bit_targets = symbol_to_bits(comm_target, mod_order)
    active_bits = int(np.log2(mod_order))
    active_llr = llr_logits[:, :active_bits]
    comm_loss = F.binary_cross_entropy_with_logits(active_llr, bit_targets)
    return lambda_comm * comm_loss

def postprocess_radar(probs, r_axis, v_axis, prob_thresh=0.7, nms_kernel=7, adaptive_thresh=True, min_peak_distance=3, max_detections=10):
    """Convert probability map to detections with adaptive threshold and strict NMS.
    
    Args:
        probs: Probability heatmap from DL model
        r_axis: Range axis in meters
        v_axis: Velocity axis in m/s
        prob_thresh: Base detection threshold (default 0.7)
        nms_kernel: NMS kernel size (default 7 for stricter NMS)
        adaptive_thresh: If True, use per-RDM adaptive threshold
        min_peak_distance: Minimum distance between peaks (reduces FPs)
        max_detections: Maximum number of detections per RDM
    """
    if adaptive_thresh:
        adaptive_th = np.percentile(probs, 99.5)
        final_thresh = max(prob_thresh, adaptive_th * 0.9)
    else:
        final_thresh = prob_thresh
    local_max = maximum_filter(probs, size=nms_kernel)
    mask = (probs >= final_thresh) & (probs == local_max)
    idxs = np.argwhere(mask)
    candidates = []
    for d_idx, r_idx in idxs:
        if r_idx < len(r_axis) and d_idx < len(v_axis):
            candidates.append({'range_m': float(r_axis[r_idx]), 'velocity_mps': float(v_axis[d_idx]), 'range_idx': int(r_idx), 'doppler_idx': int(d_idx), 'score': float(probs[d_idx, r_idx])})
    candidates.sort(key=lambda x: x['score'], reverse=True)
    detections = []
    for cand in candidates:
        too_close = False
        for det in detections:
            dr = abs(cand['range_idx'] - det['range_idx'])
            dv = abs(cand['doppler_idx'] - det['doppler_idx'])
            if dr < min_peak_distance and dv < min_peak_distance:
                too_close = True
                break
        if not too_close:
            detections.append(cand)
        if len(detections) >= max_detections:
            break
    return detections

def radar_metrics(targets, detections, match_thresh=3.0):
    """Compute radar detection metrics."""
    tp = fp = fn = 0
    matched_targets = set()
    for det in detections:
        matched = False
        for i, t in enumerate(targets):
            if i in matched_targets:
                continue
            dr = abs(det['range_m'] - t['range'])
            dv = abs(det['velocity_mps'] - t['velocity'])
            if math.sqrt(dr ** 2 + dv ** 2) < match_thresh:
                tp += 1
                matched_targets.add(i)
                matched = True
                break
        if not matched:
            fp += 1
    fn = len(targets) - len(matched_targets)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}

def cfar_metrics_from_g2(sample):
    """Get CFAR performance from G2 dataset sample."""
    targets = sample['target_info']['targets']
    cfar_dets = sample.get('cfar_detections', [])
    tp = fp = 0
    matched = set()
    for det in cfar_dets:
        found = False
        for i, t in enumerate(targets):
            if i in matched:
                continue
            dr = abs(det.get('range_m', det.get('range', 0)) - t['range'])
            dv = abs(det.get('velocity_mps', det.get('velocity', 0)) - t['velocity'])
            if math.sqrt(dr ** 2 + dv ** 2) < 5.0:
                tp += 1
                matched.add(i)
                found = True
                break
        if not found:
            fp += 1
    fn = len(targets) - len(matched)
    precision = tp / (tp + fp + 1e-08)
    recall = tp / (tp + fn + 1e-08)
    f1 = 2 * precision * recall / (precision + recall + 1e-08)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}


# =============================================================================
# Reproducibility and utility helpers
# =============================================================================


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_checkpoint_if_exists(model: nn.Module, path: Optional[str], device: torch.device) -> bool:
    if path and os.path.exists(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        return True
    return False


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def prf1_from_counts(tp: float, fp: float, fn: float, eps: float = 1e-8) -> Dict[str, float]:
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {"precision": precision, "recall": recall, "f1": f1}


def normalize_rdm(rdm: np.ndarray) -> np.ndarray:
    rdm = np.asarray(rdm, dtype=np.float32)
    if rdm.ndim == 2:
        rdm = rdm[None, :, :]
    mn = float(rdm.min())
    mx = float(rdm.max())
    return (rdm - mn) / (mx - mn + 1e-8)


def config_to_tensor(config: Dict[str, Any], override_snr_db: Optional[float] = None) -> torch.Tensor:
    """
    Unified config encoding used everywhere in this script.
    Keep this centralized so train/eval never silently diverge.
    """
    snr_db = float(config.get("snr_db", 20.0) if override_snr_db is None else override_snr_db)

    return torch.tensor([
        float(config.get("fc", 77e9)) / 1e9,
        float(config.get("bandwidth", config.get("B", 4e9))) / 1e9,
        float(config.get("num_subcarriers", config.get("N_delay", 64))) / 64.0,
        float(config.get("mod_order", 16)) / 64.0,
        snr_db / 35.0,
        float(config.get("range_resolution", 0.5)),
        float(config.get("max_range", config.get("R_max", 100.0))) / 100.0,
        float(config.get("max_velocity", 50.0)) / 50.0,
    ], dtype=torch.float32)


def get_range_velocity_axes(sample: Dict[str, Any], rdm_shape_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = rdm_shape_hw
    r_axis = np.asarray(sample.get("range_axis", np.linspace(0, 100, h)), dtype=np.float32)
    v_axis = np.asarray(sample.get("velocity_axis", np.linspace(-50, 50, w)), dtype=np.float32)
    return r_axis, v_axis


def modulation_to_num_bits(mod_order: int) -> int:
    return int(round(math.log2(mod_order)))


def get_constellation_scale(mod_order: int) -> float:
    if mod_order == 4:
        return float(np.sqrt(2))
    if mod_order == 8:
        return float(np.sqrt(6))
    if mod_order == 16:
        return float(np.sqrt(10))
    if mod_order == 64:
        return float(np.sqrt(42))
    return 1.0


# =============================================================================
# Experiment configuration
# =============================================================================


RADAR_CONFIG_GROUPS = {
    "FMCW": [
        "CN0566_TRADITIONAL",
        "Automotive_77GHz_LongRange",
        "5G_ISAC_HighBandwidth",
    ],
    "OTFS": [
        "CN0566_OTFS_ISAC",
        "AUTOMOTIVE_OTFS_ISAC",
        "OTFS_HighMobility_Wideband",
    ],
}

COMM_CONFIG_GROUPS = {
    "OFDM": {
        "4QAM": ["Automotive_77GHz_LongRange"],
        "8QAM": ["8QAM_MediumRange"],
        "16QAM": ["CN0566_TRADITIONAL", "XBand_10GHz_MediumRange", "AUTOMOTIVE_TRADITIONAL"],
        "64QAM": ["5G_ISAC_HighBandwidth"],
    },
    "OTFS": {
        "4QAM": ["CN0566_OTFS_ISAC", "AUTOMOTIVE_OTFS_ISAC"],
        "16QAM": ["OTFS_HighMobility_Wideband"],
    },
}


@dataclass
class RunConfig:
    mode: str = "eval_all"
    out_dir: str = "data/g5"
    data_root: str = "data/g5"
    seed: int = 42
    device: str = "cuda"
    batch_size: int = 8
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    train_samples: int = 300
    val_samples: int = 60
    test_samples: int = 60
    num_workers: int = 0
    radar_type: str = "all"
    comm_type: str = "all"
    qam_type: str = "all"
    channel_mode: str = "realistic"
    radar_ckpt: Optional[str] = None
    comm_ckpt: Optional[str] = None
    label_smoothing: float = 0.05
    use_focal_loss: bool = False
    radar_pos_weight: float = 5.0
    radar_sigma: float = 3.0
    eval_snr_list: Tuple[int, ...] = (5, 10, 15, 20, 25, 30)
    eval_cnr_list: Tuple[int, ...] = (0, 5, 10, 15, 20)
    eval_rcs_list: Tuple[int, ...] = (5, 10, 15, 20, 25)
    early_stop_patience: int = 8
    mixed_channel_train: bool = True
    unseen_holdout_fraction: float = 0.34
    radar_model: str = "generalized"
    comm_model: str = "adaptive"


# =============================================================================
# Losses
# =============================================================================


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.flatten(1)
        targets = targets.flatten(1)
        inter = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class FocalBCELoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=self.pos_weight)
        pt = torch.exp(-bce)
        return (self.alpha * ((1 - pt) ** self.gamma) * bce).mean()


# =============================================================================
# Communication model alternative: stable bit-logit demapper
# =============================================================================


class CommBitNet(nn.Module):
    """
    Simple, stable communication model.

    Output is bit logits, not symbol logits. This is usually better for mixed
    modulation training and easier to analyze for paper writing.
    """

    MAX_BITS = 6

    def __init__(self, in_channels: int = 5, hidden_dim: int = 256, cond_dim: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.config_encoder = ConfigEncoder(embed_dim=cond_dim)

        total_in = in_channels + cond_dim
        self.input_proj = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            ) for _ in range(3)
        ])
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, self.MAX_BITS),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, config_tensor: torch.Tensor, mod_order: Optional[int] = None) -> torch.Tensor:
        b, c, h, w = x.shape
        cond = self.config_encoder(config_tensor)

        in_ch = min(c, self.in_channels)
        feat = x[:, :in_ch].permute(0, 2, 3, 1).contiguous().view(b * h * w, in_ch)
        if in_ch < self.in_channels:
            pad = torch.zeros(b * h * w, self.in_channels - in_ch, device=feat.device, dtype=feat.dtype)
            feat = torch.cat([feat, pad], dim=1)

        cond_flat = cond.view(b, 1, 1, -1).expand(b, h, w, -1).contiguous().view(b * h * w, -1)
        z = self.input_proj(torch.cat([feat, cond_flat], dim=1))
        for blk in self.blocks:
            z = z + 0.1 * blk(z)
        out = self.output(z).view(b, h, w, self.MAX_BITS).permute(0, 3, 1, 2).contiguous()
        return out

    @staticmethod
    def bit_logits_to_symbol_logits(bit_logits: torch.Tensor, mod_order: int) -> torch.Tensor:
        active_bits = modulation_to_num_bits(mod_order)
        bit_logits = bit_logits[:, :active_bits]
        log_p1 = F.logsigmoid(bit_logits)
        log_p0 = F.logsigmoid(-bit_logits)

        b, _, h, w = bit_logits.shape
        symbol_logits = torch.zeros(b, mod_order, h, w, device=bit_logits.device, dtype=bit_logits.dtype)
        for sym in range(mod_order):
            log_prob = 0.0
            for bit_idx in range(active_bits):
                bit_val = (sym >> bit_idx) & 1
                log_prob = log_prob + (log_p1[:, bit_idx] if bit_val == 1 else log_p0[:, bit_idx])
            symbol_logits[:, sym] = log_prob
        return symbol_logits


# =============================================================================
# Dataset wrappers
# =============================================================================


class RadarSampleDataset(Dataset):
    """
    Radar dataset wrapper producing clean, consistent tensors and metadata.
    Uses AIRadar_Comm_Dataset_G2 directly so evaluation and training can share logic.
    """

    def __init__(
        self,
        config_name: str,
        num_samples: int,
        save_root: str,
        split: str,
        radar_sigma: float = 3.0,
        fixed_snr: Optional[float] = None,
        enable_clutter: bool = True,
        enable_imperfect_csi: bool = True,
        enable_rf_impairments: bool = True,
        clutter_intensity: Optional[float] = None,
        target_rcs_range: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()
        self.config_name = config_name
        self.config = RADAR_COMM_CONFIGS_G2[config_name]
        self.config_id = CONFIG_ID_MAP[config_name]
        self.radar_sigma = radar_sigma
        save_path = ensure_dir(os.path.join(save_root, split, config_name))

        self.base_ds = AIRadar_Comm_Dataset_G2(
            config_name=config_name,
            num_samples=num_samples,
            save_path=save_path,
            drawfig=False,
            fixed_snr=fixed_snr,
            enable_clutter=enable_clutter,
            enable_imperfect_csi=enable_imperfect_csi,
            enable_rf_impairments=enable_rf_impairments,
            clutter_intensity=clutter_intensity,
            target_rcs_range=target_rcs_range,
        )

    def __len__(self) -> int:
        return len(self.base_ds)

    def _create_heatmap(self, targets: List[Dict[str, Any]], r_axis: np.ndarray, v_axis: np.ndarray, shape: Tuple[int, int]) -> torch.Tensor:
        heatmap = np.zeros(shape, dtype=np.float32)
        for tgt in targets:
            r = float(tgt.get("range", 0.0))
            v = float(tgt.get("velocity", 0.0))
            r_idx = int(np.argmin(np.abs(r_axis - r)))
            v_idx = int(np.argmin(np.abs(v_axis - v)))
            if 0 <= v_idx < shape[0] and 0 <= r_idx < shape[1]:
                heatmap[v_idx, r_idx] = 1.0
        if self.radar_sigma > 0:
            heatmap = gaussian_filter(heatmap, sigma=self.radar_sigma)
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
        return torch.tensor(heatmap[None, :, :], dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base_ds[idx]
        rdm = normalize_rdm(np.asarray(sample["range_doppler_map"]))
        r_axis, v_axis = get_range_velocity_axes(sample, rdm.shape[1:])
        targets = sample.get("target_info", {}).get("targets", sample.get("targets", []))

        return {
            "radar_in": torch.tensor(rdm, dtype=torch.float32),
            "radar_target": self._create_heatmap(targets, r_axis, v_axis, rdm.shape[1:]),
            "targets": targets,
            "range_axis": r_axis,
            "velocity_axis": v_axis,
            "config_name": self.config_name,
            "config_id": self.config_id,
            "config_tensor": config_to_tensor(self.config),
            "raw_sample": sample,
        }


class CommSampleDataset(Dataset):
    """
    Communication dataset wrapper producing a consistent [5, H, W] input.
    Trains from equalized-symbol view to avoid mismatched feature pipelines.
    """

    def __init__(
        self,
        config_name: str,
        num_samples: int,
        save_root: str,
        split: str,
        fixed_snr: Optional[float] = None,
        enable_clutter: bool = True,
        enable_imperfect_csi: bool = True,
        enable_rf_impairments: bool = True,
    ):
        super().__init__()
        self.config_name = config_name
        self.config = RADAR_COMM_CONFIGS_G2[config_name]
        self.config_id = CONFIG_ID_MAP[config_name]
        save_path = ensure_dir(os.path.join(save_root, split, config_name))

        self.base_ds = AIRadar_Comm_Dataset_G2(
            config_name=config_name,
            num_samples=num_samples,
            save_path=save_path,
            drawfig=False,
            fixed_snr=fixed_snr,
            enable_clutter=enable_clutter,
            enable_imperfect_csi=enable_imperfect_csi,
            enable_rf_impairments=enable_rf_impairments,
        )

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base_ds[idx]
        comm_info = sample.get("comm_info", {})
        config = self.config
        mod_order = int(config.get("mod_order", 16))
        snr_db = float(comm_info.get("snr_db", config.get("snr_db", 20.0)))

        rx_symbols = np.asarray(comm_info.get("rx_symbols", []), dtype=np.complex64)
        channel_est = np.asarray(comm_info.get("channel_est", []), dtype=np.complex64)
        tx_ints = np.asarray(comm_info.get("tx_ints", []), dtype=np.int64)

        n_syms = int(comm_info.get("num_data_syms", config.get("num_ofdm_symbols", config.get("num_symbols", 8))))
        fft_size = int(comm_info.get("fft_size", max(1, len(rx_symbols) // max(1, n_syms))))

        if len(rx_symbols) < n_syms * fft_size:
            comm_in = torch.zeros(5, n_syms, fft_size, dtype=torch.float32)
            comm_tgt = torch.zeros(n_syms, fft_size, dtype=torch.long)
        else:
            rx_grid = rx_symbols[:n_syms * fft_size].reshape(n_syms, fft_size)

            if len(channel_est) == 0:
                h_grid = np.ones_like(rx_grid)
            elif len(channel_est) == fft_size:
                h_grid = np.tile(channel_est[None, :], (n_syms, 1))
            else:
                ce = channel_est.flatten()
                ce = np.resize(ce, fft_size)
                h_grid = np.tile(ce[None, :], (n_syms, 1))

            # rx_symbols are already equalized in the simulation
            eq_symbols = rx_grid

            scale = get_constellation_scale(mod_order)
            eq_real = np.clip(eq_symbols.real / scale, -3, 3)
            eq_imag = np.clip(eq_symbols.imag / scale, -3, 3)
            h_mag = np.abs(h_grid)
            h_mag = h_mag / (h_mag.max() + 1e-8)
            h_phase = np.angle(h_grid) / np.pi
            snr_ch = np.full_like(eq_real, snr_db / 35.0)

            comm_in = torch.tensor(np.stack([eq_real, eq_imag, h_mag, h_phase, snr_ch], axis=0), dtype=torch.float32)
            if len(tx_ints) >= n_syms * fft_size:
                comm_tgt = torch.tensor(tx_ints[:n_syms * fft_size].reshape(n_syms, fft_size), dtype=torch.long)
            else:
                comm_tgt = torch.zeros(n_syms, fft_size, dtype=torch.long)

        return {
            "comm_in": comm_in,
            "comm_target": comm_tgt,
            "config_name": self.config_name,
            "config_id": self.config_id,
            "config_tensor": config_to_tensor(config, override_snr_db=snr_db),
            "mod_order": mod_order,
            "snr_db": snr_db,
            "raw_sample": sample,
        }


class IndexedDataset(Dataset):
    """Attach dataset index/config name for balanced multi-config sampling."""

    def __init__(self, dataset: Dataset, config_name: str):
        self.dataset = dataset
        self.config_name = config_name

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        item["dataset_config_name"] = self.config_name
        return item


# =============================================================================
# Collate functions
# =============================================================================


def _get_waveform_id(cfg_name: str) -> int:
    if "OTFS" in cfg_name.upper():
        return 2  # OTFS
    elif "OFDM" in cfg_name.upper() or "TRADITIONAL" in cfg_name.upper() or "QAM" in cfg_name.upper():
        # Usually radar is FMCW for traditional, comm is OFDM
        return 0 if "RADAR" in cfg_name.upper() or "TRADITIONAL" in cfg_name.upper() else 1
    return 0  # FMCW default

def radar_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    waveform_ids = [2 if "OTFS" in b["config_name"].upper() else 0 for b in batch]
    return {
        "radar_in": torch.stack([b["radar_in"] for b in batch]),
        "radar_target": torch.stack([b["radar_target"] for b in batch]),
        "targets": [b["targets"] for b in batch],
        "range_axis": [b["range_axis"] for b in batch],
        "velocity_axis": [b["velocity_axis"] for b in batch],
        "config_name": [b["config_name"] for b in batch],
        "config_id": torch.tensor([b["config_id"] for b in batch], dtype=torch.long),
        "config_tensor": torch.stack([b["config_tensor"] for b in batch]),
        "waveform_id": torch.tensor(waveform_ids, dtype=torch.long),
        "raw_sample": [b["raw_sample"] for b in batch],
        "dataset_config_name": [b.get("dataset_config_name", b["config_name"]) for b in batch],
    }


def comm_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    waveform_ids = [2 if "OTFS" in b["config_name"].upper() else 1 for b in batch]
    
    # Map modulation order to MODULATION_IDS (1: QPSK, 2: QAM8, 3: QAM16, 4: QAM64)
    def mod_to_id(mod: int):
        if mod == 4: return 1
        if mod == 8: return 2
        if mod == 16: return 3
        if mod == 64: return 4
        return 0
    mod_ids = [mod_to_id(b["mod_order"]) for b in batch]

    return {
        "comm_in": torch.stack([b["comm_in"] for b in batch]),
        "comm_target": torch.stack([b["comm_target"] for b in batch]),
        "config_name": [b["config_name"] for b in batch],
        "config_id": torch.tensor([b["config_id"] for b in batch], dtype=torch.long),
        "config_tensor": torch.stack([b["config_tensor"] for b in batch]),
        "mod_order": torch.tensor([b["mod_order"] for b in batch], dtype=torch.long),
        "snr_db": torch.tensor([b["snr_db"] for b in batch], dtype=torch.float32),
        "waveform_id": torch.tensor(waveform_ids, dtype=torch.long),
        "mod_id": torch.tensor(mod_ids, dtype=torch.long),
        "raw_sample": [b["raw_sample"] for b in batch],
        "dataset_config_name": [b.get("dataset_config_name", b["config_name"]) for b in batch],
    }


# =============================================================================
# Builders for config splits and loaders
# =============================================================================


def choose_radar_configs(radar_type: str) -> List[str]:
    if radar_type == "all":
        return RADAR_CONFIG_GROUPS["FMCW"] + RADAR_CONFIG_GROUPS["OTFS"]
    return RADAR_CONFIG_GROUPS[radar_type]


def choose_comm_configs(comm_type: str, qam_type: str) -> List[str]:
    groups: Dict[str, List[str]] = {}
    if comm_type == "all":
        merged: Dict[str, List[str]] = defaultdict(list)
        for ct in COMM_CONFIG_GROUPS:
            for qam, cfgs in COMM_CONFIG_GROUPS[ct].items():
                merged[qam].extend(cfgs)
        groups = merged
    else:
        groups = COMM_CONFIG_GROUPS[comm_type]

    if qam_type == "all":
        flat: List[str] = []
        for cfgs in groups.values():
            flat.extend(cfgs)
        return sorted(list(dict.fromkeys(flat)))
    return groups.get(qam_type, [])


def split_seen_unseen(configs: List[str], holdout_fraction: float) -> Tuple[List[str], List[str]]:
    if len(configs) <= 1:
        return configs, []
    n_unseen = max(1, int(round(len(configs) * holdout_fraction)))
    unseen = configs[-n_unseen:]
    seen = configs[:-n_unseen]
    return seen, unseen


class HomogeneousBatchSampler(torch.utils.data.Sampler):
    """
    Yields batches of indices where all indices in a batch belong to the same underlying dataset
    in a ConcatDataset. Ensures batches don't mix different tensor shapes.
    """
    def __init__(self, datasets: List[Dataset], batch_size: int, shuffle: bool = True):
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.batches = []
        offset = 0
        for ds in datasets:
            n = len(ds)
            indices = list(range(offset, offset + n))
            if shuffle:
                random.shuffle(indices)
            for i in range(0, n, batch_size):
                self.batches.append(indices[i:i + batch_size])
            offset += n
            
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for batch in self.batches:
            yield batch
            
    def __len__(self):
        return len(self.batches)


def build_balanced_concat_loader(
    datasets_by_config: Dict[str, Dataset],
    batch_size: int,
    collate_fn,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    wrapped = []
    for cfg_name, ds in datasets_by_config.items():
        wrapped_ds = IndexedDataset(ds, cfg_name)
        wrapped.append(wrapped_ds)
    concat_ds = ConcatDataset(wrapped)
    sampler = HomogeneousBatchSampler(wrapped, batch_size, shuffle)
    return DataLoader(
        concat_ds,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


def build_eval_loaders(datasets_by_config: Dict[str, Dataset], batch_size: int, collate_fn, num_workers: int = 0) -> Dict[str, DataLoader]:
    return {
        cfg: DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
        for cfg, ds in datasets_by_config.items()
    }


# =============================================================================
# Radar train/eval
# =============================================================================


class RadarTrainer:
    def __init__(self, model: nn.Module, device: torch.device, pos_weight: float = 5.0, use_focal_loss: bool = False):
        self.model = model
        self.device = device
        self.use_focal_loss = use_focal_loss
        
        # Increase default pos_weight heavily if focal loss isn't used, since targets are extremely sparse
        # A 512x512 grid might only have 1-5 target pixels. A weight of 5.0 is far too small.
        weight = pos_weight if pos_weight > 5.0 else 500.0
        weight_tensor = torch.tensor([weight], device=device)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
        self.dice = DiceLoss()
        
        if use_focal_loss:
            self.focal = FocalBCELoss(alpha=0.25, gamma=2.0, pos_weight=weight_tensor)

    def train_one_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for batch in loader:
            radar_in = batch["radar_in"].to(self.device)
            radar_target = batch["radar_target"].to(self.device)
            config_tensor = batch["config_tensor"].to(self.device)

            optimizer.zero_grad(set_to_none=True)
            if hasattr(self.model, "forward_radar"):
                # Use ISACFoundationModel
                out = self.model.forward_radar(radar_in, config_tensor, batch["waveform_id"].to(self.device))
                logits = out["radar_logits"]
            else:
                logits = self.model(radar_in, config_tensor)
                
            if self.use_focal_loss:
                loss_bce = self.focal(logits, radar_target)
            else:
                loss_bce = self.bce(logits, radar_target)
                
            loss_dice = self.dice(logits, radar_target)
            loss = loss_bce + 0.5 * loss_dice
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

        return {"loss": total_loss / max(1, total_batches)}

    @torch.no_grad()
    def evaluate_detection(self, loaders: Dict[str, DataLoader], prob_thresh: float = 0.5) -> Dict[str, Any]:
        self.model.eval()
        per_cfg = {}
        global_counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0}

        for cfg_name, loader in loaders.items():
            counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0}
            for batch in loader:
                radar_in = batch["radar_in"].to(self.device)
                config_tensor = batch["config_tensor"].to(self.device)
                if hasattr(self.model, "forward_radar"):
                    out = self.model.forward_radar(radar_in, config_tensor, batch["waveform_id"].to(self.device))
                    logits = out["radar_logits"]
                else:
                    logits = self.model(radar_in, config_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()

                for i in range(radar_in.shape[0]):
                    detections = postprocess_radar(
                        probs[i, 0],
                        batch["range_axis"][i],
                        batch["velocity_axis"][i],
                        prob_thresh=prob_thresh,
                    )
                    m = radar_metrics(batch["targets"][i], detections)
                    counts["tp"] += m["tp"]
                    counts["fp"] += m["fp"]
                    counts["fn"] += m["fn"]

            per_cfg[cfg_name] = {**counts, **prf1_from_counts(counts["tp"], counts["fp"], counts["fn"])}
            global_counts["tp"] += counts["tp"]
            global_counts["fp"] += counts["fp"]
            global_counts["fn"] += counts["fn"]

        return {
            "global": {**global_counts, **prf1_from_counts(global_counts["tp"], global_counts["fp"], global_counts["fn"])},
            "per_config": per_cfg,
        }

    @torch.no_grad()
    def evaluate_against_cfar_sweep(
        self,
        config_name: str,
        sweep_type: str,
        sweep_values: List[int],
        num_samples: int,
        save_root: str,
    ) -> Dict[str, Any]:
        self.model.eval()
        config = RADAR_COMM_CONFIGS_G2[config_name]
        results = {
            "x": list(sweep_values),
            "dl_f1": [], "dl_precision": [], "dl_recall": [],
            "cfar_f1": [], "cfar_precision": [], "cfar_recall": [],
            "config": config_name,
            "sweep_type": sweep_type,
        }

        for value in sweep_values:
            if sweep_type == "snr":
                ds = RadarSampleDataset(
                    config_name=config_name,
                    num_samples=num_samples,
                    save_root=save_root,
                    split=f"eval_snr_{value}",
                    fixed_snr=value,
                    enable_clutter=True,
                    enable_imperfect_csi=True,
                    enable_rf_impairments=True,
                )
            elif sweep_type == "cnr":
                clutter_intensity = 0.05 * (10 ** (value / 10))
                ds = RadarSampleDataset(
                    config_name=config_name,
                    num_samples=num_samples,
                    save_root=save_root,
                    split=f"eval_cnr_{value}",
                    clutter_intensity=clutter_intensity,
                    enable_clutter=True,
                    enable_imperfect_csi=True,
                    enable_rf_impairments=True,
                )
            elif sweep_type == "rcs":
                ds = RadarSampleDataset(
                    config_name=config_name,
                    num_samples=num_samples,
                    save_root=save_root,
                    split=f"eval_rcs_{value}",
                    target_rcs_range=(value - 2, value + 2),
                    enable_clutter=True,
                    enable_imperfect_csi=True,
                    enable_rf_impairments=True,
                )
            else:
                raise ValueError(f"Unknown sweep_type: {sweep_type}")

            dl_counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0}
            cfar_counts = {"tp": 0.0, "fp": 0.0, "fn": 0.0}
            loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=radar_collate)

            for batch in loader:
                radar_in = batch["radar_in"].to(self.device)
                config_tensor = batch["config_tensor"].to(self.device)
                
                if hasattr(self.model, "forward_radar"):
                    out = self.model.forward_radar(radar_in, config_tensor, batch["waveform_id"].to(self.device))
                    logits = out["radar_logits"]
                else:
                    logits = self.model(radar_in, config_tensor)
                
                probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

                detections = postprocess_radar(probs, batch["range_axis"][0], batch["velocity_axis"][0], prob_thresh=0.5)
                m_dl = radar_metrics(batch["targets"][0], detections)
                dl_counts["tp"] += m_dl["tp"]
                dl_counts["fp"] += m_dl["fp"]
                dl_counts["fn"] += m_dl["fn"]

                m_cfar = cfar_metrics_from_g2(batch["raw_sample"][0])
                cfar_counts["tp"] += m_cfar["tp"]
                cfar_counts["fp"] += m_cfar["fp"]
                cfar_counts["fn"] += m_cfar["fn"]

            dl_stats = prf1_from_counts(dl_counts["tp"], dl_counts["fp"], dl_counts["fn"])
            cfar_stats = prf1_from_counts(cfar_counts["tp"], cfar_counts["fp"], cfar_counts["fn"])
            results["dl_f1"].append(dl_stats["f1"])
            results["dl_precision"].append(dl_stats["precision"])
            results["dl_recall"].append(dl_stats["recall"])
            results["cfar_f1"].append(cfar_stats["f1"])
            results["cfar_precision"].append(cfar_stats["precision"])
            results["cfar_recall"].append(cfar_stats["recall"])

        return results


# =============================================================================
# Communication train/eval
# =============================================================================


class CommTrainer:
    def __init__(self, model: nn.Module, device: torch.device, symbol_logits_mode: bool):
        self.model = model
        self.device = device
        self.symbol_logits_mode = symbol_logits_mode
        self.focal_bce = FocalBCELoss()

    def compute_loss(self, output: torch.Tensor, target: torch.Tensor, mod_order: int, label_smoothing: float, use_focal_loss: bool) -> torch.Tensor:
        if self.symbol_logits_mode:
            return F.cross_entropy(output, target.long(), label_smoothing=label_smoothing)

        active_bits = modulation_to_num_bits(mod_order)
        gt_bits = symbol_to_bits(target, mod_order)[:, :active_bits]
        pred_bits = output[:, :active_bits]
        if use_focal_loss:
            return self.focal_bce(pred_bits, gt_bits)
        return compute_llr_loss(output, target, mod_order, lambda_comm=1.0)

    def to_symbol_logits(self, output: torch.Tensor, mod_order: int) -> torch.Tensor:
        if self.symbol_logits_mode:
            return output
        if hasattr(self.model, "get_symbol_logits"):
            return self.model.get_symbol_logits(output, mod_order)
        return CommBitNet.bit_logits_to_symbol_logits(output, mod_order)

    def train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        label_smoothing: float = 0.0,
        use_focal_loss: bool = False,
    ) -> Dict[str, Any]:
        self.model.train()
        total_loss = 0.0
        total_ber = 0.0
        total_batches = 0
        per_cfg = defaultdict(lambda: {"loss": 0.0, "ber": 0.0, "n": 0})

        for batch in loader:
            comm_in = batch["comm_in"].to(self.device)
            comm_target = batch["comm_target"].to(self.device)
            config_tensor = batch["config_tensor"].to(self.device)
            mod_order = int(batch["mod_order"][0].item())

            optimizer.zero_grad(set_to_none=True)
            if hasattr(self.model, "forward_comm"):
                out = self.model.forward_comm(comm_in, config_tensor, batch["waveform_id"].to(self.device), batch["mod_id"].to(self.device))
                output = out["bit_logits"]
            else:
                output = self.model(comm_in, config_tensor, mod_order)
            loss = self.compute_loss(output, comm_target, mod_order, label_smoothing, use_focal_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            optimizer.step()

            with torch.no_grad():
                symbol_logits = self.to_symbol_logits(output, mod_order)
                pred = symbol_logits.argmax(dim=1)
                pred_bits = symbol_to_bits(pred, mod_order)
                gt_bits = symbol_to_bits(comm_target, mod_order)
                ber = float((pred_bits != gt_bits).float().mean().item())

            total_loss += float(loss.item())
            total_ber += ber
            total_batches += 1

            cfg_name = batch["dataset_config_name"][0]
            per_cfg[cfg_name]["loss"] += float(loss.item())
            per_cfg[cfg_name]["ber"] += ber
            per_cfg[cfg_name]["n"] += 1

        per_cfg_avg = {}
        for cfg_name, stats in per_cfg.items():
            n = max(1, stats["n"])
            per_cfg_avg[cfg_name] = {
                "loss": stats["loss"] / n,
                "ber": stats["ber"] / n,
                "n": stats["n"],
            }

        return {
            "loss": total_loss / max(1, total_batches),
            "ber": total_ber / max(1, total_batches),
            "per_config": per_cfg_avg,
        }

    @torch.no_grad()
    def evaluate(self, loaders: Dict[str, DataLoader]) -> Dict[str, Any]:
        self.model.eval()
        per_cfg = {}
        all_bers = []

        for cfg_name, loader in loaders.items():
            cfg_bers = []
            for batch in loader:
                comm_in = batch["comm_in"].to(self.device)
                comm_target = batch["comm_target"].to(self.device)
                config_tensor = batch["config_tensor"].to(self.device)
                mod_order = int(batch["mod_order"][0].item())

                if hasattr(self.model, "forward_comm"):
                    out = self.model.forward_comm(comm_in, config_tensor, batch["waveform_id"].to(self.device), batch["mod_id"].to(self.device))
                    output = out["bit_logits"]
                else:
                    output = self.model(comm_in, config_tensor, mod_order)
                symbol_logits = self.to_symbol_logits(output, mod_order)
                pred = symbol_logits.argmax(dim=1)
                pred_bits = symbol_to_bits(pred, mod_order)
                gt_bits = symbol_to_bits(comm_target, mod_order)
                ber = float((pred_bits != gt_bits).float().mean().item())
                cfg_bers.append(ber)
                all_bers.append(ber)

            per_cfg[cfg_name] = float(np.mean(cfg_bers)) if cfg_bers else 1.0

        return {
            "avg_ber": float(np.mean(all_bers)) if all_bers else 1.0,
            "per_config": per_cfg,
        }

    @torch.no_grad()
    def evaluate_snr_sweep(
        self,
        config_name: str,
        save_root: str,
        snr_list: List[int],
        num_samples: int,
        channel_mode: str = "realistic",
    ) -> Dict[str, Any]:
        self.model.eval()
        results = {"snr": list(snr_list), "dl_ber": [], "trad_ber": [], "config": config_name, "channel_mode": channel_mode}
        config = RADAR_COMM_CONFIGS_G2[config_name]
        mod_order = int(config.get("mod_order", 16))

        if channel_mode == "awgn":
            enable_clutter = False
            enable_imperfect_csi = False
            enable_rf_impairments = False
        else:
            enable_clutter = True
            enable_imperfect_csi = True
            enable_rf_impairments = False

        for snr_db in snr_list:
            ds = CommSampleDataset(
                config_name=config_name,
                num_samples=num_samples,
                save_root=save_root,
                split=f"eval_{channel_mode}_{snr_db}",
                fixed_snr=snr_db,
                enable_clutter=enable_clutter,
                enable_imperfect_csi=enable_imperfect_csi,
                enable_rf_impairments=enable_rf_impairments,
            )
            loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=comm_collate)
            dl_bers = []
            trad_bers = []

            for batch in loader:
                comm_in = batch["comm_in"].to(self.device)
                comm_target = batch["comm_target"].to(self.device)
                config_tensor = batch["config_tensor"].to(self.device)
                if hasattr(self.model, "forward_comm"):
                    out = self.model.forward_comm(comm_in, config_tensor, batch["waveform_id"].to(self.device), batch["mod_id"].to(self.device))
                    output = out["bit_logits"]
                else:
                    output = self.model(comm_in, config_tensor, mod_order)
                symbol_logits = self.to_symbol_logits(output, mod_order)
                pred = symbol_logits.argmax(dim=1)
                pred_bits = symbol_to_bits(pred, mod_order)
                gt_bits = symbol_to_bits(comm_target, mod_order)
                dl_bers.append(float((pred_bits != gt_bits).float().mean().item()))
                trad_bers.append(float(batch["raw_sample"][0].get("comm_info", {}).get("ber", 0.5)))

            results["dl_ber"].append(float(np.mean(dl_bers)) if dl_bers else 1.0)
            results["trad_ber"].append(float(np.mean(trad_bers)) if trad_bers else 1.0)

        return results


# =============================================================================
# Plotting and reporting
# =============================================================================


def plot_radar_sweep_all(results_dict: Dict[str, Any], sweep_type: str, out_path: str) -> None:
    xlabel = sweep_type.upper() + " (dB)"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    for i, (key, results) in enumerate(results_dict.items()):
        x = results["x"]
        axes[0].plot(x, results["dl_f1"], "o-", color=colors[i], linewidth=2, label=f"{key} (DL)")
        axes[0].plot(x, results["cfar_f1"], "s--", color=colors[i], linewidth=2, alpha=0.6, label=f"{key} (CFAR)")
        axes[1].plot(x, results["dl_precision"], "o-", color=colors[i], linewidth=2, label=f"{key} (DL)")
        axes[1].plot(x, results["cfar_precision"], "s--", color=colors[i], linewidth=2, alpha=0.6, label=f"{key} (CFAR)")
        axes[2].plot(x, results["dl_recall"], "o-", color=colors[i], linewidth=2, label=f"{key} (DL)")
        axes[2].plot(x, results["cfar_recall"], "s--", color=colors[i], linewidth=2, alpha=0.6, label=f"{key} (CFAR)")
        
    axes[0].set_title("F1")
    axes[1].set_title("Precision")
    axes[2].set_title("Recall")
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.0, 1.05)
    
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_radar_sweep(results: Dict[str, Any], out_path: str) -> None:
    x = results["x"]
    xlabel = results["sweep_type"].upper() + " (dB)"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(x, results["dl_f1"], "o-", linewidth=2, label="DL")
    axes[0].plot(x, results["cfar_f1"], "s--", linewidth=2, label="CFAR")
    axes[0].set_title("F1")
    axes[1].plot(x, results["dl_precision"], "o-", linewidth=2, label="DL")
    axes[1].plot(x, results["cfar_precision"], "s--", linewidth=2, label="CFAR")
    axes[1].set_title("Precision")
    axes[2].plot(x, results["dl_recall"], "o-", linewidth=2, label="DL")
    axes[2].plot(x, results["cfar_recall"], "s--", linewidth=2, label="CFAR")
    axes[2].set_title("Recall")
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.0, 1.05)
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_comm_sweep_all(snr_reports: Dict[str, Any], out_path: str, title: Optional[str] = None) -> None:
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(snr_reports)))
    for i, (key, res) in enumerate(snr_reports.items()):
        plt.semilogy(res["snr"], res["dl_ber"], "o-", color=colors[i], linewidth=2, label=f"{key} (DL)")
        plt.semilogy(res["snr"], res["trad_ber"], "s--", color=colors[i], linewidth=2, alpha=0.6, label=f"{key} (Trad)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(title or "BER vs SNR - All Configs")
    plt.grid(True, alpha=0.3, which="both")
    plt.ylim(1e-4, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_comm_sweep(results: Dict[str, Any], out_path: str, title: Optional[str] = None) -> None:
    plt.figure(figsize=(8, 5))
    plt.semilogy(results["snr"], results["dl_ber"], "o-", linewidth=2, label="DL")
    plt.semilogy(results["snr"], results["trad_ber"], "s--", linewidth=2, label="Traditional")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(title or f"BER vs SNR - {results['config']}")
    plt.grid(True, alpha=0.3, which="both")
    plt.ylim(1e-4, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def generate_markdown_report(summary: Dict[str, Any], out_path: str) -> None:
    lines = [
        "# G3 Refactored Evaluation Report",
        "",
        "## 1. Experiment Summary",
        "",
        "```json",
        json.dumps(summary.get("run_config", {}), indent=2),
        "```",
        "",
    ]

    radar = summary.get("radar", {})
    if radar:
        lines += [
            "## 2. Radar Results",
            "",
            f"Seen-config F1: **{radar.get('seen', {}).get('global', {}).get('f1', 0):.4f}**  ",
            f"Unseen-config F1: **{radar.get('unseen', {}).get('global', {}).get('f1', 0):.4f}**  ",
            "",
        ]
        per_cfg = radar.get("seen", {}).get("per_config", {})
        if per_cfg:
            lines += ["### Seen-config per-config metrics", "", "| Config | Precision | Recall | F1 |", "|---|---:|---:|---:|"]
            for cfg, m in per_cfg.items():
                lines.append(f"| {cfg} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |")
            lines.append("")

    comm = summary.get("comm", {})
    if comm:
        lines += [
            "## 3. Communication Results",
            "",
            f"Seen-config avg BER: **{comm.get('seen', {}).get('avg_ber', 1):.4e}**  ",
            f"Unseen-config avg BER: **{comm.get('unseen', {}).get('avg_ber', 1):.4e}**  ",
            "",
        ]
        per_cfg = comm.get("seen", {}).get("per_config", {})
        if per_cfg:
            lines += ["### Seen-config per-config BER", "", "| Config | BER |", "|---|---:|"]
            for cfg, v in per_cfg.items():
                lines.append(f"| {cfg} | {v:.4e} |")
            lines.append("")

    lines += [
        "## 4. Paper-facing takeaways",
        "",
        "- Training uses balanced multi-configuration sampling rather than per-configuration block training.",
        "- Validation is split into seen-config and unseen-config settings to support generalization claims.",
        "- Radar is evaluated at the detection level using post-processed detections and CFAR baselines.",
        "- Communication is evaluated using BER and SNR sweeps against a classical baseline stored in the dataset.",
        "",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =============================================================================
# Training/evaluation orchestration
# =============================================================================


def build_radar_datasets(cfg: RunConfig) -> Tuple[Dict[str, Dataset], Dict[str, Dataset], Dict[str, Dataset], List[str], List[str]]:
    configs = choose_radar_configs(cfg.radar_type)
    seen_configs, unseen_configs = split_seen_unseen(configs, cfg.unseen_holdout_fraction)
    if not seen_configs:
        seen_configs = configs
        unseen_configs = []

    train_sets = {}
    val_seen_sets = {}
    test_unseen_sets = {}

    for config_name in seen_configs:
        train_sets[config_name] = RadarSampleDataset(
            config_name, cfg.train_samples, cfg.data_root, "train", cfg.radar_sigma,
            enable_clutter=True, enable_imperfect_csi=True, enable_rf_impairments=True,
        )
        val_seen_sets[config_name] = RadarSampleDataset(
            config_name, cfg.val_samples, cfg.data_root, "val_seen", cfg.radar_sigma,
            enable_clutter=True, enable_imperfect_csi=True, enable_rf_impairments=True,
        )

    for config_name in unseen_configs:
        test_unseen_sets[config_name] = RadarSampleDataset(
            config_name, cfg.test_samples, cfg.data_root, "test_unseen", cfg.radar_sigma,
            enable_clutter=True, enable_imperfect_csi=True, enable_rf_impairments=True,
        )

    return train_sets, val_seen_sets, test_unseen_sets, seen_configs, unseen_configs


def build_comm_datasets(cfg: RunConfig) -> Tuple[Dict[str, Dataset], Dict[str, Dataset], Dict[str, Dataset], List[str], List[str]]:
    configs = choose_comm_configs(cfg.comm_type, cfg.qam_type)
    seen_configs, unseen_configs = split_seen_unseen(configs, cfg.unseen_holdout_fraction)
    if not seen_configs:
        seen_configs = configs
        unseen_configs = []

    train_sets = {}
    val_seen_sets = {}
    test_unseen_sets = {}

    for config_name in seen_configs:
        if cfg.mixed_channel_train:
            train_awgn = CommSampleDataset(
                config_name, cfg.train_samples // 2, cfg.data_root, "train_awgn",
                enable_clutter=False, enable_imperfect_csi=False, enable_rf_impairments=False,
            )
            train_real = CommSampleDataset(
                config_name, cfg.train_samples - cfg.train_samples // 2, cfg.data_root, "train_realistic",
                enable_clutter=True, enable_imperfect_csi=True, enable_rf_impairments=True,
            )
            train_sets[config_name] = ConcatDataset([IndexedDataset(train_awgn, config_name), IndexedDataset(train_real, config_name)])
        else:
            train_sets[config_name] = CommSampleDataset(
                config_name, cfg.train_samples, cfg.data_root, "train",
                enable_clutter=True, enable_imperfect_csi=True, enable_rf_impairments=True,
            )

        val_seen_sets[config_name] = CommSampleDataset(
            config_name, cfg.val_samples, cfg.data_root, "val_seen",
            enable_clutter=True, enable_imperfect_csi=True, enable_rf_impairments=True,
        )

    for config_name in unseen_configs:
        test_unseen_sets[config_name] = CommSampleDataset(
            config_name, cfg.test_samples, cfg.data_root, "test_unseen",
            enable_clutter=True, enable_imperfect_csi=True, enable_rf_impairments=True,
        )

    return train_sets, val_seen_sets, test_unseen_sets, seen_configs, unseen_configs


def train_radar_pipeline(cfg: RunConfig, device: torch.device, model_name: str = "generalized") -> Dict[str, Any]:
    out_dir = ensure_dir(os.path.join(cfg.out_dir, "radar"))
    train_sets, val_seen_sets, test_unseen_sets, seen_configs, unseen_configs = build_radar_datasets(cfg)

    train_loader = build_balanced_concat_loader(train_sets, cfg.batch_size, radar_collate, cfg.num_workers, shuffle=True)
    val_seen_loaders = build_eval_loaders(val_seen_sets, cfg.batch_size, radar_collate, cfg.num_workers)
    test_unseen_loaders = build_eval_loaders(test_unseen_sets, cfg.batch_size, radar_collate, cfg.num_workers)

    model = create_radar_model(model_name, device)
    if cfg.radar_ckpt:
        load_checkpoint_if_exists(model, cfg.radar_ckpt, device)

    # Enable focal loss by default to handle extreme target sparsity
    trainer = RadarTrainer(model, device, pos_weight=cfg.radar_pos_weight, use_focal_loss=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best = {"f1": -1.0, "epoch": -1}
    best_path = os.path.join(out_dir, "radar_best.pt")
    patience = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_stats = trainer.train_one_epoch(train_loader, optimizer)
        val_seen = trainer.evaluate_detection(val_seen_loaders)
        val_f1 = val_seen["global"]["f1"]
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_seen_f1": val_f1,
            "val_seen_precision": val_seen["global"]["precision"],
            "val_seen_recall": val_seen["global"]["recall"],
        })

        print(f"[Radar][Epoch {epoch:03d}] loss={train_stats['loss']:.4f} val_seen_f1={val_f1:.4f}")

        if val_f1 > best["f1"]:
            best = {"f1": val_f1, "epoch": epoch}
            torch.save({"model": model.state_dict(), "best": best, "seen_configs": seen_configs}, best_path)
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"[Radar] Early stopping at epoch {epoch}")
                break

    load_checkpoint_if_exists(model, best_path, device)
    final_seen = trainer.evaluate_detection(val_seen_loaders)
    final_unseen = trainer.evaluate_detection(test_unseen_loaders) if test_unseen_loaders else {"global": {}, "per_config": {}}

    eval_dir = ensure_dir(os.path.join(out_dir, "eval"))
    snr_results_all = {}
    cnr_results_all = {}
    rcs_results_all = {}
    for config_name in seen_configs:
        cfg_eval_dir = ensure_dir(os.path.join(eval_dir, config_name))
        print(f"[Radar] Evaluating sweeps for {config_name}")
        
        snr_results = trainer.evaluate_against_cfar_sweep(
            config_name=config_name,
            sweep_type="snr",
            sweep_values=list(cfg.eval_snr_list),
            num_samples=15,
            save_root=cfg.data_root,
        )
        snr_results_all[config_name] = snr_results
        plot_radar_sweep(snr_results, os.path.join(cfg_eval_dir, "radar_snr_vs_cfar.png"))

        cnr_results = trainer.evaluate_against_cfar_sweep(
            config_name=config_name,
            sweep_type="cnr",
            sweep_values=list(cfg.eval_cnr_list),
            num_samples=15,
            save_root=cfg.data_root,
        )
        cnr_results_all[config_name] = cnr_results
        plot_radar_sweep(cnr_results, os.path.join(cfg_eval_dir, "radar_cnr_vs_cfar.png"))

        rcs_results = trainer.evaluate_against_cfar_sweep(
            config_name=config_name,
            sweep_type="rcs",
            sweep_values=list(cfg.eval_rcs_list),
            num_samples=15,
            save_root=cfg.data_root,
        )
        rcs_results_all[config_name] = rcs_results
        plot_radar_sweep(rcs_results, os.path.join(cfg_eval_dir, "radar_rcs_vs_cfar.png"))

    plot_radar_sweep_all(snr_results_all, "snr", os.path.join(eval_dir, "radar_snr_all_configs.png"))
    plot_radar_sweep_all(cnr_results_all, "cnr", os.path.join(eval_dir, "radar_cnr_all_configs.png"))
    plot_radar_sweep_all(rcs_results_all, "rcs", os.path.join(eval_dir, "radar_rcs_all_configs.png"))

    summary = {
        "checkpoint": best_path,
        "param_count": count_parameters(model),
        "seen_configs": seen_configs,
        "unseen_configs": unseen_configs,
        "history": history,
        "seen": final_seen,
        "unseen": final_unseen,
        "snr_sweeps": snr_results_all,
        "cnr_sweeps": cnr_results_all,
        "rcs_sweeps": rcs_results_all,
    }
    save_json(summary, os.path.join(out_dir, "summary.json"))
    return summary


def create_comm_model(model_name: str, device: torch.device) -> Tuple[nn.Module, bool]:
    model_name = model_name.lower()
    if model_name == "adaptive":
        return AdaptiveCommNet(base_ch=64, cond_dim=64).to(device), True
    elif model_name == "bit":
        return CommBitNet(in_channels=5, hidden_dim=256, cond_dim=64).to(device), False
    elif model_name == "isac":
        return ISACFoundationModel(comm_in_channels=5, base_ch=64, cond_dim=64, config_dim=8).to(device), False
    raise ValueError(f"Unsupported comm model: {model_name}")


def train_comm_pipeline(cfg: RunConfig, device: torch.device, model_name: str = "adaptive") -> Dict[str, Any]:
    out_dir = ensure_dir(os.path.join(cfg.out_dir, "comm"))
    train_sets, val_seen_sets, test_unseen_sets, seen_configs, unseen_configs = build_comm_datasets(cfg)

    # Balanced sampling over configs. ConcatDataset entries already wrapped for mixed-channel case.
    if all(isinstance(ds, ConcatDataset) for ds in train_sets.values()):
        wrapped = []
        for cfg_name, ds in train_sets.items():
            wrapped.append(ds)
        concat_ds = ConcatDataset(wrapped)
        sampler = HomogeneousBatchSampler(wrapped, cfg.batch_size, shuffle=True)
        train_loader = DataLoader(concat_ds, batch_sampler=sampler, collate_fn=comm_collate, num_workers=cfg.num_workers)
    else:
        train_loader = build_balanced_concat_loader(train_sets, cfg.batch_size, comm_collate, cfg.num_workers, shuffle=True)

    val_seen_loaders = build_eval_loaders(val_seen_sets, cfg.batch_size, comm_collate, cfg.num_workers)
    test_unseen_loaders = build_eval_loaders(test_unseen_sets, cfg.batch_size, comm_collate, cfg.num_workers)

    model, symbol_logits_mode = create_comm_model(model_name, device)
    if cfg.comm_ckpt:
        load_checkpoint_if_exists(model, cfg.comm_ckpt, device)

    trainer = CommTrainer(model, device, symbol_logits_mode=symbol_logits_mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best = {"ber": float("inf"), "epoch": -1}
    best_path = os.path.join(out_dir, "comm_best.pt")
    patience = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_stats = trainer.train_one_epoch(
            train_loader,
            optimizer,
            label_smoothing=cfg.label_smoothing,
            use_focal_loss=cfg.use_focal_loss and (not symbol_logits_mode),
        )
        val_seen = trainer.evaluate(val_seen_loaders)
        val_ber = val_seen["avg_ber"]
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_ber": train_stats["ber"],
            "val_seen_ber": val_ber,
            "per_config_train": train_stats["per_config"],
            "per_config_val": val_seen["per_config"],
        })

        print(f"[Comm][Epoch {epoch:03d}] loss={train_stats['loss']:.4f} train_ber={train_stats['ber']:.4e} val_seen_ber={val_ber:.4e}")

        if val_ber < best["ber"]:
            best = {"ber": val_ber, "epoch": epoch}
            torch.save({"model": model.state_dict(), "best": best, "seen_configs": seen_configs, "symbol_logits_mode": symbol_logits_mode}, best_path)
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"[Comm] Early stopping at epoch {epoch}")
                break

    load_checkpoint_if_exists(model, best_path, device)
    final_seen = trainer.evaluate(val_seen_loaders)
    final_unseen = trainer.evaluate(test_unseen_loaders) if test_unseen_loaders else {"avg_ber": 1.0, "per_config": {}}

    eval_dir = ensure_dir(os.path.join(out_dir, "eval"))
    snr_reports = {}
    for config_name in seen_configs:
        cfg_eval_dir = ensure_dir(os.path.join(eval_dir, config_name))
        for channel_mode in ["awgn", "realistic"]:
            key = f"{config_name}_{channel_mode}"
            print(f"[Comm] Evaluating SNR sweep for {key}")
            res = trainer.evaluate_snr_sweep(
                config_name=config_name,
                save_root=cfg.data_root,
                snr_list=list(cfg.eval_snr_list),
                num_samples=15,
                channel_mode=channel_mode,
            )
            snr_reports[key] = res
            plot_comm_sweep(res, os.path.join(cfg_eval_dir, f"ber_snr_{channel_mode}.png"), title=f"{config_name} - {channel_mode}")

    plot_comm_sweep_all({k: v for k, v in snr_reports.items() if "awgn" in k}, os.path.join(eval_dir, "comm_ber_snr_awgn_all_configs.png"), title="BER vs SNR (AWGN) - All Configs")
    plot_comm_sweep_all({k: v for k, v in snr_reports.items() if "realistic" in k}, os.path.join(eval_dir, "comm_ber_snr_realistic_all_configs.png"), title="BER vs SNR (Realistic) - All Configs")

    summary = {
        "checkpoint": best_path,
        "param_count": count_parameters(model),
        "seen_configs": seen_configs,
        "unseen_configs": unseen_configs,
        "history": history,
        "seen": final_seen,
        "unseen": final_unseen,
        "snr_sweeps": snr_reports,
        "symbol_logits_mode": symbol_logits_mode,
        "model_name": model_name,
    }
    save_json(summary, os.path.join(out_dir, "summary.json"))
    return summary


def run_full_evaluation(cfg: RunConfig, device: torch.device, radar_model_name: str = "generalized", comm_model_name: str = "adaptive") -> Dict[str, Any]:
    ensure_dir(cfg.out_dir)
    summary = {"run_config": asdict(cfg)}

    radar_summary_path = os.path.join(cfg.out_dir, "radar", "summary.json")
    comm_summary_path = os.path.join(cfg.out_dir, "comm", "summary.json")

    if os.path.exists(radar_summary_path):
        with open(radar_summary_path, "r", encoding="utf-8") as f:
            summary["radar"] = json.load(f)
        
        # Regenerate radar plots from loaded summary if eval_all is called
        radar_eval_dir = ensure_dir(os.path.join(cfg.out_dir, "radar", "eval"))
        snr_results_all = summary["radar"].get("snr_sweeps", {})
        cnr_results_all = summary["radar"].get("cnr_sweeps", {})
        rcs_results_all = summary["radar"].get("rcs_sweeps", {})
        
        for config_name in snr_results_all.keys():
            cfg_eval_dir = ensure_dir(os.path.join(radar_eval_dir, config_name))
            if config_name in snr_results_all:
                plot_radar_sweep(snr_results_all[config_name], os.path.join(cfg_eval_dir, "radar_snr_vs_cfar.png"))
            if config_name in cnr_results_all:
                plot_radar_sweep(cnr_results_all[config_name], os.path.join(cfg_eval_dir, "radar_cnr_vs_cfar.png"))
            if config_name in rcs_results_all:
                plot_radar_sweep(rcs_results_all[config_name], os.path.join(cfg_eval_dir, "radar_rcs_vs_cfar.png"))
                
        if snr_results_all:
            plot_radar_sweep_all(snr_results_all, "snr", os.path.join(radar_eval_dir, "radar_snr_all_configs.png"))
        if cnr_results_all:
            plot_radar_sweep_all(cnr_results_all, "cnr", os.path.join(radar_eval_dir, "radar_cnr_all_configs.png"))
        if rcs_results_all:
            plot_radar_sweep_all(rcs_results_all, "rcs", os.path.join(radar_eval_dir, "radar_rcs_all_configs.png"))
    else:
        summary["radar"] = train_radar_pipeline(cfg, device, model_name=radar_model_name)

    if os.path.exists(comm_summary_path):
        with open(comm_summary_path, "r", encoding="utf-8") as f:
            summary["comm"] = json.load(f)
            
        # Regenerate comm plots from loaded summary if eval_all is called
        comm_eval_dir = ensure_dir(os.path.join(cfg.out_dir, "comm", "eval"))
        snr_reports = summary["comm"].get("snr_sweeps", {})
        
        for key, res in snr_reports.items():
            # key is format: {config_name}_{channel_mode}
            parts = key.rsplit('_', 1)
            if len(parts) == 2:
                config_name, channel_mode = parts
                cfg_eval_dir = ensure_dir(os.path.join(comm_eval_dir, config_name))
                plot_comm_sweep(res, os.path.join(cfg_eval_dir, f"ber_snr_{channel_mode}.png"), title=f"{config_name} - {channel_mode}")
                
        awgn_reports = {k: v for k, v in snr_reports.items() if "awgn" in k}
        realistic_reports = {k: v for k, v in snr_reports.items() if "realistic" in k}
        
        if awgn_reports:
            plot_comm_sweep_all(awgn_reports, os.path.join(comm_eval_dir, "comm_ber_snr_awgn_all_configs.png"), title="BER vs SNR (AWGN) - All Configs")
        if realistic_reports:
            plot_comm_sweep_all(realistic_reports, os.path.join(comm_eval_dir, "comm_ber_snr_realistic_all_configs.png"), title="BER vs SNR (Realistic) - All Configs")
    else:
        summary["comm"] = train_comm_pipeline(cfg, device, model_name=comm_model_name)

    save_json(summary, os.path.join(cfg.out_dir, "full_summary.json"))
    generate_markdown_report(summary, os.path.join(cfg.out_dir, "REPORT.md"))
    return summary


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Refactored G3 radar/comm training and evaluation")
    parser.add_argument("--mode", choices=["train_radar", "train_comm", "eval_all"], default="eval_all")
    parser.add_argument("--out_dir", type=str, default="data/g6")
    parser.add_argument("--data_root", type=str, default="data/g6")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_samples", type=int, default=300)
    parser.add_argument("--val_samples", type=int, default=60)
    parser.add_argument("--test_samples", type=int, default=60)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--radar_type", choices=["FMCW", "OTFS", "all"], default="FMCW")
    parser.add_argument("--comm_type", choices=["OFDM", "OTFS", "all"], default="OFDM")
    parser.add_argument("--qam_type", choices=["4QAM", "8QAM", "16QAM", "64QAM", "all"], default="4QAM")
    parser.add_argument("--channel_mode", choices=["awgn", "realistic"], default="realistic")
    parser.add_argument("--radar_ckpt", type=str, default=None)
    parser.add_argument("--comm_ckpt", type=str, default=None)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--radar_pos_weight", type=float, default=5.0)
    parser.add_argument("--radar_sigma", type=float, default=3.0)
    parser.add_argument("--early_stop_patience", type=int, default=8)
    parser.add_argument("--unseen_holdout_fraction", type=float, default=0.34)
    parser.add_argument("--radar_model", choices=["generalized", "robustg3", "crnn", "transformer", "dualpath", "isac"], default="isac")
    parser.add_argument("--comm_model", choices=["adaptive", "bit", "isac"], default="isac")
    args = parser.parse_args()
    return RunConfig(**vars(args))


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    ensure_dir(cfg.out_dir)
    ensure_dir(cfg.data_root)

    print("=" * 80)
    print("G3 REFACTORED PIPELINE")
    print(json.dumps(asdict(cfg), indent=2))
    print("=" * 80)

    t0 = time.time()
    if cfg.mode == "train_radar":
        train_radar_pipeline(cfg, device, model_name=cfg.radar_model)
    elif cfg.mode == "train_comm":
        train_comm_pipeline(cfg, device, model_name=cfg.comm_model)
    else:
        run_full_evaluation(cfg, device, radar_model_name=cfg.radar_model, comm_model_name=cfg.comm_model)

    print(f"Done in {(time.time() - t0) / 60.0:.2f} minutes")


if __name__ == "__main__":
    main()

"""
# 1. Train Radar
python AIRadar/AIradar_comm_model_g6.py --mode train_radar --radar_type all --comm_type all --qam_type all --radar_model isac --comm_model isac --out_dir data/g6_full --data_root data/g6_full

# 2. Train Comm
python AIRadar/AIradar_comm_model_g6.py --mode train_comm --radar_type all --comm_type all --qam_type all --radar_model isac --comm_model isac --out_dir data/g6_full --data_root data/g6_full

# 3. Evaluate All (Generates REPORT.md)
python AIRadar/AIradar_comm_model_g6.py --mode eval_all --radar_type all --comm_type all --qam_type all --radar_model isac --comm_model isac --out_dir data/g6_full --data_root data/g6_full


data/g6_full/
├── radar/
│   ├── radar_best.pt
│   ├── summary.json
│   └── eval/
│       ├── CN0566_TRADITIONAL/
│       │   ├── radar_snr_vs_cfar.png
│       │   ├── radar_cnr_vs_cfar.png
│       │   └── radar_rcs_vs_cfar.png
│       ├── Automotive_77GHz_LongRange/
│       │   └── ...
│       ├── radar_snr_all_configs.png   <-- Master comparison figure
│       ├── radar_cnr_all_configs.png
│       └── radar_rcs_all_configs.png
├── comm/
│   ├── comm_best.pt
│   ├── summary.json
│   └── eval/
│       ├── CN0566_TRADITIONAL/
│       │   ├── ber_snr_awgn.png
│       │   └── ber_snr_realistic.png
│       ├── Automotive_77GHz_LongRange/
│       │   └── ...
│       ├── comm_ber_snr_awgn_all_configs.png       <-- Master comparison figure
│       └── comm_ber_snr_realistic_all_configs.png

#new full running
# Full Radar Training
python AIRadar/AIradar_comm_model_g6.py --mode train_radar --radar_type all --comm_type all --qam_type all --radar_model isac --comm_model isac --out_dir data/g6_final --data_root data/g6_final --train_samples 1000 --val_samples 200 --test_samples 200 --epochs 50

# Full Comm Training
python AIRadar/AIradar_comm_model_g6.py --mode train_comm --radar_type all --comm_type all --qam_type all --radar_model isac --comm_model isac --out_dir data/g6_final --data_root data/g6_final --train_samples 1000 --val_samples 200 --test_samples 200 --epochs 50

#results in data/g6_final
"""
#!/usr/bin/env python3
"""
ISAC Foundation Model v2

Recommended unified architecture for FMCW / OFDM / OTFS that avoids the main
failure mode of naive joint training: hard sharing all high-level features.

Core idea
---------
1. Modality-specific input stem
   - Radar branch: accepts radar tensors such as range-Doppler maps or other
     radar feature maps.
   - Communication branch: accepts communication tensors such as equalized IQ,
     channel magnitude/phase, SNR maps, or OTFS delay-Doppler grids.

2. Shared physics stem
   - Learns low-level signal structures that can be shared across tasks and
     waveform families: local spectral patterns, amplitude/phase consistency,
     noise/interference signatures, and hardware distortion patterns.

3. Task-specific adapters + experts
   - Radar and communication do NOT fully share the mid/high-level stack.
   - This reduces gradient interference while still allowing a single model
     family and shared low-level representation.

4. Explicit conditioning
   - configuration tensor, waveform id, task id, modulation order, and SNR can
     be injected via FiLM-style conditioning.

This file focuses on model definition, tensor shapes, and forward contracts.
It does not include the full training loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Constants and ID conventions
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
    "NONE": 0,
    "QPSK": 1,
    "QAM8": 2,
    "QAM16": 3,
    "QAM64": 4,
}


# =============================================================================
# Configuration dataclasses for shape clarity
# =============================================================================


@dataclass
class RadarForwardSpec:
    """
    Radar input contract.

    x shape:
        [B, C_r, H, W]

    Typical examples:
        FMCW range-Doppler map:
            C_r = 1 or small number of channels
            H = range bins
            W = Doppler bins

        OTFS radar map / delay-Doppler feature map:
            C_r = 1..4
            H = delay/range bins
            W = Doppler bins

    output shape:
        radar_logits: [B, 1, H, W]
            dense heatmap / segmentation style detection output
    """
    in_channels: int = 1


@dataclass
class CommForwardSpec:
    """
    Communication input contract.

    x shape:
        [B, C_c, H, W]

    Typical examples:
        OFDM feature grid:
            C_c = 5
            channels = [eq_real, eq_imag, H_mag, H_phase, snr]
            H = num_symbols
            W = num_subcarriers

        OTFS feature grid:
            C_c = 5..8
            channels may include [real, imag, H_mag, H_phase, snr, delay mask, doppler mask, ...]
            H = Doppler bins or symbols
            W = delay bins or subcarriers

    output shape:
        bit_logits: [B, max_bits, H, W]
            max_bits typically = 6 to support up to 64-QAM
    """
    in_channels: int = 5
    max_bits: int = 6


# =============================================================================
# Small utility blocks
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_gn(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1,
                 p: int | None = None, padding_mode: str = "zeros"):
        super().__init__()
        if p is None:
            p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p,
                      bias=False, padding_mode=padding_mode),
            _make_gn(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, padding_mode: str = "zeros", scale: float = 0.1):
        super().__init__()
        self.scale = scale
        self.conv1 = ConvNormAct(channels, channels, k=3, padding_mode=padding_mode)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False, padding_mode=padding_mode),
            _make_gn(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        return self.act(x + self.scale * y)


class MLP(nn.Module):
    def __init__(self, dims, dropout: float = 0.0):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionEncoderV2(nn.Module):
    """
    Better condition encoder:
    - continuous config is preserved
    - discrete ids do not overwrite config information
    - gated fusion instead of naive summation
    """
    def __init__(
        self,
        config_dim: int = 8,
        cond_dim: int = 128,
        num_waveforms: int = 3,
        num_tasks: int = 2,
        num_modulations: int = 5,
    ):
        super().__init__()
        self.config_branch = MLP([config_dim, cond_dim, cond_dim], dropout=0.0)

        self.wave_embed = nn.Embedding(num_waveforms, cond_dim // 2)
        self.task_embed = nn.Embedding(num_tasks, cond_dim // 4)
        self.mod_embed = nn.Embedding(num_modulations, cond_dim // 4)

        disc_dim = cond_dim // 2 + cond_dim // 4 + cond_dim // 4
        self.disc_branch = MLP([disc_dim, cond_dim, cond_dim], dropout=0.0)

        self.fuse = MLP([cond_dim * 2, cond_dim, cond_dim], dropout=0.0)
        self.gate = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        config_tensor: torch.Tensor,
        waveform_id: torch.Tensor,
        task_id: torch.Tensor,
        mod_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg_feat = self.config_branch(config_tensor)

        if mod_id is None:
            mod_id = torch.zeros_like(task_id)

        disc_feat = torch.cat([
            self.wave_embed(waveform_id),
            self.task_embed(task_id),
            self.mod_embed(mod_id),
        ], dim=-1)
        disc_feat = self.disc_branch(disc_feat)

        fused_in = torch.cat([cfg_feat, disc_feat], dim=-1)
        fused = self.fuse(fused_in)
        gate = self.gate(fused_in)

        # Keep config branch alive; do not let discrete embeddings dominate.
        cond = gate * cfg_feat + (1.0 - gate) * fused
        return cond


class FiLM2d(nn.Module):
    """
    Feature-wise affine modulation for 2D feature maps.

    Input:
        x:    [B, C, H, W]
        cond: [B, D]
    Output:
        y:    [B, C, H, W]
    """
    def __init__(self, channels: int, cond_dim: int, gamma_limit: float = 0.5):
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, channels * 2)
        self.gamma_limit = gamma_limit

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.to_gamma_beta(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = torch.tanh(gamma) * self.gamma_limit
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class ConditionEncoder(nn.Module):
    """
    Encodes physical config + waveform/task/modulation IDs into one condition vector.

    Inputs:
        config_tensor: [B, config_dim]
        waveform_id:   [B]
        task_id:       [B]
        mod_id:        [B] or None

    Output:
        cond:          [B, cond_dim]
    """
    def __init__(
        self,
        config_dim: int = 8,
        cond_dim: int = 128,
        num_waveforms: int = 3,
        num_tasks: int = 2,
        num_modulations: int = 5,
    ):
        super().__init__()
        self.config_mlp = MLP([config_dim, cond_dim, cond_dim], dropout=0.0)
        self.wave_embed = nn.Embedding(num_waveforms, cond_dim)
        self.task_embed = nn.Embedding(num_tasks, cond_dim)
        self.mod_embed = nn.Embedding(num_modulations, cond_dim)
        self.fuse = MLP([cond_dim, cond_dim], dropout=0.0)

    def forward(
        self,
        config_tensor: torch.Tensor,
        waveform_id: torch.Tensor,
        task_id: torch.Tensor,
        mod_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond = self.config_mlp(config_tensor)
        cond = cond + self.wave_embed(waveform_id) + self.task_embed(task_id)
        if mod_id is not None:
            cond = cond + self.mod_embed(mod_id)
        return self.fuse(cond)


# =============================================================================
# Input encoders
# =============================================================================


class RadarInputEncoder(nn.Module):
    """
    Radar-specific input stem.

    Input:
        x: [B, C_r, H, W]
        waveform_id: [B]
    Output:
        z: [B, stem_ch, H, W]
    """
    def __init__(self, in_channels: int = 1, stem_ch: int = 64):
        super().__init__()
        self.stem_ch = stem_ch
        self.net = nn.Sequential(
            ConvNormAct(in_channels, stem_ch // 2, k=3),
            ConvNormAct(stem_ch // 2, stem_ch, k=3),
            ResidualConvBlock(stem_ch),
        )
        self.otfs_net = nn.Sequential(
            ConvNormAct(in_channels, stem_ch // 2, k=3, padding_mode='circular'),
            ConvNormAct(stem_ch // 2, stem_ch, k=5, padding_mode='circular'),
            ResidualConvBlock(stem_ch, padding_mode='circular'),
        )

    def forward(self, x: torch.Tensor, waveform_id: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.shape[0], self.stem_ch, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        is_otfs = (waveform_id == 2)
        is_other = ~is_otfs
        
        if is_otfs.any():
            # Apply 2D FFT magnitude to extract robust features for OTFS delay-doppler maps
            x_otfs = x[is_otfs]
            fft_feat = torch.fft.fft2(x_otfs, norm="ortho")
            x_otfs_combined = torch.abs(fft_feat) + x_otfs
            out[is_otfs] = self.otfs_net(x_otfs_combined)
        if is_other.any():
            out[is_other] = self.net(x[is_other])
            
        return out


class CommInputEncoder(nn.Module):
    """
    Communication-specific input stem.

    Input:
        x: [B, C_c, H, W]
    Output:
        z: [B, stem_ch, H, W]
    """
    def __init__(self, in_channels: int = 5, stem_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct(in_channels, stem_ch // 2, k=1),
            ConvNormAct(stem_ch // 2, stem_ch, k=3),
            ResidualConvBlock(stem_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CommInputEncoderV2(nn.Module):
    """
    Separate OFDM and OTFS input stems.
    OFDM: emphasize local symbol/subcarrier neighborhoods.
    OTFS: emphasize 2D delay-doppler structure.
    """
    def __init__(self, in_channels: int = 5, stem_ch: int = 64):
        super().__init__()
        self.stem_ch = stem_ch

        self.ofdm_net = nn.Sequential(
            ConvNormAct(in_channels, stem_ch // 2, k=1, padding_mode="zeros"),
            ConvNormAct(stem_ch // 2, stem_ch, k=3, padding_mode="zeros"),
            ResidualConvBlock(stem_ch, padding_mode="zeros"),
        )

        self.otfs_net = nn.Sequential(
            ConvNormAct(in_channels, stem_ch // 2, k=3, padding_mode="circular"),
            ConvNormAct(stem_ch // 2, stem_ch, k=5, padding_mode="circular"),
            ResidualConvBlock(stem_ch, padding_mode="circular"),
        )

    def forward(self, x: torch.Tensor, waveform_id: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        out = torch.zeros(b, self.stem_ch, h, w, device=x.device, dtype=x.dtype)

        is_otfs = waveform_id == 2
        is_ofdm = ~is_otfs

        if is_ofdm.any():
            out[is_ofdm] = self.ofdm_net(x[is_ofdm])

        if is_otfs.any():
            x_otfs = x[is_otfs]
            # Optional: lightweight spectral enhancement
            fft_feat = torch.fft.fft2(x_otfs[:, :2], norm="ortho")
            amp = torch.abs(fft_feat).mean(dim=1, keepdim=True)
            x_aug = x_otfs.clone()
            x_aug[:, :1] = x_aug[:, :1] + amp
            out[is_otfs] = self.otfs_net(x_aug)

        return out
# =============================================================================
# Shared physics stem
# =============================================================================


class SharedPhysicsStem(nn.Module):
    """
    Shared low-level representation learner.

    This is the part that can reasonably be shared between radar and communication.
    It should learn waveform-agnostic, physics-aware low-level features rather than
    highly task-specific semantic features.

    Input:
        x:    [B, C, H, W]
        cond: [B, D]
    Output:
        f1:   [B, base_ch, H, W]
        f2:   [B, 2*base_ch, H/2, W/2]
        f3:   [B, 4*base_ch, H/4, W/4]
    """
    def __init__(self, in_ch: int = 64, base_ch: int = 64, cond_dim: int = 128):
        super().__init__()
        self.block1 = nn.Sequential(
            ConvNormAct(in_ch, base_ch, k=3),
            ResidualConvBlock(base_ch),
        )
        self.block2 = nn.Sequential(
            ConvNormAct(base_ch, base_ch * 2, k=3, s=2),
            ResidualConvBlock(base_ch * 2),
        )
        self.block3 = nn.Sequential(
            ConvNormAct(base_ch * 2, base_ch * 4, k=3, s=2),
            ResidualConvBlock(base_ch * 4),
        )

        self.film1 = FiLM2d(base_ch, cond_dim)
        self.film2 = FiLM2d(base_ch * 2, cond_dim)
        self.film3 = FiLM2d(base_ch * 4, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f1 = self.block1(x)              # [B, C, H, W]
        f1 = self.film1(f1, cond)

        f2 = self.block2(f1)            # [B, 2C, H/2, W/2]
        f2 = self.film2(f2, cond)

        f3 = self.block3(f2)            # [B, 4C, H/4, W/4]
        f3 = self.film3(f3, cond)
        return f1, f2, f3


# =============================================================================
# Task adapters
# =============================================================================


class TaskAdapter(nn.Module):
    """
    Lightweight adapter to reduce task conflict after shared stem.

    Input:
        x:    [B, C, H, W]
        cond: [B, D]
    Output:
        y:    [B, C, H, W]
    """
    def __init__(self, channels: int, bottleneck_ratio: int = 4, cond_dim: int = 128):
        super().__init__()
        hidden = max(8, channels // bottleneck_ratio)
        self.down = nn.Conv2d(channels, hidden, kernel_size=1)
        self.up = nn.Conv2d(hidden, channels, kernel_size=1)
        self.act = nn.GELU()
        self.film = FiLM2d(channels, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y = self.down(x)
        y = self.act(y)
        y = self.up(y)
        y = self.film(y, cond)
        return x + 0.1 * y


# =============================================================================
# Radar expert and head
# =============================================================================


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.fuse = nn.Sequential(
            ConvNormAct(out_ch + skip_ch, out_ch, k=3),
            ResidualConvBlock(out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


class RadarExpert(nn.Module):
    """
    Radar-specific decoder.

    Inputs:
        f1:   [B, C,   H,   W]
        f2:   [B, 2C,  H/2, W/2]
        f3:   [B, 4C,  H/4, W/4]
        cond: [B, D]

    Output:
        z:    [B, C, H, W]
    """
    def __init__(self, base_ch: int = 64, cond_dim: int = 128):
        super().__init__()
        self.adapter3 = TaskAdapter(base_ch * 4, cond_dim=cond_dim)
        self.adapter2 = TaskAdapter(base_ch * 2, cond_dim=cond_dim)
        self.adapter1 = TaskAdapter(base_ch, cond_dim=cond_dim)

        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch)

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        f3 = self.adapter3(f3, cond)
        f2 = self.adapter2(f2, cond)
        f1 = self.adapter1(f1, cond)
        z = self.up2(f3, f2)           # [B, 2C, H/2, W/2]
        z = self.up1(z, f1)            # [B, C, H, W]
        return z


class RadarHead(nn.Module):
    """
    Output radar dense detection logits.

    Input:
        z: [B, C, H, W]
    Output:
        radar_logits: [B, 1, H, W]
    """
    def __init__(self, in_ch: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            ConvNormAct(in_ch, in_ch, k=3),
            nn.Conv2d(in_ch, 1, kernel_size=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


# =============================================================================
# Communication expert and head
# =============================================================================


class CommExpert(nn.Module):
    """
    Communication-specific feature processor.

    Inputs:
        f1:   [B, C,   H,   W]
        f2:   [B, 2C,  H/2, W/2]
        f3:   [B, 4C,  H/4, W/4]
        cond: [B, D]

    Output:
        z:    [B, C, H, W]

    Design choice:
        Communication branch keeps more local symbol-grid structure and avoids
        a full U-Net style semantic decoder. We fuse multi-scale features back
        to full grid resolution.
    """
    def __init__(self, base_ch: int = 64, cond_dim: int = 128):
        super().__init__()
        self.adapter3 = TaskAdapter(base_ch * 4, cond_dim=cond_dim)
        self.adapter2 = TaskAdapter(base_ch * 2, cond_dim=cond_dim)
        self.adapter1 = TaskAdapter(base_ch, cond_dim=cond_dim)

        self.proj3 = nn.Conv2d(base_ch * 4, base_ch, kernel_size=1)
        self.proj2 = nn.Conv2d(base_ch * 2, base_ch, kernel_size=1)
        self.proj1 = nn.Conv2d(base_ch, base_ch, kernel_size=1)

        self.fuse = nn.Sequential(
            ConvNormAct(base_ch * 3, base_ch, k=3),
            ResidualConvBlock(base_ch),
            ResidualConvBlock(base_ch),
        )

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        f3 = self.adapter3(f3, cond)
        f2 = self.adapter2(f2, cond)
        f1 = self.adapter1(f1, cond)

        p1 = self.proj1(f1)   # [B, C, H, W]
        p2 = self.proj2(f2)   # [B, C, H/2, W/2]
        p3 = self.proj3(f3)   # [B, C, H/4, W/4]

        p2 = F.interpolate(p2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        p3 = F.interpolate(p3, size=f1.shape[-2:], mode="bilinear", align_corners=False)

        z = torch.cat([p1, p2, p3], dim=1)  # [B, 3C, H, W]
        z = self.fuse(z)                     # [B, C, H, W]
        return z


class CommHead(nn.Module):
    """
    Output bit logits for demapping.

    Input:
        z: [B, C, H, W]
    Output:
        bit_logits: [B, max_bits, H, W]

    Notes:
        - max_bits = 6 supports up to 64-QAM.
        - At training/eval time, only the first log2(M) channels are active.
    """
    def __init__(self, in_ch: int = 64, hidden_ch: int = 64, max_bits: int = 6):
        super().__init__()
        self.max_bits = max_bits
        self.head = nn.Sequential(
            ConvNormAct(in_ch, hidden_ch, k=3),
            nn.Conv2d(hidden_ch, max_bits, kernel_size=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)

    @staticmethod
    def bit_logits_to_symbol_logits(bit_logits: torch.Tensor, mod_order: int) -> torch.Tensor:
        """
        Convert bit logits to symbol logits in a numerically stable way.

        Input:
            bit_logits: [B, max_bits, H, W]
        Output:
            symbol_logits: [B, mod_order, H, W]
        """
        active_bits = int(round(math.log2(mod_order)))
        active = bit_logits[:, :active_bits]           # [B, active_bits, H, W]
        log_p1 = F.logsigmoid(active)
        log_p0 = F.logsigmoid(-active)

        b, _, h, w = active.shape
        symbol_logits = torch.zeros(b, mod_order, h, w, device=bit_logits.device, dtype=bit_logits.dtype)

        for sym in range(mod_order):
            log_prob = 0.0
            for bit_idx in range(active_bits):
                bit_val = (sym >> bit_idx) & 1
                log_prob = log_prob + (log_p1[:, bit_idx] if bit_val == 1 else log_p0[:, bit_idx])
            symbol_logits[:, sym] = log_prob
        return symbol_logits

MOD_ID_TO_ORDER = {
    1: 4,   # QPSK / 4QAM
    2: 8,   # 8QAM
    3: 16,  # 16QAM
    4: 64,  # 64QAM
}


class CommAdapterHead(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, out_bits: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct(in_ch, hidden_ch, k=3),
            ResidualConvBlock(hidden_ch),
            nn.Conv2d(hidden_ch, out_bits, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModulationAwareCommHead(nn.Module):
    """
    Shared pre-head + modulation-specific adapter heads.
    Output always [B, max_bits, H, W], but inactive bits are zero padded.
    """
    def __init__(self, in_ch: int = 64, hidden_ch: int = 64, max_bits: int = 6):
        super().__init__()
        self.max_bits = max_bits

        self.shared = nn.Sequential(
            ConvNormAct(in_ch, hidden_ch, k=3),
            ResidualConvBlock(hidden_ch),
        )

        self.head_4 = CommAdapterHead(hidden_ch, hidden_ch, out_bits=2)
        self.head_8 = CommAdapterHead(hidden_ch, hidden_ch, out_bits=3)
        self.head_16 = CommAdapterHead(hidden_ch, hidden_ch, out_bits=4)
        self.head_64 = CommAdapterHead(hidden_ch, hidden_ch, out_bits=6)

    def forward(self, z: torch.Tensor, mod_id: torch.Tensor) -> torch.Tensor:
        base = self.shared(z)
        b, _, h, w = base.shape
        out = torch.zeros(b, self.max_bits, h, w, device=base.device, dtype=base.dtype)

        for mid, order in MOD_ID_TO_ORDER.items():
            mask = (mod_id == mid)
            if not mask.any():
                continue

            feat = base[mask]
            if order == 4:
                logits = self.head_4(feat)
            elif order == 8:
                logits = self.head_8(feat)
            elif order == 16:
                logits = self.head_16(feat)
            else:
                logits = self.head_64(feat)

            out[mask, :logits.shape[1]] = logits

        return out

    @staticmethod
    def bit_logits_to_symbol_logits(bit_logits: torch.Tensor, mod_order: int) -> torch.Tensor:
        active_bits = int(round(math.log2(mod_order)))
        active = bit_logits[:, :active_bits]
        log_p1 = F.logsigmoid(active)
        log_p0 = F.logsigmoid(-active)

        b, _, h, w = active.shape
        symbol_logits = torch.zeros(b, mod_order, h, w, device=bit_logits.device, dtype=bit_logits.dtype)

        for sym in range(mod_order):
            log_prob = 0.0
            for bit_idx in range(active_bits):
                bit_val = (sym >> bit_idx) & 1
                log_prob = log_prob + (log_p1[:, bit_idx] if bit_val == 1 else log_p0[:, bit_idx])
            symbol_logits[:, sym] = log_prob

        return symbol_logits

# =============================================================================
# Main unified model
# =============================================================================


class ISACFoundationModel(nn.Module):
    """
    Unified model family with shared low-level stem and task-specific experts.

    This is NOT a naive fully shared backbone.
    Shared part: low-level physics stem only.
    Separate part: task adapters + experts + heads.

    ----------------------
    Radar forward contract
    ----------------------
    Inputs:
        x_radar:       [B, C_r, H, W]
        config_tensor: [B, config_dim]
        waveform_id:   [B]            e.g. FMCW=0, OTFS=2

    Output dict:
        {
            "radar_logits": [B, 1, H, W],
            "shared_f1":    [B, base_ch, H, W],
            "shared_f2":    [B, 2*base_ch, H/2, W/2],
            "shared_f3":    [B, 4*base_ch, H/4, W/4],
            "radar_feat":   [B, base_ch, H, W],
            "cond":         [B, cond_dim],
        }

    ------------------------------
    Communication forward contract
    ------------------------------
    Inputs:
        x_comm:        [B, C_c, H, W]
        config_tensor: [B, config_dim]
        waveform_id:   [B]            OFDM=1, OTFS=2
        mod_id:        [B]            e.g. QAM16=3

    Output dict:
        {
            "bit_logits":   [B, max_bits, H, W],
            "shared_f1":    [B, base_ch, H, W],
            "shared_f2":    [B, 2*base_ch, H/2, W/2],
            "shared_f3":    [B, 4*base_ch, H/4, W/4],
            "comm_feat":    [B, base_ch, H, W],
            "cond":         [B, cond_dim],
        }
    """
    def __init__(
        self,
        radar_in_channels: int = 1,
        comm_in_channels: int = 5,
        config_dim: int = 8,
        base_ch: int = 64,
        cond_dim: int = 128,
        max_bits: int = 6,
    ):
        super().__init__()
        self.base_ch = base_ch
        self.cond_dim = cond_dim
        self.max_bits = max_bits

        self.cond_encoder = ConditionEncoder(config_dim=config_dim, cond_dim=cond_dim)

        self.radar_input = RadarInputEncoder(in_channels=radar_in_channels, stem_ch=base_ch)
        self.comm_input = CommInputEncoderV2(in_channels=comm_in_channels, stem_ch=base_ch)

        self.shared_stem = SharedPhysicsStem(in_ch=base_ch, base_ch=base_ch, cond_dim=cond_dim)

        self.radar_expert = RadarExpert(base_ch=base_ch, cond_dim=cond_dim)
        self.comm_expert = CommExpert(base_ch=base_ch, cond_dim=cond_dim)

        self.radar_head = RadarHead(in_ch=base_ch)
        self.comm_head = ModulationAwareCommHead(in_ch=base_ch, hidden_ch=base_ch, max_bits=max_bits)

    def forward_radar(
        self,
        x_radar: torch.Tensor,
        config_tensor: torch.Tensor,
        waveform_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_radar:       [B, C_r, H, W]
            config_tensor: [B, config_dim]
            waveform_id:   [B]

        Returns dict with shapes:
            radar_logits: [B, 1, H, W]
            shared_f1:    [B, C, H, W]
            shared_f2:    [B, 2C, H/2, W/2]
            shared_f3:    [B, 4C, H/4, W/4]
            radar_feat:   [B, C, H, W]
            cond:         [B, D]
        """
        b = x_radar.shape[0]
        task_id = torch.full((b,), TASK_IDS["RADAR"], device=x_radar.device, dtype=torch.long)
        cond = self.cond_encoder(config_tensor, waveform_id, task_id, mod_id=None)   # [B, D]

        x0 = self.radar_input(x_radar, waveform_id)                                   # [B, C, H, W]
        f1, f2, f3 = self.shared_stem(x0, cond)                                       # multi-scale shared features
        radar_feat = self.radar_expert(f1, f2, f3, cond)                              # [B, C, H, W]
        radar_logits = self.radar_head(radar_feat)                                    # [B, 1, H, W]

        if radar_logits.shape[-2:] != x_radar.shape[-2:]:
            radar_logits = F.interpolate(radar_logits, size=x_radar.shape[-2:], mode="bilinear", align_corners=False)

        return {
            "radar_logits": radar_logits,
            "shared_f1": f1,
            "shared_f2": f2,
            "shared_f3": f3,
            "radar_feat": radar_feat,
            "cond": cond,
        }

    def forward_comm(
        self,
        x_comm: torch.Tensor,
        config_tensor: torch.Tensor,
        waveform_id: torch.Tensor,
        mod_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        b = x_comm.shape[0]
        task_id = torch.full((b,), TASK_IDS["COMM"], device=x_comm.device, dtype=torch.long)

        cond = self.cond_encoder(
            config_tensor=config_tensor,
            waveform_id=waveform_id,
            task_id=task_id,
            mod_id=mod_id,
        )

        x0 = self.comm_input(x_comm, waveform_id)      # changed
        f1, f2, f3 = self.shared_stem(x0, cond)
        comm_feat = self.comm_expert(f1, f2, f3, cond)
        bit_logits = self.comm_head(comm_feat, mod_id) # changed

        if bit_logits.shape[-2:] != x_comm.shape[-2:]:
            bit_logits = F.interpolate(bit_logits, size=x_comm.shape[-2:], mode="bilinear", align_corners=False)

        return {
            "bit_logits": bit_logits,
            "shared_f1": f1,
            "shared_f2": f2,
            "shared_f3": f3,
            "comm_feat": comm_feat,
            "cond": cond,
        }


# =============================================================================
# Example tensor shapes and usage
# =============================================================================


def _example_usage() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Shared model
    model = ISACFoundationModel(
        radar_in_channels=1,
        comm_in_channels=5,
        config_dim=8,
        base_ch=64,
        cond_dim=128,
        max_bits=6,
    ).to(device)

    # -------------------------------------------------------------------------
    # Example 1: FMCW radar
    # x_radar shape: [B, 1, range_bins, doppler_bins]
    # -------------------------------------------------------------------------
    B = 2
    radar_x = torch.randn(B, 1, 256, 128, device=device)
    radar_config = torch.randn(B, 8, device=device)
    radar_waveform_id = torch.full((B,), WAVEFORM_IDS["FMCW"], dtype=torch.long, device=device)

    radar_out = model.forward_radar(
        x_radar=radar_x,
        config_tensor=radar_config,
        waveform_id=radar_waveform_id,
    )

    print("=== FMCW radar ===")
    for k, v in radar_out.items():
        print(k, tuple(v.shape))
    # Expected:
    # radar_logits: [B, 1, 256, 128]
    # shared_f1:    [B, 64, 256, 128]
    # shared_f2:    [B, 128, 128, 64]
    # shared_f3:    [B, 256, 64, 32]
    # radar_feat:   [B, 64, 256, 128]
    # cond:         [B, 128]

    # -------------------------------------------------------------------------
    # Example 2: OFDM communication
    # x_comm shape: [B, 5, num_symbols, num_subcarriers]
    # channels = [eq_real, eq_imag, H_mag, H_phase, snr]
    # -------------------------------------------------------------------------
    comm_x = torch.randn(B, 5, 14, 64, device=device)
    comm_config = torch.randn(B, 8, device=device)
    comm_waveform_id = torch.full((B,), WAVEFORM_IDS["OFDM"], dtype=torch.long, device=device)
    comm_mod_id = torch.full((B,), MODULATION_IDS["QAM16"], dtype=torch.long, device=device)

    comm_out = model.forward_comm(
        x_comm=comm_x,
        config_tensor=comm_config,
        waveform_id=comm_waveform_id,
        mod_id=comm_mod_id,
    )

    print("=== OFDM comm ===")
    for k, v in comm_out.items():
        print(k, tuple(v.shape))
    # Expected:
    # bit_logits:   [B, 6, 14, 64]
    # shared_f1:    [B, 64, 14, 64]
    # shared_f2:    [B, 128, 7, 32]
    # shared_f3:    [B, 256, 4, 16]  (depends on padding/stride rounding)
    # comm_feat:    [B, 64, 14, 64]
    # cond:         [B, 128]

    # Convert bit logits to 16-QAM symbol logits
    symbol_logits = model.comm_head.bit_logits_to_symbol_logits(comm_out["bit_logits"], mod_order=16)
    print("symbol_logits", tuple(symbol_logits.shape))
    # Expected: [B, 16, 14, 64]

    # -------------------------------------------------------------------------
    # Example 3: OTFS communication
    # x_comm shape can still be [B, C, H, W], e.g. delay-Doppler grid features
    # -------------------------------------------------------------------------
    otfs_x = torch.randn(B, 5, 32, 32, device=device)
    otfs_config = torch.randn(B, 8, device=device)
    otfs_waveform_id = torch.full((B,), WAVEFORM_IDS["OTFS"], dtype=torch.long, device=device)
    otfs_mod_id = torch.full((B,), MODULATION_IDS["QPSK"], dtype=torch.long, device=device)

    otfs_out = model.forward_comm(
        x_comm=otfs_x,
        config_tensor=otfs_config,
        waveform_id=otfs_waveform_id,
        mod_id=otfs_mod_id,
    )

    print("=== OTFS comm ===")
    for k, v in otfs_out.items():
        print(k, tuple(v.shape))


if __name__ == "__main__":
    _example_usage()

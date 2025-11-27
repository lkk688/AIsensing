import torch
import numpy as np
from ..config import DEVICE, C0
from ..utils import to_torch

def fmcw_torch(pts, its, vels, sp):
    """
    Simulate FMCW radar return and process it to generate a Range-Doppler map.
    
    Args:
        pts (torch.Tensor): Point cloud positions (N_points, 3).
        its (torch.Tensor): Intensities (N_points,).
        vels (torch.Tensor): Velocities (N_points, 3).
        sp (SystemParams): System parameters.
        
    Returns:
        tuple: (rd_db,) Range-Doppler map in dB.
    """
    M, N = sp.M, sp.N
    iq = torch.zeros((M, N), dtype=torch.complex64, device=DEVICE)
    
    if len(pts) == 0:
        return (np.zeros((M, N//2)),)

    # Relative position and range
    P = pts - to_torch([0, 0, sp.H])
    R = torch.norm(P, dim=1)
    
    # Filter close points
    mask = R > 0.1
    P, R, vels, its = P[mask], R[mask], vels[mask], its[mask]
    
    # Amplitude model: Intensity / Range^2
    amp = torch.where(its==255, 1e6, 1e-1) / (R**2 + 1e-6)
    
    # Radial velocity: projection of velocity vector onto range vector
    vr  = torch.sum(P/R.unsqueeze(1)*vels, dim=1)

    # FMCW Signal Model
    t_f = torch.arange(N, device=DEVICE) / sp.fs             # fast-time samples within chirp
    t_s = torch.arange(M, device=DEVICE) * sp.T_chirp        # slow-time (chirp index spacing)
    k_r = 2 * sp.slope / C0                                  # beat freq constant = k_r * R
    k_v = 2 / sp.lambda_m                                    # Doppler freq constant = k_v * v

    # Batch processing to avoid OOM
    BATCH = 4096
    for i in range(0, len(R), BATCH):
        rb, vrb, ab = R[i:i+BATCH], vr[i:i+BATCH], amp[i:i+BATCH]
        # Phase = 2*pi * ( (2*Slope*R/c)*t_fast + (2*v/lambda)*t_slow )
        phase = 2j * np.pi * ( (k_r * rb[:, None, None]) * t_f[None, None, :] + 
                               (k_v * vrb[:, None, None]) * t_s[None, :, None] )
        iq += torch.sum(ab[:, None, None] * torch.exp(phase), dim=0)

    # Add noise + 1st-order MTI (Moving Target Indication) filter
    iq = iq + (torch.randn(M, N, device=DEVICE) + 1j * torch.randn(M, N, device=DEVICE)) * 1e-4
    iq[1:] -= iq[:-1].clone()
    iq[0] = 0

    # Windowing
    w_r = torch.hann_window(N, device=DEVICE)
    w_d = torch.hann_window(M, device=DEVICE)
    iq = iq * (w_d[:, None] * w_r[None, :])

    # Range FFT (one-sided 0..N/2-1)
    RFFT = torch.fft.fft(iq, dim=1)
    RFFT = RFFT[:, :N//2]
    
    # Doppler FFT with fftshift
    RD = torch.fft.fftshift(torch.fft.fft(RFFT, dim=0), dim=0)

    RD_mag = torch.abs(RD).clamp_min(1e-12)
    rd_db = 20 * torch.log10(RD_mag).cpu().numpy()
    
    return (rd_db,)

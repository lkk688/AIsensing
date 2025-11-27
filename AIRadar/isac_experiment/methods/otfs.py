import torch
import numpy as np
from ..config import DEVICE, C0
from ..utils import to_torch
from .comm_utils import qpsk_gray_mod, qpsk_gray_demod, awgn, rand_bits

def otfs_torch(pts, its, vels, sp):
    """
    Simulate OTFS radar return (Toy mapping for simulation).
    
    Args:
        pts (torch.Tensor): Point cloud positions.
        its (torch.Tensor): Intensities.
        vels (torch.Tensor): Velocities.
        sp (SystemParams): System parameters.
        
    Returns:
        tuple: (rd_db,) Delay-Doppler map in dB.
    """
    M, N = sp.M, sp.N
    H = torch.zeros((M, N), dtype=torch.complex64, device=DEVICE)
    
    if len(pts) == 0:
        return (np.zeros((M, N)),)

    P = pts - to_torch([0, 0, sp.H])
    R = torch.norm(P, dim=1)
    
    mask = R > 0.1
    P, R, vels, its = P[mask], R[mask], vels[mask], its[mask]
    
    amp = torch.where(its==255, 1e6, 1e-1) / (R**2 + 1e-6)
    vr  = torch.sum(P/R.unsqueeze(1)*vels, dim=1)

    # Resolution cells
    k_res = C0 / (2 * sp.fs)
    l_res = (sp.lambda_m / 2) * (sp.fs / (sp.M * sp.N))

    # Map Range/Velocity to Delay/Doppler indices
    k = torch.clamp((R / k_res).long(), 0, N-1)
    l = torch.clamp((vr / l_res).long() + M//2, 0, M-1)
    
    # Accumulate energy in the Delay-Doppler grid
    H.view(-1).scatter_add_(0, (l*N + k).view(-1), amp.to(torch.complex64))
    
    # Add noise
    H += (torch.randn(M, N, device=DEVICE) + 1j * torch.randn(M, N, device=DEVICE)) * 1e-4

    rd_db = (20 * torch.log10(torch.abs(H).clamp_min(1e-12))).cpu().numpy()
    return (rd_db,)

def otfs_mod(bits, M=64, N=256, cp_len=32, rng=None):
    """
    OTFS Modulation.
    
    Args:
        bits (np.ndarray): Input bits.
        M (int): Number of Doppler bins (subcarriers in time).
        N (int): Number of Delay bins (subcarriers in freq).
        cp_len (int): Cyclic Prefix length.
        rng (np.random.Generator): Random number generator.
        
    Returns:
        tuple: (tx_signal, X_dd_symbols)
    """
    if rng is None:
        rng = np.random.default_rng()
        
    # bits -> X_dd (QPSK)
    bits = bits.reshape(M*N, 2)[:M*N].reshape(M, N, 2)
    X_dd = qpsk_gray_mod(bits)                       # (M,N)

    # ISFFT (Inverse Symplectic Finite Fourier Transform)
    # DD -> TF: along N (delay) use FFT, along M (doppler/time slots) use IFFT
    X_tf = np.fft.ifft(np.fft.fft(X_dd, n=N, axis=1, norm='ortho'), n=M, axis=0, norm='ortho')  # (M,N)

    # Heisenberg Transform (TF -> time) 
    # With rectangular pulses, this is equivalent to OFDM per time slot
    tx = np.fft.ifft(X_tf, n=N, axis=1, norm='ortho')  # (M,N) time samples per slot
    
    if cp_len > 0:
        cp = tx[:, -cp_len:]
        tx = np.concatenate([cp, tx], axis=1)          # (M,N+cp)
        
    return tx, X_dd

def otfs_demod(rx, M=64, N=256, cp_len=32):
    """
    OTFS Demodulation.
    
    Args:
        rx (np.ndarray): Received signal.
        M (int): Number of Doppler bins.
        N (int): Number of Delay bins.
        cp_len (int): Cyclic Prefix length.
        
    Returns:
        np.ndarray: Demodulated Delay-Doppler symbols.
    """
    if cp_len > 0:
        rx = rx[:, cp_len:cp_len+N]
        
    # Wigner Transform (Time -> TF)
    Y_tf = np.fft.fft(rx, n=N, axis=1, norm='ortho')           # (M,N)
    
    # SFFT (Symplectic Finite Fourier Transform)
    # TF -> DD: along M use FFT, along N use IFFT
    Y_dd = np.fft.ifft(np.fft.fft(Y_tf, n=M, axis=0, norm='ortho'), n=N, axis=1, norm='ortho')
    
    return Y_dd  # (M,N)

def otfs_tx_rx_ber(ebn0_db, M=64, N=256, cp_len=32, rng=None):
    """
    Simulate OTFS transmission and reception to calculate BER.
    """
    if rng is None:
        rng = np.random.default_rng()
    bits_per_sym = 2
    nbits = M * N * bits_per_sym
    bits = rand_bits(nbits, rng)
    tx, Xdd = otfs_mod(bits, M=M, N=N, cp_len=cp_len, rng=rng)
    cp_ratio = cp_len / N
    rx = awgn(tx, ebn0_db, bits_per_sym=bits_per_sym, cp_ratio=cp_ratio, rng=rng)
    Ydd = otfs_demod(rx, M=M, N=N, cp_len=cp_len)
    
    hard_bits = qpsk_gray_demod(Ydd.reshape(-1))
    hard_bits = hard_bits.reshape(-1, 2)
    bits_hat = hard_bits.reshape(-1)
    bits_hat = bits_hat[:len(bits)]
    
    ber = np.mean(bits != bits_hat)
    return ber

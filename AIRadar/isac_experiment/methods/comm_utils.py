import numpy as np

def rand_bits(n, rng):
    """Generate random bits (0 or 1)."""
    return rng.integers(0, 2, size=n, dtype=np.uint8)

def qpsk_gray_mod(bits):
    """
    QPSK modulation with Gray mapping.
    
    Mapping:
    00 -> (1+1j)/sqrt(2)
    01 -> (-1+1j)/sqrt(2)
    11 -> (-1-1j)/sqrt(2)
    10 -> (1-1j)/sqrt(2)
    
    Args:
        bits (np.ndarray): Input bits of shape (..., 2).
        
    Returns:
        np.ndarray: Complex symbols.
    """
    b0 = bits[..., 0]
    b1 = bits[..., 1]
    
    # I component
    I = np.where((b0==0) & (b1==0),  1.0,
        np.where((b0==0) & (b1==1), -1.0,
        np.where((b0==1) & (b1==1), -1.0,  1.0)))
        
    # Q component
    Q = np.where((b0==0) & (b1==0),  1.0,
        np.where((b0==0) & (b1==1),  1.0,
        np.where((b0==1) & (b1==1), -1.0, -1.0)))
        
    s = (I + 1j*Q) / np.sqrt(2.0)  # Normalize energy to 1
    return s

def qpsk_gray_demod(symbols):
    """
    QPSK demodulation with Gray mapping (Hard decision).
    
    Args:
        symbols (np.ndarray): Received complex symbols.
        
    Returns:
        np.ndarray: Demodulated bits of shape (..., 2).
    """
    I = np.real(symbols)
    Q = np.imag(symbols)
    
    # Decision boundaries are at axes
    b0 = (Q < 0).astype(np.uint8)
    b1 = (I < 0).astype(np.uint8)
    return np.stack([b0, b1], axis=-1)

def awgn(x, ebn0_db, bits_per_sym, cp_ratio=0.0, rng=None):
    """
    Add Additive White Gaussian Noise (AWGN) to the signal.
    
    Args:
        x (np.ndarray): Input complex signal (assumed Es=1).
        ebn0_db (float): Desired Eb/N0 in dB.
        bits_per_sym (int): Bits per symbol (e.g., 2 for QPSK).
        cp_ratio (float): Cyclic Prefix ratio (overhead).
        rng (np.random.Generator): Random number generator.
        
    Returns:
        np.ndarray: Signal with noise.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    
    # Effective rate accounts for CP overhead
    r_eff = bits_per_sym * (1.0 / (1.0 + cp_ratio))
    
    Es = 1.0   # Assumed normalized energy
    Eb = Es / r_eff
    N0 = Eb / ebn0
    sigma2_complex = N0  # Total noise variance
    
    # Generate complex noise
    n = (rng.normal(scale=np.sqrt(sigma2_complex/2), size=x.shape)
         + 1j*rng.normal(scale=np.sqrt(sigma2_complex/2), size=x.shape))
         
    return x + n

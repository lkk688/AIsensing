import numpy as np
from .comm_utils import qpsk_gray_mod, qpsk_gray_demod, awgn, rand_bits

def ofdm_mod(bits, Nfft=256, cp_len=32, rng=None):
    """
    OFDM Modulation (QPSK).
    
    Args:
        bits (np.ndarray): Input bits.
        Nfft (int): FFT size.
        cp_len (int): Cyclic Prefix length.
        rng (np.random.Generator): Random number generator.
        
    Returns:
        tuple: (tx_signal, symbols)
    """
    if rng is None:
        rng = np.random.default_rng()
        
    bits = bits.reshape(-1, 2)  # pairs for QPSK
    nsym = bits.shape[0] // Nfft
    bits = bits[:nsym*Nfft].reshape(nsym, Nfft, 2)
    
    syms = qpsk_gray_mod(bits)           # (nsym, Nfft)
    
    # IFFT with unitary scale (norm='ortho')
    x = np.fft.ifft(syms, n=Nfft, axis=1, norm='ortho')  # (nsym, Nfft)
    
    # Add Cyclic Prefix
    if cp_len > 0:
        cp = x[:, -cp_len:]
        x_cp = np.concatenate([cp, x], axis=1)           # (nsym, Nfft+cp_len)
    else:
        x_cp = x
        
    return x_cp, syms

def ofdm_demod(rx, Nfft=256, cp_len=32):
    """
    OFDM Demodulation.
    
    Args:
        rx (np.ndarray): Received signal.
        Nfft (int): FFT size.
        cp_len (int): Cyclic Prefix length.
        
    Returns:
        np.ndarray: Demodulated frequency domain symbols.
    """
    if cp_len > 0:
        rx = rx[:, cp_len:cp_len+Nfft]
        
    # FFT with unitary scale
    Sy = np.fft.fft(rx, n=Nfft, axis=1, norm='ortho')  # (nsym, Nfft)
    return Sy

def ofdm_tx_rx_ber(ebn0_db, Nfft=256, cp_len=32, n_ofdm_sym=200, rng=None):
    """
    Simulate OFDM transmission and reception to calculate BER.
    """
    if rng is None:
        rng = np.random.default_rng()
    bits_per_sym = 2
    nbits = n_ofdm_sym * Nfft * bits_per_sym
    bits = rand_bits(nbits, rng)
    tx, ref_syms = ofdm_mod(bits, Nfft=Nfft, cp_len=cp_len, rng=rng)
    cp_ratio = cp_len / Nfft
    rx = awgn(tx, ebn0_db, bits_per_sym=bits_per_sym, cp_ratio=cp_ratio, rng=rng)
    Sy = ofdm_demod(rx, Nfft=Nfft, cp_len=cp_len)
    
    # Hard decision
    hard_bits = qpsk_gray_demod(Sy.reshape(-1))
    hard_bits = hard_bits.reshape(-1, 2)
    bits_hat = hard_bits[:len(bits) // 2].reshape(-1) # bits is flattened in original? No, bits is (nbits,)
    # Wait, bits in original was generated as (nbits,)
    # In ofdm_mod, bits is reshaped to (-1, 2).
    # Let's check original code logic.
    
    # Original:
    # bits = _rand_bits(nbits, rng) -> (nbits,)
    # tx, ... = ofdm_mod(bits, ...) -> inside it reshapes
    # hard_bits = _qpsk_gray_demod(...) -> (..., 2)
    # bits_hat = hard_bits[:len(bits)].reshape(-1) -> This looks wrong if len(bits) is nbits.
    # hard_bits is (N_symbols, 2). Flattened it is (N_symbols * 2).
    # If nbits matches exactly, then len(bits) == len(hard_bits.flatten()).
    
    # Let's stick to original logic but fix variable names.
    # bits is (nbits,)
    
    bits_hat = hard_bits.reshape(-1)
    # Truncate if necessary (though with full symbols it shouldn't be)
    bits_hat = bits_hat[:len(bits)]
    
    ber = np.mean(bits != bits_hat)
    return ber

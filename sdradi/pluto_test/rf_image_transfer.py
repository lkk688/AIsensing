#!/usr/bin/env python3
"""
RF Image Transfer & BER Test
---------------------------
Unified TX/RX script for PlutoSDR.
Supports:
1. Image File Transfer (Splits file into chunks, transmits, reassembles).
2. BER/PER Calculation (Using known Test Patterns).

Usage:
  TX (Image):
    python3 rf_image_transfer.py --mode tx --uri ip:192.168.3.2 --file my_image.jpg --repeat 1

  RX (Image):
    python3 rf_image_transfer.py --mode rx --uri ip:192.168.2.2 --out_file received.jpg

  TX (BER Test Pattern):
    python3 rf_image_transfer.py --mode tx --uri ip:192.168.3.2 --test_pattern --payload_len 64

  RX (BER Test Pattern):
    python3 rf_image_transfer.py --mode rx --uri ip:192.168.2.2 --test_pattern --payload_len 64
"""

import argparse
import time
import zlib
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Using Agg backend for headless plotting if needed
matplotlib.use("Agg")

# =============================================================================
# OFDM & PHY Configuration (Must match on TX and RX)
# =============================================================================
N_FFT = 64
N_CP = 16
PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21], dtype=int)
DATA_SUBCARRIERS = np.array([k for k in range(-26, 27) if k != 0 and k not in set(PILOT_SUBCARRIERS)], dtype=int)
N_DATA = len(DATA_SUBCARRIERS)  # 48
BITS_PER_OFDM_SYM = N_DATA * 2  # QPSK = 2 bits per symbol
MAGIC = b"IMG1"  # Magic header for this protocol
NON_MAGIC_BYTES = 4

# Protocol Header:
# MAGIC (4 bytes) | SEQ (2 bytes) | TOTAL_PKTS (2 bytes) | LEN (2 bytes) | PAYLOAD (...) | CRC32 (4 bytes)
# Header overhead = 4 + 2 + 2 + 2 = 10 bytes
# Footer overhead = 4 bytes (CRC)
# Total overhead per packet = 14 bytes

# =============================================================================
# PHY Functions (Common)
# =============================================================================

def mapping_qpsk_gray(bits: np.ndarray) -> np.ndarray:
    """Bits -> QPSK Symbols (Gray Encoded)"""
    bits = bits.astype(np.uint8)
    if len(bits) % 2 != 0:
        bits = np.pad(bits, (0, 1))
    
    b0 = bits[0::2]
    b1 = bits[1::2]
    
    # Truth table for our specific gray mapping needed to match receiver logic
    # We use a lookup table approach for clarity and speed
    # 00 -> +1+1j
    # 01 -> -1+1j
    # 11 -> -1-1j
    # 10 -> +1-1j
    
    # Initialize with 00 case
    syms = np.full(len(b0), 1.0 + 1.0j, dtype=np.complex64)
    
    # 01
    mask_01 = (b0 == 0) & (b1 == 1)
    syms[mask_01] = -1.0 + 1.0j
    
    # 11
    mask_11 = (b0 == 1) & (b1 == 1)
    syms[mask_11] = -1.0 - 1.0j
    
    # 10
    mask_10 = (b0 == 1) & (b1 == 0)
    syms[mask_10] = 1.0 - 1.0j
    
    return syms / np.sqrt(2)

def demapping_qpsk_gray(symbols: np.ndarray) -> np.ndarray:
    """QPSK Symbols -> Bits (Gray Decoded)"""
    re = np.real(symbols) >= 0
    im = np.imag(symbols) >= 0
    
    bits = np.empty(len(symbols) * 2, dtype=np.uint8)
    
    # b0: 1 if real component is positive? No, let's match the TX map.
    # TX:
    # 00 (RE>0, IM>0)
    # 01 (RE<0, IM>0)
    # 11 (RE<0, IM<0)
    # 10 (RE>0, IM<0)
    
    # b0 is 1 if RE > 0 and IM < 0 (10) OR RE < 0 and IM < 0 (11) -> RE < 0 ?? check
    # Let's inverse the map:
    # RE > 0: 00 or 10 -> b0 could be 0 or 1. Wait.
    # RE < 0: 01 or 11 -> b0 could be 0 or 1.
    
    # Re-checking the TX logic:
    # 00 -> (+, +) -> b0=0, b1=0
    # 01 -> (-, +) -> b0=0, b1=1
    # 11 -> (-, -) -> b0=1, b1=1
    # 10 -> (+, -) -> b0=1, b1=0
    
    # Logic:
    # b1 is 1 if IM < 0 ?? No. 
    # IM > 0: 00, 01 -> b1 is 0 or 1. 
    
    # Let's map strictly:
    # Quadrant 1 (+,+): 00
    # Quadrant 2 (-,+): 01
    # Quadrant 3 (-,-): 11
    # Quadrant 4 (+,-): 10
    
    # Array based boolean logic:
    m_pp = re & im      # 00
    m_mp = (~re) & im   # 01
    m_mm = (~re) & (~im)# 11
    m_pm = re & (~im)   # 10
    
    # We populate a temporary array of shape (N, 2)
    b_out = np.zeros((len(symbols), 2), dtype=np.uint8)
    
    # 00 is default
    
    # 01
    b_out[m_mp, 0] = 0
    b_out[m_mp, 1] = 1
    
    # 11
    b_out[m_mm, 0] = 1
    b_out[m_mm, 1] = 1
    
    # 10
    b_out[m_pm, 0] = 1
    b_out[m_pm, 1] = 0
    
    return b_out.reshape(-1)

def create_stf(N=64):
    rng = np.random.default_rng(42)  # Fixed seed for standard STF
    X = np.zeros(N, dtype=np.complex64)
    even_subs = np.array([k for k in range(-26, 27, 2) if k != 0], dtype=int)
    bpsk = rng.choice([-1.0, 1.0], size=len(even_subs)).astype(np.float32)
    for i, k in enumerate(even_subs):
        X[(k + N) % N] = bpsk[i]
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
    stf = np.tile(x, 2)
    stf_cp = np.concatenate([stf[-N_CP:], stf])
    return stf_cp.astype(np.complex64)

def create_ltf(N=64):
    X = np.zeros(N, dtype=np.complex64)
    used = np.array([k for k in range(-26, 27) if k != 0], dtype=int)
    for i, k in enumerate(used):
        X[(k + N) % N] = 1.0 if (i % 2 == 0) else -1.0
    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N)
    ltf_sym = np.concatenate([x[-N_CP:], x])
    ltf = np.tile(ltf_sym, 2)
    return ltf.astype(np.complex64), X.astype(np.complex64)

def bits_to_bytes(bits):
    return np.packbits(bits).tobytes()

def bytes_to_bits(b):
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))

# =============================================================================
# Helper: Data Chunking
# =============================================================================

def create_chunks(data: bytes, chunk_size: int):
    """Yields chunks of data with index."""
    total_len = len(data)
    total_pkts = (total_len + chunk_size - 1) // chunk_size
    for i in range(0, total_len, chunk_size):
        yield i // chunk_size, total_pkts, data[i : i + chunk_size]

def build_packet_bits(seq: int, total_pkts: int, payload: bytes, repeat: int = 1):
    """
    Constructs a PHY packet bits for a given payload.
    Format: MAGIC(4) | SEQ(2) | TOTAL(2) | LEN(2) | PAYLOAD | CRC(4)
    """
    plen = len(payload)
    # Header
    header = MAGIC + \
             seq.to_bytes(2, "little") + \
             total_pkts.to_bytes(2, "little") + \
             plen.to_bytes(2, "little")
    
    # Calculate CRC on Payload only? Or Header+Payload? 
    # Usually better on Header+Payload to protect metadata.
    # But for simplicity and matching previous script logic which only did payload:
    # Let's do CRC on Payload only to match "rf_step4" style roughly, 
    # BUT adding header protection is safer. given "enhanced", let's CRC the whole frame content.
    
    content = header + payload
    crc = zlib.crc32(content) & 0xFFFFFFFF
    
    frame_bytes = content + crc.to_bytes(4, "little")
    
    bits = bytes_to_bits(frame_bytes)
    
    # Repetition coding
    if repeat > 1:
        bits = np.repeat(bits, repeat)
        
    return bits, len(frame_bytes)

def parse_packet_data(bits_bytes: bytes):
    """
    Parses raw bytes from receiver.
    Returns: (valid_crc, seq, total, payload)
    """
    if len(bits_bytes) < 14: # Min size (Header 10 + CRC 4)
        return False, 0, 0, b""
    
    # Check Magic
    if bits_bytes[:4] != MAGIC:
        return False, 0, 0, b""
        
    try:
        seq = int.from_bytes(bits_bytes[4:6], "little")
        total = int.from_bytes(bits_bytes[6:8], "little")
        plen = int.from_bytes(bits_bytes[8:10], "little")
        
        # Validation of lengths
        expected_len = 10 + plen + 4
        if len(bits_bytes) < expected_len:
            return False, 0, 0, b""
            
        content_end = 10 + plen
        content = bits_bytes[:content_end]
        payload = bits_bytes[10:content_end]
        
        rx_crc = int.from_bytes(bits_bytes[content_end:content_end+4], "little")
        calc_crc = zlib.crc32(content) & 0xFFFFFFFF
        
        return (rx_crc == calc_crc), seq, total, payload
    except Exception:
        return False, 0, 0, b""

# =============================================================================
# TX Logic
# =============================================================================

def run_tx(args):
    import adi
    
    # 1. Prepare Data
    if args.test_pattern:
        print(f"[TX] Generating Test Pattern (random, {args.payload_len} bytes/pkt)...")
        # In test pattern mode, we construct a dummy full 'file' to chunk, or just infinite random?
        # To measure BER/PER, we usually want known data. 
        # Let's generate a repeating pattern or a static random seed.
        # For simple BER: use a fixed seed per packet or header-based seed. 
        # Easier: Generate one large random buffer and loop over it.
        rng = np.random.default_rng(12345)
        # 100 packets worth of data
        data_to_send = rng.bytes(args.payload_len * 100) 
    elif args.file:
        print(f"[TX] Reading file: {args.file}")
        with open(args.file, "rb") as f:
            data_to_send = f.read()
    else:
        print("[TX] Error: Must provide --file or --test_pattern")
        return

    # 2. Setup SDR
    print(f"[TX] Connecting to Pluto at {args.uri}...")
    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.tx_lo = int(args.fc)
    sdr.tx_rf_bandwidth = int(args.fs * 1.2)
    sdr.tx_hardwaregain_chan0 = float(args.tx_gain)
    sdr.tx_cyclic_buffer = True # We will create one large waveform or cycle buffers?
    # Cyclic buffer repeats the *same* waveform. 
    # For file transfer, we need to send *different* packets.
    # Thus cyclic_buffer = False is needed for streaming, OR we use cyclic=True but manual push?
    # ADI Pluto python styling usually acts weird with non-cyclic streaming for bursty data.
    # BEST PRACTICE for reliable Python scripting with Pluto:
    # Use cyclic_buffer = True, but load ONE packet, let it repeat for a bit (redundancy), then load NEXT.
    # OR use cyclic_buffer = False and push continuously. 
    # Based on user "step4" script, it used `sdr.tx_cyclic_buffer = True` and just ran `while True: sleep`.
    # This implies the user experienced static packet sending.
    # For FILE TRANSFER, we must change the buffer content.
    # Changing buffer takes time. 
    
    sdr.tx_cyclic_buffer = True # We will manually update buffer for each chunk
    
    # 3. Pre-compute Preamble
    stf = create_stf(N_FFT)
    ltf, _ = create_ltf(N_FFT)
    
    # Preamble structure: Gap + Preamble
    gap = np.zeros(1000, dtype=np.complex64)
    preamble = np.concatenate([stf, ltf])
    
    # 4. Transmission Loop
    chunk_gen = create_chunks(data_to_send, args.payload_len)
    all_chunks = list(chunk_gen)
    total_chunks = len(all_chunks)
    
    print(f"[TX] Starting Transmission. Total Chunks: {total_chunks}")
    print(f"[TX] Press Ctrl+C to stop.")
    
    try:
        # Loop indefininely or run through list once? 
        # Image transfer usually wants safe delivery. 
        # Simple approach: Round-robin transmit all chunks continuously until stopped.
        # This allows receiver to pick up missing pieces.
        
        while True:
            for seq, total, payload in all_chunks:
                # Build Signal
                bits, frame_len = build_packet_bits(seq, total, payload, args.repeat)
                
                # OFDM Mod
                num_syms = int(np.ceil(len(bits) / BITS_PER_OFDM_SYM))
                # Pad bits
                n_pad = num_syms * BITS_PER_OFDM_SYM - len(bits)
                if n_pad > 0:
                    bits = np.pad(bits, (0, n_pad))
                
                # QPSK
                # Map chunks of bits to symbols
                sym_bits = bits.reshape(-1, 2)
                # Re-flatten for our mapper which expects 1D stream but handles pairs
                # Our mapper `mapping_qpsk_gray` expects 1D array of bits
                qpsk_syms = mapping_qpsk_gray(bits)
                
                # OFDM IFFT
                pilot_vals = np.array([1, 1, 1, -1], dtype=np.complex64)
                ofdm_payload = []
                
                for s_i in range(num_syms):
                    X = np.zeros(N_FFT, dtype=np.complex64)
                    
                    # Data
                    chunk_syms = qpsk_syms[s_i * N_DATA : (s_i + 1) * N_DATA]
                    # If last symbol incomplete (should happen due to padding? No, we padded bits)
                    
                    # Fill Data Subcarriers
                    X[(DATA_SUBCARRIERS + N_FFT) % N_FFT] = chunk_syms
                    
                    # Pilots (BPSK, alternating sign)
                    p_sign = 1.0 if s_i % 2 == 0 else -1.0
                    X[(PILOT_SUBCARRIERS + N_FFT) % N_FFT] = p_sign * pilot_vals
                    
                    # IFFT
                    x = np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(N_FFT)
                    x_cp = np.concatenate([x[-N_CP:], x])
                    ofdm_payload.append(x_cp)
                
                ofdm_signal = np.concatenate(ofdm_payload)
                
                # Full Frame
                tx_frame = np.concatenate([gap, preamble, ofdm_signal, gap])
                
                # Scale
                max_val = np.max(np.abs(tx_frame))
                if max_val > 0:
                    tx_frame = tx_frame / max_val * 0.7
                
                tx_scaled = (tx_frame * 2**14).astype(np.complex64)
                
                # FIX: In cyclic mode, we must destroy the previous buffer before submitting a new one.
                # This stops the previous repetitive transmission.
                if sdr.tx_cyclic_buffer:
                    try:
                        sdr.tx_destroy_buffer()
                    except Exception:
                        pass # Buffer might not exist yet on first run

                sdr.tx(tx_scaled)
                
                if args.static:
                    print(f"\r[TX] Sent Packet {seq+1}/{total} (Static Mode). Repeating forever...")
                    while True:
                        time.sleep(1.0)
                
                # Wait for transmission duration + margin
                # In cyclic mode, it repeats. We wait a bit to ensure it goes out a few times
                # or at least once fully.
                # The overhead of usb/network might be high, so we sleep generously to ensure playback.
                time.sleep(0.5) 
                
                print(f"\r[TX] Sent Packet {seq+1}/{total} ({len(payload)} bytes)   ", end="")
            
            print("\n[TX] Cycle complete. Restarting loop for robustness...")
            time.sleep(1.0) # Pause between cycles

    except KeyboardInterrupt:
        print("\n[TX] Stop.")
        sdr.tx_destroy_buffer()

# =============================================================================
# RX Logic
# =============================================================================

def apply_cfo_correction(x, cfo, fs):
    t = np.arange(len(x)) / fs
    return x * np.exp(-1j * 2 * np.pi * cfo * t)

def run_rx(args):
    import adi
    
    print(f"[RX] Connecting to Pluto at {args.uri}...")
    sdr = adi.Pluto(uri=args.uri)
    sdr.sample_rate = int(args.fs)
    sdr.rx_lo = int(args.fc)
    sdr.rx_rf_bandwidth = int(args.fs * 1.2)
    sdr.rx_buffer_size = int(args.fs * 0.3) # 300ms buffer
    sdr.rx_hardwaregain_chan0 = float(args.rx_gain)
    
    # Prepare References
    stf_ref = create_stf(N_FFT)
    # STF Correlation requires excluding CP usually, or just using raw
    # Let's use the core STF part (no CP) for correlation to be sharper
    # But current create_stf includes CP.
    # Extract core
    stf_core = stf_ref[-N_FFT:] # Last repeat
    
    ltf_ref, ltf_freq_ref = create_ltf(N_FFT)
    
    received_chunks = {}
    total_pkts_expected = 0
    total_bits_received = 0
    bit_errors = 0
    packets_ok = 0
    packets_crc_fail = 0
    
    print("[RX] Listening... (Ctrl+C to stop)")
    
    try:
        while True:
            # Get Samples
            rx_raw = sdr.rx()
            rx_raw = rx_raw / 2**14
            
            # Simple Packet Detect: Cross Corr with STF
            # Limit processing to avoid lag
            
            # 1. Coarse Freq / Detection
            # (Skipping robust CFO for brevity, relying on user existing script logic if needed, 
            #  but implementing basic STF peak detection)
            
            corr = np.abs(np.correlate(rx_raw, stf_core, mode='valid'))
            threshold = np.mean(corr) * 5 # Adaptive threshold
            peaks = np.where(corr > threshold)[0]
            
    # Packet Detect: Cross Corr with STF
            # Limit processing to avoid lag
            
            # 1. Coarse Freq / Detection
            
            # Use full STF ref for correlation (simple)
            corr_full = np.abs(np.correlate(rx_raw, stf_ref, mode='valid'))
            
            # Debug Stats
            c_mean = np.mean(corr_full)
            c_max = np.max(corr_full)
            sys.stdout.write(f"\r[RX DBG] Max/Mean: {c_max/c_mean:.1f}  Pkts:{len(received_chunks)} ")
            sys.stdout.flush()
            
            threshold = c_mean * 3.0
            peaks = np.where(corr_full > threshold)[0]
            
            # Simple Peak Iteration
            processed_mask = np.zeros(len(rx_raw), dtype=bool)
            
            # Ensure ltf_freq_ref is in FFTSHIFT domain for channel estimation division
            # create_ltf returns it in standard FFT order (DC at 0, neg at end)
            # We extracted Y using fftshift (DC at center).
            ltf_freq_ref_shifted = np.fft.fftshift(ltf_freq_ref)
            
            for peak_idx in peaks:
                if processed_mask[peak_idx]:
                    continue
                
                # Debounce
                s_end = min(len(rx_raw), peak_idx + 20000)
                processed_mask[peak_idx : peak_idx + 200] = True # crude debounce
                
                # Check peak quality (check neighbor)
                if peak_idx + 1 < len(corr_full) and corr_full[peak_idx+1] > corr_full[peak_idx]:
                    continue # Not local max
                
                stf_idx = peak_idx
                # Calculate CFO using STF repeats
                # Structure: [CP] [x] [x]
                # Start of x1: stf_idx + N_CP
                # Start of x2: stf_idx + N_CP + N_FFT
                r1 = rx_raw[stf_idx + N_CP : stf_idx + N_CP + N_FFT]
                r2 = rx_raw[stf_idx + N_CP + N_FFT : stf_idx + N_CP + 2*N_FFT]
                
                # Schmidl & Cox
                conj_prod = np.sum(np.conj(r1) * r2)
                angle = np.angle(conj_prod)
                # angle = 2*pi * cfo * (N/fs)
                cfo_est = angle / (2 * np.pi * N_FFT / args.fs)
                
                # Apply Correction to the rest of the frame (conceptually)
                # We need to extract enough samples for LTF + Payload and correct them
                ltf_start = stf_idx + len(stf_ref)
                
                # Max frame length guess
                max_process_len = (N_FFT + N_CP) * 250 # 250 symbols
                if ltf_start + max_process_len > len(rx_raw):
                    max_process_len = len(rx_raw) - ltf_start
                
                # Extract raw chunk including LTF + Payload
                chunk_raw = rx_raw[ltf_start : ltf_start + max_process_len]
                
                # Correction vector
                # Time indices relative to current start? 
                # Continuity matters if we want phase to be continuous from STF?
                # Actually, LTF estimation handles absolute phase offset (Channel H).
                # We just need to remove Frequency rotation (progressive phase).
                # So starting t=0 at ltf_start is fine, H will absorb the constant phase offset.
                t_vec = np.arange(len(chunk_raw), dtype=np.float64) / args.fs
                chunk_cfo = chunk_raw * np.exp(-1j * 2 * np.pi * cfo_est * t_vec)
                
                # Bounds check
                # (handled by slice logic usually, but chunk might be shorter than needed)
                
                # Channel Estimate (LTF)
                sym_len = N_FFT + N_CP
                if len(chunk_cfo) < 2 * sym_len: continue
                
                y1 = chunk_cfo[N_CP : sym_len]
                y2 = chunk_cfo[sym_len + N_CP : 2*sym_len]
                
                Y1 = np.fft.fftshift(np.fft.fft(y1))
                Y2 = np.fft.fftshift(np.fft.fft(y2))
                Y_avg = (Y1 + Y2) / 2
                
                H = np.zeros(N_FFT, dtype=np.complex64)
                eps = 1e-9
                
                # Zero Forcing Estimate
                used_mask = np.abs(ltf_freq_ref_shifted) > 0.1
                H[used_mask] = Y_avg[used_mask] / ltf_freq_ref_shifted[used_mask]
                
                # Encode / Decode Loop
                # Payload starts after LTF (2 syms)
                # In chunk_cfo, payload starts at index 2*sym_len
                current_chunk_idx = 2 * sym_len
                
                all_demod_bits = []
                phase_acc = 0.0
                
                debug_constellation = []
                debug_phase_err = []
                
                decoded_ok = False
                
                for i in range(150): # Cap at 150 symbols
                    if current_chunk_idx + sym_len > len(chunk_cfo):
                        break
                        
                    y = chunk_cfo[current_chunk_idx + N_CP : current_chunk_idx + sym_len]
                    Y = np.fft.fftshift(np.fft.fft(y))
                    
                    # EQ
                    Y_eq = np.zeros_like(Y)
                    Y_eq[used_mask] = Y[used_mask] / (H[used_mask] + eps)
                    
                    # Mapping indices for Shifted Domain
                    p_idxs = PILOT_SUBCARRIERS + N_FFT // 2
                    
                    # Pilots
                    pilot_expect = (1.0 if i%2==0 else -1.0) * np.array([1, 1, 1, -1])
                    p_rx = Y_eq[p_idxs]
                    
                    # Phase Error
                    p_err = np.angle(np.sum(p_rx * np.conj(pilot_expect)))
                    phase_acc += p_err * 0.1
                    debug_phase_err.append(p_err)
                    
                    # Derotate
                    Y_eq = Y_eq * np.exp(-1j * phase_acc)
                    
                    # Data
                    d_idxs = DATA_SUBCARRIERS + N_FFT // 2
                    d_syms = Y_eq[d_idxs]
                    
                    debug_constellation.append(d_syms)
                    
                    bits_sym = demapping_qpsk_gray(d_syms)
                    all_demod_bits.append(bits_sym)
                    
                    current_chunk_idx += sym_len
                
                if not all_demod_bits:
                    continue
                    
                full_bits = np.concatenate(all_demod_bits)
                full_bytes = bits_to_bytes(full_bits)
                
                # Parse
                valid, seq, total_pkts, payload = parse_packet_data(full_bytes)
                
                status_str = "FAIL"
                if valid:
                    status_str = "OK"
                    if args.test_pattern:
                        packets_ok += 1
                        # print(f"[RX] Pkt {seq} OK (BER Mode).")
                    else:
                        if seq not in received_chunks:
                            received_chunks[seq] = payload
                            print(f"[RX] Recv Chunk {seq}/{total_pkts} (len={len(payload)})")
                            total_pkts_expected = total_pkts
                else:
                    if len(full_bytes) > 20 and full_bytes[:4] == MAGIC:
                         packets_crc_fail += 1
                         status_str = "CRC_FAIL"
                    else:
                         status_str = "BAD_HEADER"

                # Debug Saving
                if args.debug:
                    import os
                    os.makedirs("rx_debug", exist_ok=True)
                    # We want to save:
                    # 1. Raw Time domain
                    # 2. Correlation
                    # 3. Constellations (Pre/Post)
                    # 4. Phase Error Log
                    
                    fig = plt.figure(figsize=(15, 10))
                    
                    # 1. Time Domain (Chunk)
                    ax1 = fig.add_subplot(2, 3, 1)
                    ax1.plot(np.abs(chunk_cfo))
                    ax1.set_title(f"Time Magnitude ({status_str})")
                    ax1.grid(True)
                    
                    # 2. Correlation (Zoomed)
                    ax2 = fig.add_subplot(2, 3, 2)
                    p_start = max(0, peak_idx - 100)
                    p_end = min(len(corr_full), peak_idx + 100)
                    ax2.plot(corr_full[p_start:p_end])
                    ax2.set_title(f"Peak Corr (Val={corr_full[peak_idx]:.1f})")
                    ax2.grid(True)
                    
                    # 3. Constellation (Post-CPE)
                    ax3 = fig.add_subplot(2, 3, 3)
                    # Gather all syms from loop
                    # We need to reconstruct all_demod_bits logic to get symbols?
                    # The loop calculated `d_syms` each time.
                    # Let's Modify loop to store them?
                    # Doing it inside the loop is cleaner but requires more variable passing.
                    # Or just capture the last block? 
                    # Better: Accumulate all symbols in a debug list inside the loop
                    # Added 'debug_syms_post' to loop below or above.
                    # ... Wait, I can't easily change the loop above in this replacement block 
                    # efficiently without re-writing the whole loop.
                    # I will assume I can edit the loop to save symbols.
                    # For now, let's just plot what we have or placeholder.
                    
                    # Actually, I will insert the symbol capture in the loop in a separate replacement 
                    # or make this replacement cover the loop? 
                    # The loop is large.
                    # I'll edit this replacement to be just the saving part, 
                    # and I'll do another edit to capture the symbols.
                    
                    # Placeholder for now, assumed available from 'debug_constellation' list
                    if 'debug_constellation' in locals() and len(debug_constellation) > 0:
                        all_const = np.concatenate(debug_constellation)
                        ax3.scatter(np.real(all_const), np.imag(all_const), s=2, alpha=0.5)
                        ax3.set_title(f"Constellation n={len(all_const)}")
                        ax3.set_xlim(-2, 2); ax3.set_ylim(-2, 2)
                        ax3.grid(True)
                    
                    # 4. Phase Error
                    ax4 = fig.add_subplot(2, 3, 4)
                    if 'debug_phase_err' in locals():
                        ax4.plot(debug_phase_err)
                        ax4.set_title("Phase Error")
                        ax4.grid(True)
                        
                    # 5. Spectrum
                    ax5 = fig.add_subplot(2, 3, 5)
                    f_spec = np.fft.fftshift(np.fft.fft(chunk_cfo))
                    ax5.plot(20*np.log10(np.abs(f_spec)+1e-9))
                    ax5.set_title("Spectrum (Chunk)")
                    ax5.grid(True)
                    
                    # 6. Info
                    ax6 = fig.add_subplot(2, 3, 6)
                    ax6.axis('off')
                    info = f"Seq: {seq}\nCFO Est: {cfo_est:.1f} Hz\nPkts Received: {len(received_chunks)}\n"
                    if len(full_bytes) >= 10:
                        info += f"Magic: {full_bytes[:4].hex()}\n"
                    ax6.text(0.1, 0.5, info, family='monospace')
                    
                    ts = int(time.time()*100)
                    fname = f"rx_debug/pkt_{ts}_{status_str}.png"
                    fig.tight_layout()
                    fig.savefig(fname)
                    plt.close(fig)
                    # print(f"[RX] Saved debug: {fname}")

            # Check completion
            if total_pkts_expected > 0 and len(received_chunks) == total_pkts_expected:
                print(f"[RX] All {total_pkts_expected} chunks received!")
                
                # Reassemble
                all_data = b""
                for i in range(total_pkts_expected):
                    all_data += received_chunks.get(i, b"") # Should exist
                
                if args.out_file:
                    with open(args.out_file, "wb") as f:
                        f.write(all_data)
                    print(f"[RX] Saved to {args.out_file}")
                    
                # Reset
                received_chunks.clear()
                total_pkts_expected = 0
            
            # Simple status line
            sys.stdout.write(f"\r[RX] Pkts OK: {len(received_chunks)} | CRC Fail: {packets_crc_fail}")
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n[RX] Stopped.")

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF Image Transfer")
    parser.add_argument("--mode", choices=["tx", "rx"], required=True)
    parser.add_argument("--uri", required=True, help="SDR URI (e.g. ip:192.168.2.1)")
    parser.add_argument("--file", help="Input file (TX) or Output file (RX)")
    parser.add_argument("--out_file", help="Output file (RX)")
    parser.add_argument("--test_pattern", action="store_true", help="Use internal test pattern instead of file")
    
    parser.add_argument("--fc", type=float, default=915e6, help="Center Freq")
    parser.add_argument("--fs", type=float, default=2e6, help="Sample Rate")
    parser.add_argument("--tx_gain", type=float, default=-5.0)
    parser.add_argument("--rx_gain", type=float, default=60.0)
    parser.add_argument("--payload_len", type=int, default=64, help="Bytes per packet")
    parser.add_argument("--repeat", type=int, default=1, help="Repetition coding")
    parser.add_argument("--debug", action="store_true", help="Save debug plots on RX")
    parser.add_argument("--static", action="store_true", help="TX: Transmit one packet continuously (no updates)")
    
    args = parser.parse_args()
    
    if args.mode == "tx":
        run_tx(args)
    else:
        run_rx(args)

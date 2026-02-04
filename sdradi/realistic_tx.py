import numpy as np
import time
import argparse
import sys
import adi

from sdr_video_comm import OFDMTransceiver, OFDMConfig, OTFSTransceiver, OTFSConfig, WaveformType

def generate_zc_sequence(length, root):
    """Generate Zadoff-Chu sequence."""
    n = np.arange(length)
    return np.exp(-1j * np.pi * root * n * (n + 1) / length)

def main():
    parser = argparse.ArgumentParser(description="Realistic TX (OFDM Transceiver)")
    parser.add_argument("--ip", type=str, default="ip:192.168.2.2", help="SDR IP") 
    parser.add_argument("--uri", type=str, default="ip:192.168.3.2", help="SDR URI") # Fixed URI
    parser.add_argument("--gain", type=int, default=0, help="TX Gain")
    parser.add_argument("--waveform", type=str, default="ofdm", choices=["ofdm", "otfs"], help="Waveform type")
    args = parser.parse_args()

    # Config
    if args.waveform == "ofdm":
        cfg = OFDMConfig(mod_order=2) # BPSK
        tr = OFDMTransceiver(cfg)
        print("Selected Waveform: OFDM (BPSK)")
    else:
        cfg = OTFSConfig(mod_order=2) # BPSK
        tr = OTFSTransceiver(cfg)
        print("Selected Waveform: OTFS (BPSK)")
    
    from sdr_video_comm import PacketFramer

    # Generate Payload (Framed)
    np.random.seed(42)
    
    if args.waveform == "ofdm":
        # bits_per_frame = 48 * 14 * 2 = 1344
        # 10 frames = 13440 bits = 1680 bytes
        total_bits = cfg.bits_per_frame * 10 
    else:
        # OTFS: bits_per_frame = 256 * 64 * 2 = 32768
        # 1 frame = 32768 bits = 4096 bytes
        total_bits = cfg.bits_per_frame
        
    total_bytes = total_bits // 8
    print(f"Generating {total_bytes} bytes of framed payload (Seed 42)...")
    
    # Create packets to fill the space
    # Max packet size ~200 bytes
    payload_buffer = bytearray()
    seq = 0
    
    while len(payload_buffer) < total_bytes:
        remaining = total_bytes - len(payload_buffer)
        # Overhead is 12 bytes. Min payload 1 byte.
        if remaining < PacketFramer.OVERHEAD + 1:
            # Not enough space for another packet, fill with padding
            payload_buffer.extend(b'\x00' * remaining)
            break
            
        data_len = min(200, remaining - PacketFramer.OVERHEAD)
        # Random data
        p_data = bytes(np.random.randint(0, 256, data_len, dtype=np.uint8))
        
        frame_bytes = PacketFramer.frame(p_data, seq)
        payload_buffer.extend(frame_bytes)
        seq += 1
        
    print(f"Generated {seq} packets. Total bytes: {len(payload_buffer)}")
    
    # Convert bytes to bits
    # unpack bits: (N,) uint8 -> (N*8,) uint8
    tx_bits = np.unpackbits(np.frombuffer(payload_buffer, dtype=np.uint8))
    
    # Modulate
    start = time.time()
    payload_signal = tr.modulate(tx_bits)
    print(f"Modulation took {time.time()-start:.4f}s")
    
    # Frame Construction
    # [ZC (127)] [Guard (50)] [Tone (256)] [Guard (50)] [Payload]
    zc = generate_zc_sequence(127, 25)
    guard = np.zeros(50, dtype=complex)
    t_tone = np.arange(256)
    tone = np.exp(1j * 2 * np.pi * t_tone / 16.0)
    
    # Scaling
    max_payload = np.max(np.abs(payload_signal))
    if max_payload > 0:
        payload_signal = payload_signal / max_payload * 0.8
    
    silence = np.zeros(500, dtype=complex) # Loop padding
    
    frame = np.concatenate([zc, guard, tone, guard, payload_signal, silence])
    
    # SDR Setup
    print(f"Connecting to SDR {args.uri}...")
    sdr = adi.Pluto(args.uri)
    sdr.tx_lo = int(2.3e9) + 6561 # Match RX tuning
    sdr.tx_cyclic_buffer = True
    sdr.tx_hardwaregain_chan0 = args.gain
    sdr.sample_rate = int(2e6)
    sdr.tx_rf_bandwidth = int(1e6)
    
    # Transmit
    frame = frame.astype(np.complex64)
    # Scaling for Pluto (0-1 range? No, 2^14. ADI takes care if normalize=True?)
    # ADI python: if normalize=True, it scales to max.
    
    # Scale for DAC (14-bit)
    frame = frame * (2**14) * 0.5 # 0.5 backoff to be safe
    frame = frame.astype(np.complex64)
    
    print(f"Transmitting {len(frame)} samples cyclically...")
    sdr.tx_cyclic_buffer = True
    sdr.tx(frame)
    # sdr.tx_destroy_buffer() # Don't destroy if cyclic
    
    # Keep alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()

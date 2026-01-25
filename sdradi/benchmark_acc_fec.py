import time
import numpy as np
import torch
import sys
import os
sys.path.append('/Developer/AIsensing/sdradi')
from sdr_video_comm import SDRVideoLink, FECConfig, FECType, WaveformType
# sdr_ldpc has LDPC5GEncoder, but we might test the wrapper from sdr_video_comm
from sdr_ldpc import LDPC5GEncoder as TorchLDPCEncoder

def benchmark_component_latency(num_runs=100, packet_size=8192):
    print(f"\n=== Component Benchmark (Packet Size: {packet_size} bits, {num_runs} runs) ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 1. FastGPUFEC (Repetition)
    print("--- FastGPUFEC (Repetition 7x) ---")
    try:
        from sdr_video_comm import FastGPUFEC
        fec_rep = FastGPUFEC(repetitions=7, device=device)
        bits = np.random.randint(0, 2, packet_size)
        
        # Warmup
        for _ in range(10):
            e = fec_rep.encode(bits)
            d = fec_rep.decode(e)
            
        t_enc = []
        t_dec = []
        for _ in range(num_runs):
            t0 = time.time()
            encoded = fec_rep.encode(bits)
            t1 = time.time()
            decoded = fec_rep.decode(encoded)
            t2 = time.time()
            t_enc.append((t1-t0)*1000)
            t_dec.append((t2-t1)*1000)
            
        print(f"Encode: {np.mean(t_enc):.3f} ms")
        print(f"Decode: {np.mean(t_dec):.3f} ms")
    except Exception as e:
        print(f"Skipped: {e}")

    # 2. LDPC (PyTorch)
    print("--- LDPC (PyTorch, Rate 1/2) ---")
    try:
        # We assume sdr_video_comm wrapper uses our sdr_ldpc
        # Let's instantiate via SDRVideoLink logic or direct import
        # We need the Wrapper from sdr_video_comm to handle numpy<->torch conversion overhead too
        from sdr_video_comm import LDPC5GCoder, FECConfig, FECType
        config = FECConfig(enabled=True, fec_type=FECType.LDPC)
        fec_ldpc = LDPC5GCoder(config)
        
        if fec_ldpc.backend != 'torch':
            print("WARNING: Not using Torch backend for benchmark!")
            
        bits = np.random.randint(0, 2, packet_size)
        
        # Warmup
        for _ in range(10):
            e = fec_ldpc.encode(bits)
            d = fec_ldpc.decode(e)
            
        t_enc = []
        t_dec = []
        for _ in range(num_runs):
            t0 = time.time()
            encoded = fec_ldpc.encode(bits)
            t1 = time.time()
            # Feed LLRs or bits? Wrapper takes bits
            decoded = fec_ldpc.decode(encoded) 
            t2 = time.time()
            t_enc.append((t1-t0)*1000)
            t_dec.append((t2-t1)*1000)
            
        print(f"Encode: {np.mean(t_enc):.3f} ms")
        print(f"Decode: {np.mean(t_dec):.3f} ms")
        
    except Exception as e:
        print(f"Skipped: {e}")

def benchmark_system_throughput(num_frames=50):
    print(f"\n=== System Loopback Benchmark ({num_frames} frames) ===")
    
    modes = [
        ("None", FECConfig(enabled=False)),
        ("Repetition", FECConfig(enabled=True, fec_type=FECType.REPETITION, repetitions=7)),
        ("LDPC-Torch", FECConfig(enabled=True, fec_type=FECType.LDPC))
    ]
    
    for name, config in modes:
        print(f"\n Testing Mode: {name}")
        try:
            link = SDRVideoLink(
                waveform=WaveformType.OFDM,
                fec_config=config,
                simulation_mode=True
            )
            
            # Generate dummy frame data (approx 10KB/frame for modest quality)
            # In simulation, we might process raw bytes.
            # simulate_video_transmission usually takes a file.
            # Let's call process_frame directly to avoid file IO overhead.
            
            # Mock dummy video frame (compressed JPEG bytes)
            dummy_frame = np.random.randint(0, 255, 10000, dtype=np.uint8).tobytes()
            
            t_start = time.time()
            bits_processed = 0
            
            for _ in range(num_frames):
                # TX: Frame -> Bits -> Symbols -> Waveform
                # RX: Waveform -> Symbols -> Bits -> Frame
                
                # We'll use the lower level `_transmit_packet` loop if possible, 
                # but `test_loopback` logic in SDRVideoLink?
                # Actually, simulate_video_transmission runs a thread.
                # Let's use the transceiver components manually for synchronization.
                
                # 1. Encode Frame
                bits = link.video_codec.bytes_to_bits(dummy_frame)
                
                # 2. FEC Encode
                encoded_bits = link.fec_codec.encode(bits)
                
                # 3. Modulate (OFDM)
                tx_signal = link.ofdm.modulate(encoded_bits)
                
                # 4. Channel (Identity for speed test)
                rx_signal = tx_signal 
                
                # 5. Demodulate
                demod_bits, _ = link.ofdm.demodulate(rx_signal)
                
                # 6. FEC Decode
                decoded_bits = link.fec_codec.decode(demod_bits)
                
                # 7. Reassemble
                _ = link.video_codec.bits_to_bytes(decoded_bits)
                
                bits_processed += len(bits)
                
            t_end = time.time()
            duration = t_end - t_start
            fps = num_frames / duration
            kbps = (bits_processed / 1000) / duration
            
            print(f"FPS: {fps:.2f}")
            print(f"Throughput: {kbps:.2f} kbps")
            print(f"Avg Latency per Frame: {duration/num_frames*1000:.2f} ms")
            
        except Exception as e:
            print(f"Failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    benchmark_component_latency()
    benchmark_system_throughput()

import adi
import numpy as np
import time
import argparse
import scipy.signal

# Configuration
DEFAULT_IP = "ip:192.168.2.2"
FC = 2300006561 
FS = 2000000
BW = 1000000
DEFAULT_GAIN = 15

class SimpleOFDMRX:
    def __init__(self):
        self.fft_size = 64
        self.cp_length = 32
        self.active_carriers = np.array([-26, -25, -24, -23, -22, -20, -19, -18, -17, -16, 
                                         -15, -14, -13, -12, -11, -10, -9, -8, -6, -5, -4, -3, -2, -1,
                                         1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 
                                         16, 17, 18, 19, 20, 22, 23, 24, 25, 26])
        self.pilots = np.array([-21, -7, 7, 21])
        self.pilot_values = np.array([1+1j, 1-1j, 1+1j, 1-1j])
        
    def demap_qpsk(self, symbols):
        # Slicing
        bits = []
        for s in symbols:
            # 00 -> 1+j (Q1)
            # 01 -> -1+j (Q2)
            # 11 -> 1-j (Q4) ? No wait.
            # My TX map:
            # 00 -> 1+1j
            # 01 -> -1+1j
            # 11 -> 1-1j 
            # 10 -> -1-1j
            
            # Real > 0 -> b1=0 (or 1?)
            # Let's check TX:
            # b0=0, b1=0 -> 1+1j  => Re>0, Im>0
            # b0=0, b1=1 -> -1+1j => Re<0, Im>0
            # b0=1, b1=0 -> 1-1j  => Re>0, Im<0
            # b0=1, b1=1 -> -1-1j => Re<0, Im<0
            
            re = s.real
            im = s.imag
            
            b0 = 0 if re > 0 else 1 # Wait. 1+1j (00) has Re>0. So if Re>0, b0=0. If Re<0, b0=1.
            # BUT 1-1j (10) has Re>0. So Re>0 -> b0 could be 0 or 1?
            # Let's re-read TX:
            # 00 -> 1+1j (Re+, Im+)
            # 01 -> -1+1j (Re-, Im+)
            # 10 -> -1-1j (Re-, Im-)
            # 11 -> 1-1j (Re+, Im-)  <-- This is gray mapping?
            # TX Code I wrote:
            # 00 -> 1+1j
            # 01 -> -1+1j
            # 11 -> 1-1j  <-- Wait, 11 is 1-1j?
            # 10 -> -1-1j
            
            # Let's invert:
            # Re > 0: Could be 00 or 11 (Wait, 11 is 1-1j). So Re>0 -> b0?
            
            # 00: +,+
            # 01: -,+
            # 10: -,-
            # 11: +,-
            
            # Logic:
            # If Re > 0: b0 = ? 
            #   00 (+) vs 01 (-) vs 10 (-) vs 11 (+)
            #   So Re > 0 implies 00 or 11.
            #   Re < 0 implies 01 or 10.
            #   So if Re > 0, first bit is NOT consistent? 
            #   My TX mapping: b0 refers to first bit?
            #   bits[i] = b0. 
            #   00 -> b0=0. 1+1j. Re>0.
            #   01 -> b0=0. -1+1j. Re<0.
            #   So b0 depends on Re??? No.
            #   Let's check b1 (2nd bit).
            #   00 -> b1=0. Im>0.
            #   01 -> b1=1. Im>0.
            #   So Im>0 implies b1 could be 0 or 1.
            
            # This mapping is WEIRD.
            # 00 (++), 01 (-+), 10 (--), 11 (+-)
            # Q1(00), Q2(01), Q3(10), Q4(11)
            
            if re > 0 and im > 0: bits.extend([0, 0])
            elif re < 0 and im > 0: bits.extend([0, 1])
            elif re < 0 and im < 0: bits.extend([1, 1]) # TX says 11 is 1-1j?? No TX said 10 is -1-1j
            elif re > 0 and im < 0: bits.extend([1, 0]) # TX says 11 is 1-1j. 
            
            # Re-read TX Code:
            # elif b0==1 and b1==1: s = -1-1j  => 11 is (-,-)
            # elif b0==1 and b1==0: s = 1-1j   => 10 is (+,-)  <-- My comments were wrong, code was:
            # b0=1, b1=0 -> 1-1j.
            # So 11 is -1-1j. 10 is 1-1j.
            
            # Correct RX Map:
            # (+,+) -> 00
            # (-,+) -> 01
            # (-,-) -> 11
            # (+,-) -> 10
            
        return np.array(bits)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=DEFAULT_IP, help="SDR IP URI")
    parser.add_argument("--gain", default=DEFAULT_GAIN, type=int, help="RX Gain")
    args = parser.parse_args()

    np.random.seed(42) # Set seed to match TX (Assume TX uses 42) - Wait I need to update TX
    
    print(f"Connecting RX to {args.ip}...")
    sdr = adi.Pluto(uri=args.ip)
    sdr.sample_rate = int(FS)
    sdr.rx_lo = int(FC)
    sdr.rx_rf_bandwidth = int(BW)
    sdr.rx_buffer_size = 65536
    
    # Gain
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = int(args.gain)
    
    # Generate EXPECTED bits
    # We need to match the frame generation exactly
    tx_ofdm = SimpleOFDMRX() # Just to get params
    # We need the random logic 
    num_symbols = 14
    bits_per_symbol = 2
    num_data_carriers = len(tx_ofdm.active_carriers)
    total_bits = num_symbols * num_data_carriers * bits_per_symbol
    expected_bits = np.random.randint(0, 2, total_bits)
    
    # Preamble for Sync
    n = np.arange(127)
    root = 25
    zc = np.exp(-1j * np.pi * root * n * (n + 1) / 127)
    
    while True:
        samples = sdr.rx()
        
        # 1. Sync (Correllation)
        corr = np.correlate(samples, zc, mode='valid')
        peaks = np.abs(corr)
        peak_idx = np.argmax(peaks)
        peak_val = peaks[peak_idx]
        
        if peak_val < 30: # Threshold
            print(".", end="", flush=True)
            continue
            
        print(f"\nLocked! Peak={peak_val:.1f}")
        
        # CFO Estimation (Using Preamble)
        # Preamble is ZC followed by ZC.
        # Length of ZC is 127.
        # We can compare phase of samples[peak_idx : peak_idx+127] and samples[peak_idx+127 : peak_idx+254]
        # Actually ZC is repeated.
        
        # Extract the two halves of the preamble
        # Note: peak_idx points to START of match?
        # If we correlated with ZC, peak_idx is where ZC starts.
        # Since preamble is ZC, ZC.
        # If peak is first ZC, then r1 = samples[peak:peak+127], r2 = samples[peak+127:peak+254]
        
        r1 = samples[peak_idx : peak_idx + 127]
        r2 = samples[peak_idx + 127 : peak_idx + 254]
        
        if len(r1) == 127 and len(r2) == 127:
            # Calculate Phase Difference
            # delta_phi = angle( sum( conj(r1) * r2 ) )
            corr_cfo = np.sum(np.conj(r1) * r2)
            angle = np.angle(corr_cfo)
            # angle = 2*pi*f_off * T_seq
            # T_seq = 127 / FS
            # f_off = angle / (2*pi * T_seq)
            
            t_seq = 127.0 / FS
            cfo_est = angle / (2 * np.pi * t_seq)
            
            print(f"CFO Est: {cfo_est:.1f} Hz")
            
            # Apply Correction
            t = np.arange(len(samples)) / FS
            samples = samples * np.exp(-1j * 2 * np.pi * cfo_est * t)
        
        # Timing Search
        best_ber = 1.0
        best_offset = 0
        
        # ZC Auto-Correllation Peak is usually at index 0 (perfect alignment) of the 'valid' result
        # But we need to know where the FRAME starts relative to that.
        # If we Correlated Samples with ZC.
        # Peak Index `k` means Samples[k : k+127] matches ZC.
        # Our Frame: [ZC, ZC, Gap, Body]
        # Preamble is 2 ZCs.
        # If we match the first ZC, the Body starts at k + 254 + 100.
        # If we match the second ZC, the Body starts at k + 127 + 100.
        
        # We will try a range of coarse offsets to cover both possibilities + fine timing
        # Coarse: First ZC vs Second ZC
        
        candidates = []
        # Assumption 1: Peak is First ZC
        candidates.append(peak_idx + 127*2 + 100)
        # Assumption 2: Peak is Second ZC
        # candidates.append(peak_idx + 127 + 100) # Less likely if we pick Argmax and signal is clean-ish
        
        # Fine Timing Search (+/- 4 samples)
        search_offsets = range(-4, 5)
        
        for base_start in candidates:
            for fine_off in search_offsets:
                payload_start = base_start + fine_off
                
                # Extract and Decode
                rx_bits = []
                current_idx = payload_start
                fft_size = 64
                cp = 32
                sym_len = fft_size + cp
                
                valid_frame = True
                
                # Reuse H est from first symbol? No, estimate per frame.
                # Actually, we should estimate H once per frame using pilots of ALL symbols or averages.
                # Simplified: Estimate per symbol (Block pilots)
                
                for i in range(num_symbols):
                    if current_idx + sym_len > len(samples):
                        valid_frame = False
                        break
                    
                    sym_time = samples[current_idx + cp : current_idx + sym_len]
                    current_idx += sym_len
                    
                    # FFT
                    sym_freq = np.fft.fftshift(np.fft.fft(sym_time / np.sqrt(fft_size)))
                    
                    # Pilots
                    pilots_rx = sym_freq[tx_ofdm.pilots + 32]
                    h_est = pilots_rx / tx_ofdm.pilot_values
                    h_avg = np.mean(h_est)
                    
                    # Eq
                    sym_eq = sym_freq / h_avg
                    
                    # Data
                    data_rx = sym_eq[tx_ofdm.active_carriers + 32]
                    
                    # Demap
                    bits = SimpleOFDMRX().demap_qpsk(data_rx)
                    rx_bits.extend(bits)
                
                if not valid_frame: continue
                
                rx_bits = np.array(rx_bits)
                L = min(len(rx_bits), len(expected_bits))
                if L == 0: continue
                
                errors = np.sum(rx_bits[:L] != expected_bits[:L])
                ber = errors / L
                
                if ber < best_ber:
                    best_ber = ber
                    best_offset = fine_off
                    best_rx = rx_bits[:20] # Capture first 20 for debug
                    
        print(f"Best BER: {best_ber:.5f} (Offset {best_offset})")
        if best_ber < 0.2:
            print(f"SUCCESS! Link Functional. Rx: {best_rx} Exp: {expected_bits[:20]}")
            # return # Optional: Exit on success
        else:
            print(f"Debug: Exp {expected_bits[:10]}")
            print(f"       Rx  {best_rx[:10] if 'best_rx' in locals() else 'None'}")

if __name__ == "__main__":
    main()

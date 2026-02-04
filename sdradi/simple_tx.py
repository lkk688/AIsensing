import adi
import numpy as np
import time
import argparse

# Configuration matching previous successful probe
SDR_IP = "ip:192.168.3.2"
FC = 2300006561 # Corrected Frequency
FS = 2000000    # 2 MHz
BW = 1000000    # 1 MHz
TX_GAIN = -5

class SimpleOFDM:
    def __init__(self):
        self.fft_size = 64
        self.cp_length = 32
        self.active_carriers = np.array([-26, -25, -24, -23, -22, -20, -19, -18, -17, -16, 
                                         -15, -14, -13, -12, -11, -10, -9, -8, -6, -5, -4, -3, -2, -1,
                                         1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 
                                         16, 17, 18, 19, 20, 22, 23, 24, 25, 26])
        self.pilots = np.array([-21, -7, 7, 21])
        self.pilot_values = np.array([1+1j, 1-1j, 1+1j, 1-1j])
        self.all_carriers = np.arange(-32, 32)
        
    def qpsk_modul(self, bits):
        # 00 -> 1+j, 01 -> -1+j, 10 -> -1-j, 11 -> 1-j
        # Simple mapping
        syms = []
        for i in range(0, len(bits), 2):
            b0 = bits[i]
            b1 = bits[i+1]
            if b0==0 and b1==0: s = 1+1j
            elif b0==0 and b1==1: s = -1+1j
            elif b0==1 and b1==1: s = -1-1j
            elif b0==1 and b1==0: s = 1-1j
            syms.append(s)
        return np.array(syms) / np.sqrt(2) # Normalize

    def generate_frame(self, num_symbols=14):
        # 1. Preamble (Zadoff-Chu) for Sync
        n = np.arange(127)
        root = 25
        zc = np.exp(-1j * np.pi * root * n * (n + 1) / 127)
        preamble = np.concatenate([np.zeros(64), zc, np.zeros(65)]) # Pad to roughly resemble standard
        # Actually standard ZC is time domain. Let's send Time Domain ZC directly.
        # Reuse probe preamble logic for robust sync
        time_preamble = np.concatenate([zc, zc]) # 2 repetitions
        
        # 2. OFDM Symbols
        ofdm_samples = []
        
        # Generate Random Bits
        bits_per_symbol = 2 # QPSK
        num_data_carriers = len(self.active_carriers)
        total_bits = num_symbols * num_data_carriers * bits_per_symbol
        tx_bits = np.random.randint(0, 2, total_bits)
        
        symbols = self.qpsk_modul(tx_bits)
        
        idx = 0
        for s in range(num_symbols):
            freq_data = np.zeros(self.fft_size, dtype=complex)
            
            # Map Data
            chunk = symbols[idx : idx+num_data_carriers]
            idx += num_data_carriers
            freq_data[self.active_carriers + 32] = chunk # Shift to 0..63
            
            # Map Pilots
            freq_data[self.pilots + 32] = self.pilot_values
            
            # IFFT
            time_sym = np.fft.ifft(np.fft.ifftshift(freq_data)) * np.sqrt(self.fft_size)
            
            # CP
            cp = time_sym[-self.cp_length:]
            ofdm_sym = np.concatenate([cp, time_sym])
            # ofdm_samples.append(ofdm_samples) # BUG REMOVED
            ofdm_samples.append(ofdm_sym)

        body = np.concatenate(ofdm_samples)
        
        # Combine
        # Gap
        gap = np.zeros(100, dtype=complex)
        
        frame = np.concatenate([time_preamble, gap, body, gap])
        
        return frame, tx_bits

def main():
    print("Starting Simple TX (No CV2/Torch)...")
    np.random.seed(42) # Sync with RX
    
    # 1. Setup SDR
    sdr = adi.Pluto(uri=SDR_IP)
    sdr.sample_rate = int(FS)
    sdr.tx_lo = int(FC)
    sdr.tx_rf_bandwidth = int(BW)
    
    # 2. High Power Config
    sdr.tx_enabled_channels = [0]
    sdr.tx_hardwaregain_chan0 = int(TX_GAIN)
    sdr.tx_hardwaregain_chan1 = int(TX_GAIN) # Dual gain fix
    
    # 3. Generate Signal
    ofdm = SimpleOFDM()
    frame, bits = ofdm.generate_frame()
    
    # Normalize
    max_val = np.max(np.abs(frame))
    if max_val > 0:
        frame = frame / max_val * 0.8 # 0.8 Amplitude
        
    data = frame * (2**14)
    
    # 4. Padding for DMA
    MIN_BUFFER = 32768 * 2
    if len(data) < MIN_BUFFER:
        repeats = int(np.ceil(MIN_BUFFER / len(data)))
        data = np.tile(data, repeats)
        
    print(f"Transmitting {len(data)} samples...")
    
    sdr.tx_cyclic_buffer = True
    sdr.tx(data)
    
    # Keep alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sdr.tx_destroy_buffer()

if __name__ == "__main__":
    main()

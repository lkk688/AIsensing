import numpy as np
import time
import cv2
import sys
import matplotlib.pyplot as plt
from sdr_video_comm import SDRVideoLink, SDRConfig, OFDMConfig, FECConfig, FECType, WaveformType

def generate_test_image(width=640, height=480):
    # Create a gradient image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        img[i, :, 0] = i % 255  # Blue gradient
    for j in range(width):
        img[:, j, 1] = j % 255  # Green gradient
    img[:, :, 2] = 255          # Red channel constant
    
    # Add some text
    cv2.putText(img, "PlutoSDR Video Test", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return img

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def test_hardware_video_loopback():
    print("=== PlutoSDR Hardware Video Loopback Test ===")
    
    # 1. Config
    # 1. Config
    sdr_config = SDRConfig.load_from_json()
    print(f"Loaded Config: FC={sdr_config.fc/1e9}G, FS={sdr_config.fs/1e6}M, TX={sdr_config.tx_gain}, RX={sdr_config.rx_gain}")
    
    # Ensure correct device ip if it differs (usually handled by json, but safety check)
    if sdr_config.sdr_ip == 'ip:192.168.86.40': # Default override
         sdr_config.sdr_ip = 'ip:192.168.2.1'
         sdr_config.device = 'pluto'
    
    # Use OFDM + LDPC for robustness
    # Default OFDM is 16-QAM. Let's make it QPSK for robustness
    # Override Sample Rate to 1 MSPS to prevent USB Timeouts
    sdr_config.fs = 1e6
    sdr_config.bandwidth = 1e6
    # Default OFDM is 16-QAM. Let's make it QPSK (mod_order=4) for initial link test
    ofdm_config = OFDMConfig(mod_order=4)
    # Try Repetition Code for maximum robustness
    fec_config = FECConfig(enabled=True, fec_type=FECType.REPETITION, repetitions=7)
    
    link = SDRVideoLink(
        sdr_config=sdr_config,
        ofdm_config=ofdm_config,
        fec_config=fec_config,
        waveform=WaveformType.OFDM,
        simulation_mode=False
    )
    
    if not link.connect_sdr():
        print("Failed to connect to SDR.")
        return
        
    # Enable Digital Loopback (Internal) for verification
    print("Enabling Internal Digital Loopback... (Set to 0 for RF)")
    link.sdr.sdr.loopback = 0 # 0=Disable (RF), 1=Digital, 2=RF

        
    # 2. Prepare Data
    print("Generating test frame...")
    original_frame = generate_test_image()
    cv2.imwrite("tx_video_frame.jpg", original_frame)
    
    # Convert to JPEG bytes -> Bits (Direct mode)
    # Convert to JPEG bytes -> Bits (Direct mode)
    # Quality=30 for small payload (~5KB)
    _, jpg_encoded = cv2.imencode('.jpg', original_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
    jpg_bytes = jpg_encoded.tobytes()
    # Use helper to convert to bits
    tx_bits = link.video_codec.bytes_to_bits(jpg_bytes)
    
    print(f"Frame Size: {len(tx_bits)/8/1024:.2f} KB ({len(tx_bits)} bits)")
    
    # Encode (FEC) -> Modulate
    encoded_bits = link.fec_codec.encode(tx_bits)
    tx_signal = link.transceiver.modulate(encoded_bits)
    
    print(f"Modulated Symbols: {len(tx_signal)}")
    
    # Add Preamble
    preamble = link._generate_preamble()
    full_tx_signal = np.concatenate([preamble, tx_signal, np.zeros(500)]) # Padding
    
    # 3. Transmit (Cyclic)
    print("Transmitting (Cyclic Mode)...")
    # Need to access low-level SDR for cyclic mode w/o destroying buffer immediately
    # SDRVideoLink.transmit doesn't support cyclic kwarg, so we use link.sdr directly
    link.sdr.SDR_TX_send(full_tx_signal, cyclic=True)
    
    time.sleep(1.0) # Let it stabilize
    
    # 4. Receive
    print("Receiving...")
    # Capture enough samples to guarantee finding the repetition
    # Capture enough samples to guarantee finding the repetition
    # 3x is sufficient for cyclic sync (guarantees >1 full copy) and reduces USB load
    num_rx = len(full_tx_signal) * 3
    link.sdr.SDR_RX_setup(n_SAMPLES=num_rx)
    rx_signal = link.sdr.SDR_RX_receive()
    
    # Handle cyclic wrap by doubling buffer
    rx_signal = np.concatenate([rx_signal, rx_signal])
    
    # Stop TX
    link.sdr.SDR_TX_stop()
    
    print(f"Captured {len(rx_signal)} samples.")
    
    # 5. Process
    print("Synchronizing and Correcting CFO...")
    try:
        # Use robust synchronization from the link class
        # This performs Peak Detection -> CFO Estimation -> Correction -> Payload Extraction
        payload_signal = link._synchronize(rx_signal)
        
        # Determine strict length (standardize to TX length)
        # Note: _synchronize returns everything after preamble. We crop to expected length.
        mod_len = len(tx_signal)
        if len(payload_signal) < mod_len:
             print(f"Warning: Payload too short ({len(payload_signal)} < {mod_len})")
             # Pad?
             payload_signal = np.concatenate([payload_signal, np.zeros(mod_len - len(payload_signal), dtype=complex)])
        else:
             payload_signal = payload_signal[:mod_len]

        # Demodulate
        print("Demodulating...")
        demod_bits, metrics = link.transceiver.demodulate(payload_signal)
        
        # FEC Decode
        print("FEC Decoding...")
        decoded_bits = link.fec_codec.decode(demod_bits)
        
        # BER Check
        min_len = min(len(tx_bits), len(decoded_bits))
        errors = np.sum(tx_bits[:min_len] != decoded_bits[:min_len])
        ber = errors / min_len
        print(f"Loopback BER: {ber:.5f} ({errors} errors)")
        
        # Reassemble to Bytes
        decoded_bytes = link.video_codec.bits_to_bytes(decoded_bits)
        
        # Save received
        with open("rx_video_payload.bin", "wb") as f:
            f.write(decoded_bytes)
            
        # Decode Image (Direct)
        print("Decoding JPEG...")
        rx_array = np.frombuffer(decoded_bytes, dtype=np.uint8)
        rx_frame = cv2.imdecode(rx_array, cv2.IMREAD_COLOR)
        
        if rx_frame is None:
            print("FAILURE: Could not decode JPEG frame (Corruption too high).")
        else:
            cv2.imwrite("rx_video_frame.jpg", rx_frame)
            print("Saved 'rx_video_frame.jpg'")
            
            # Metrics
            psnr = calculate_psnr(original_frame, rx_frame)
            print(f"SUCCESS! PSNR: {psnr:.2f} dB")
            
            if psnr > 30:
                print("Video Quality: Excellent")
            elif psnr > 20:
                print("Video Quality: Good")
            else:
                print("Video Quality: Poor")
                
    except Exception as e:
        print(f"Processing Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hardware_video_loopback()

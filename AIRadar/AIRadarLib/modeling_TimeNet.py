import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from AIRadarLib.ofdm_decoder import OFDMDecoder

# === Learnable FFT Block ===
class LearnableFFT(nn.Module):
    """
    Learnable FFT block that can be trained to perform FFT-like operations.
    
    This module implements a learnable linear transformation that approximates the FFT.
    It maintains separate real and imaginary weight matrices to perform complex multiplication.
    
    Mathematical formulation:
    For input x = a + bi and weights W = c + di:
    output = (a⊗c - b⊗d) + (a⊗d + b⊗c)i
    where ⊗ represents matrix multiplication
    """
    def __init__(self, input_len, output_len):
        """
        Initialize the LearnableFFT module.
        
        Args:
            input_len: Length of the input signal
            output_len: Length of the output signal
        """
        super().__init__()
        # Initialize real and imaginary weight matrices with Glorot initialization
        self.real = nn.Parameter(torch.randn(input_len, output_len) / math.sqrt(input_len))
        self.imag = nn.Parameter(torch.randn(input_len, output_len) / math.sqrt(input_len))

    def forward(self, real_input, imag_input):
        """
        Forward pass of the LearnableFFT module.
        
        Args:
            real_input: Real part of the input signal [B, input_len]
            imag_input: Imaginary part of the input signal [B, input_len]
            
        Returns:
            Complex output as real and imaginary parts [B, output_len, 2]
        """
        # Perform complex matrix multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        r_part = torch.matmul(real_input, self.real) - torch.matmul(imag_input, self.imag)
        i_part = torch.matmul(real_input, self.imag) + torch.matmul(imag_input, self.real)
        
        # Stack real and imaginary parts along the last dimension
        return torch.stack([r_part, i_part], dim=-1)

# === OFDM Demodulation Module ===
class OFDMDemodulator(nn.Module):
    """
    OFDM demodulation module that can be trained to perform OFDM demodulation.
    
    This module implements OFDM demodulation by:
    1. Removing cyclic prefix (if present)
    2. Performing FFT on each OFDM symbol
    3. Extracting data from active subcarriers
    
    Mathematical formulation:
    For OFDM symbol y with cyclic prefix of length cp:
    1. Remove CP: y' = y[cp:cp+N]
    2. Perform FFT: Y = FFT(y')
    3. Extract data: X = Y[active_carriers]
    """
    def __init__(self, fft_size, cp_length=0, learnable=True):
        """
        Initialize the OFDMDemodulator module.
        
        Args:
            fft_size: Size of the FFT
            cp_length: Length of the cyclic prefix
            learnable: Whether to use learnable FFT or standard FFT
        """
        super().__init__()
        self.fft_size = fft_size
        self.cp_length = cp_length
        self.learnable = learnable
        
        if learnable:
            self.fft = LearnableFFT(fft_size, fft_size)

    def forward(self, x):
        """
        Forward pass of the OFDMDemodulator module.
        
        Args:
            x: Input signal with shape [B, num_symbols, symbol_length, 2]
               where symbol_length = fft_size + cp_length
               
        Returns:
            Demodulated OFDM symbols with shape [B, num_symbols, fft_size, 2]
        """
        batch_size, num_symbols, symbol_length, _ = x.shape
        
        # Remove cyclic prefix if present
        if self.cp_length > 0:
            x = x[:, :, self.cp_length:self.cp_length + self.fft_size, :]
        
        # Reshape for processing
        x = x.reshape(batch_size * num_symbols, self.fft_size, 2)
        real_part, imag_part = x[..., 0], x[..., 1]
        
        if self.learnable:
            # Use learnable FFT
            output = self.fft(real_part, imag_part)
        else:
            # Use standard FFT
            complex_input = torch.complex(real_part, imag_part)
            complex_output = torch.fft.fft(complex_input, dim=1)
            output = torch.stack([complex_output.real, complex_output.imag], dim=-1)
        
        # Reshape back to original dimensions
        return output.reshape(batch_size, num_symbols, self.fft_size, 2)


# === RadarTimeNet: processes time-domain IQ signals ===
class RadarTimeNet(nn.Module):
    """
    Deep learning model for processing time-domain IQ signals to extract range-Doppler maps.
    
    This model can process raw IQ time-domain signals, perform demodulation (including OFDM
    demodulation if applicable), and output a range-Doppler map. It can be initialized with
    traditional range-Doppler map calculation capabilities through pretraining.
    
    The processing pipeline includes:
    1. Time-domain preprocessing with 3D convolutions
    2. Demodulation (mixing) with reference signal
    3. Range FFT processing
    4. Doppler FFT processing
    5. Post-processing with 2D convolutions
    """
    def __init__(self, num_rx=2, num_chirps=64, samples_per_chirp=64, 
                 out_doppler_bins=64, out_range_bins=64, use_learnable_fft=True,
                 support_ofdm=True, ofdm_modulation='qpsk'):
        """
        Initialize the RadarTimeNet module.
        
        Args:
            num_rx: Number of receive antennas
            num_chirps: Number of chirps in the input signal
            samples_per_chirp: Number of samples per chirp
            out_doppler_bins: Number of Doppler bins in the output
            out_range_bins: Number of range bins in the output
            use_learnable_fft: Whether to use learnable FFT or standard FFT
            support_ofdm: Whether to support OFDM demodulation
        """
        super().__init__()
        self.num_rx = num_rx
        self.num_chirps = num_chirps
        self.samples_per_chirp = samples_per_chirp
        self.out_doppler_bins = out_doppler_bins
        self.out_range_bins = out_range_bins
        self.use_learnable_fft = use_learnable_fft
        self.support_ofdm = support_ofdm
        self.ofdm_modulation = ofdm_modulation
        
        # === Time-domain preprocessing ===
        # Input shape: [B, num_rx, num_chirps, samples_per_chirp, 2]
        # Output shape: [B, 32, num_rx, num_chirps, samples_per_chirp]
        self.time_conv = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        # === Demodulation module (mixing with reference) ===
        # Learnable complex multiplication for demodulation
        # Implements y = x * conj(ref) where x is the received signal and ref is the reference signal
        self.demod_weights = nn.Parameter(torch.randn(2, 2) / math.sqrt(2))
        
        # === Range FFT processing ===
        # Process each chirp with range FFT
        if use_learnable_fft:
            self.range_fft = LearnableFFT(samples_per_chirp, out_range_bins)
        else:
            self.range_fft = None
            
        # === Doppler FFT processing ===
        # Process each range bin with Doppler FFT
        if use_learnable_fft:
            self.doppler_fft = LearnableFFT(num_chirps, out_doppler_bins)
        else:
            self.doppler_fft = None
            
        # === OFDM demodulation module ===
        if support_ofdm:
            self.ofdm_demod = OFDMDemodulator(samples_per_chirp, cp_length=0, learnable=use_learnable_fft)
            
            # OFDM detection head
            self.ofdm_head = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 2, kernel_size=1)
            )
            
            # OFDM symbol decoder for bit extraction
            self.ofdm_decoder = OFDMDecoder(
                fft_size=samples_per_chirp,
                num_symbols=num_chirps,
                use_channel_estimation=True
            )
        
        # === Post-processing for range-Doppler map ===
        # Process the range-Doppler map with 2D convolutions
        self.rd_conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final output layer
        self.output = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1)
        )
        
        # Initialize with FFT-like weights
        self._init_fft_weights()
        
    def _init_fft_weights(self):
        """
        Initialize the learnable FFT weights to mimic the standard FFT.
        This helps the model converge faster during training.
        """
        if self.use_learnable_fft and self.range_fft is not None:
            # Initialize range FFT weights
            N = self.samples_per_chirp
            for k in range(self.out_range_bins):
                for n in range(N):
                    angle = -2 * np.pi * k * n / N
                    self.range_fft.real.data[n, k] = np.cos(angle) / np.sqrt(N)
                    self.range_fft.imag.data[n, k] = np.sin(angle) / np.sqrt(N)
            
            # Initialize Doppler FFT weights
            N = self.num_chirps
            for k in range(self.out_doppler_bins):
                for n in range(N):
                    angle = -2 * np.pi * k * n / N
                    self.doppler_fft.real.data[n, k] = np.cos(angle) / np.sqrt(N)
                    self.doppler_fft.imag.data[n, k] = np.sin(angle) / np.sqrt(N)
        
        # Initialize demodulation weights for complex conjugate multiplication
        self.demod_weights.data = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.float32)

    def complex_multiply(self, x, y):
        """
        Perform complex multiplication between two tensors.
        
        Args:
            x: First tensor with shape [..., 2] (real, imag)
            y: Second tensor with shape [..., 2] (real, imag)
            
        Returns:
            Complex product with shape [..., 2]
        """
        # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        real = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
        imag = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
        return torch.stack([real, imag], dim=-1)
    
    def complex_conjugate(self, x):
        """
        Compute the complex conjugate of a tensor.
        
        Args:
            x: Input tensor with shape [..., 2] (real, imag)
            
        Returns:
            Complex conjugate with shape [..., 2]
        """
        return torch.stack([x[..., 0], -x[..., 1]], dim=-1)
    
    def demodulate(self, rx_signal, ref_signal=None):
        """
        Demodulate the received signal by mixing with the reference signal.
        
        Args:
            rx_signal: Received signal with shape [..., 2]
            ref_signal: Reference signal with shape [..., 2], if None, use learnable demodulation
            
        Returns:
            Demodulated signal with shape [..., 2]
        """
        if ref_signal is not None:
            # Use provided reference signal
            # y = x * conj(ref)
            return self.complex_multiply(rx_signal, self.complex_conjugate(ref_signal))
        else:
            # Use learnable demodulation
            # Apply the demodulation weights to the input
            # [B, num_rx, num_chirps, samples_per_chirp, 2]
            batch_size = rx_signal.shape[0]
            rx_signal_flat = rx_signal.reshape(-1, 2)  # Flatten all dimensions except the last
            demod_signal_flat = torch.matmul(rx_signal_flat, self.demod_weights)
            return demod_signal_flat.reshape(*rx_signal.shape)
    
    def apply_range_fft(self, x):
        """
        Apply range FFT to the input signal.
        
        The range FFT converts the time-domain signal to the range domain.
        For FMCW radar, the frequency after mixing is proportional to the target range.
        
        Mathematical formulation:
        Range FFT: X[k] = ∑_{n=0}^{N-1} x[n] * e^{-j2πkn/N}
        
        Args:
            x: Input signal with shape [B, num_rx, num_chirps, samples_per_chirp, 2]
            
        Returns:
            Range spectrum with shape [B, num_rx, num_chirps, out_range_bins, 2]
        """
        batch_size, num_rx, num_chirps, samples_per_chirp, _ = x.shape
        
        # Reshape for processing
        x_reshaped = x.reshape(batch_size * num_rx * num_chirps, samples_per_chirp, 2)
        real_part, imag_part = x_reshaped[..., 0], x_reshaped[..., 1]
        
        if self.use_learnable_fft and self.range_fft is not None:
            # Use learnable FFT
            range_spectrum = self.range_fft(real_part, imag_part)
        else:
            # Use standard FFT
            complex_input = torch.complex(real_part, imag_part)
            complex_output = torch.fft.fft(complex_input, n=self.out_range_bins, dim=1)
            range_spectrum = torch.stack([complex_output.real, complex_output.imag], dim=-1)
        
        # Reshape back to original dimensions
        return range_spectrum.reshape(batch_size, num_rx, num_chirps, self.out_range_bins, 2)
    
    def apply_doppler_fft(self, x):
        """
        Apply Doppler FFT to the input signal.
        
        The Doppler FFT converts the chirp-domain signal to the Doppler domain.
        For FMCW radar, the phase change across chirps is proportional to the target velocity.
        
        Mathematical formulation:
        Doppler FFT: X[k] = ∑_{n=0}^{N-1} x[n] * e^{-j2πkn/N}
        
        Args:
            x: Input signal with shape [B, num_rx, num_chirps, out_range_bins, 2]
            
        Returns:
            Range-Doppler map with shape [B, num_rx, out_doppler_bins, out_range_bins, 2]
        """
        batch_size, num_rx, num_chirps, range_bins, _ = x.shape
        
        # Transpose to put chirps in the right dimension for FFT
        x_transposed = x.permute(0, 1, 3, 2, 4)  # [B, num_rx, range_bins, num_chirps, 2]
        
        # Reshape for processing
        x_reshaped = x_transposed.reshape(batch_size * num_rx * range_bins, num_chirps, 2)
        real_part, imag_part = x_reshaped[..., 0], x_reshaped[..., 1]
        
        if self.use_learnable_fft and self.doppler_fft is not None:
            # Use learnable FFT
            doppler_spectrum = self.doppler_fft(real_part, imag_part)
        else:
            # Use standard FFT
            complex_input = torch.complex(real_part, imag_part)
            complex_output = torch.fft.fft(complex_input, n=self.out_doppler_bins, dim=1)
            # Apply FFT shift to center the Doppler spectrum
            complex_output = torch.fft.fftshift(complex_output, dim=1)
            doppler_spectrum = torch.stack([complex_output.real, complex_output.imag], dim=-1)
        
        # Reshape back to original dimensions
        return doppler_spectrum.reshape(batch_size, num_rx, range_bins, self.out_doppler_bins, 2).permute(0, 1, 3, 2, 4)
    
    def process_ofdm(self, x, is_ofdm=False, modulation=None):
        """
        Process OFDM signal if applicable.
        
        Args:
            x: Input signal with shape [B, num_rx, num_chirps, samples_per_chirp, 2]
            is_ofdm: Whether the input signal is OFDM modulated
            modulation: Modulation scheme to use for decoding (overrides self.ofdm_modulation if provided)
            
        Returns:
            Tuple of:
            - OFDM demodulated signal with shape [B, 2, out_doppler_bins, out_range_bins]
            - Decoded bits with shape [B, num_symbols * num_active_subcarriers * bits_per_symbol]
        """
        if not self.support_ofdm or not is_ofdm:
            return None
            
        batch_size, num_rx, num_chirps, samples_per_chirp, _ = x.shape
        
        # Reshape for OFDM processing
        # Treat chirps as OFDM symbols
        x_ofdm = x.reshape(batch_size * num_rx, num_chirps, samples_per_chirp, 2)
        
        # Apply OFDM demodulation
        ofdm_demod = self.ofdm_demod(x_ofdm)
        
        # Reshape to [B*num_rx, 2, num_chirps, samples_per_chirp]
        ofdm_demod = ofdm_demod.permute(0, 3, 1, 2)
        
        # Apply OFDM detection head
        ofdm_output = self.ofdm_head(ofdm_demod.reshape(batch_size * num_rx, 2, num_chirps, samples_per_chirp))
        
        # Reshape to [B, 2, out_doppler_bins, out_range_bins]
        ofdm_map = ofdm_output.reshape(batch_size, num_rx, 2, num_chirps, samples_per_chirp).mean(dim=1)
        
        # Decode OFDM symbols to bits
        modulation_scheme = modulation if modulation is not None else self.ofdm_modulation
        decoded_bits = self.ofdm_decoder(ofdm_map, modulation_scheme)
        
        return ofdm_map, decoded_bits

    def forward(self, x, ref_signal=None, is_ofdm=False, modulation=None):
        """
        Forward pass of the RadarTimeNet module.
        
        Args:
            x: Input signal with shape [B, num_rx, num_chirps, samples_per_chirp, 2]
            ref_signal: Optional reference signal for demodulation
            is_ofdm: Whether the input signal is OFDM modulated
            modulation: Modulation scheme to use for OFDM decoding (overrides self.ofdm_modulation if provided)
            
        Returns:
            Range-Doppler map with shape [B, 2, out_doppler_bins, out_range_bins]
            If is_ofdm is True and support_ofdm is True, also returns:
            - OFDM map with shape [B, 2, out_doppler_bins, out_range_bins]
            - Decoded bits with shape [B, num_symbols * num_active_subcarriers * bits_per_symbol]
        """
        # Input shape: [B, num_rx, num_chirps, samples_per_chirp, 2]
        batch_size = x.shape[0]
        
        # === Step 1: Time-domain preprocessing ===
        # Permute to [B, 2, num_rx, num_chirps, samples_per_chirp]
        x = x.permute(0, 4, 1, 2, 3)
        
        # Apply 3D convolution for time-domain preprocessing
        # Output shape: [B, 32, num_rx, num_chirps, samples_per_chirp]
        x = self.time_conv(x)
        
        # Permute back to [B, num_rx, num_chirps, samples_per_chirp, 2]
        # and combine the channel dimension with batch for processing
        x = torch.cat([x[:, :16], x[:, 16:]], dim=-1).permute(0, 2, 3, 4, 1)
        
        # === Step 2: Demodulation (mixing with reference) ===
        # Output shape: [B, num_rx, num_chirps, samples_per_chirp, 2]
        x = self.demodulate(x, ref_signal)
        
        # === Step 3: Range FFT processing ===
        # Output shape: [B, num_rx, num_chirps, out_range_bins, 2]
        x = self.apply_range_fft(x)
        
        # === Step 4: Doppler FFT processing ===
        # Output shape: [B, num_rx, out_doppler_bins, out_range_bins, 2]
        x = self.apply_doppler_fft(x)
        
        # === Process OFDM if applicable ===
        ofdm_map = None
        decoded_bits = None
        if self.support_ofdm and is_ofdm:
            ofdm_map, decoded_bits = self.process_ofdm(x, is_ofdm, modulation)
        
        # === Step 5: Post-processing ===
        # Average across receive antennas
        # Output shape: [B, out_doppler_bins, out_range_bins, 2]
        x = x.mean(dim=1)
        
        # Permute to [B, 2, out_doppler_bins, out_range_bins] for 2D convolution
        x = x.permute(0, 3, 1, 2)
        
        # Apply 2D convolution for post-processing
        # Output shape: [B, 64, out_doppler_bins, out_range_bins]
        x = self.rd_conv(x)
        
        # Final output layer
        # Output shape: [B, 2, out_doppler_bins, out_range_bins]
        x = self.output(x)
        
        if self.support_ofdm and is_ofdm:
            return x, ofdm_map, decoded_bits
        else:
            return x


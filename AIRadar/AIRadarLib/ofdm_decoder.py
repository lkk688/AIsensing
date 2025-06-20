import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class OFDMSymbolDecoder(nn.Module):
    """
    OFDM symbol decoder that can decode symbols from the OFDM map returned by RadarTimeNet.
    
    This module implements OFDM symbol decoding by:
    1. Extracting symbols from the OFDM map
    2. Performing constellation demapping based on the specified modulation scheme
    3. Converting constellation points to bits
    
    Supported modulation schemes:
    - BPSK: 1 bit per symbol
    - QPSK: 2 bits per symbol
    - QAM16: 4 bits per symbol
    - QAM64: 6 bits per symbol
    - QAM256: 8 bits per symbol
    """
    def __init__(self, fft_size, num_subcarriers=None, dc_null=True, guard_bands=None):
        """
        Initialize the OFDMSymbolDecoder module.
        
        Args:
            fft_size: Size of the FFT used in OFDM
            num_subcarriers: Number of active subcarriers (if None, use all except DC)
            dc_null: Whether the DC subcarrier is nulled
            guard_bands: List of two integers specifying left and right guard band sizes
                         (if None, no guard bands are used)
        """
        super().__init__()
        self.fft_size = fft_size
        
        # Determine active subcarriers
        if num_subcarriers is None:
            num_subcarriers = fft_size - (1 if dc_null else 0)
            
        self.num_subcarriers = num_subcarriers
        self.dc_null = dc_null
        
        # Calculate guard bands
        if guard_bands is None:
            guard_bands = [0, 0]
        self.guard_left, self.guard_right = guard_bands
        
        # Calculate active subcarrier indices
        self.active_subcarriers = self._calculate_active_subcarriers()
        
        # Define constellation mappings for different modulation schemes
        self.constellation_maps = {
            'bpsk': self._create_bpsk_constellation(),
            'qpsk': self._create_qpsk_constellation(),
            'qam16': self._create_qam16_constellation(),
            'qam64': self._create_qam64_constellation(),
            'qam256': self._create_qam256_constellation()
        }
        
        # Define bits per symbol for each modulation scheme
        self.bits_per_symbol = {
            'bpsk': 1,
            'qpsk': 2,
            'qam16': 4,
            'qam64': 6,
            'qam256': 8
        }
    
    def _calculate_active_subcarriers(self):
        """
        Calculate the indices of active subcarriers based on configuration.
        
        Returns:
            Tensor of active subcarrier indices
        """
        # Start with all subcarriers
        all_indices = torch.arange(self.fft_size)
        
        # Apply guard bands
        valid_indices = all_indices[self.guard_left:self.fft_size-self.guard_right]
        
        # Remove DC if needed
        if self.dc_null:
            dc_index = self.fft_size // 2
            valid_indices = torch.cat([valid_indices[:dc_index], valid_indices[dc_index+1:]])
        
        # Ensure we have the right number of subcarriers
        if len(valid_indices) > self.num_subcarriers:
            # If we have too many, take from the middle
            start_idx = (len(valid_indices) - self.num_subcarriers) // 2
            valid_indices = valid_indices[start_idx:start_idx+self.num_subcarriers]
        
        return valid_indices
    
    def _create_bpsk_constellation(self):
        """
        Create BPSK constellation mapping.
        
        Returns:
            Tensor of constellation points and corresponding bit patterns
        """
        # BPSK: {0: -1, 1: 1}
        points = torch.tensor([-1.0, 1.0], dtype=torch.float32)
        bits = torch.tensor([[0], [1]], dtype=torch.int8)
        return points, bits
    
    def _create_qpsk_constellation(self):
        """
        Create QPSK constellation mapping.
        
        Returns:
            Tensor of constellation points and corresponding bit patterns
        """
        # QPSK: {00: -1-1j, 01: -1+1j, 10: 1-1j, 11: 1+1j}
        real = torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=torch.float32)
        imag = torch.tensor([-1.0, 1.0, -1.0, 1.0], dtype=torch.float32)
        points = torch.complex(real, imag) / math.sqrt(2)
        bits = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.int8)
        return points, bits
    
    def _create_qam16_constellation(self):
        """
        Create 16-QAM constellation mapping.
        
        Returns:
            Tensor of constellation points and corresponding bit patterns
        """
        # 16-QAM: 4 bits per symbol, points at (-3, -3), (-3, -1), ..., (3, 3)
        values = torch.tensor([-3.0, -1.0, 1.0, 3.0], dtype=torch.float32)
        real, imag = torch.meshgrid(values, values, indexing='ij')
        points = torch.complex(real.flatten(), imag.flatten()) / math.sqrt(10)  # Normalize energy
        
        # Generate Gray-coded bit patterns
        gray_map = torch.tensor([0, 1, 3, 2], dtype=torch.int8)
        bits_real = torch.zeros((16, 2), dtype=torch.int8)
        bits_imag = torch.zeros((16, 2), dtype=torch.int8)
        
        for i in range(4):
            idx = torch.arange(i*4, (i+1)*4)
            bits_real[idx, :] = torch.tensor([[gray_map[i] >> 1, gray_map[i] & 1]] * 4)
            
        for i in range(4):
            idx = torch.arange(i, 16, 4)
            bits_imag[idx, :] = torch.tensor([[gray_map[i] >> 1, gray_map[i] & 1]] * 4)
        
        bits = torch.cat([bits_real, bits_imag], dim=1)
        return points, bits
    
    def _create_qam64_constellation(self):
        """
        Create 64-QAM constellation mapping.
        
        Returns:
            Tensor of constellation points and corresponding bit patterns
        """
        # 64-QAM: 6 bits per symbol
        values = torch.tensor([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=torch.float32)
        real, imag = torch.meshgrid(values, values, indexing='ij')
        points = torch.complex(real.flatten(), imag.flatten()) / math.sqrt(42)  # Normalize energy
        
        # Generate Gray-coded bit patterns (simplified approach)
        bits = torch.zeros((64, 6), dtype=torch.int8)
        for i in range(64):
            gray_i = i ^ (i >> 1)  # Convert to Gray code
            for j in range(6):
                bits[i, 5-j] = (gray_i >> j) & 1
        
        return points, bits
    
    def _create_qam256_constellation(self):
        """
        Create 256-QAM constellation mapping.
        
        Returns:
            Tensor of constellation points and corresponding bit patterns
        """
        # 256-QAM: 8 bits per symbol
        values = torch.tensor([-15.0, -13.0, -11.0, -9.0, -7.0, -5.0, -3.0, -1.0, 
                              1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0], dtype=torch.float32)
        real, imag = torch.meshgrid(values, values, indexing='ij')
        points = torch.complex(real.flatten(), imag.flatten()) / math.sqrt(170)  # Normalize energy
        
        # Generate Gray-coded bit patterns (simplified approach)
        bits = torch.zeros((256, 8), dtype=torch.int8)
        for i in range(256):
            gray_i = i ^ (i >> 1)  # Convert to Gray code
            for j in range(8):
                bits[i, 7-j] = (gray_i >> j) & 1
        
        return points, bits
    
    def demapping(self, symbols, modulation='qpsk'):
        """
        Perform constellation demapping to convert symbols to bits.
        
        Args:
            symbols: Complex symbols to demap with shape [..., num_symbols]
            modulation: Modulation scheme ('bpsk', 'qpsk', 'qam16', 'qam64', 'qam256')
            
        Returns:
            Tensor of bits with shape [..., num_symbols * bits_per_symbol]
        """
        # Get constellation for the specified modulation
        if modulation.lower() not in self.constellation_maps:
            raise ValueError(f"Unsupported modulation scheme: {modulation}")
            
        constellation, bit_patterns = self.constellation_maps[modulation.lower()]
        bits_per_symbol = self.bits_per_symbol[modulation.lower()]
        
        # Flatten the input for processing
        original_shape = symbols.shape
        symbols_flat = symbols.reshape(-1)
        
        # Calculate Euclidean distance to each constellation point
        # symbols_flat: [N], constellation: [M]
        # Expand dimensions for broadcasting: [N, 1] - [1, M] = [N, M]
        distances = torch.abs(symbols_flat.unsqueeze(1) - constellation.unsqueeze(0))
        
        # Find the closest constellation point
        _, indices = torch.min(distances, dim=1)
        
        # Get the corresponding bit patterns
        bits_flat = bit_patterns[indices]
        
        # Reshape to original dimensions with expanded bit dimension
        new_shape = original_shape[:-1] + (original_shape[-1] * bits_per_symbol,)
        bits = bits_flat.reshape(new_shape)
        
        return bits
    
    def forward(self, ofdm_map, modulation='qpsk'):
        """
        Forward pass of the OFDMSymbolDecoder module.
        
        Args:
            ofdm_map: OFDM map with shape [B, 2, num_symbols, fft_size]
                      where the second dimension is [real, imag]
            modulation: Modulation scheme ('bpsk', 'qpsk', 'qam16', 'qam64', 'qam256')
            
        Returns:
            Decoded bits with shape [B, num_symbols * num_active_subcarriers * bits_per_symbol]
        """
        batch_size, _, num_symbols, fft_size = ofdm_map.shape
        
        # Convert to complex representation
        ofdm_complex = torch.complex(ofdm_map[:, 0], ofdm_map[:, 1])  # [B, num_symbols, fft_size]
        
        # Extract active subcarriers
        active_indices = self.active_subcarriers.to(ofdm_complex.device)
        symbols = torch.index_select(ofdm_complex, dim=-1, index=active_indices)  # [B, num_symbols, num_active_subcarriers]
        
        # Reshape for demapping
        symbols = symbols.reshape(batch_size, -1)  # [B, num_symbols * num_active_subcarriers]
        
        # Perform constellation demapping
        bits = self.demapping(symbols, modulation)  # [B, num_symbols * num_active_subcarriers * bits_per_symbol]
        
        return bits


class OFDMDecoder(nn.Module):
    """
    Complete OFDM decoder that can process the OFDM map returned by RadarTimeNet.
    
    This module combines symbol extraction and decoding with optional channel estimation
    and equalization, and supports different modulation schemes.
    """
    def __init__(self, fft_size, num_symbols, num_subcarriers=None, dc_null=True, 
                 guard_bands=None, use_channel_estimation=True):
        """
        Initialize the OFDMDecoder module.
        
        Args:
            fft_size: Size of the FFT used in OFDM
            num_symbols: Number of OFDM symbols
            num_subcarriers: Number of active subcarriers (if None, use all except DC)
            dc_null: Whether the DC subcarrier is nulled
            guard_bands: List of two integers specifying left and right guard band sizes
                         (if None, no guard bands are used)
            use_channel_estimation: Whether to use channel estimation and equalization
        """
        super().__init__()
        self.fft_size = fft_size
        self.num_symbols = num_symbols
        self.use_channel_estimation = use_channel_estimation
        
        # Symbol decoder for constellation demapping
        self.symbol_decoder = OFDMSymbolDecoder(fft_size, num_subcarriers, dc_null, guard_bands)
        
        # Channel estimation module (if enabled)
        if use_channel_estimation:
            self.channel_estimator = nn.Sequential(
                nn.Conv2d(2, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 2, kernel_size=1)
            )
    
    def equalize_channel(self, ofdm_map, channel_estimate):
        """
        Perform channel equalization using the estimated channel response.
        
        Args:
            ofdm_map: OFDM map with shape [B, 2, num_symbols, fft_size]
            channel_estimate: Channel estimate with shape [B, 2, num_symbols, fft_size]
            
        Returns:
            Equalized OFDM map with shape [B, 2, num_symbols, fft_size]
        """
        # Convert to complex representation
        ofdm_complex = torch.complex(ofdm_map[:, 0], ofdm_map[:, 1])
        channel_complex = torch.complex(channel_estimate[:, 0], channel_estimate[:, 1])
        
        # Perform equalization (division in complex domain)
        equalized_complex = ofdm_complex / (channel_complex + 1e-10)  # Add small value for numerical stability
        
        # Convert back to real/imag representation
        equalized_map = torch.stack([equalized_complex.real, equalized_complex.imag], dim=1)
        
        return equalized_map
    
    def forward(self, ofdm_map, modulation='qpsk'):
        """
        Forward pass of the OFDMDecoder module.
        
        Args:
            ofdm_map: OFDM map with shape [B, 2, num_symbols, fft_size]
                      where the second dimension is [real, imag]
            modulation: Modulation scheme ('bpsk', 'qpsk', 'qam16', 'qam64', 'qam256')
            
        Returns:
            Decoded bits with shape [B, num_symbols * num_active_subcarriers * bits_per_symbol]
        """
        # Perform channel estimation if enabled
        if self.use_channel_estimation:
            channel_estimate = self.channel_estimator(ofdm_map)
            ofdm_map = self.equalize_channel(ofdm_map, channel_estimate)
        
        # Decode symbols to bits
        bits = self.symbol_decoder(ofdm_map, modulation)
        
        return bits
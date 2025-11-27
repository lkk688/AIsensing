import numpy as np
from scipy.constants import c
from numpy.fft import fft, ifft, fftshift
from scipy.signal.windows import hann
from scipy.interpolate import interp1d

class Channel:
    def __init__(self, fs, fc, n_paths=1, path_loss_exponent=2.5, no_channel=False):
        self.no_channel = no_channel
        self.path_loss_exponent = path_loss_exponent
        if self.no_channel:
            return
        self.fs = fs
        self.fc = fc

    def apply_channel_effects(self, signal, R, v):
        if self.no_channel:
            return signal

        # Apply path loss
        path_loss = (4 * np.pi * R / (c / self.fc)) ** -self.path_loss_exponent
        signal_with_loss = signal * np.sqrt(path_loss)
        
        return signal_with_loss


class RadarSystem:
    def __init__(self, fc=77e9, T_chirp=20e-6, N_samples=1024, N_chirps=128, N_subcarriers=64, zero_pad_factor=8):
        self.fc = fc
        self.T = T_chirp
        self.Ns = N_samples
        self.Nc = N_chirps
        self.lambda_c = c / fc
        self.fs = self.Ns / self.T
        self.v_max = self.lambda_c / (4 * T_chirp)
        self.zero_pad = zero_pad_factor * N_subcarriers

        # Corrected range axis for OFDM radar
        self.subcarrier_spacing = self.fs / N_subcarriers
        self.range_res = c / (2 * self.subcarrier_spacing * N_subcarriers)
        self.range_axis = np.arange(self.zero_pad // 2) * self.range_res

        self.velocity_axis = fftshift(np.fft.fftfreq(N_chirps, d=self.T)) * self.lambda_c / 2
        self.t_fast = np.arange(self.Ns) / self.fs
        self.t_slow = np.arange(self.Nc) * self.T

import pyldpc

class CommSystem:
    def __init__(self, N_subcarriers=64, CP_len=16, N_chirps=128, ldpc_n=120, ldpc_maxiter=20, n_pilots=4):
        self.N_sub = N_subcarriers
        self.CP = CP_len
        self.n_pilots = n_pilots
        self.N_sym = N_chirps
        self.OFDM_len = self.N_sub + self.CP
        self.ldpc_n = ldpc_n
        self.ldpc_maxiter = ldpc_maxiter

        self.H, self.G = pyldpc.make_ldpc(self.ldpc_n, 3, 6, systematic=True, sparse=True)
        self.ldpc_k = self.G.shape[1]

        # Define pilot and data indices
        self.pilot_indices = np.arange(0, self.N_sub, self.N_sub // self.n_pilots, dtype=int)
        self.data_indices = np.array(sorted(list(set(range(self.N_sub)) - set(self.pilot_indices))))

        # Create known pilot symbols
        self.pilot_values = np.array([1+1j, 1-1j, -1+1j, -1-1j] * (self.n_pilots // 4 + 1))[:self.n_pilots]

        self.bits = np.random.randint(0, 2, (self.N_sym, self.ldpc_k))
        self.qpsk = np.array([self.modulate_ofdm_symbol(b) for b in self.bits])

    def modulate_ofdm_symbol(self, data_bits):
        coded_bits = pyldpc.encode(self.G, data_bits, 1000)
        qpsk_data = (1 - 2 * coded_bits[0::2]) + 1j * (1 - 2 * coded_bits[1::2])
        qpsk_data /= np.sqrt(2)  # Normalize to unit power
        
        ofdm_symbol = np.zeros(self.N_sub, dtype=np.complex128)
        ofdm_symbol[self.pilot_indices] = self.pilot_values
        ofdm_symbol[self.data_indices[:len(qpsk_data)]] = qpsk_data
        
        return ofdm_symbol

    def decode_qpsk(self, rx_qpsk, nVar, snr, channel_estimates):
        # rx_qpsk contains equalized QPSK data symbols of shape [num_chirps, ldpc_n/2]
        # Convert equalized QPSK symbols to Gaussian-corrupted bits in codeword space (0..1).
        # With Tx normalization 1/sqrt(2): s_real ≈ (1-2*b0)/sqrt(2) + n_r; map to y_bit0 ≈ b0 + noise.
        # Derivation: a_hat = sqrt(2)*s_real ≈ (1-2*b0) + sqrt(2)*n_r; y_bit0 = (1 - a_hat)/2 ≈ b0 - (sqrt(2)/2)*n_r.
        # Therefore, var_y = (sqrt(2)/2)^2 * Var(n_r) = (1/2) * Var(n_r). With complex n_eq, Var(n_r)=E[|n_eq|^2]/2.
        # So var_y = (E[|n_eq|^2])/4 and SNR_linear = 1/var_y = 4/E[|n_eq|^2].
        num_chirps, num_data_symbols = rx_qpsk.shape
        y_bits = np.zeros((num_chirps, self.ldpc_n), dtype=np.float64)
        # Map to bit-space observations in [0,1]
        y_bits[:, 0::2] = (1.0 - np.sqrt(2.0) * rx_qpsk.real) / 2.0
        y_bits[:, 1::2] = (1.0 - np.sqrt(2.0) * rx_qpsk.imag) / 2.0

        # nVar may be scalar or per-chirp array of equalized complex noise power E[|n_eq|^2].
        # Convert to SNR dB per chirp: SNR = 4 / sigma2_eq
        if np.ndim(nVar) == 0:
            sigma2_eq = float(nVar)
            snr_db_vec = np.full((num_chirps,), 10 * np.log10(4.0 / (sigma2_eq + 1e-12)))
        else:
            nVar = np.squeeze(nVar)
            if nVar.ndim == 2 and nVar.shape[1] == 1:
                nVar = nVar[:, 0]
            if nVar.shape[0] != num_chirps:
                sigma2_eq = float(np.mean(nVar))
                snr_db_vec = np.full((num_chirps,), 10 * np.log10(4.0 / (sigma2_eq + 1e-12)))
            else:
                sigma2_eq_vec = nVar
                snr_db_vec = 10 * np.log10(4.0 / (sigma2_eq_vec + 1e-12))

        # Decode each chirp independently using pyldpc
        decoded_codewords = []
        for i in range(num_chirps):
            y = y_bits[i]
            snr_db_i = float(snr_db_vec[i])
            codeword_bits = pyldpc.decode(self.H, y, snr_db_i, maxiter=self.ldpc_maxiter)
            decoded_codewords.append(codeword_bits)
        decoded_codewords = np.array(decoded_codewords)

        # Extract message bits from decoded codewords
        decoded_bits = np.array([pyldpc.get_message(self.G, c) for c in decoded_codewords])
        return decoded_bits

    def generate_ofdm_baseband(self):
        ofdm_symbols = []
        for sym in self.qpsk:
            freq = np.zeros(self.N_sub, dtype=complex)
            freq[:len(sym)] = sym
            time = ifft(freq)
            time_cp = np.concatenate([time[-self.CP:], time])
            ofdm_symbols.append(time_cp)
        return np.stack(ofdm_symbols)

def fractional_delay(signal, delay, fs):
    t = np.arange(len(signal)) / fs
    t_delayed = t - delay
    interp = interp1d(t, signal, kind='cubic', bounds_error=False, fill_value=0.0)
    return interp(t_delayed)
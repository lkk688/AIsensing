from isac_lib import RadarSystem, CommSystem, Channel, fractional_delay
from scipy.signal import resample
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
import os
import numpy as np
from scipy.constants import c
from numpy.fft import fft, fftshift
from scipy.interpolate import interp1d

class ISACSimulator:
    def __init__(self, radar_params, comm_params, channel_params, R_true, v_true, SNR_dB=20, sir_db=-20):
        self.radar = RadarSystem(**radar_params)
        self.comm = CommSystem(**comm_params)
        self.channel = Channel(fs=self.radar.fs, fc=self.radar.fc, **channel_params)
        self.R_true = R_true
        self.v_true = v_true
        self.SNR_dB = SNR_dB
        self.sir_db = sir_db
        self.ber = None
        # Add a small random delay to the leakage path to make it more realistic
        self.leakage_delay = np.random.uniform(0, 1/self.radar.fs)

    def simulate(self):
        # --- Transmit Signal Generation ---
        ofdm_base = self.comm.generate_ofdm_baseband()
        ofdm = resample(ofdm_base, self.radar.Ns, axis=1)
        L = self.radar.Ns

        phase_chirp = 2 * np.pi * self.radar.fc * self.radar.t_fast[:L]
        tx_chirp = np.exp(1j * phase_chirp)

        # --- Channel Simulation ---
        target_signal_only = np.zeros((self.radar.Nc, L), dtype=np.complex128)
        leakage_signal_all_chirps = np.zeros((self.radar.Nc, L), dtype=np.complex128)
        fd_true = -2 * self.v_true / self.radar.lambda_c
        doppler_fast_component = np.exp(1j * 2 * np.pi * fd_true * self.radar.t_fast[:L])

        # --- Define a static leakage channel based on SIR ---
        path_loss_target = (4 * np.pi * self.R_true / (c / self.radar.fc)) ** -self.channel.path_loss_exponent
        sir_lin = 10**(self.sir_db / 10)
        leakage_power_factor = path_loss_target / sir_lin
        # Static leakage channel coefficient
        self.leakage_coeff = np.sqrt(leakage_power_factor) * (np.random.randn() + 1j*np.random.randn())/np.sqrt(2)

        for m, t in enumerate(self.radar.t_slow):
            tx_signal_m = ofdm[m] * tx_chirp

            # --- Target Path ---
            tau_true = 2 * (self.R_true + self.v_true * t) / c
            delayed_tx_signal = fractional_delay(tx_signal_m, tau_true, self.radar.fs)
            doppler_slow_component = np.exp(1j * 2 * np.pi * fd_true * t)
            target_signal_only[m] = self.channel.apply_channel_effects(delayed_tx_signal * doppler_slow_component * doppler_fast_component, self.R_true, self.v_true)

            # --- Leakage Path (Self-Interference) ---
            delayed_leakage = fractional_delay(tx_signal_m, self.leakage_delay, self.radar.fs)
            leakage_signal_all_chirps[m] = delayed_leakage * self.leakage_coeff

        # --- Add Noise ---
        power_target = np.mean(np.abs(target_signal_only)**2)
        snr_lin = 10 ** (self.SNR_dB / 10)
        self.nVar = power_target / snr_lin
        noise = (np.random.randn(*target_signal_only.shape) + 1j * np.random.randn(*target_signal_only.shape)) * np.sqrt(self.nVar / 2)
        
        # --- Final Received Signal ---
        rx_signal = target_signal_only + leakage_signal_all_chirps + noise

        # --- Pilot-based Radar Processing and Communication Decoding ---
        full_channel_est_matrix = np.zeros((self.radar.Nc, self.comm.N_sub), dtype=np.complex128)
        rx_qpsk = []
        rx_qpsk_wo_leak = []
        leakage_channel_est_freq = None

        def estimate_cp_start(x, N, CP):
            L = len(x)
            max_n = L - (N + CP)
            if max_n <= 0:
                return 0
            metrics = []
            for n in range(max_n):
                a = x[n:n+CP]
                b = x[n+N:n+N+CP]
                metrics.append(np.abs(np.vdot(a, b)))
            return int(np.argmax(metrics))

        for m in range(self.radar.Nc):
            # --- Downconversion to Baseband ---
            rx_baseband_m = rx_signal[m, :] * np.conj(tx_chirp)
            
            # Compute leakage baseband for this chirp
            leakage_bb_m = leakage_signal_all_chirps[m, :] * np.conj(tx_chirp)
            # Subtract leakage in time domain before OFDM demodulation
            rx_bb_wo_leak_m = rx_baseband_m - leakage_bb_m

            # --- OFDM Demodulation with CP-based timing alignment ---
            downsampled_symbol = resample(rx_bb_wo_leak_m, self.comm.OFDM_len)
            cp_start = estimate_cp_start(downsampled_symbol, self.comm.N_sub, self.comm.CP)
            symbol_no_cp = downsampled_symbol[cp_start + self.comm.CP : cp_start + self.comm.CP + self.comm.N_sub]
            fft_output_wo_leak = fft(symbol_no_cp)
            rx_qpsk_wo_leak.append(fft_output_wo_leak)

            # --- Channel Estimation using linear amplitude and phase fit across pilots ---
            pilots_rx_m = fft_output_wo_leak[self.comm.pilot_indices]
            H_pilots = pilots_rx_m / self.comm.pilot_values
            k_p = self.comm.pilot_indices.astype(float)
            k_all = np.arange(self.comm.N_sub, dtype=float)

            # Fit amplitude (magnitude) linearly and enforce non-negativity
            amp_p = np.abs(H_pilots)
            a1, a0 = np.polyfit(k_p, amp_p, 1)
            amp_all = a0 + a1 * k_all
            amp_all = np.clip(amp_all, 1e-6, None)

            # Fit unwrapped phase linearly
            phase_p = np.unwrap(np.angle(H_pilots))
            p1, p0 = np.polyfit(k_p, phase_p, 1)
            phase_all = p0 + p1 * k_all

            full_channel_est_matrix[m, :] = amp_all * np.exp(1j * phase_all)

        # --- Coherent Leakage Subtraction ---
        # We already subtracted leakage per-chirp; treat channel estimate as target-only
        self.leakage_channel_est = np.zeros(self.comm.N_sub, dtype=np.complex128)
        self.target_channel_est_matrix = full_channel_est_matrix

        # --- Radar Processing on Target-Only Channel Estimate ---
        range_profile = fft(self.target_channel_est_matrix, n=self.radar.zero_pad, axis=1)
        range_profile = range_profile[:, :self.radar.zero_pad // 2]
        rdm = fftshift(np.fft.ifft(range_profile, axis=0), axes=0)
        self.RDM = 20 * np.log10(np.abs(rdm) + 1e-6)
        self.RDM_peak = np.max(self.RDM)
        self.RDM_peak_coords = np.unravel_index(np.argmax(self.RDM), self.RDM.shape)
        i, j = self.RDM_peak_coords
        self.R_det = self.radar.range_axis[j]
        self.v_det = self.radar.velocity_axis[i]

        # --- Communication Processing ---
        rx_qpsk_raw = np.array(rx_qpsk_wo_leak)

        # --- Equalization for Comm Data using fitted channel ---
        equalized_symbols = []
        for i in range(self.comm.N_sym):
            equalized_symbol = rx_qpsk_raw[i, :] / (full_channel_est_matrix[i, :] + 1e-12)
            equalized_symbols.append(equalized_symbol)

        self.rx_qpsk = np.array(equalized_symbols)
        self.rx_qpsk_wo_leak = rx_qpsk_raw

        # --- Final Steps ---
        self.cfar_detections = self.run_cfar(threshold_dB=10, guard_cells=2, training_cells=8)
        self.ber = self._calculate_ber()

    def run_cfar(self, threshold_dB=10, guard_cells=2, training_cells=8):
        RDM = self.RDM
        detections = []
        mag = RDM.copy()
        local_max = maximum_filter(mag, size=3)
        candidates = (mag == local_max) & (mag > threshold_dB)
        idxs = np.argwhere(candidates)

        for i, j in idxs:
            R = self.radar.range_axis[j]
            v = self.radar.velocity_axis[i]
            detections.append({
                'i': i, 'j': j,
                'range': R,
                'velocity': v,
                'mag_dB': RDM[i, j]
            })

        self.cfar_hit = any(abs(d['range'] - self.R_true) < self.radar.range_res * 2 and
                            abs(d['velocity'] - self.v_true) < 1.0 for d in detections)
        return detections

    def _calculate_ber(self):
        # Equalized data symbols and pilot residuals using pilot-interpolated channel
        data_symbols_rx = self.rx_qpsk[:, self.comm.data_indices]
        num_data_symbols = self.comm.ldpc_n // 2
        equalized_data_to_decode = data_symbols_rx[:, :num_data_symbols]
        channel_est_data = self.target_channel_est_matrix[:, self.comm.data_indices[:num_data_symbols]]
        eq_pilots = self.rx_qpsk[:, self.comm.pilot_indices]

        # Estimate equalized noise variance per-chirp from pilot residuals
        pilot_err = eq_pilots - self.comm.pilot_values[np.newaxis, :]
        nVar_eq_per_chirp = np.mean(np.abs(pilot_err)**2, axis=1) + 1e-12

        # The decoder expects equalized symbols and noise variance; use measured per-chirp variance
        # Debug: compute effective SNR in dB seen by decoder and TX/RX symbol mismatch
        snr_eff_db = 10 * np.log10(4.0 / (nVar_eq_per_chirp + 1e-12))
        tx_symbols = np.array([sym[self.comm.data_indices[:num_data_symbols]] for sym in self.comm.qpsk])
        mse = np.mean(np.abs(equalized_data_to_decode - tx_symbols[:equalized_data_to_decode.shape[0]])**2)
        print(f"Decoder SNR per-chirp (dB): min={snr_eff_db.min():.2f}, max={snr_eff_db.max():.2f}, mean={snr_eff_db.mean():.2f}; EQ vs TX MSE={mse:.3e}")
        decoded_bits = self.comm.decode_qpsk(equalized_data_to_decode, nVar_eq_per_chirp[:, np.newaxis], self.SNR_dB, channel_est_data)
        tx_bits = self.comm.bits
        
        num_chirps = min(tx_bits.shape[0], decoded_bits.shape[0])
        num_bits = min(tx_bits.shape[1], decoded_bits.shape[1])

        tx_bits_ref = tx_bits[:num_chirps, :num_bits]
        decoded_bits_ref = decoded_bits[:num_chirps, :num_bits]

        bit_errors = np.sum(tx_bits_ref != decoded_bits_ref)
        total_bits = tx_bits_ref.size

        return bit_errors / total_bits if total_bits > 0 else 0

    def report(self):
        print(f"🎯 Ground Truth: Range = {self.R_true:.2f} m, Velocity = {self.v_true:.2f} m/s")
        print(f"✅ Detected    : Range = {self.R_det:.2f} m, Velocity = {self.v_det:.2f} m/s")
        print(f"📏 Errors      : ΔR = {abs(self.R_true - self.R_det):.2f} m, Δv = {abs(self.v_true - self.v_det):.2f} m/s")
        if self.ber is not None:
            print(f"📶 OFDM BER    : {self.ber:.2e}")
        print(f"📡 CFAR HIT    : {'Yes' if self.cfar_hit else 'No'}")
        print(f"🌟 RDM Peak    : Coord = {self.RDM_peak_coords}, Mag = {self.RDM_peak:.2f} dB")

    def plot_rdm(self, save_path=None):
        plt.figure(figsize=(10, 6))
        plt.imshow(self.RDM, extent=[self.radar.range_axis[0], self.radar.range_axis[-1],
                                     self.radar.velocity_axis[0], self.radar.velocity_axis[-1]],
                   origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(label='Magnitude (dB)')
        plt.xlabel('Range (m)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Range-Doppler Map (ISAC + CFAR + Peak)')

        # Ground truth and peak detection
        plt.scatter(self.R_det, self.v_det, color='white', marker='x', s=100, label='Detected')
        plt.scatter(self.R_true, self.v_true, facecolors='none', edgecolors='red', s=120, label='Ground Truth')

        # RDM peak location
        peak_i, peak_j = self.RDM_peak_coords
        peak_R = self.radar.range_axis[peak_j]
        peak_v = self.radar.velocity_axis[peak_i]
        plt.scatter(peak_R, peak_v, color='yellow', marker='*', s=150, label='RDM Peak')

        # CFAR detections
        for d in self.cfar_detections:
            plt.scatter(d['range'], d['velocity'], color='lime', s=40, marker='o')

        plt.legend()
        plt.tight_layout()
        
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, 'range_doppler_map.png'))
        else:
            plt.show()

def plot_ber_curve(radar_params, comm_params, channel_params, snr_range_db, save_path=None):
    bers = []
    for snr_db in snr_range_db:
        R_true = np.random.uniform(10, radar_params['R_max'] - 10)
        v_max = (c / radar_params['fc']) / (4 * radar_params['T_chirp'])
        v_true = np.random.uniform(-v_max + 1, v_max - 1)
        sim = ISACSimulator(radar_params, comm_params, channel_params, R_true, v_true, SNR_dB=snr_db)
        sim.simulate()
        bers.append(sim.ber)
    
    plt.figure()
    plt.semilogy(snr_range_db, bers, 'o-')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('BER vs. SNR')
    plt.grid(True)
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, 'ber_curve.png'))
    else:
        plt.show()


if __name__ == "__main__":
    print("Starting simulation...")
    comm_params = {
    'N_subcarriers': 64, 'CP_len': 16, 'N_chirps': 128,
    'ldpc_n': 108, 'ldpc_maxiter': 120, 'n_pilots': 8
}
    radar_params = {
        'fc': 77e9, 'T_chirp': 20e-6, 'N_samples': 1024,
        'N_chirps': 128, 'N_subcarriers': comm_params['N_subcarriers'], 'zero_pad_factor': 8
    }
    channel_params = {
        'n_paths': 5, 'path_loss_exponent': 2.5, 'no_channel': False
    }
    v_max = (c / radar_params['fc']) / (4 * radar_params['T_chirp'])
    v_true = np.random.uniform(-v_max + 1, v_max - 1)

    sim = ISACSimulator(radar_params, comm_params, channel_params, 0, v_true) # R_true is temporary
    R_true = np.random.uniform(10, sim.radar.range_axis[-1] -10)
    sim.R_true = R_true
    sim.simulate()
    sim.report()
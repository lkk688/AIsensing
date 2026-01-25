import sys
import time
import numpy as np
import adi
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QGroupBox, QLabel, QLineEdit, QComboBox, QSlider, QPushButton, 
                             QCheckBox, QFormLayout)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ==============================================================================
# Signal Generation Utils
# ==============================================================================
def generate_sine_wave(fs, freq, duration, amplitude=1.0):
    t = np.arange(int(fs * duration)) / fs
    # Complex exponential: e^(j*2*pi*f*t) = cos(...) + j*sin(...)
    # This generates a single sideband tone
    signal = amplitude * np.exp(1j * 2 * np.pi * freq * t)
    return signal.astype(np.complex64)

def generate_two_tone(fs, f1, f2, duration, amplitude=0.5):
    t = np.arange(int(fs * duration)) / fs
    s1 = amplitude * np.exp(1j * 2 * np.pi * f1 * t)
    s2 = amplitude * np.exp(1j * 2 * np.pi * f2 * t)
    return (s1 + s2).astype(np.complex64)

# ==============================================================================
# SDR Worker Thread
# ==============================================================================
class SDRWorker(QThread):
    data_received = pyqtSignal(np.ndarray, float) # rx_data, actual_fs
    status_message = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.sdr = None
        self.ip = "ip:192.168.2.1"
        self.fc = 2400e6
        self.fs = 2e6 # Low default for safety
        self.tx_gain = -50
        self.rx_gain = 20
        self.tx_signal_type = "Sine"
        self.tx_freq = 100e3 # 100 kHz offset
        self.tx_enabled = False
        
        self.current_config = {} # Cache to avoid redundant SPI writes
        
    def run(self):
        self.running = True
        
        # Connect
        try:
            self.status_message.emit(f"Connecting to {self.ip}...")
            # Use specific Pluto class or generic one? Using Pluto for now.
            self.sdr = adi.Pluto(uri=self.ip)
            self.status_message.emit("Connected.")
        except Exception as e:
            self.status_message.emit(f"Connection Failed: {e}")
            self.running = False
            return

        # Initial Setup
        self.update_sdr_config()
        
        buffer_size = 1024 * 4
        self.sdr.rx_buffer_size = buffer_size
        
        while self.running:
            try:
                # Apply runtime config changes
                self.update_sdr_config()
                self.update_tx_signal()
                
                # Receive
                # Start timing
                rx = self.sdr.rx()
                
                # Emit data
                self.data_received.emit(rx, self.fs)
                
                # Sleep a bit to limit UI refresh rate ~30fps
                time.sleep(0.03)
                
            except Exception as e:
                self.status_message.emit(f"Error: {e}")
                time.sleep(1)

        # Cleanup
        if self.sdr:
            self.sdr.tx_destroy_buffer()
            # self.sdr.rx_destroy_buffer()
            del self.sdr
            
    def update_sdr_config(self):
        if not self.sdr: return
        
        # Helper to set if changed
        def set_param(name, value):
            if self.current_config.get(name) != value:
                setattr(self.sdr, name, value)
                self.current_config[name] = value
                
        try:
            # Basic params
            # Note: Changing sample rate usually requires buffer reset or care
            if self.current_config.get('sample_rate') != int(self.fs):
                self.sdr.sample_rate = int(self.fs)
                self.current_config['sample_rate'] = int(self.fs)
                
            set_param('rx_lo', int(self.fc))
            set_param('tx_lo', int(self.fc))
            set_param('tx_hardwaregain_chan0', int(self.tx_gain))
            set_param('rx_hardwaregain_chan0', int(self.rx_gain))
            
            # Ensure bandwidth follows sample rate (Nyquist)
            bw = int(self.fs)
            set_param('rx_rf_bandwidth', bw)
            set_param('tx_rf_bandwidth', bw)
            
        except Exception as e:
            print(f"Config Error: {e}")

    def update_tx_signal(self):
        if not self.sdr: return
        
        # Check if we need to update TX buffer
        # We use a dirty flag logic or just check if params changed
        # For simplicity, we regenerate if params changed significantly
        
        cfg_key = (self.tx_signal_type, self.tx_freq, self.tx_enabled, self.fs)
        if self.current_config.get('tx_state') == cfg_key:
            return
            
        if not self.tx_enabled or self.tx_signal_type == "OFF":
            self.sdr.tx_destroy_buffer()
        else:
            # Generate signal
            N = 1024 * 16
            if self.tx_signal_type == "Sine":
                sig = generate_sine_wave(self.fs, self.tx_freq, duration=N/self.fs)
            elif self.tx_signal_type == "Two-Tone":
                sig = generate_two_tone(self.fs, self.tx_freq, self.tx_freq*2, duration=N/self.fs)
            elif self.tx_signal_type == "Noise":
                sig = (np.random.randn(N) + 1j*np.random.randn(N)) * 0.1
                sig = sig.astype(np.complex64)
            else:
                sig = np.zeros(N, dtype=np.complex64)
            
            # Scale to hardware range (Pluto expects 2^14 range)
            sig *= 2**14 * 0.5 # 0.5 headroom
            
            self.sdr.tx_cyclic_buffer = True
            self.sdr.tx(sig)
            
        self.current_config['tx_state'] = cfg_key

    def stop(self):
        self.running = False
        self.wait()

# ==============================================================================
# UI Implementation
# ==============================================================================
class SDRDiagnosticsApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SDR Hardware Diagnostics")
        self.resize(1200, 800)
        
        self.worker = SDRWorker()
        self.worker.data_received.connect(self.update_plots)
        self.worker.status_message.connect(self.log_status)
        
        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        layout = QHBoxLayout(main_widget)
        
        # --- Left Panel: Controls ---
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setFixedWidth(300)
        
        # 1. Connection
        grp_conn = QGroupBox("Connection")
        form_conn = QFormLayout()
        self.txt_ip = QLineEdit("ip:192.168.2.1")
        self.btn_connect = QPushButton("Connect / Start")
        self.btn_connect.setCheckable(True)
        self.btn_connect.clicked.connect(self.toggle_connection)
        form_conn.addRow("URI:", self.txt_ip)
        form_conn.addRow(self.btn_connect)
        grp_conn.setLayout(form_conn)
        control_layout.addWidget(grp_conn)
        
        # 2. SDR Settings
        grp_sdr = QGroupBox("Radio Settings")
        form_sdr = QFormLayout()
        
        # FC
        self.spin_fc = QLineEdit("2400") # MHz
        self.spin_fc.editingFinished.connect(self.update_params)
        form_sdr.addRow("Freq (MHz):", self.spin_fc)
        
        # FS
        self.combo_fs = QComboBox()
        self.combo_fs.addItems(["1.0", "2.0", "3.0", "4.0", "6.0", "10.0"]) # MSPS
        self.combo_fs.setCurrentText("2.0")
        self.combo_fs.currentTextChanged.connect(self.update_params)
        form_sdr.addRow("Rate (MSPS):", self.combo_fs)
        
        # Gains
        self.slider_rx_gain = QSlider(Qt.Orientation.Horizontal)
        self.slider_rx_gain.setRange(0, 70)
        self.slider_rx_gain.setValue(20)
        self.slider_rx_gain.valueChanged.connect(self.update_params)
        self.lbl_rx_gain = QLabel("20 dB")
        form_sdr.addRow("RX Gain:", self.lbl_rx_gain)
        form_sdr.addRow(self.slider_rx_gain)
        
        self.slider_tx_gain = QSlider(Qt.Orientation.Horizontal)
        self.slider_tx_gain.setRange(-80, 0)
        self.slider_tx_gain.setValue(-50)
        self.slider_tx_gain.valueChanged.connect(self.update_params)
        self.lbl_tx_gain = QLabel("-50 dB")
        form_sdr.addRow("TX Gain:", self.lbl_tx_gain)
        form_sdr.addRow(self.slider_tx_gain)
        
        grp_sdr.setLayout(form_sdr)
        control_layout.addWidget(grp_sdr)
        
        # 3. Test Signal
        grp_sig = QGroupBox("TX Test Signal")
        form_sig = QFormLayout()
        
        self.chk_tx_enable = QCheckBox("Enable TX")
        self.chk_tx_enable.toggled.connect(self.update_params)
        
        self.combo_sig_type = QComboBox()
        self.combo_sig_type.addItems(["Sine", "Two-Tone", "Noise"])
        self.combo_sig_type.currentTextChanged.connect(self.update_params)
        
        self.slider_sig_freq = QSlider(Qt.Orientation.Horizontal)
        self.slider_sig_freq.setRange(-500, 500) # kHz
        self.slider_sig_freq.setValue(100)
        self.slider_sig_freq.valueChanged.connect(self.update_params)
        self.lbl_sig_freq = QLabel("100 kHz")
        
        form_sig.addRow(self.chk_tx_enable)
        form_sig.addRow("Type:", self.combo_sig_type)
        form_sig.addRow("Offset:", self.lbl_sig_freq)
        form_sig.addRow(self.slider_sig_freq)
        
        grp_sig.setLayout(form_sig)
        control_layout.addWidget(grp_sig)
        
        # Status Label
        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setWordWrap(True)
        control_layout.addWidget(self.lbl_status)
        
        control_layout.addStretch()
        layout.addWidget(control_panel)
        
        # --- Right Panel: Plots ---
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        
        # Matplotlib Figure
        self.fig = Figure(figsize=(5, 8), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax_time = self.fig.add_subplot(211)
        self.ax_freq = self.fig.add_subplot(212)
        
        self.line_i, = self.ax_time.plot([], [], 'r-', label='Real (I)')
        self.line_q, = self.ax_time.plot([], [], 'b-', label='Imag (Q)')
        self.ax_time.set_title("Time Domain")
        self.ax_time.legend(loc='upper right')
        self.ax_time.grid(True)
        
        self.line_psd, = self.ax_freq.plot([], [], 'k-')
        self.ax_freq.set_title("Spectrum (PSD)")
        self.ax_freq.set_xlabel("Frequency (MHz)")
        self.ax_freq.set_ylabel("dB")
        self.ax_freq.grid(True)
        
        self.peak_marker, = self.ax_freq.plot([], [], 'rx')
        self.txt_peak = self.ax_freq.text(0, 0, "")
        
        plot_layout.addWidget(self.canvas)
        layout.addWidget(plot_panel)
        
    def toggle_connection(self):
        if self.btn_connect.isChecked():
            # Start
            self.worker.ip = self.txt_ip.text()
            self.update_params() # push initial ui values to worker
            self.worker.start()
            self.btn_connect.setText("Disconnect")
        else:
            # Stop
            self.worker.stop()
            self.btn_connect.setText("Connect / Start")
            
    def update_params(self):
        # UI -> Worker
        try:
            self.worker.fc = float(self.spin_fc.text()) * 1e6
            self.worker.fs = float(self.combo_fs.currentText()) * 1e6
            
            self.worker.tx_gain = self.slider_tx_gain.value()
            self.lbl_tx_gain.setText(f"{self.worker.tx_gain} dB")
            
            self.worker.rx_gain = self.slider_rx_gain.value()
            self.lbl_rx_gain.setText(f"{self.worker.rx_gain} dB")
            
            self.worker.tx_enabled = self.chk_tx_enable.isChecked()
            self.worker.tx_signal_type = self.combo_sig_type.currentText()
            
            freq_khz = self.slider_sig_freq.value()
            self.worker.tx_freq = freq_khz * 1e3
            self.lbl_sig_freq.setText(f"{freq_khz} kHz")
            
        except ValueError:
            pass # Parsing error
            
    def log_status(self, msg):
        self.lbl_status.setText(msg)
        
    def update_plots(self, rx_data, fs):
        # Decimate for plotting speed if too large
        if len(rx_data) > 1000:
            plot_data = rx_data[:1000]
        else:
            plot_data = rx_data
            
        # 1. Time Domain
        self.line_i.set_data(np.arange(len(plot_data)), plot_data.real)
        self.line_q.set_data(np.arange(len(plot_data)), plot_data.imag)
        self.ax_time.set_xlim(0, len(plot_data))
        self.ax_time.set_ylim(-2000, 2000) # Hardware is 12-bit (shifted), usually ranges +/- 2048
        
        # Auto-scale time if needed (optional, keeping fixed for flicker-free exp)
        if np.max(np.abs(plot_data)) > 2000:
             self.ax_time.set_ylim(-4000, 4000)
        
        # 2. Frequency Domain
        # FFT on full buffer
        N = len(rx_data)
        fft_data = np.fft.fftshift(np.fft.fft(rx_data))
        freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
        psd = 20 * np.log10(np.abs(fft_data) + 1e-12)
        
        self.line_psd.set_data(freqs/1e6, psd)
        self.ax_freq.set_xlim(freqs[0]/1e6, freqs[-1]/1e6)
        
        # Peak Detection
        peak_idx = np.argmax(psd)
        peak_freq = freqs[peak_idx]
        peak_pwr = psd[peak_idx]
        
        self.peak_marker.set_data([peak_freq/1e6], [peak_pwr])
        self.txt_peak.set_position((peak_freq/1e6, peak_pwr))
        self.txt_peak.set_text(f"Peak: {peak_freq/1e3:.1f} kHz")
        
        # Dynamic Y limit
        self.ax_freq.set_ylim(np.min(psd)-10, np.max(psd)+10)
        
        self.canvas.draw()

def main():
    app = QApplication(sys.argv)
    window = SDRDiagnosticsApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

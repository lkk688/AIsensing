import sys
import time
import numpy as np
import adi
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QLabel, QCheckBox, QComboBox, QSlider, QGroupBox, QPushButton,
    QFrame, QLineEdit, QFormLayout, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QMutex
from PyQt6.QtGui import QColor, QFont

# ==============================================================================
# Signal Generation Utils
# ==============================================================================
def generate_sine_wave(fs, freq, duration, amplitude=1.0):
    t = np.arange(int(fs * duration)) / fs
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
        self.fs = 3e6 
        self.tx_gain = -50
        self.rx_gain = 20
        self.tx_signal_type = "Sine"
        self.tx_freq = 100e3 
        self.tx_enabled = False
        
        self.current_config = {} 
        
    def run(self):
        self.running = True
        
        # Connect
        try:
            self.status_message.emit(f"Connecting to {self.ip}...")
            self.sdr = adi.Pluto(uri=self.ip)
            self.status_message.emit("Connected.")
        except Exception as e:
            self.status_message.emit(f"Connection Failed: {e}")
            self.running = False
            return

        # Initial Setup
        self.update_sdr_config()
        
        # Robust buffer size
        buffer_size = 1024 * 16
        self.sdr.rx_buffer_size = buffer_size
        if hasattr(self.sdr, "_rxadc") and hasattr(self.sdr._rxadc, "set_kernel_buffers_count"):
             self.sdr._rxadc.set_kernel_buffers_count(4)
        
        while self.running:
            try:
                # Apply runtime config changes
                self.update_sdr_config()
                self.update_tx_signal()
                
                # Receive
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
            if hasattr(self.sdr, 'tx_destroy_buffer'):
                self.sdr.tx_destroy_buffer()
            if hasattr(self.sdr, 'rx_destroy_buffer'):
                self.sdr.rx_destroy_buffer()
            del self.sdr
            
    def update_sdr_config(self):
        if not self.sdr: return
        
        def set_param(name, value):
            if self.current_config.get(name) != value:
                setattr(self.sdr, name, value)
                self.current_config[name] = value
                
        try:
            if self.current_config.get('sample_rate') != int(self.fs):
                self.sdr.sample_rate = int(self.fs)
                self.current_config['sample_rate'] = int(self.fs)
                
            set_param('rx_lo', int(self.fc))
            set_param('tx_lo', int(self.fc))
            set_param('tx_hardwaregain_chan0', int(self.tx_gain))
            set_param('rx_hardwaregain_chan0', int(self.rx_gain))
            
            bw = int(self.fs)
            set_param('rx_rf_bandwidth', bw)
            set_param('tx_rf_bandwidth', bw)
            
        except Exception as e:
            print(f"Config Error: {e}")

    def update_tx_signal(self):
        if not self.sdr: return
        
        cfg_key = (self.tx_signal_type, self.tx_freq, self.tx_enabled, self.fs)
        if self.current_config.get('tx_state') == cfg_key:
            return
            
        if not self.tx_enabled or self.tx_signal_type == "OFF":
            if hasattr(self.sdr, 'tx_destroy_buffer'):
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
            elif self.tx_signal_type == "Packet":
                 preamble = np.array([1, 1, 1, 1, -1, -1, 1, 1] * 10, dtype=np.complex64)
                 payload = np.random.choice([1, -1], 1024).astype(np.complex64)
                 sig = np.concatenate([preamble, payload])
                 sig = np.tile(sig, int(np.ceil(N / len(sig))))[:N]
            else:
                sig = np.zeros(N, dtype=np.complex64)
            
            sig *= 2**14 * 0.5 
            
            self.sdr.tx_cyclic_buffer = True
            self.sdr.tx(sig)
            
        self.current_config['tx_state'] = cfg_key

    def set_digital_loopback(self, enabled):
        if not self.sdr: return
        try:
            import iio
            ctx = self.sdr.ctx
            phy = ctx.find_device('ad9361-phy')
            if phy:
                val = '1' if enabled else '0'
                if 'loopback' in phy.debug_attrs:
                    phy.debug_attrs['loopback'].value = val
                    self.status_message.emit(f"Digital Loopback: {'ON' if enabled else 'OFF'}")
                else:
                    self.status_message.emit("Loopback attribute not found")
        except Exception as e:
            self.status_message.emit(f"Loopback Error: {e}")

    def run_mode_test(self):
        if not self.sdr: return
        try:
            ctx = self.sdr.ctx
            phy = ctx.find_device('ad9361-phy')
            if phy and "ensm_mode" in phy.attrs:
                phy.attrs["ensm_mode"].value = "fdd"
                current = phy.attrs["ensm_mode"].value
                self.status_message.emit(f"Mode Test: Set FDD -> Read: {current}")
            else:
                self.status_message.emit("Mode Test: ENSM mode attr not found")
        except Exception as e:
            self.status_message.emit(f"Mode Test Error: {e}")

    def stop(self):
        self.running = False
        self.wait()

# ==============================================================================
# UI Implementation (Styled like radarappwdevice5.py)
# ==============================================================================
class SDRDiagnosticsApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SDR Diagnostics Pro")
        self.setGeometry(100, 100, 1400, 800)
        
        # Dark Theme Styling
        self.setStyleSheet("""
            QMainWindow {background-color: #1e1e1e; color: #e0e0e0;}
            QLabel {color: #ffffff; font-size: 13px;}
            QGroupBox {
                border: 1px solid #555; 
                margin-top: 20px; 
                color: #4CAF50; 
                font-weight: bold;
                padding-top: 15px;
            }
            QGroupBox::title {subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px;}
            QCheckBox {color: #dddddd; padding: 3px;}
            QComboBox {background-color: #333; color: white; border: 1px solid #555; padding: 4px;}
            QLineEdit {background-color: #333; color: white; border: 1px solid #555; padding: 4px;}
            QPushButton {
                background-color: #007ACC; color: white; border: none; padding: 6px; font-weight: bold;
            }
            QPushButton:hover {background-color: #0098FF;}
            QPushButton:pressed {background-color: #005C99;}
            QPushButton:checked {background-color: #FF5555;}
            QSlider::groove:horizontal {border: 1px solid #999; height: 8px; background: #333; margin: 2px 0;}
            QSlider::handle:horizontal {background: #007ACC; border: 1px solid #5c5c5c; width: 18px; margin: -2px 0; border-radius: 3px;}
        """)
        
        self.worker = SDRWorker()
        self.worker.data_received.connect(self.update_plots)
        self.worker.status_message.connect(self.log_status)
        
        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # --- Sidebar ---
        sidebar = QFrame()
        sidebar.setFixedWidth(340)
        sidebar.setStyleSheet("background-color: #262626; border-right: 1px solid #444;")
        sidebar_layout = QVBoxLayout(sidebar)
        
        title = QLabel("SDR DIAGNOSTICS")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #4CAF50; margin-bottom: 15px;")
        sidebar_layout.addWidget(title)
        
        # 1. Connection
        grp_conn = QGroupBox("Device Connection")
        form_conn = QFormLayout()
        self.txt_ip = QLineEdit("ip:192.168.2.1")
        self.btn_connect = QPushButton("Connect / Start")
        self.btn_connect.setCheckable(True)
        self.btn_connect.clicked.connect(self.toggle_connection)
        form_conn.addRow("URI:", self.txt_ip)
        form_conn.addRow(self.btn_connect)
        grp_conn.setLayout(form_conn)
        sidebar_layout.addWidget(grp_conn)
        
        # 2. SDR Settings
        grp_sdr = QGroupBox("Radio Configuration")
        form_sdr = QFormLayout()
        
        self.spin_fc = QLineEdit("2400") # MHz
        self.spin_fc.editingFinished.connect(self.update_params)
        
        self.combo_fs = QComboBox()
        self.combo_fs.addItems(["1.0", "2.0", "3.0", "4.0", "6.0", "10.0"]) # MSPS
        self.combo_fs.setCurrentText("3.0")
        self.combo_fs.currentTextChanged.connect(self.update_params)
        
        self.slider_rx_gain = QSlider(Qt.Orientation.Horizontal)
        self.slider_rx_gain.setRange(0, 70)
        self.slider_rx_gain.setValue(20)
        self.slider_rx_gain.valueChanged.connect(self.update_params)
        self.lbl_rx_gain = QLabel("20 dB")
        
        self.slider_tx_gain = QSlider(Qt.Orientation.Horizontal)
        self.slider_tx_gain.setRange(-80, 0)
        self.slider_tx_gain.setValue(-50)
        self.slider_tx_gain.valueChanged.connect(self.update_params)
        self.lbl_tx_gain = QLabel("-50 dB")
        
        form_sdr.addRow("Freq (MHz):", self.spin_fc)
        form_sdr.addRow("Rate (MSPS):", self.combo_fs)
        form_sdr.addRow("RX Gain:", self.lbl_rx_gain)
        form_sdr.addRow(self.slider_rx_gain)
        form_sdr.addRow("TX Gain:", self.lbl_tx_gain)
        form_sdr.addRow(self.slider_tx_gain)
        grp_sdr.setLayout(form_sdr)
        sidebar_layout.addWidget(grp_sdr)
        
        # 3. Test Signal
        grp_sig = QGroupBox("TX Test Signal")
        form_sig = QFormLayout()
        
        self.chk_tx_enable = QCheckBox("Enable Transmitter")
        self.chk_tx_enable.toggled.connect(self.update_params)
        
        self.combo_sig_type = QComboBox()
        self.combo_sig_type.addItems(["Sine", "Two-Tone", "Noise", "Packet"])
        self.combo_sig_type.currentTextChanged.connect(self.update_params)
        
        self.slider_sig_freq = QSlider(Qt.Orientation.Horizontal)
        self.slider_sig_freq.setRange(-500, 500)
        self.slider_sig_freq.setValue(100)
        self.slider_sig_freq.valueChanged.connect(self.update_params)
        self.lbl_sig_freq = QLabel("100 kHz")
        
        form_sig.addRow(self.chk_tx_enable)
        form_sig.addRow("Signal Type:", self.combo_sig_type)
        form_sig.addRow("Tone Offset:", self.lbl_sig_freq)
        form_sig.addRow(self.slider_sig_freq)
        grp_sig.setLayout(form_sig)
        sidebar_layout.addWidget(grp_sig)
        
        # 4. Diagnostics
        grp_diag = QGroupBox("Diagnostics")
        layout_diag = QVBoxLayout()
        self.chk_loopback = QCheckBox("Digital Loopback (BIST)")
        self.chk_loopback.toggled.connect(self.toggle_loopback)
        self.btn_mode_test = QPushButton("Test Mode Switching (FDD)")
        self.btn_mode_test.clicked.connect(self.trigger_mode_test)
        layout_diag.addWidget(self.chk_loopback)
        layout_diag.addWidget(self.btn_mode_test)
        grp_diag.setLayout(layout_diag)
        sidebar_layout.addWidget(grp_diag)
        
        sidebar_layout.addStretch()
        
        # Status Bar
        self.lbl_status = QLabel("Ready.")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #aaa; font-style: italic; padding: 10px; border-top: 1px solid #444;")
        sidebar_layout.addWidget(self.lbl_status)
        
        main_layout.addWidget(sidebar)
        
        # --- Plots Area ---
        plot_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Time Domain Plot
        self.plot_time = pg.PlotWidget(title="Time Domain (I/Q)")
        self.plot_time.setBackground('#1e1e1e')
        self.plot_time.showGrid(x=True, y=True, alpha=0.3)
        self.plot_time.setLabel('left', 'Amplitude')
        self.plot_time.setLabel('bottom', 'Sample')
        self.plot_time.addLegend()
        self.curve_i = self.plot_time.plot(pen=pg.mkPen('#FF5555', width=1), name='Real (I)')
        self.curve_q = self.plot_time.plot(pen=pg.mkPen('#55AAFF', width=1), name='Imag (Q)')
        plot_splitter.addWidget(self.plot_time)
        
        # Frequency Domain Plot
        self.plot_freq = pg.PlotWidget(title="Frequency Domain (PSD)")
        self.plot_freq.setBackground('#1e1e1e')
        self.plot_freq.showGrid(x=True, y=True, alpha=0.3)
        self.plot_freq.setLabel('left', 'Power (dB)')
        self.plot_freq.setLabel('bottom', 'Frequency (MHz)')
        self.curve_psd = self.plot_freq.plot(pen=pg.mkPen('#00FF00', width=1))
        plot_splitter.addWidget(self.plot_freq)
        
        main_layout.addWidget(plot_splitter)
        
    def toggle_connection(self):
        if self.btn_connect.isChecked():
            self.worker.ip = self.txt_ip.text()
            self.update_params() 
            self.worker.start()
            self.btn_connect.setText("Disconnect")
            self.btn_connect.setStyleSheet("background-color: #CC3333;")
        else:
            self.worker.stop()
            self.btn_connect.setText("Connect / Start")
            self.btn_connect.setStyleSheet("background-color: #007ACC;")
            
    def update_params(self):
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
            pass 

    def toggle_loopback(self, checked):
        if self.worker.running:
            self.worker.set_digital_loopback(checked)
        else:
            self.log_status("Connect first to enable loopback.")
            self.chk_loopback.setChecked(False)

    def trigger_mode_test(self):
        if self.worker.running:
            self.worker.run_mode_test()
        else:
            self.log_status("Connect first to run tests.")
            
    def log_status(self, msg):
        self.lbl_status.setText(msg)
        
    def update_plots(self, rx_data, fs):
        # Time Domain
        limit = 1000
        view_data = rx_data[:limit]
        self.curve_i.setData(view_data.real)
        self.curve_q.setData(view_data.imag)
        
        # Frequency Domain
        N = len(rx_data)
        fft_data = np.fft.fftshift(np.fft.fft(rx_data))
        freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
        psd = 20 * np.log10(np.abs(fft_data) + 1e-12)
        
        self.curve_psd.setData(freqs/1e6, psd)

def main():
    app = QApplication(sys.argv)
    window = SDRDiagnosticsApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

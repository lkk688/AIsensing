#!/usr/bin/env python
"""
sdr_video_ui.py - Real-time UI Dashboard for SDR Video Communication

PyQt6-based dashboard showing:
- Live video feeds (TX source, RX decoded)
- Real-time metrics (BER, SNR, throughput)
- Constellation diagram
- OFDM vs OTFS waveform comparison

Usage:
    python sdr_video_ui.py
    python sdr_video_ui.py --device adrv9009 --ip ip:192.168.86.40
"""

import sys
import os

# CRITICAL: Fix OpenCV/PyQt6 Qt plugin conflict
# Must be done BEFORE importing cv2 or PyQt6
# This tells Qt to use system Qt, not OpenCV's bundled version
if 'QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ:
    del os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']

import numpy as np
from dataclasses import dataclass
from typing import Optional
import time
import threading
import queue

# Check for PyQt6 FIRST (before OpenCV which bundles conflicting Qt)
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QLabel, QPushButton, QComboBox, QGroupBox,
        QSlider, QSpinBox, QTabWidget, QTextEdit, QProgressBar,
        QFrame, QSplitter, QStatusBar
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
    from PyQt6.QtGui import QImage, QPixmap, QFont, QPalette, QColor
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("[Error] PyQt6 not installed. Run: pip install pyqt6")

# Check for matplotlib (after PyQt6)
MATPLOTLIB_AVAILABLE = False
FigureCanvas = None
Figure = None
if PYQT_AVAILABLE:
    try:
        import matplotlib
        matplotlib.use('QtAgg')
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        MATPLOTLIB_AVAILABLE = True
    except ImportError:
        print("[Warning] Matplotlib not available for plotting")

# Check for OpenCV AFTER PyQt6 to avoid Qt conflicts
# Also disable OpenCV's Qt backend if present
CV2_AVAILABLE = False
try:
    # Prevent OpenCV from using its Qt
    os.environ['QT_QPA_PLATFORM'] = ''
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("[Warning] OpenCV not available. Video features disabled.")

# Import our communication module
try:
    from sdr_video_comm import (
        SDRVideoLink, SDRConfig, OFDMConfig, OTFSConfig, VideoConfig,
        WaveformType, FECType, FECConfig, QAMModulator
    )
    COMM_AVAILABLE = True
except ImportError:
    COMM_AVAILABLE = False
    print("[Warning] sdr_video_comm not available")



# ==============================================================================
# Dark Theme Stylesheet
# ==============================================================================

DARK_STYLE = """
QMainWindow {
    background-color: #1e1e2e;
}
QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', 'Ubuntu', sans-serif;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 10px;
    font-weight: bold;
    background-color: #313244;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px 0 5px;
    color: #89b4fa;
}
QPushButton {
    background-color: #45475a;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    color: #cdd6f4;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #585b70;
}
QPushButton:pressed {
    background-color: #89b4fa;
    color: #1e1e2e;
}
QPushButton:checked {
    background-color: #a6e3a1;
    color: #1e1e2e;
}
QComboBox {
    background-color: #45475a;
    border: 1px solid #585b70;
    border-radius: 4px;
    padding: 4px 8px;
}
QComboBox::drop-down {
    border: none;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #45475a;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #89b4fa;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QProgressBar {
    border: none;
    border-radius: 4px;
    background-color: #45475a;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #a6e3a1;
    border-radius: 4px;
}
QLabel#metric {
    font-size: 24px;
    font-weight: bold;
    color: #f9e2af;
}
QLabel#metric_label {
    font-size: 12px;
    color: #9399b2;
}
QStatusBar {
    background-color: #313244;
    color: #9399b2;
}
"""


# ==============================================================================
# Matplotlib Canvas for Constellation/BER Plots
# ==============================================================================

# Only define MplCanvas if matplotlib is available
if MATPLOTLIB_AVAILABLE and FigureCanvas is not None:
    class MplCanvas(FigureCanvas):
        """Matplotlib canvas widget for embedding plots."""
        
        def __init__(self, parent=None, width=5, height=4, dpi=100):
            self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#1e1e2e')
            self.axes = self.fig.add_subplot(111)
            self.axes.set_facecolor('#313244')
            self.axes.tick_params(colors='#cdd6f4')
            self.axes.spines['bottom'].set_color('#45475a')
            self.axes.spines['top'].set_color('#45475a')
            self.axes.spines['left'].set_color('#45475a')
            self.axes.spines['right'].set_color('#45475a')
            super().__init__(self.fig)
else:
    MplCanvas = None



# ==============================================================================
# Video Worker Thread
# ==============================================================================

class VideoWorker(QThread):
    """Background thread for video capture and processing."""
    
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self.running = False
        self.cap = None
    
    def run(self):
        if not CV2_AVAILABLE:
            return
        
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True
        
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_ready.emit(frame)
            time.sleep(1/30)  # ~30 FPS
        
        self.cap.release()
    
    def stop(self):
        self.running = False
        self.wait()


# ==============================================================================
# Communication Worker Thread
# ==============================================================================

class CommWorker(QThread):
    """Background thread for SDR communication."""
    
    metrics_updated = pyqtSignal(dict)
    rx_frame_ready = pyqtSignal(np.ndarray)
    constellation_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, link: 'SDRVideoLink'):
        super().__init__()
        self.link = link
        self.running = False
        self.tx_queue = queue.Queue(maxsize=5)
        self.mode = 'idle'  # 'idle', 'loopback', 'transmit'
    
    def run(self):
        self.running = True
        last_update = time.time()
        
        while self.running:
            if self.mode == 'loopback':
                # Run continuous loopback test
                results = self.link.loopback_test(num_bits=10000)
                
                # Extract constellation for plotting
                if 'constellation' in results:
                    self.constellation_ready.emit(results['constellation'])
                
                # Emit metrics
                metrics = {
                    'ber': results['ber'],
                    'snr_db': results.get('snr_db', 0),
                    'waveform': results.get('waveform', 'unknown'),
                }
                self.metrics_updated.emit(metrics)
                
                time.sleep(0.5)  # Update every 500ms
            
            elif self.mode == 'transmit':
                try:
                    frame = self.tx_queue.get(timeout=0.1)
                    # TODO: Encode and transmit frame
                except queue.Empty:
                    pass
            
            else:
                time.sleep(0.1)
    
    def stop(self):
        self.running = False
        self.wait()
    
    def set_mode(self, mode: str):
        self.mode = mode


class VideoSimWorker(QThread):
    """Worker thread for video file simulation."""
    
    # Signals: tx_frame (RGB), rx_frame (BGR from OpenCV), metrics dict
    frame_processed = pyqtSignal(object, object, dict)
    progress = pyqtSignal(int, str)  # percent, status text
    finished = pyqtSignal(dict)  # final stats
    
    def __init__(self, link, video_path: str, snr_db: float, channel_type: str, max_frames: int = 100):
        super().__init__()
        self.link = link
        self.video_path = video_path
        self.snr_db = snr_db
        self.channel_type = channel_type.lower()
        self.max_frames = max_frames
        self.running = True
    
    def run(self):
        """Process video file through communication pipeline."""
        try:
            import cv2
        except ImportError:
            self.finished.emit({'error': 'OpenCV not available'})
            return
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.finished.emit({'error': f'Cannot open video: {self.video_path}'})
            return
        
        total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), self.max_frames)
        
        stats = {
            'frames_processed': 0,
            'frames_decoded': 0,
            'total_bits': 0,
            'total_errors': 0,
            'avg_ber': 0.0,
            'avg_psnr_db': 0.0,
        }
        
        ber_values = []
        psnr_values = []
        frame_idx = 0
        
        while self.running and frame_idx < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Simulate transmission through pipeline
            decoded_frame, metrics = self.link.simulate_frame_transmission(
                frame, self.snr_db, self.channel_type
            )
            
            stats['frames_processed'] += 1
            stats['total_bits'] += metrics.get('total_bits', 0)
            stats['total_errors'] += metrics.get('bit_errors', 0)
            
            if metrics.get('decode_success'):
                stats['frames_decoded'] += 1
                psnr_values.append(metrics.get('psnr_db', 0))
            
            ber_values.append(metrics.get('ber', 1.0))
            
            # Convert BGR to RGB for display
            tx_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Emit frame update
            self.frame_processed.emit(tx_rgb, decoded_frame, metrics)
            
            # Emit progress
            percent = int((frame_idx + 1) / total_frames * 100)
            status = f"Frame {frame_idx + 1}/{total_frames} | BER: {metrics.get('ber', 0):.2e}"
            self.progress.emit(percent, status)
            
            frame_idx += 1
            
            # Small delay to allow UI updates
            time.sleep(0.05)
        
        cap.release()
        
        # Calculate averages
        if ber_values:
            stats['avg_ber'] = float(np.mean(ber_values))
        if psnr_values:
            stats['avg_psnr_db'] = float(np.mean(psnr_values))
        
        stats['decode_rate'] = stats['frames_decoded'] / stats['frames_processed'] if stats['frames_processed'] > 0 else 0
        
        self.finished.emit(stats)
    
    def stop(self):
        self.running = False
        self.wait()


# ==============================================================================
# Main Window
# ==============================================================================

class SDRVideoUI(QMainWindow):
    """Main UI window for SDR Video Communication."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SDR Video Communication - OFDM/OTFS")
        self.setMinimumSize(1400, 900)
        self.setStyleSheet(DARK_STYLE)
        
        # Initialize communication link
        if COMM_AVAILABLE:
            self.link = SDRVideoLink(
                sdr_config=SDRConfig(),
                ofdm_config=OFDMConfig(),
                otfs_config=OTFSConfig(),
                waveform=WaveformType.OFDM
            )
        else:
            self.link = None
        
        # Workers
        self.video_worker = None
        self.comm_worker = None
        
        # Current metrics
        self.current_ber = 0.0
        self.current_snr = 0.0
        self.current_throughput = 0.0
        
        # Setup UI
        self._setup_ui()
        self._setup_timers()
    
    def _setup_ui(self):
        """Create the UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Left panel: Video feeds
        left_panel = self._create_video_panel()
        main_layout.addWidget(left_panel, stretch=3)
        
        # Right panel: Controls and metrics
        right_panel = self._create_control_panel()
        main_layout.addWidget(right_panel, stretch=2)
        
        # Status bar
        self.statusBar().showMessage("Ready | SDR: Disconnected")
    
    def _create_video_panel(self) -> QWidget:
        """Create video feed panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("📹 Video Feeds")
        title.setFont(QFont('Segoe UI', 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #89b4fa;")
        layout.addWidget(title)
        
        # Video feeds grid
        feeds = QHBoxLayout()
        
        # TX Video
        tx_group = QGroupBox("TX Source")
        tx_layout = QVBoxLayout(tx_group)
        self.tx_video_label = QLabel()
        self.tx_video_label.setMinimumSize(320, 240)
        self.tx_video_label.setStyleSheet("background-color: #11111b; border-radius: 8px;")
        self.tx_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tx_video_label.setText("No Camera")
        tx_layout.addWidget(self.tx_video_label)
        feeds.addWidget(tx_group)
        
        # RX Video
        rx_group = QGroupBox("RX Decoded")
        rx_layout = QVBoxLayout(rx_group)
        self.rx_video_label = QLabel()
        self.rx_video_label.setMinimumSize(320, 240)
        self.rx_video_label.setStyleSheet("background-color: #11111b; border-radius: 8px;")
        self.rx_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rx_video_label.setText("No Signal")
        rx_layout.addWidget(self.rx_video_label)
        feeds.addWidget(rx_group)
        
        layout.addLayout(feeds)
        
        # Constellation diagram
        const_group = QGroupBox("📊 Constellation Diagram")
        const_layout = QVBoxLayout(const_group)
        
        if MATPLOTLIB_AVAILABLE:
            self.constellation_canvas = MplCanvas(self, width=6, height=4, dpi=100)
            const_layout.addWidget(self.constellation_canvas)
            self._init_constellation_plot()
        else:
            const_layout.addWidget(QLabel("Matplotlib not available"))
        
        layout.addWidget(const_group)
        
        return panel
    
    def _create_control_panel(self) -> QWidget:
        """Create control and metrics panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Metrics display
        metrics_group = QGroupBox("📈 Real-time Metrics")
        metrics_layout = QGridLayout(metrics_group)
        
        # BER
        self.ber_value = QLabel("0.00e+00")
        self.ber_value.setObjectName("metric")
        ber_label = QLabel("Bit Error Rate")
        ber_label.setObjectName("metric_label")
        metrics_layout.addWidget(self.ber_value, 0, 0)
        metrics_layout.addWidget(ber_label, 1, 0)
        
        # SNR
        self.snr_value = QLabel("0.0 dB")
        self.snr_value.setObjectName("metric")
        snr_label = QLabel("Signal-to-Noise Ratio")
        snr_label.setObjectName("metric_label")
        metrics_layout.addWidget(self.snr_value, 0, 1)
        metrics_layout.addWidget(snr_label, 1, 1)
        
        # Throughput
        self.throughput_value = QLabel("0.0 kbps")
        self.throughput_value.setObjectName("metric")
        throughput_label = QLabel("Throughput")
        throughput_label.setObjectName("metric_label")
        metrics_layout.addWidget(self.throughput_value, 0, 2)
        metrics_layout.addWidget(throughput_label, 1, 2)
        
        layout.addWidget(metrics_group)
        
        # Waveform selection
        waveform_group = QGroupBox("📻 Waveform Selection")
        waveform_layout = QVBoxLayout(waveform_group)
        
        wf_row = QHBoxLayout()
        wf_row.addWidget(QLabel("Waveform:"))
        self.waveform_combo = QComboBox()
        self.waveform_combo.addItems(["OFDM", "OTFS"])
        self.waveform_combo.currentTextChanged.connect(self._on_waveform_changed)
        wf_row.addWidget(self.waveform_combo)
        waveform_layout.addLayout(wf_row)
        
        mod_row = QHBoxLayout()
        mod_row.addWidget(QLabel("Modulation:"))
        self.mod_combo = QComboBox()
        self.mod_combo.addItems(["QPSK (4)", "16-QAM", "64-QAM"])
        self.mod_combo.setCurrentIndex(1)  # Default 16-QAM
        waveform_layout.addLayout(mod_row)
        mod_row.addWidget(self.mod_combo)
        
        layout.addWidget(waveform_group)
        
        # FEC Configuration
        fec_group = QGroupBox("🛡️ Error Correction (FEC)")
        fec_layout = QVBoxLayout(fec_group)
        
        fec_row = QHBoxLayout()
        fec_row.addWidget(QLabel("Type:"))
        self.fec_combo = QComboBox()
        self.fec_combo.addItems(["None", "Repetition (Fast GPU)", "LDPC (5G NR)", "Convolutional"])
        self.fec_combo.setCurrentIndex(1)  # Default to Repetition
        self.fec_combo.currentTextChanged.connect(self._on_fec_changed)
        fec_row.addWidget(self.fec_combo)
        fec_layout.addLayout(fec_row)
        
        layout.addWidget(fec_group)
        
        # SDR Configuration
        sdr_group = QGroupBox("🔧 SDR Configuration")
        sdr_layout = QGridLayout(sdr_group)
        
        sdr_layout.addWidget(QLabel("Device:"), 0, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["adrv9009", "ad9361", "pluto"])
        sdr_layout.addWidget(self.device_combo, 0, 1)
        
        sdr_layout.addWidget(QLabel("IP:"), 1, 0)
        self.ip_label = QLabel("ip:192.168.86.40")
        self.ip_label.setStyleSheet("color: #9399b2;")
        sdr_layout.addWidget(self.ip_label, 1, 1)
        
        sdr_layout.addWidget(QLabel("Frequency:"), 2, 0)
        self.freq_label = QLabel("2.4 GHz")
        self.freq_label.setStyleSheet("color: #9399b2;")
        sdr_layout.addWidget(self.freq_label, 2, 1)
        
        layout.addWidget(sdr_group)
        
        # Control buttons
        controls_group = QGroupBox("🎮 Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        btn_row1 = QHBoxLayout()
        self.connect_btn = QPushButton("🔌 Connect SDR")
        self.connect_btn.clicked.connect(self._on_connect_sdr)
        btn_row1.addWidget(self.connect_btn)
        
        self.camera_btn = QPushButton("📷 Start Camera")
        self.camera_btn.clicked.connect(self._on_toggle_camera)
        btn_row1.addWidget(self.camera_btn)
        controls_layout.addLayout(btn_row1)
        
        btn_row2 = QHBoxLayout()
        self.loopback_btn = QPushButton("🔄 Loopback Test")
        self.loopback_btn.setCheckable(True)
        self.loopback_btn.clicked.connect(self._on_toggle_loopback)
        btn_row2.addWidget(self.loopback_btn)
        
        self.ber_btn = QPushButton("📊 BER Sweep")
        self.ber_btn.clicked.connect(self._on_ber_sweep)
        btn_row2.addWidget(self.ber_btn)
        controls_layout.addLayout(btn_row2)
        
        layout.addWidget(controls_group)
        
        # Video File Simulation
        video_sim_group = QGroupBox("🎬 Video File Simulation")
        video_sim_layout = QVBoxLayout(video_sim_group)
        
        # File selection row
        file_row = QHBoxLayout()
        self.video_path_label = QLabel("No file selected")
        self.video_path_label.setStyleSheet("color: #9399b2;")
        file_row.addWidget(self.video_path_label, stretch=1)
        
        self.select_file_btn = QPushButton("📁 Select Video")
        self.select_file_btn.clicked.connect(self._on_select_video_file)
        file_row.addWidget(self.select_file_btn)
        video_sim_layout.addLayout(file_row)
        
        # SNR slider
        snr_row = QHBoxLayout()
        snr_row.addWidget(QLabel("SNR:"))
        self.snr_slider = QSlider(Qt.Orientation.Horizontal)
        self.snr_slider.setMinimum(0)
        self.snr_slider.setMaximum(30)
        self.snr_slider.setValue(20)
        self.snr_slider.valueChanged.connect(self._on_snr_changed)
        snr_row.addWidget(self.snr_slider)
        self.snr_display = QLabel("20 dB")
        self.snr_display.setMinimumWidth(50)
        snr_row.addWidget(self.snr_display)
        video_sim_layout.addLayout(snr_row)
        
        # Channel type
        channel_row = QHBoxLayout()
        channel_row.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["AWGN", "Rayleigh", "Rician"])
        channel_row.addWidget(self.channel_combo)
        video_sim_layout.addLayout(channel_row)
        
        # Start simulation button
        self.start_sim_btn = QPushButton("▶️ Start Video Simulation")
        self.start_sim_btn.clicked.connect(self._on_start_video_simulation)
        video_sim_layout.addWidget(self.start_sim_btn)
        
        # Progress
        self.sim_progress = QProgressBar()
        self.sim_progress.setValue(0)
        video_sim_layout.addWidget(self.sim_progress)
        
        # Stats label
        self.sim_stats_label = QLabel("Ready")
        self.sim_stats_label.setStyleSheet("color: #a6e3a1;")
        video_sim_layout.addWidget(self.sim_stats_label)
        
        layout.addWidget(video_sim_group)
        
        # Video file path storage
        self.selected_video_path = None
        self.sim_running = False
        
        # BER plot
        ber_group = QGroupBox("📉 BER vs SNR")
        ber_layout = QVBoxLayout(ber_group)
        
        if MATPLOTLIB_AVAILABLE:
            self.ber_canvas = MplCanvas(self, width=5, height=3, dpi=100)
            ber_layout.addWidget(self.ber_canvas)
        else:
            ber_layout.addWidget(QLabel("Matplotlib not available"))
        
        layout.addWidget(ber_group)
        
        layout.addStretch()
        
        return panel
    
    def _init_constellation_plot(self):
        """Initialize empty constellation plot."""
        ax = self.constellation_canvas.axes
        ax.clear()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xlabel('In-Phase', color='#cdd6f4')
        ax.set_ylabel('Quadrature', color='#cdd6f4')
        ax.set_title('Constellation', color='#cdd6f4')
        ax.grid(True, alpha=0.3, color='#45475a')
        ax.axhline(0, color='#45475a', linewidth=0.5)
        ax.axvline(0, color='#45475a', linewidth=0.5)
        self.constellation_canvas.draw()
    
    def _setup_timers(self):
        """Setup update timers."""
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self._update_display)
        self.metrics_timer.start(500)  # Update every 500ms
    
    def _on_waveform_changed(self, text):
        """Handle waveform selection change."""
        if self.link:
            self.link.waveform = WaveformType.OFDM if text == "OFDM" else WaveformType.OTFS
            self.statusBar().showMessage(f"Waveform: {text}")
    
    def _on_connect_sdr(self):
        """Connect to SDR device."""
        if not self.link:
            self.statusBar().showMessage("Communication module not available")
            return
        
        success = self.link.connect_sdr()
        if success:
            self.statusBar().showMessage("SDR: Connected")
            self.connect_btn.setText("✅ Connected")
            self.connect_btn.setEnabled(False)
        else:
            self.statusBar().showMessage("SDR: Using simulation mode")
            self.connect_btn.setText("⚡ Simulation Mode")
    
    def _on_toggle_camera(self):
        """Toggle camera capture."""
        if not CV2_AVAILABLE:
            self.statusBar().showMessage("OpenCV not available")
            return
        
        if self.video_worker is None or not self.video_worker.isRunning():
            self.video_worker = VideoWorker()
            self.video_worker.frame_ready.connect(self._update_tx_video)
            self.video_worker.start()
            self.camera_btn.setText("⏹ Stop Camera")
        else:
            self.video_worker.stop()
            self.camera_btn.setText("📷 Start Camera")
            self.tx_video_label.setText("No Camera")
    
    def _on_toggle_loopback(self):
        """Toggle continuous loopback test."""
        if not self.link:
            self.loopback_btn.setChecked(False)
            return
        
        if self.loopback_btn.isChecked():
            # Start loopback
            self.comm_worker = CommWorker(self.link)
            self.comm_worker.metrics_updated.connect(self._update_metrics)
            self.comm_worker.constellation_ready.connect(self._update_constellation)
            self.comm_worker.set_mode('loopback')
            self.comm_worker.start()
            self.loopback_btn.setText("⏹ Stop Loopback")
        else:
            # Stop loopback
            if self.comm_worker:
                self.comm_worker.stop()
                self.comm_worker = None
            self.loopback_btn.setText("🔄 Loopback Test")
    
    def _on_ber_sweep(self):
        """Run BER sweep test."""
        if not self.link:
            self.statusBar().showMessage("Communication module not available")
            return
        
        self.statusBar().showMessage("Running BER sweep...")
        self.ber_btn.setEnabled(False)
        
        # Run in separate thread
        def run_sweep():
            results = self.link.ber_sweep()
            self._plot_ber_results(results)
            self.ber_btn.setEnabled(True)
            self.statusBar().showMessage("BER sweep complete")
        
        threading.Thread(target=run_sweep, daemon=True).start()
    
    def _update_tx_video(self, frame):
        """Update TX video display."""
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(
            self.tx_video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.tx_video_label.setPixmap(scaled)
    
    def _update_metrics(self, metrics):
        """Update metrics display from worker thread."""
        self.current_ber = metrics.get('ber', 0)
        self.current_snr = metrics.get('snr_db', 0)
    
    def _update_constellation(self, symbols):
        """Update constellation plot."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        ax = self.constellation_canvas.axes
        ax.clear()
        
        ax.scatter(
            symbols.real, symbols.imag,
            c='#89b4fa', s=10, alpha=0.5
        )
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xlabel('In-Phase', color='#cdd6f4')
        ax.set_ylabel('Quadrature', color='#cdd6f4')
        ax.set_title('RX Constellation', color='#cdd6f4')
        ax.grid(True, alpha=0.3, color='#45475a')
        ax.axhline(0, color='#45475a', linewidth=0.5)
        ax.axvline(0, color='#45475a', linewidth=0.5)
        ax.set_facecolor('#313244')
        
        self.constellation_canvas.draw()
    
    def _plot_ber_results(self, results):
        """Plot BER sweep results."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        ax = self.ber_canvas.axes
        ax.clear()
        
        ax.semilogy(results['snr'], results['ber_ofdm'], 'b-o', 
                    label='OFDM', linewidth=2, color='#89b4fa')
        ax.semilogy(results['snr'], results['ber_otfs'], 'r-s',
                    label='OTFS', linewidth=2, color='#f38ba8')
        
        ax.set_xlabel('SNR (dB)', color='#cdd6f4')
        ax.set_ylabel('BER', color='#cdd6f4')
        ax.set_title('BER vs SNR', color='#cdd6f4')
        ax.legend(facecolor='#313244', edgecolor='#45475a', labelcolor='#cdd6f4')
        ax.grid(True, alpha=0.3, color='#45475a', which='both')
        ax.set_ylim([1e-5, 1])
        ax.set_facecolor('#313244')
        ax.tick_params(colors='#cdd6f4')
        
        self.ber_canvas.draw()
    
    def _update_display(self):
        """Periodic display update."""
        self.ber_value.setText(f"{self.current_ber:.2e}")
        self.snr_value.setText(f"{self.current_snr:.1f} dB")
        self.throughput_value.setText(f"{self.current_throughput:.1f} kbps")
    
    # ========== Video Simulation Methods ==========
    
    def _on_select_video_file(self):
        """Open file dialog to select video file."""
        from PyQt6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.webm);;All Files (*)"
        )
        
        if file_path:
            self.selected_video_path = file_path
            # Show just filename
            name = file_path.split('/')[-1]
            if len(name) > 30:
                name = name[:27] + "..."
            self.video_path_label.setText(name)
            self.video_path_label.setToolTip(file_path)
            self.statusBar().showMessage(f"Selected: {file_path}")
    
    def _on_snr_changed(self, value):
        """Update SNR display when slider changes."""
        self.snr_display.setText(f"{value} dB")
    
    def _on_fec_changed(self, text):
        """Handle FEC selection change."""
        self.statusBar().showMessage(f"FEC selected: {text}")

    def _on_start_video_simulation(self):
        """Start video file simulation through communication pipeline."""
        if self.sim_running:
            # Stop simulation
            self.sim_running = False
            self.start_sim_btn.setText("▶️ Start Video Simulation")
            return
        
        if not self.selected_video_path:
            self.statusBar().showMessage("Please select a video file first")
            return
        
        if not COMM_AVAILABLE:
            self.statusBar().showMessage("Communication module not available")
            return
        
        # Get parameters
        snr_db = self.snr_slider.value()
        channel_type = self.channel_combo.currentText().lower()
        waveform = WaveformType.OTFS if self.waveform_combo.currentIndex() == 1 else WaveformType.OFDM
        
        # Determine FEC config
        fec_idx = self.fec_combo.currentIndex()
        fec_config = FECConfig(enabled=False)
        
        if fec_idx == 1:  # Repetition
            fec_config = FECConfig(enabled=True, fec_type=FECType.REPETITION, repetitions=7)
        elif fec_idx == 2:  # LDPC
            fec_config = FECConfig(enabled=True, fec_type=FECType.LDPC, code_rate="1/2")
        elif fec_idx == 3:  # Convolutional
            fec_config = FECConfig(enabled=True, fec_type=FECType.CONVOLUTIONAL)
        
        # Create link with new params
        # Always recreate to ensure config is applied
        self.link = SDRVideoLink(
            waveform=waveform, 
            fec_config=fec_config,
            simulation_mode=True
        )
        
        self.sim_running = True
        self.start_sim_btn.setText("⏹️ Stop Simulation")
        self.sim_progress.setValue(0)
        self.sim_stats_label.setText("Starting...")
        
        # Start worker thread
        self.sim_worker = VideoSimWorker(
            self.link,
            self.selected_video_path,
            snr_db,
            channel_type
        )
        self.sim_worker.frame_processed.connect(self._on_sim_frame)
        self.sim_worker.progress.connect(self._on_sim_progress)
        self.sim_worker.finished.connect(self._on_sim_finished)
        self.sim_worker.start()

        # Update Config Display
        if self.link and self.link.sdr_config:
            cfg = self.link.sdr_config
            self.ip_title_label = self.findChild(QLabel, "ip_label") # Assuming I can find it or I need to store it
            # Actually, I should store references in _create_control_panel
            # But simpler: Just update self.ip_label which I stored earlier
            self.ip_label.setText(cfg.sdr_ip)
            self.freq_label.setText(f"{cfg.fc/1e9:.1f} GHz")
            # Select device in combo
            index = self.device_combo.findText(cfg.device)
            if index >= 0:
                self.device_combo.setCurrentIndex(index)
    
    def _on_sim_frame(self, tx_frame, rx_frame, metrics):
        """Handle processed frame from simulation."""
        # Update TX display
        if tx_frame is not None:
            h, w, ch = tx_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(tx_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            scaled = pixmap.scaled(
                self.tx_video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.tx_video_label.setPixmap(scaled)
        
        # Update RX display
        if rx_frame is not None:
            # Convert BGR to RGB for Qt
            rgb_frame = rx_frame[:, :, ::-1].copy()
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            scaled = pixmap.scaled(
                self.rx_video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.rx_video_label.setPixmap(scaled)
        
        # Update metrics
        self.current_ber = metrics.get('ber', 0)
        self.current_snr = metrics.get('snr_db', 0)
        psnr = metrics.get('psnr_db', 0)
        
        self.sim_stats_label.setText(
            f"BER: {self.current_ber:.2e} | PSNR: {psnr:.1f} dB | "
            f"{'✓ Decoded' if metrics.get('decode_success') else '✗ Failed'}"
        )
    
    def _on_sim_progress(self, percent, stats_text):
        """Update simulation progress."""
        self.sim_progress.setValue(percent)
        self.statusBar().showMessage(stats_text)
    
    def _on_sim_finished(self, stats):
        """Handle simulation completion."""
        self.sim_running = False
        self.start_sim_btn.setText("▶️ Start Video Simulation")
        
        if 'error' in stats:
            self.sim_stats_label.setText(f"Error: {stats['error']}")
        else:
            decode_rate = stats.get('decode_rate', 0) * 100
            avg_ber = stats.get('avg_ber', 0)
            avg_psnr = stats.get('avg_psnr_db', 0)
            self.sim_stats_label.setText(
                f"Done: {decode_rate:.1f}% decoded | Avg BER: {avg_ber:.2e} | Avg PSNR: {avg_psnr:.1f} dB"
            )
        
        self.sim_progress.setValue(100)
    
    def closeEvent(self, event):
        """Clean up on close."""
        if self.video_worker:
            self.video_worker.stop()
        if self.comm_worker:
            self.comm_worker.stop()
        if hasattr(self, 'sim_worker') and self.sim_worker:
            self.sim_worker.stop()
        event.accept()


# ==============================================================================
# Main
# ==============================================================================

def main():
    if not PYQT_AVAILABLE:
        print("Error: PyQt6 is required. Install with: pip install pyqt6")
        sys.exit(1)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = SDRVideoUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

import sys
import os
import time
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QLabel, QCheckBox, QComboBox, QSlider, QGroupBox, QPushButton,
    QSplitter, QFrame, QLineEdit, QFormLayout, QSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QMutex, QRectF
from PyQt6.QtGui import QColor, QFont
import matplotlib.pyplot as plt # For colormap matching if needed, but we used pg.colormap

# Add path to find myradar
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from myradar_all_in_one_v2 import RadarConfig, AIRadarDataset, RadarEngine

class RadarWorker(QThread):
    """
    Worker thread that handles data acquisition (Sim/Loopback/Hardware) and Processing (RDM/CFAR).
    """
    data_ready = pyqtSignal(object)   # (rdm_dB, detections, targets, false_alarms, r_axis, v_axis)
    status_update = pyqtSignal(str)   # Status messages
    diag_update = pyqtSignal(dict)    # Diagnostics from loopback/hardware

    def __init__(self, config_name='config_cn0566'):
        super().__init__()
        self.running = False
        self.mode = 'simulation'  # 'simulation', 'digital_loopback', 'hardware'
        self.config_name = config_name
        self._cfg = RadarConfig(config_name=config_name)
        self.engine = None
        self.mutex = QMutex()

        # Connection settings
        self.sdr_ip = "ip:192.168.2.1"
        self.phaser_ip = "ip:phaser.local"

        # Runtime flags
        self.enable_impairments = True
        self.enable_compensation = True
        self.enable_phase_tracking = True
        self.enable_os_cfar = True
        self.cfar_threshold = 25
        self.sim_target_speed = 5.0
        self.num_targets = 2

        # Background Subtraction
        self.bg_frame = None
        self.trigger_cal = False
        self.enable_clutter_removal = False
        self.min_range = 0.0
        self.detection_mode = 'CFAR'  # 'CFAR' or 'DL'

        # AIRadarDataset for simulation mode (direct control)
        self.dataset = AIRadarDataset(num_samples=1,
                                      config_name=self.config_name)

    def calibrate_background(self):
        self.trigger_cal = True

    def _build_targets(self, timestep):
        """Build moving targets for simulation/loopback modes."""
        targets = []
        angle = timestep * 0.1
        r1 = 50 + 20 * np.sin(angle * 0.5)
        v1 = self.sim_target_speed * np.cos(angle * 0.5)
        targets.append({'range': r1, 'velocity': v1, 'rcs': 20,
                        'azimuth': 0, 'elevation': 0})
        if self.num_targets > 1:
            for i in range(1, self.num_targets):
                offset = i * (2 * np.pi / 5)
                r_i = 30 + 15 * np.sin(angle * 0.3 + offset)
                v_i = 3.0 * np.cos(angle * 0.3 + offset)
                targets.append({'range': r_i, 'velocity': v_i, 'rcs': 15,
                                'azimuth': 0, 'elevation': 0})
        return targets

    def run(self):
        self.running = True
        self.status_update.emit(f"Starting Worker in {self.mode} mode...")

        # For loopback/hardware, create and start RadarEngine
        if self.mode in ('digital_loopback', 'hardware'):
            try:
                self.status_update.emit(f"Connecting ({self.mode})...")
                self.engine = RadarEngine(
                    config_name=self.config_name,
                    mode=self.mode,
                    sdr_ip=self.sdr_ip,
                    phaser_ip=self.phaser_ip
                )
                self.engine.start()
                self.status_update.emit(f"{self.mode} engine started")
            except Exception as e:
                self.status_update.emit(f"Engine start failed: {e}")
                self.running = False
                return

        timestep = 0

        while self.running:
            t_start = time.time()
            try:
                targets = []
                false_alarms = []

                if self.mode == 'simulation':
                    # --- Simulation: direct dataset control ---
                    self.dataset.config['enable_compensation'] = self.enable_compensation
                    self.dataset.config['enable_phase_noise_tracking'] = self.enable_phase_tracking
                    self.dataset.config['cfar_type'] = 'OS' if self.enable_os_cfar else 'CA'
                    self.dataset.cfar_params['threshold_offset'] = self.cfar_threshold
                    self.dataset.cfar_params['min_range_m'] = self.min_range
                    self.dataset.apply_realistic_effects = self.enable_impairments
                    self.dataset.config['disable_impairments'] = not self.enable_impairments

                    targets = self._build_targets(timestep)
                    beat = self.dataset.simulate_fmcw_signal(targets, snr_db=25)
                    if beat.ndim == 3 and beat.shape[0] == 1:
                        beat = beat[0]

                    # Background subtraction
                    if self.trigger_cal:
                        self.bg_frame = beat.copy()
                        self.trigger_cal = False
                        self.status_update.emit("Background Calibrated")
                    if self.enable_clutter_removal and self.bg_frame is not None:
                        if beat.shape == self.bg_frame.shape:
                            beat = beat - self.bg_frame
                        else:
                            self.bg_frame = None

                    if beat.size == 0 or beat.ndim < 2:
                        timestep += 1
                        continue

                    current_Ns = beat.shape[-1]
                    if current_Ns != self.dataset.Ns:
                        self.dataset.Ns = current_Ns

                    rdm_db = self.dataset.compute_rdm(beat)

                    if self.detection_mode == 'DL' and hasattr(self.dataset, 'run_dl_detection'):
                        detections = self.dataset.run_dl_detection(rdm_db)
                    else:
                        detections = self.dataset.cfar_detection(rdm_db)

                    false_alarms = self._check_false_alarms(detections, targets)
                    r_axis = self.dataset.range_axis
                    v_axis = self.dataset.velocity_axis

                elif self.mode == 'digital_loopback':
                    # --- Digital Loopback: use RadarEngine ---
                    self.engine.update_processor_params(
                        cfar_threshold=self.cfar_threshold,
                        cfar_type='OS' if self.enable_os_cfar else 'CA',
                        enable_compensation=self.enable_compensation,
                        enable_phase_tracking=self.enable_phase_tracking,
                        min_range=self.min_range
                    )
                    targets = self._build_targets(timestep)
                    rdm_db, detections, targets, diag, r_axis, v_axis = \
                        self.engine.get_frame(targets=targets, snr_db=25)
                    false_alarms = self._check_false_alarms(detections, targets)
                    self.diag_update.emit(diag)

                elif self.mode == 'hardware':
                    # --- Hardware: use RadarEngine ---
                    self.engine.update_processor_params(
                        cfar_threshold=self.cfar_threshold,
                        cfar_type='OS' if self.enable_os_cfar else 'CA',
                        enable_compensation=self.enable_compensation,
                        enable_phase_tracking=self.enable_phase_tracking,
                        min_range=self.min_range
                    )
                    rdm_db, detections, targets, diag, r_axis, v_axis = \
                        self.engine.get_frame()
                    self.diag_update.emit(diag)

                self.data_ready.emit((rdm_db, detections, targets, false_alarms,
                                      r_axis, v_axis))

                timestep += 1
                elapsed = time.time() - t_start
                sleep_time = max(0, 0.05 - elapsed)
                time.sleep(sleep_time)

            except Exception as e:
                self.status_update.emit(f"Loop Error: {str(e)[:80]}")
                time.sleep(1)

    def _check_false_alarms(self, detections, targets):
        R_THRESH = 3.0
        V_THRESH = 1.0
        fas = []
        for det in detections:
            is_match = False
            dr = det['range_m']
            dv = det['velocity_mps']
            for t in targets:
                tr = t['range']
                tv = t['velocity']
                if abs(dr - tr) < R_THRESH and abs(dv - tv) < V_THRESH:
                    is_match = True
                    break
            if not is_match:
                fas.append(det)
        return fas

    def update_steering(self, angle):
        if self.engine and self.mode == 'hardware':
            try:
                self.engine.steer_beam(angle)
            except Exception as e:
                print(f"Steering Failed: {e}")

    def stop(self):
        self.running = False
        if self.engine is not None:
            try:
                self.engine.stop()
            except Exception:
                pass
            self.engine = None
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RadarSensing V2 - Professional")
        self.setGeometry(100, 100, 1600, 900)
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
            QSpinBox {background-color: #333; color: white; border: 1px solid #555; padding: 4px;}
        """)
        
        self.worker = RadarWorker()
        self.worker.data_ready.connect(self.update_plots)
        self.worker.status_update.connect(self.update_status)
        self.worker.diag_update.connect(self.update_diagnostics)
        
        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # --- Sidebar ---
        sidebar = QFrame()
        sidebar.setFixedWidth(320)
        sidebar.setStyleSheet("background-color: #262626; border-right: 1px solid #444;")
        sidebar_layout = QVBoxLayout(sidebar)
        
        title = QLabel("RADAR CONTROL")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #4CAF50; margin-bottom: 15px;")
        sidebar_layout.addWidget(title)
        
        # Connection
        conn_group = QGroupBox("Device Connection")
        conn_layout = QFormLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Simulation", "Digital Loopback", "CN0566 Hardware"])
        self.mode_combo.currentTextChanged.connect(self.change_mode)

        self.txt_sdr_ip = QLineEdit("ip:192.168.2.1")
        self.txt_phaser_ip = QLineEdit("ip:phaser.local")
        
        conn_layout.addRow("Mode:", self.mode_combo)
        conn_layout.addRow("SDR IP:", self.txt_sdr_ip)
        conn_layout.addRow("Phaser IP:", self.txt_phaser_ip)
        
        self.btn_run = QPushButton("START")
        self.btn_run.setCheckable(True)
        self.btn_run.setStyleSheet("""
            QPushButton {background-color: #4CAF50; color: white; border: none; padding: 10px; border-radius: 4px; font-weight: bold; font-size: 14px;} 
            QPushButton:checked {background-color: #F44336; text-decoration: none;}
            QPushButton:hover {background-color: #5cb860;}
        """)
        self.btn_run.clicked.connect(self.toggle_run)
        conn_layout.addRow(self.btn_run)
        conn_group.setLayout(conn_layout)
        sidebar_layout.addWidget(conn_group)
        
        # Hardware Steering
        steer_group = QGroupBox("Beam Steering")
        steer_layout = QVBoxLayout()
        self.lbl_steer = QLabel("Angle: 0°")
        self.slider_steer = QSlider(Qt.Orientation.Horizontal)
        self.slider_steer.setRange(-80, 80)
        self.slider_steer.setValue(0)
        self.slider_steer.setTickInterval(10)
        self.slider_steer.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_steer.valueChanged.connect(self.update_steering)
        self.slider_steer.setEnabled(False) # Disabled in Sim
        steer_layout.addWidget(self.lbl_steer)
        steer_layout.addWidget(self.slider_steer)
        steer_group.setLayout(steer_layout)
        sidebar_layout.addWidget(steer_group)
        
        # Sim Control
        sim_group = QGroupBox("Simulation Control")
        sim_layout = QFormLayout()
        self.spin_targets = QSpinBox()
        self.spin_targets.setRange(1, 10)
        self.spin_targets.setValue(2)
        self.spin_targets.valueChanged.connect(self.update_worker_params)
        sim_layout.addRow("Num Targets:", self.spin_targets)
        sim_group.setLayout(sim_layout)
        sidebar_layout.addWidget(sim_group)
        
        # DSP
        dsp_group = QGroupBox("DSP & Robustness")
        dsp_layout = QVBoxLayout()
        self.chk_impairments = QCheckBox("Simulate Impairments")
        self.chk_impairments.setChecked(True)
        self.chk_compensation = QCheckBox("Enable Compensation")
        self.chk_compensation.setChecked(True)
        self.chk_phasenoise = QCheckBox("Phase Noise Tracking")
        self.chk_phasenoise.setChecked(True)

        self.chk_clutter = QCheckBox("Background Subtraction")
        self.chk_clutter.setChecked(False)
        self.btn_cal_bg = QPushButton("Calibrate BG")
        self.btn_cal_bg.setStyleSheet("background-color: #555; border: 1px solid #777; padding: 4px;")
        self.btn_cal_bg.clicked.connect(self.calibrate_bg)
        
        self.chk_impairments.toggled.connect(self.update_worker_params)
        self.chk_compensation.toggled.connect(self.update_worker_params)
        self.chk_phasenoise.toggled.connect(self.update_worker_params)
        self.chk_clutter.toggled.connect(self.update_worker_params)
        
        dsp_layout.addWidget(self.chk_impairments)
        dsp_layout.addWidget(self.chk_compensation)
        dsp_layout.addWidget(self.chk_phasenoise)
        dsp_layout.addWidget(self.chk_clutter)
        dsp_layout.addWidget(self.btn_cal_bg)
        dsp_group.setLayout(dsp_layout)
        sidebar_layout.addWidget(dsp_group)
        
        # Detection
        det_group = QGroupBox("Detection")
        det_layout = QVBoxLayout()
        
        # Mode Selector
        self.combo_det_mode = QComboBox()
        self.combo_det_mode.addItems(["CFAR", "Deep Learning"])
        self.combo_det_mode.currentTextChanged.connect(self.update_worker_params)
        det_layout.addWidget(QLabel("Algorithm:"))
        det_layout.addWidget(self.combo_det_mode)
        
        self.chk_oscfar = QCheckBox("Enable OS-CFAR")
        self.chk_oscfar.setChecked(True)
        self.chk_oscfar.toggled.connect(self.update_worker_params)
        
        self.lbl_thresh = QLabel("Threshold: 25 dB")
        self.slider_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh.setRange(5, 50)
        self.slider_thresh.setValue(25)
        self.slider_thresh.valueChanged.connect(self.update_worker_params)
        
        det_layout.addWidget(self.chk_oscfar)
        det_layout.addWidget(self.lbl_thresh)
        det_layout.addWidget(self.slider_thresh)
        
        self.lbl_min_range = QLabel("Min Range: 0 m")
        self.slider_min_range = QSlider(Qt.Orientation.Horizontal)
        self.slider_min_range.setRange(0, 20)
        self.slider_min_range.setValue(0)
        self.slider_min_range.valueChanged.connect(self.update_worker_params)
        det_layout.addWidget(self.lbl_min_range)
        det_layout.addWidget(self.slider_min_range)
        
        det_group.setLayout(det_layout)
        sidebar_layout.addWidget(det_group)

        # Legend Info
        legend_group = QGroupBox("Legend")
        legend_layout = QVBoxLayout()
        legend_layout.addWidget(QLabel("● Green: Ground Truth"))
        legend_layout.addWidget(QLabel("● Cyan:  Detections"))
        legend_layout.addWidget(QLabel("● Red:   False Alarms/Noise"))
        legend_group.setLayout(legend_layout)
        sidebar_layout.addWidget(legend_group)
        
        sidebar_layout.addStretch()

        # Diagnostics display (loopback/hardware)
        self.lbl_diag = QLabel("")
        self.lbl_diag.setWordWrap(True)
        self.lbl_diag.setStyleSheet("color: #88ccff; font-size: 11px; padding: 4px;")
        sidebar_layout.addWidget(self.lbl_diag)

        self.lbl_info = QLabel("Ready")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("color: #aaa; border-top: 1px solid #444; padding-top: 10px;")
        sidebar_layout.addWidget(self.lbl_info)
        
        main_layout.addWidget(sidebar)
        
        # --- Visualization ---
        viz_splitter = QSplitter(Qt.Orientation.Vertical)
        viz_splitter.setHandleWidth(2)
        
        # 3D Plot
        self.plot3d = gl.GLViewWidget()
        self.plot3d.setCameraPosition(distance=180, elevation=30, azimuth=45)
        self.plot3d.setBackgroundColor((20, 20, 20))
        
        # Surface with Color
        # Initial dummy data
        init_z = np.zeros((10, 10))
        init_x = np.linspace(0, 1, 10)
        init_y = np.linspace(0, 1, 10)
        
        # Use simple shader options or manual colors
        self.surface = gl.GLSurfacePlotItem(x=init_x, y=init_y, z=init_z, computeNormals=True, smooth=True, shader='shaded')
        self.plot3d.addItem(self.surface)
        
        # 3DMarkers
        self.scatter_3d_gt = gl.GLScatterPlotItem(color=(0,1,0,1), size=15, pxMode=True)
        self.scatter_3d_det = gl.GLScatterPlotItem(color=(0,1,1,1), size=12, pxMode=True)
        self.scatter_3d_fp = gl.GLScatterPlotItem(color=(1,0,0,1), size=15, pxMode=True)
        self.plot3d.addItem(self.scatter_3d_gt)
        self.plot3d.addItem(self.scatter_3d_det)
        self.plot3d.addItem(self.scatter_3d_fp)
        
        # Grid - Cover Range (0-150m) and Velocity (-15 to 15m/s)
        # Size needs to be slightly larger than max range/vel
        # GLGridItem is centered at (0,0,0) by default
        grid = gl.GLGridItem()
        grid.setSize(x=40, y=160, z=0) # Width (Vel)=40, Height (Range)=160
        grid.setSpacing(x=5, y=10)
        grid.translate(0, 80, 0) # Shift Y up by 80 to cover 0..160
        self.plot3d.addItem(grid)
        
        # Add overlay legend for 3D? Or just put it in the layout?
        # Let's add a layout for the 3D container to hold a label
        
        viz_splitter.addWidget(self.plot3d)
        
        # 2D Plot
        self.plot2d_widget = pg.PlotWidget(title="Range-Velocity Map (BEV)")
        self.plot2d_widget.setBackground('#181818')
        self.plot2d_widget.setLabel('bottom', 'Velocity', units='m/s', **{'color': '#ddd'})
        self.plot2d_widget.setLabel('left', 'Range', units='m', **{'color': '#ddd'})
        self.plot2d_widget.getAxis('bottom').setPen('#666')
        self.plot2d_widget.getAxis('left').setPen('#666')
        self.plot2d_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot2d_widget.addLegend(offset=(10, 10))
        
        self.img_item = pg.ImageItem()
        self.plot2d_widget.addItem(self.img_item)
        
        # Custom Colormap for RDM (Match Viridis-style roughly if needed, or stick to bwr/jet)
        # Using Viridis-like
        pos = np.array([0.0, 0.4, 0.7, 1.0])
        color = np.array([
            [68, 1, 84, 255],   # Dark Purple
            [49, 104, 142, 255], # Blue
            [53, 183, 121, 255], # Green
            [253, 231, 37, 255]  # Yellow
        ], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        self.img_item.setLookupTable(cmap.getLookupTable())
        
        # 2D Markers
        self.scatter_gt = pg.ScatterPlotItem(pen=pg.mkPen(None), brush=pg.mkBrush('g'), symbol='o', size=14, name="Ground Truth")
        self.scatter_det = pg.ScatterPlotItem(pen=pg.mkPen('c', width=2), brush=pg.mkBrush(None), symbol='x', size=12, name="Detections")
        self.scatter_fp = pg.ScatterPlotItem(pen=pg.mkPen('r', width=2), brush=pg.mkBrush(None), symbol='t', size=16, name="False Alarms")
        
        self.plot2d_widget.addItem(self.scatter_gt)
        self.plot2d_widget.addItem(self.scatter_det)
        self.plot2d_widget.addItem(self.scatter_fp)
        
        viz_splitter.addWidget(self.plot2d_widget)
        viz_splitter.setSizes([500, 400])
        main_layout.addWidget(viz_splitter)
        
    def toggle_run(self, checked):
        if checked:
            self.btn_run.setText("STOP")
            self.btn_run.setStyleSheet("QPushButton {background-color: #F44336; color: white; border: none; padding: 10px; border-radius: 4px; font-weight: bold; font-size: 14px;}")
            self.worker.sdr_ip = self.txt_sdr_ip.text()
            self.worker.phaser_ip = self.txt_phaser_ip.text()
            self.worker.start()
        else:
            self.btn_run.setText("START")
            self.btn_run.setStyleSheet("QPushButton {background-color: #4CAF50; color: white; border: none; padding: 10px; border-radius: 4px; font-weight: bold; font-size: 14px;}")
            self.worker.stop()
            
    def change_mode(self, text):
        if "Hardware" in text:
            self.worker.mode = 'hardware'
            self.chk_impairments.setEnabled(False)
            self.spin_targets.setEnabled(False)
            self.slider_steer.setEnabled(True)
            self.txt_phaser_ip.setEnabled(True)
            self.lbl_info.setText("Mode: CN0566 Hardware")
        elif "Loopback" in text:
            self.worker.mode = 'digital_loopback'
            self.chk_impairments.setEnabled(False)
            self.spin_targets.setEnabled(True)
            self.slider_steer.setEnabled(False)
            self.txt_phaser_ip.setEnabled(False)
            self.lbl_info.setText("Mode: Digital Loopback (Pluto SDR)")
        else:
            self.worker.mode = 'simulation'
            self.chk_impairments.setEnabled(True)
            self.spin_targets.setEnabled(True)
            self.slider_steer.setEnabled(False)
            self.txt_phaser_ip.setEnabled(True)
            self.lbl_info.setText("Mode: Simulation")
            
    def update_steering(self):
        angle = self.slider_steer.value()
        self.lbl_steer.setText(f"Angle: {angle}°")
        self.worker.update_steering(angle)

    def calibrate_bg(self):
        self.worker.calibrate_background()
            
    def update_worker_params(self):
        self.worker.enable_impairments = self.chk_impairments.isChecked()
        self.worker.enable_compensation = self.chk_compensation.isChecked()
        self.worker.enable_phase_tracking = self.chk_phasenoise.isChecked()
        self.worker.enable_clutter_removal = self.chk_clutter.isChecked()
        
        mode = self.combo_det_mode.currentText()
        self.worker.detection_mode = 'DL' if mode == "Deep Learning" else 'CFAR'
        
        self.worker.enable_os_cfar = self.chk_oscfar.isChecked()
        self.worker.cfar_threshold = self.slider_thresh.value()
        self.worker.min_range = self.slider_min_range.value()
        self.worker.num_targets = self.spin_targets.value()
        self.lbl_thresh.setText(f"Threshold: {self.worker.cfar_threshold} dB")
        self.lbl_min_range.setText(f"Min Range: {self.worker.min_range} m")
            
    def update_status(self, msg):
        self.lbl_info.setText(msg)

    def update_diagnostics(self, diag):
        parts = []
        if 'rx_power_dB' in diag:
            parts.append(f"RX: {diag['rx_power_dB']:.1f} dB")
        if 'tx_rx_correlation' in diag:
            parts.append(f"Corr: {diag['tx_rx_correlation']:.3f}")
        if 'num_detections' in diag:
            parts.append(f"Det: {diag['num_detections']}")
        if 'frame_number' in diag:
            parts.append(f"Frame: {diag['frame_number']}")
        self.lbl_diag.setText(" | ".join(parts))

    def update_plots(self, data):
        rdm_db, detections, targets, false_alarms, r_axis, v_axis = data
        
        # 1. Update 2D Image
        if len(v_axis) > 1 and len(r_axis) > 1:
            v0 = float(v_axis[0])
            r0 = float(r_axis[0])
            v_range = float(v_axis[-1] - v0)
            r_range = float(r_axis[-1] - r0)
            
            self.img_item.setImage(rdm_db, autoLevels=False, levels=[-80, 0])
            self.img_item.setRect(QRectF(v0, r0, v_range, r_range))
        
        # 2D Markers
        if targets:
            tx = [t['velocity'] for t in targets]
            ty = [t['range'] for t in targets]
            self.scatter_gt.setData(tx, ty)
        else:
            self.scatter_gt.clear()
            
        if detections:
            dx = [d['velocity_mps'] for d in detections]
            dy = [d['range_m'] for d in detections]
            self.scatter_det.setData(dx, dy)
        else:
            self.scatter_det.clear()
            
        if false_alarms:
            fx = [f['velocity_mps'] for f in false_alarms]
            fy = [f['range_m'] for f in false_alarms]
            self.scatter_fp.setData(fx, fy)
        else:
            self.scatter_fp.clear()

        # 3. Update 3D Surface
        # Downsample for perf
        stride = 2 
        z_data = rdm_db[::stride, ::stride]
        x_data = v_axis[::stride]
        y_data = r_axis[::stride]
        
        # Color Map Calculation (Viridis Style)
        # Normalize Z: -80 dB to 0 dB
        z_norm = (z_data + 80) / 80.0
        z_norm = np.clip(z_norm, 0, 1)
        
        # Use pyqtgraph colormap
        cmap = pg.colormap.get('viridis')
        colors = cmap.map(z_norm, mode='float') # Returns (Rows, Cols, 4)
        
        self.surface.setData(x=x_data, y=y_data, z=z_data, colors=colors)
        
        # 3D Markers (x, y, z)
        z_offset = 2.0 
        
        if targets:
            # Find Z value (approx) or use fixed? Reference uses Z val.
            # Using fixed Z at top of dynamic range or constant offset might be cleaner
            # But matching Z is better.
            pos_gt = []
            for t in targets:
                # Find nearest bin
                v_idx = int((np.abs(v_axis - t['velocity'])).argmin())
                r_idx = int((np.abs(r_axis - t['range'])).argmin())
                z_val = rdm_db[v_idx, r_idx] if (0 <= v_idx < rdm_db.shape[0] and 0 <= r_idx < rdm_db.shape[1]) else -50
                pos_gt.append([t['velocity'], t['range'], z_val + 2])
            
            self.scatter_3d_gt.setData(pos=np.array(pos_gt))
        else:
            self.scatter_3d_gt.setData(pos=None)
            
        if detections:
            pos_det = np.array([[d['velocity_mps'], d['range_m'], d['power'] + 2] for d in detections])
            self.scatter_3d_det.setData(pos=pos_det)
        else:
            self.scatter_3d_det.setData(pos=None)
            
        if false_alarms:
            pos_fp = np.array([[f['velocity_mps'], f['range_m'], f['power'] + 5] for f in false_alarms])
            self.scatter_3d_fp.setData(pos=pos_fp)
        else:
            self.scatter_3d_fp.setData(pos=None)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

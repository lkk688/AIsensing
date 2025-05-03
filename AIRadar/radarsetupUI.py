import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QTableWidget, QTableWidgetItem, QGroupBox
)
from PyQt5.QtCore import Qt

class RadarParameterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FMCW Radar Parameter Tuner")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Parameter sliders
        self.bandwidth_slider = self.create_slider(1, 500, 100, "Bandwidth (MHz)")
        self.chirp_slider = self.create_slider(1, 1000, 100, "Chirp Duration (us)")
        self.range_slider = self.create_slider(10, 1000, 100, "Max Range (m)")
        self.samplerate_slider = self.create_slider(1, 500, 50, "Sample Rate (MHz)")

        # Add sliders to layout
        layout.addWidget(self.bandwidth_slider["group"])
        layout.addWidget(self.chirp_slider["group"])
        layout.addWidget(self.range_slider["group"])
        layout.addWidget(self.samplerate_slider["group"])

        # Table for metrics
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        layout.addWidget(self.metrics_table)

        self.setLayout(layout)

        # Connect sliders to update function
        for slider in [self.bandwidth_slider, self.chirp_slider, self.range_slider, self.samplerate_slider]:
            slider["slider"].valueChanged.connect(self.update_metrics)

        self.update_metrics()

    def create_slider(self, min_val, max_val, init_val, label):
        group = QGroupBox(label)
        vbox = QVBoxLayout()
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(init_val)
        slider.setTickInterval((max_val - min_val) // 10)
        slider.setTickPosition(QSlider.TicksBelow)
        value_label = QLabel(f"{init_val}")
        slider.valueChanged.connect(lambda val: value_label.setText(str(val)))
        vbox.addWidget(slider)
        vbox.addWidget(value_label)
        group.setLayout(vbox)
        return {"group": group, "slider": slider, "label": value_label}

    def update_metrics(self):
        # Get parameter values
        bandwidth = self.bandwidth_slider["slider"].value() * 1e6  # Hz
        chirp_duration = self.chirp_slider["slider"].value() * 1e-6  # seconds
        max_range = self.range_slider["slider"].value()  # meters
        sample_rate = self.samplerate_slider["slider"].value() * 1e6  # Hz
        c = 3e8

        # Calculate metrics
        slope = bandwidth / chirp_duration
        range_resolution = c / (2 * bandwidth)
        f_beat_max = 2 * slope * max_range / c
        nyquist = sample_rate / 2
        freq_wrap = "No" if f_beat_max <= nyquist else "Yes"

        metrics = [
            ("Bandwidth", f"{bandwidth/1e6:.2f} MHz"),
            ("Chirp Duration", f"{chirp_duration*1e6:.2f} us"),
            ("Max Range", f"{max_range:.2f} m"),
            ("Sample Rate", f"{sample_rate/1e6:.2f} MHz"),
            ("Chirp Slope", f"{slope/1e12:.2f} THz/s"),
            ("Range Resolution", f"{range_resolution:.2f} m"),
            ("Max Beat Frequency", f"{f_beat_max/1e6:.2f} MHz"),
            ("Nyquist Frequency", f"{nyquist/1e6:.2f} MHz"),
            ("Frequency Wraparound", freq_wrap),
        ]

        self.metrics_table.setRowCount(len(metrics))
        for i, (name, value) in enumerate(metrics):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(name))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(value))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = RadarParameterGUI()
    gui.show()
    sys.exit(app.exec_())

#pip3 install PyQt5
#python3 radar_gui.py
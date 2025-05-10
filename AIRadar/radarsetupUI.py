import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QTableWidget, QTableWidgetItem, QGroupBox,
    QPushButton, QTextEdit
)
from PyQt5.QtCore import Qt

PARAMETER_EXPLANATIONS = {
    "Bandwidth": {
        "text": "Bandwidth determines the frequency span of the radar signal. Increasing bandwidth improves range resolution.",
        "latex": r"Range\ Resolution = \frac{c}{2B}",
        "contributes": ["Range Resolution"]
    },
    "Chirp Duration": {
        "text": "Chirp duration is the time for one frequency sweep. Longer chirps can improve SNR but reduce maximum beat frequency.",
        "latex": r"Slope = \frac{B}{T_{chirp}}",
        "contributes": ["Chirp Slope", "Max Beat Frequency"]
    },
    "Max Range": {
        "text": "Maximum range is the farthest distance the radar can detect, determined by chirp duration and speed of light.",
        "latex": r"T_{chirp} = \frac{2 \cdot R_{max}}{c}",
        "contributes": ["Chirp Duration"]
    },
    "Sample Rate": {
        "text": "Sample rate is how fast the ADC samples the signal. It must be at least twice the maximum beat frequency (Nyquist).",
        "latex": r"f_{Nyquist} = \frac{f_{s}}{2}",
        "contributes": ["Nyquist Frequency", "Frequency Wraparound"]
    },
    "Chirp Slope": {
        "text": "Chirp slope is the rate of frequency change during a chirp. It affects the beat frequency for a given range.",
        "latex": r"Slope = \frac{B}{T_{chirp}}",
        "contributes": ["Max Beat Frequency"]
    },
    "Range Resolution": {
        "text": "Range resolution is the minimum distance between two distinguishable targets. It improves with higher bandwidth.",
        "latex": r"Range\ Resolution = \frac{c}{2B}",
        "contributes": ["Bandwidth"]
    },
    "Max Beat Frequency": {
        "text": "Maximum beat frequency is the highest frequency difference between TX and RX signals for the farthest target.",
        "latex": r"f_{beat,max} = \frac{2 \cdot Slope \cdot R_{max}}{c}",
        "contributes": ["Chirp Slope", "Max Range"]
    },
    "Nyquist Frequency": {
        "text": "Nyquist frequency is half the sample rate. Beat frequencies above this will alias and cause errors.",
        "latex": r"f_{Nyquist} = \frac{f_{s}}{2}",
        "contributes": ["Sample Rate"]
    },
    "Frequency Wraparound": {
        "text": "Frequency wraparound (aliasing) occurs if the max beat frequency exceeds the Nyquist frequency. Increase sample rate or reduce max range/bandwidth to avoid.",
        "latex": r"f_{beat,max} \leq f_{Nyquist}",
        "contributes": ["Sample Rate", "Max Beat Frequency"]
    }
}

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
        self.metrics_table = QTableWidget(0, 3)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value", "Explain"])
        layout.addWidget(self.metrics_table)

        # Explanation area
        self.explanation_box = QTextEdit()
        self.explanation_box.setReadOnly(True)
        layout.addWidget(self.explanation_box)

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
            btn = QPushButton("Explain")
            btn.clicked.connect(lambda _, n=name: self.show_explanation(n))
            self.metrics_table.setCellWidget(i, 2, btn)

    def show_explanation(self, metric_name):
        info = PARAMETER_EXPLANATIONS.get(metric_name, None)
        if info:
            text = f"<b>{metric_name}</b><br>{info['text']}<br><br>"
            text += f"<b>LaTeX:</b> <br><span style='font-family:monospace;'>{info['latex']}</span><br><br>"
            text += "<b>Contributing Parameters:</b><ul>"
            for p in info.get("contributes", []):
                text += f"<li>{p}</li>"
            text += "</ul>"
            self.explanation_box.setHtml(text)
        else:
            self.explanation_box.setText("No explanation available.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = RadarParameterGUI()
    gui.show()
    sys.exit(app.exec_())

#pip3 install PyQt5
#python3 radar_gui.py
import sys
import numpy as np
Runtime="QT5"
if Runtime=="Side6": #"QT5" Side6 and QT5 works in Mac
   from PyQt5.QtWidgets import QApplication, QWidget, QLabel
   #from PyQt5.QtGui import QIcon
   #from PyQt5.QtCore import pyqtSlot
elif Runtime=="QT5":
   #from PyQt5.QtWidgets import QApplication, QWidget, QLabel
   from PyQt5.QtCore import Qt
   from PyQt5.QtWidgets import *
elif Runtime=="QT6":
   #from PyQt6.QtWidgets import QApplication, QWidget, QLabel
   from PyQt6.QtCore import Qt
   from PyQt6.QtWidgets import *
elif Runtime=="Side6":
   from PySide6 import QtCore, QtGui, QtWidgets
   from PySide6.QtWidgets import QApplication, QWidget, QLabel

#from PyQt5.QtCore import Qt
#from PyQt5.QtWidgets import *
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

c = 3e8
default_chirp_bw = 500e6
signal_freq = 100e3
plot_freq = 100e3    # x-axis freq range to plot
sample_rate = 0.6e6 #2e6
fft_size = 1024 * 8
N = 2048#int(my_sdr.rx_buffer_size)
num_slices = 50     # this sets how much time will be displayed on the waterfall plot
N_frame = fft_size
freq = np.linspace(-sample_rate / 2, sample_rate / 2, int(N_frame))

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MyRadar Test")
        self.setGeometry(0, 0, 400, 400)  # (x,y, width, height)
        #self.setFixedWidth(600)
        self.setWindowState(QtCore.Qt.WindowMaximized)
        self.num_rows = 12
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False) #remove the window's close button
        self.UiComponents()
        self.show()

    # method for components
    def UiComponents(self):
        widget = QWidget()

        global layout, signal_freq, plot_freq
        layout = QGridLayout()

        # Control Panel
        control_label = QLabel("Radar Targeting")
        font = control_label.font()
        font.setPointSize(24)
        control_label.setFont(font)
        font.setPointSize(12)
        control_label.setAlignment(Qt.AlignHCenter)  # | Qt.AlignVCenter)
        layout.addWidget(control_label, 0, 0, 1, 2)

        # Check boxes
        self.thresh_check = QCheckBox("Plot CFAR Threshold")
        font = self.thresh_check.font()
        font.setPointSize(10)
        self.thresh_check.setFont(font)
        self.thresh_check.stateChanged.connect(self.change_thresh)
        layout.addWidget(self.thresh_check, 2, 0)
        
        self.cfar_check = QCheckBox("Apply CFAR Threshold")
        font = self.cfar_check.font()
        self.cfar_check.setFont(font)
        self.cfar_check.stateChanged.connect(self.change_cfar)
        layout.addWidget(self.cfar_check, 2, 1)

        # Chirp bandwidth slider
        self.bw_slider = QSlider(Qt.Horizontal)
        self.bw_slider.setMinimum(100)
        self.bw_slider.setMaximum(500)
        self.bw_slider.setValue(int(default_chirp_bw / 1e6))
        self.bw_slider.setTickInterval(50)
        self.bw_slider.setMaximumWidth(200)
        self.bw_slider.setTickPosition(QSlider.TicksBelow)
        self.bw_slider.valueChanged.connect(self.get_range_res)
        layout.addWidget(self.bw_slider, 4, 0)

        self.set_bw = QPushButton("Set Chirp Bandwidth")
        self.set_bw.setMaximumWidth(200)
        self.set_bw.pressed.connect(self.set_range_res)
        layout.addWidget(self.set_bw, 5, 0, 1, 1)
        
        self.quit_button = QPushButton("Quit")
        self.quit_button.pressed.connect(self.end_program)
        layout.addWidget(self.quit_button, 30, 0, 4, 4)
        
        #CFAR Sliders
        self.cfar_bias = QSlider(Qt.Horizontal)
        self.cfar_bias.setMinimum(0)
        self.cfar_bias.setMaximum(100)
        self.cfar_bias.setValue(40) #25
        self.cfar_bias.setTickInterval(5)
        self.cfar_bias.setMaximumWidth(200)
        self.cfar_bias.setTickPosition(QSlider.TicksBelow)
        self.cfar_bias.valueChanged.connect(self.get_cfar_values)
        layout.addWidget(self.cfar_bias, 8, 0)
        self.cfar_bias_label = QLabel("CFAR Bias (dB): %0.0f" % (self.cfar_bias.value()))
        self.cfar_bias_label.setFont(font)
        self.cfar_bias_label.setAlignment(Qt.AlignLeft)
        self.cfar_bias_label.setMinimumWidth(100)
        self.cfar_bias_label.setMaximumWidth(200)
        layout.addWidget(self.cfar_bias_label, 8, 1)
        
        self.cfar_guard = QSlider(Qt.Horizontal)
        self.cfar_guard.setMinimum(1)
        self.cfar_guard.setMaximum(40)
        self.cfar_guard.setValue(27) #15
        self.cfar_guard.setTickInterval(4)
        self.cfar_guard.setMaximumWidth(200)
        self.cfar_guard.setTickPosition(QSlider.TicksBelow)
        self.cfar_guard.valueChanged.connect(self.get_cfar_values)
        layout.addWidget(self.cfar_guard, 10, 0)
        self.cfar_guard_label = QLabel("Num Guard Cells: %0.0f" % (self.cfar_guard.value()))
        self.cfar_guard_label.setFont(font)
        self.cfar_guard_label.setAlignment(Qt.AlignLeft)
        self.cfar_guard_label.setMinimumWidth(100)
        self.cfar_guard_label.setMaximumWidth(200)
        layout.addWidget(self.cfar_guard_label, 10, 1)
        
        self.cfar_ref = QSlider(Qt.Horizontal)
        self.cfar_ref.setMinimum(1)
        self.cfar_ref.setMaximum(100)
        self.cfar_ref.setValue(16)
        self.cfar_ref.setTickInterval(10)
        self.cfar_ref.setMaximumWidth(200)
        self.cfar_ref.setTickPosition(QSlider.TicksBelow)
        self.cfar_ref.valueChanged.connect(self.get_cfar_values)
        layout.addWidget(self.cfar_ref, 12, 0)
        self.cfar_ref_label = QLabel("Num Ref Cells: %0.0f" % (self.cfar_ref.value()))
        self.cfar_ref_label.setFont(font)
        self.cfar_ref_label.setAlignment(Qt.AlignLeft)
        self.cfar_ref_label.setMinimumWidth(100)
        self.cfar_ref_label.setMaximumWidth(200)
        layout.addWidget(self.cfar_ref_label, 12, 1)


        # waterfall level slider
        self.low_slider = QSlider(Qt.Horizontal)
        self.low_slider.setMinimum(-100)
        self.low_slider.setMaximum(0)
        self.low_slider.setValue(-100)
        self.low_slider.setTickInterval(20)
        self.low_slider.setMaximumWidth(200)
        self.low_slider.setTickPosition(QSlider.TicksBelow)
        self.low_slider.valueChanged.connect(self.get_water_levels)
        layout.addWidget(self.low_slider, 16, 0)

        self.high_slider = QSlider(Qt.Horizontal)
        self.high_slider.setMinimum(-100)
        self.high_slider.setMaximum(0)
        self.high_slider.setValue(0)
        self.high_slider.setTickInterval(20)#5
        self.high_slider.setMaximumWidth(200)
        self.high_slider.setTickPosition(QSlider.TicksBelow)
        self.high_slider.valueChanged.connect(self.get_water_levels)
        layout.addWidget(self.high_slider, 18, 0)

        self.water_label = QLabel("Waterfall Intensity Levels")
        self.water_label.setFont(font)
        self.water_label.setAlignment(Qt.AlignCenter)
        self.water_label.setMinimumWidth(100)
        self.water_label.setMaximumWidth(200)
        layout.addWidget(self.water_label, 15, 0,1,1)
        self.low_label = QLabel("LOW LEVEL: %0.0f" % (self.low_slider.value()))
        self.low_label.setFont(font)
        self.low_label.setAlignment(Qt.AlignLeft)
        self.low_label.setMinimumWidth(100)
        self.low_label.setMaximumWidth(200)
        layout.addWidget(self.low_label, 16, 1)
        self.high_label = QLabel("HIGH LEVEL: %0.0f" % (self.high_slider.value()))
        self.high_label.setFont(font)
        self.high_label.setAlignment(Qt.AlignLeft)
        self.high_label.setMinimumWidth(100)
        self.high_label.setMaximumWidth(200)
        layout.addWidget(self.high_label, 18, 1)

        self.steer_slider = QSlider(Qt.Horizontal)
        self.steer_slider.setMinimum(-80)
        self.steer_slider.setMaximum(80)
        self.steer_slider.setValue(0)
        self.steer_slider.setTickInterval(20)
        self.steer_slider.setMaximumWidth(200)
        self.steer_slider.setTickPosition(QSlider.TicksBelow)
        self.steer_slider.valueChanged.connect(self.get_steer_angle)
        layout.addWidget(self.steer_slider, 22, 0)
        self.steer_title = QLabel("Receive Steering Angle")
        self.steer_title.setFont(font)
        self.steer_title.setAlignment(Qt.AlignCenter)
        self.steer_title.setMinimumWidth(100)
        self.steer_title.setMaximumWidth(200)
        layout.addWidget(self.steer_title, 21, 0)
        self.steer_label = QLabel("%0.0f DEG" % (self.steer_slider.value()))
        self.steer_label.setFont(font)
        self.steer_label.setAlignment(Qt.AlignLeft)
        self.steer_label.setMinimumWidth(100)
        self.steer_label.setMaximumWidth(200)
        layout.addWidget(self.steer_label, 22, 1,1,2)

        # FFT plot
        self.fft_plot = pg.plot()
        self.fft_plot.setMinimumWidth(600)
        self.fft_curve = self.fft_plot.plot(freq, pen={'color':'y', 'width':2})
        self.fft_threshold = self.fft_plot.plot(freq, pen={'color':'r', 'width':2})
        title_style = {"size": "20pt"}
        label_style = {"color": "#FFF", "font-size": "14pt"}
        self.fft_plot.setLabel("bottom", text="Frequency", units="Hz", **label_style)
        self.fft_plot.setLabel("left", text="Magnitude", units="dB", **label_style)
        self.fft_plot.setTitle("Received Signal - Frequency Spectrum", **title_style)
        layout.addWidget(self.fft_plot, 0, 2, self.num_rows, 1)
        self.fft_plot.setYRange(-60, 0)
        self.fft_plot.setXRange(signal_freq, signal_freq+plot_freq)

        # Waterfall plot
        self.waterfall = pg.PlotWidget()
        self.imageitem = pg.ImageItem()
        self.waterfall.addItem(self.imageitem)
        # Use a viridis colormap
        pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        color = np.array([[68, 1, 84,255], [59, 82, 139,255], [33, 145, 140,255], [94, 201, 98,255], [253, 231, 37,255]], dtype=np.ubyte)
        lut = pg.ColorMap(pos, color).getLookupTable(0.0, 1.0, 256)
        self.imageitem.setLookupTable(lut)
        self.imageitem.setLevels([0,1])
        # self.imageitem.scale(0.35, sample_rate / (N))  # this is deprecated -- we have to use setTransform instead
        tr = QtGui.QTransform()
        tr.translate(0,-sample_rate/2)
        tr.scale(0.35, sample_rate / (N))
        #tr.scale(0.35, sample_rate / fft_size)
        self.imageitem.setTransform(tr)
        zoom_freq = 35e3
        self.waterfall.setRange(yRange=(signal_freq, signal_freq + zoom_freq))
        self.waterfall.setTitle("Waterfall Spectrum", **title_style)
        self.waterfall.setLabel("left", "Frequency", units="Hz", **label_style)
        self.waterfall.setLabel("bottom", "Time", units="sec", **label_style)
        layout.addWidget(self.waterfall, 0 + self.num_rows + 1, 2, self.num_rows, 1)
        self.img_array = np.ones((num_slices, fft_size))*(-100)

        widget.setLayout(layout)
        # setting this widget as central widget of the main window
        self.setCentralWidget(widget)

    def get_range_res(self):
        """ Updates the slider bar label with Chirp bandwidth and range resolution
		Returns:
			None
		"""
        bw = self.bw_slider.value() * 1e6
        range_res = c / (2 * bw)

    def get_cfar_values(self):
        """ Updates the cfar values
		Returns:
			None
		"""
        self.cfar_bias_label.setText("CFAR Bias (dB): %0.0f" % (self.cfar_bias.value()))
        self.cfar_guard_label.setText("Num Guard Cells: %0.0f" % (self.cfar_guard.value()))
        self.cfar_ref_label.setText("Num Ref Cells: %0.0f" % (self.cfar_ref.value()))


    def get_water_levels(self):
        """ Updates the waterfall intensity levels
		Returns:
			None
		"""
        if self.low_slider.value() > self.high_slider.value():
            self.low_slider.setValue(self.high_slider.value())
        self.low_label.setText("LOW LEVEL: %0.0f" % (self.low_slider.value()))
        self.high_label.setText("HIGH LEVEL: %0.0f" % (self.high_slider.value()))

    def get_steer_angle(self):
        """ Updates the steering angle readout
		Returns:
			None
		"""
        self.steer_label.setText("%0.0f DEG" % (self.steer_slider.value()))
        phase_delta = (
            2
            * 3.14159
            * 10.25e9
            * 0.014
            * np.sin(np.radians(self.steer_slider.value()))
            / (3e8)
        )
        # phase_delta = (2 * 3.14159 * output_freq * my_phaser.element_spacing
        #     * np.sin(np.radians(self.steer_slider.value()))
        #     / (3e8)
        # )
        #my_phaser.set_beam_phase_diff(np.degrees(phase_delta))

    def set_range_res(self):
        """ Sets the Chirp bandwidth
		Returns:
			None
		"""
        global dist, slope, signal_freq, plot_freq
        bw = self.bw_slider.value() * 1e6
      #   slope = bw / ramp_time_s
      #   dist = (freq - signal_freq) * c / (2 * slope)
      #   my_phaser.freq_dev_range = int(bw / 4)  # frequency deviation range in Hz
      #   my_phaser.enable = 0

    def end_program(self):
        """ Gracefully shutsdown the program and Radio
		Returns:
			None
		"""
        #close device
        self.close()

    def change_thresh(self, state):
        """ Toggles between showing cfar threshold values
		Args:
			state (QtCore.Qt.Checked) : State of check box
		Returns:
			None
		"""
        global plot_threshold
        plot_state = win.fft_plot.getViewBox().state
        if state == QtCore.Qt.Checked:
            plot_threshold = True
        else:
            plot_threshold = False

    def change_cfar(self, state):
        """ Toggles between enabling/disabling CFAR
		Args:
			state (QtCore.Qt.Checked) : State of check box
		Returns:
			None
		"""
        global cfar_toggle
        if state == QtCore.Qt.Checked:
            cfar_toggle = True
        else:
            cfar_toggle = False



# create pyqt5 app
App = QApplication(sys.argv)

# create the instance of our Window
win = Window()
index = 0

def update():
   """ Updates the FFT in the window
	Returns:
		None
	"""
   print('Do updates')


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

# start the app
sys.exit(App.exec())
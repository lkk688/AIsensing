# importing the required libraries 
from PyQt5.QtGui import * 
#from PyQt5.QtWidgets import * 
import sys 
from PyQt5.QtCore import Qt
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import *
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import numpy as np
from scipy import ndimage

#fix the error of `np.float` was a deprecated alias for the builtin `float`
np.float = float    
np.int = int   #module 'numpy' has no attribute 'int'
np.object = object    #module 'numpy' has no attribute 'object'
np.bool = bool    #module 'numpy' has no attribute 'bool'


def loadData(N_frame):
    with open('./data/radardata5s-1101fast3move.npy', 'rb') as f:
        alldata = np.load(f)
    print(len(alldata))
    totallen=len(alldata)
    Ntotalframe=int(totallen/N_frame)-1
    return alldata, Ntotalframe
    # for i in range(Ntotalframe):
    #     data = alldata[i*N_frame:(i+1)*N_frame]

def showspectrum(data):
    # fc = int(100e3 / (fs / N_frame)) * (fs / N_frame) #300KHz
    # ts =1.0/fs
    # t = np.arange(0, N_frame * ts, ts)
    # i = np.cos(2 * np.pi * t * fc) * 2 ** 14
    # q = np.sin(2 * np.pi * t * fc) * 2 ** 14
    # iq_100k = 1 * (i + 1j * q)
    #data=data[0:N_frame] #* iq_100k
    win_funct = np.blackman(len(data))
    y = data * win_funct
    sp = np.absolute(np.fft.fft(y))
    sp = np.fft.fftshift(sp)
    s_mag = np.abs(sp) / np.sum(win_funct)
    s_mag = np.maximum(s_mag, 10 ** (-15))
    s_dbfs = 20 * np.log10(s_mag / (2 ** 11))

    """there's a scaling issue on the y-axis of the waterfallcthe data is off by 300kHz.  To fix, I'm just shifting the freq"""
    fc = int(300e3 / (fs / N_frame)) * (fs / N_frame) #300KHz
    ts =1.0/fs
    t = np.arange(0, N_frame * ts, ts)
    i = np.cos(2 * np.pi * t * fc) * 2 ** 14
    q = np.sin(2 * np.pi * t * fc) * 2 ** 14
    iq_300k = 1 * (i + 1j * q)
    data_shift = data * iq_300k
    y = data_shift * win_funct
    sp = np.absolute(np.fft.fft(y))
    sp = np.fft.fftshift(sp)
    s_mag = np.abs(sp) / np.sum(win_funct)
    s_mag = np.maximum(s_mag, 10 ** (-15))
    s_dbfs_shift = 20 * np.log10(s_mag / (2 ** 11))
    return s_dbfs, s_dbfs_shift


def rangedoppler(data, n_s=600):
    #n_s = 600
    n_r = int(len(data)/n_s)-1
    table = np.zeros((n_r, n_s)) #150 chirps,1000 samples/chirp
    for chirp_nr in range(n_r):
        table[chirp_nr, :] = data[(chirp_nr*n_s):(n_s*(chirp_nr+1))]
    #fft_output = np.fft.fft2(table)
    #2D FFT and Velocity-Distance Relationship
    Z_fft2 = abs(np.fft.fft2(table)) #
    #Data_fft2 = Z_fft2[0:int(n_r/2),0:int(n_s/2)] #get half
    Data_fft2 = Z_fft2[0:int(n_r/2),0:int(n_s/2)]#70:120] #get half 0:int(n_s/2)
    return Data_fft2, table

c = 3e8
default_rf_bw = 500e6
sample_rate = 0.6e6 #0.6M
fs = int(sample_rate) #0.6MHz
fft_size = 1024 * 16
N_frame = fft_size
freq = np.linspace(-fs / 2, fs / 2, int(N_frame))
N = int(fft_size)
signal_freq = 100e3 #100K
fc = int(signal_freq / (fs / N)) * (fs / N) #100KHz
ts = 1 / float(fs)
t = np.arange(0, N * ts, ts)

BW = 500e6
ramp_time = 1e3  # us
ramp_time_s = ramp_time / 1e6
slope = BW / ramp_time_s
Nr = int(ramp_time_s * fs) #Number ADC sampling points in each chirp


alldata, Ntotalframe = loadData(N_frame)



class Window(QMainWindow): 
    def __init__(self): 
        super().__init__() 
  
        # set the title 
        self.setWindowTitle("RadarSensing") 
  
        # setting  the geometry of window 
        # setGeometry(left, top, width, height) 
        #self.setGeometry(100, 60, 1000, 800) 
        self.setGeometry(100, 100, 1800, 1200) #x-coordinate, y-coordinate, width of window, height of window
  
        # creating a label widget 
        #self.widget = QLabel('Hello', self) 
        self.num_rows = 12
        self.UiComponents()
        
        self.currentindex = 0
        self.create_timer()
  
        # show all the widgets 
        self.show() 
    
    # method for components
    def UiComponents(self):
        widget = QWidget()

        global layout
        layout = QGridLayout()

        # Control Panel
        control_label = QLabel("Kaikai Liu @ San Jose State University")
        font = control_label.font()
        font.setPointSize(15)
        control_label.setFont(font)
        control_label.setAlignment(Qt.AlignHCenter)  # | Qt.AlignVCenter)
        layout.addWidget(control_label, 0, 0, 1, 2) #row, col, rowspan=1, colspan=2

        # Check boxes
        self.x_axis_check = QCheckBox("Toggle Range/Frequency x-axis")
        font = self.x_axis_check.font()
        font.setPointSize(12)
        self.x_axis_check.setFont(font)

        self.x_axis_check.stateChanged.connect(self.change_x_axis)
        layout.addWidget(self.x_axis_check, 1, 0) #row=2, col=0

        # Range resolution
        # Changes with the RF BW slider
        self.range_res_label = QLabel(
            "B<sub>RF</sub>: %0.2f MHz - R<sub>res</sub>: %0.2f m"
            % (default_rf_bw / 1e6, c / (2 * default_rf_bw))
        )
        font = self.range_res_label.font()
        font.setPointSize(12)
        self.range_res_label.setFont(font)
        self.range_res_label.setAlignment(Qt.AlignCenter) #Qt.AlignRight
        self.range_res_label.setMinimumWidth(500)
        layout.addWidget(self.range_res_label, 2, 0, 1, 2)

        # RF bandwidth slider
        self.bw_slider = QSlider(Qt.Horizontal)
        self.bw_slider.setMinimum(100)
        self.bw_slider.setMaximum(500)
        self.bw_slider.setValue(int(default_rf_bw / 1e6))
        self.bw_slider.setTickInterval(50)
        self.bw_slider.setTickPosition(QSlider.TicksBelow)
        self.bw_slider.valueChanged.connect(self.get_range_res)
        layout.addWidget(self.bw_slider, 3, 0, 1, 2)

        self.set_bw = QPushButton("Set RF Bandwidth")
        #self.set_bw.pressed.connect(self.set_range_res)
        layout.addWidget(self.set_bw, 4, 0, 1, 2)

        #waterfall title
        self.water_label = QLabel("Waterfall Intensity Levels")
        self.water_label.setFont(font)
        self.water_label.setAlignment(Qt.AlignCenter)
        self.water_label.setMinimumWidth(300)
        layout.addWidget(self.water_label, 5, 0, 1, 2)

        # waterfall level slider
        self.low_slider = QSlider(Qt.Horizontal)
        self.low_slider.setMinimum(-100)
        self.low_slider.setMaximum(100)
        self.low_slider.setValue(-20) #20
        self.low_slider.setTickInterval(20)
        self.low_slider.setTickPosition(QSlider.TicksBelow)
        #self.low_slider.valueChanged.connect(self.get_water_levels)
        layout.addWidget(self.low_slider, 6, 0)
        self.low_label = QLabel("LOW LEVEL: %0.0f" % (self.low_slider.value()))
        self.low_label.setFont(font)
        self.low_label.setAlignment(Qt.AlignLeft)
        self.low_label.setMinimumWidth(300)
        layout.addWidget(self.low_label, 6, 1)

        self.high_slider = QSlider(Qt.Horizontal)
        self.high_slider.setMinimum(-100)
        self.high_slider.setMaximum(100)
        self.high_slider.setValue(30) #60
        self.high_slider.setTickInterval(20)
        self.high_slider.setTickPosition(QSlider.TicksBelow)
        #self.high_slider.valueChanged.connect(self.get_water_levels)
        layout.addWidget(self.high_slider, 7, 0)
        self.high_label = QLabel("HIGH LEVEL: %0.0f" % (self.high_slider.value()))
        self.high_label.setFont(font)
        self.high_label.setAlignment(Qt.AlignLeft)
        self.high_label.setMinimumWidth(300)
        layout.addWidget(self.high_label, 7, 1)

        #Steering angle
        self.steer_title = QLabel("Receive Steering Angle")
        self.steer_title.setFont(font)
        self.steer_title.setAlignment(Qt.AlignCenter)
        self.steer_title.setMinimumWidth(300)
        layout.addWidget(self.steer_title, 8, 0, 1, 2)

        self.steer_slider = QSlider(Qt.Horizontal)
        self.steer_slider.setMinimum(-80)
        self.steer_slider.setMaximum(80)
        self.steer_slider.setValue(0)
        self.steer_slider.setTickInterval(20)
        self.steer_slider.setTickPosition(QSlider.TicksBelow)
        #self.steer_slider.valueChanged.connect(self.get_steer_angle)
        layout.addWidget(self.steer_slider, 9, 0)
        self.steer_label = QLabel("%0.0f DEG" % (self.steer_slider.value()))
        self.steer_label.setFont(font)
        self.steer_label.setAlignment(Qt.AlignLeft)
        self.steer_label.setMinimumWidth(300)
        layout.addWidget(self.steer_label, 9, 1)

        #add polar plot
        self.polarplot = pg.plot()
        self.polarplot.setBackground("w")
        self.polarplot.setAspectLocked()
        #self.polarplot.setMinimumWidth(600)
        # Add polar grid lines
        self.polarplot.addLine(x=0, pen=0.2)
        self.polarplot.addLine(y=0, pen=0.2)
        for r in range(2, 20, 2):
            circle = pg.QtGui.QGraphicsEllipseItem(-r, -r, r * 2, r * 2)
            circle.setPen(pg.mkPen(0.2))
            self.polarplot.addItem(circle)
        # rectangle = pg.Qt.QtCore.QRectF(10.0, 20.0, 80.0, 60.0)
        # startAngle = 30 * 16
        # spanAngle = 120 * 16
        # painter = QPainter(self)
        # #The startAngle and spanAngle must be specified in 1/16th of a degree, i.e. a full circle equals 5760 (16 * 360).
        # painter.drawPie(rectangle, startAngle, spanAngle)
        # self.polarplot.addItem(painter)
        # make polar data
        theta = np.linspace(0, 2 * np.pi, 100)
        radius = np.random.normal(loc=10, size=100)
        # Transform to cartesian and plot
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        self.polarplot.plot(x, y)
        layout.addWidget(self.polarplot, 11, 0, 8,2)

        title_style = {"size": "12pt"} #{"color": "#FFF", "size": "14pt"}
        label_style = {"font-size": "10pt"} #{"color": "#FFF", "font-size": "10pt"}

        # time plot
        self.plot_graph = pg.PlotWidget()
        self.setCentralWidget(self.plot_graph)
        self.plot_graph.setBackground("w")
        pen = pg.mkPen(color=(255, 0, 0))
        self.line = self.plot_graph.plot(t, pen=pen)
        self.plot_graph.setLabel("bottom", text="Time", units="s", **label_style)
        self.plot_graph.setLabel("left", text="Amplitude", units="adc", **label_style)
        self.plot_graph.setTitle("Received Signal Time Series", **title_style)
        layout.addWidget(self.plot_graph, 0, 2, 10, 1) #row=2, col=2m rowspan=4, colspan=1

        # FFT plot
        self.fft_plot = pg.plot()
        self.fft_plot.setMinimumWidth(1200)
        self.fft_plot.setBackground("w") #'k' is black https://www.pythonguis.com/tutorials/plotting-pyqtgraph/
        #self.fft_curve = self.fft_plot.plot(freq, pen="y", width=6)
        self.fft_curve = self.fft_plot.plot(freq, pen=pg.mkPen('b'), width=6)
        self.fft_plot.setLabel("bottom", text="Frequency", units="Hz", **label_style)
        self.fft_plot.setLabel("left", text="Magnitude", units="dB", **label_style)
        self.fft_plot.setTitle("Received Signal Spectrum", **title_style)
        layout.addWidget(self.fft_plot, 10, 2, 10, 1) #10, 2, 10, 1 row=2, col=2m rowspan=4, colspan=1
        self.fft_plot.setYRange(-80, 0)
        self.fft_plot.setXRange(100e3, 200e3)

        # Waterfall plot
        self.num_slices = 200
        self.img_array = np.zeros((self.num_slices, fft_size))
        self.waterfall = pg.PlotWidget()
        self.waterfall.setBackground("w")
        self.waterfall.setMinimumWidth(1200)
        self.imageitem = pg.ImageItem() #(image=self.img_array)
        self.waterfall.addItem(self.imageitem)
        # self.imageitem.scale(0.35, sample_rate / (N))  # this is deprecated -- we have to use setTransform instead
        tr = QtGui.QTransform()
        tr.scale(0.35, sample_rate / (N)) ## scale horizontal and vertical axes
        self.imageitem.setTransform(tr)
        zoom_freq = 40e3
        self.waterfall.setRange(yRange=(100e3, 100e3 + zoom_freq))#starting from 100K
        self.waterfall.setTitle("Waterfall Spectrum", **title_style)
        self.waterfall.setLabel("left", "Frequency", units="Hz")
        self.waterfall.setLabel("bottom", "Time", units="sec")
        layout.addWidget(self.waterfall, 0, 3, 10, 1)
        
        # creating image view  object
        self.imv = pg.ImageView()
        #self.imv.setTitle("Ranger Doppler", **title_style)
        #self.imv.setBackground("w")
        self.imv.setMinimumWidth(1200)
        # Create random 3D data set with noisy signals
        # img = pg.gaussianFilter(np.random.normal(
        #     size=(200, 200)), (5, 5)) * 20 + 100
        # setting new axis to image
        img = np.random.normal(size=(60, 60))#[np.newaxis, :, :] #(1, 200, 200)
        # Displaying the data and assign each frame a time value from 1.0 to 3.0
        self.imv.setImage(img)
        layout.addWidget(self.imv, 10, 3, 10, 1)


        widget.setLayout(layout)
        # setting this widget as central widget of the main window
        self.setCentralWidget(widget)
    
    def create_timer(self, timestep=300):  #300 milliseconds
        # Add a timer to simulate new temperature measurements
        self.timer = QtCore.QTimer()
        self.timer.setInterval(timestep)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
    
    def update_plot(self):
        #self.fft_curve
        print("update plot percentage: ", self.currentindex/Ntotalframe)
        #alldata, Ntotalframe
        if self.currentindex>=Ntotalframe:
            print("Finished")
            self.currentindex=0
        currentdata = alldata[self.currentindex*N_frame:(self.currentindex+1)*N_frame]
        self.line.setData(t, currentdata.real)

        sepcf, s_dbfs_shift = showspectrum(currentdata)
        

        self.fft_curve.setData(freq, sepcf)

        #Roll array elements along a given axis. Elements that roll beyond the last position are re-introduced at the first.
        self.img_array = np.roll(self.img_array, 1, axis=0)
        self.img_array[1] = s_dbfs_shift
        self.imageitem.setLevels([self.low_slider.value(), self.high_slider.value()])
        self.imageitem.setImage(self.img_array, autoLevels=False)
        #self.imageitem.setImage(self.img_array, autoLevels=True)

        
        # rddata=rddata/max(rddata)
        # print(rddata)
        #endframeid=(self.currentindex*5+1)*N_frame
        Ntimes=5
        if self.currentindex>Ntimes:
            currentdata2=alldata[max(0, (self.currentindex-Ntimes)*N_frame):(self.currentindex+1)*N_frame]
            rddata, table=rangedoppler(currentdata2, n_s=Nr) #81 300
            resampled_rd=ndimage.zoom(rddata, zoom=(20,5))
            self.imv.setImage(resampled_rd)

        self.currentindex = self.currentindex + 1

  
    def change_x_axis(self, state):
        """ Toggles between showing frequency and range for the x-axis
		Args:
			state (QtCore.Qt.Checked) : State of check box
		Returns:
			None
		"""
        global plot_dist, slope
        #plot_state = win.fft_plot.getViewBox().state
        if state == QtCore.Qt.Checked:
            print("Range axis")
            # plot_dist = True
            # range_x = (100e3) * c / (4 * slope)
            # self.fft_plot.setXRange(0, range_x)
        else:
            print("Frequency axis")
            # plot_dist = False
            # self.fft_plot.setXRange(100e3, 200e3)
    
    def get_range_res(self):
        """ Updates the slider bar label with RF bandwidth and range resolution
		Returns:
			None
		"""
        bw = self.bw_slider.value() * 1e6
        range_res = c / (2 * bw)
        self.range_res_label.setText(
            "B<sub>RF</sub>: %0.2f MHz - R<sub>res</sub>: %0.2f m"
            % (bw / 1e6, c / (2 * bw))
        )

# create pyqt5 app 
App = QApplication(sys.argv) 
  
# create the instance of our Window 
window = Window() 

# timer = QtCore.QTimer()
# timer.timeout.connect(update)
# timer.start(0) #A QTimer with a timeout interval of 0 will time out as soon as all the events in the window system's event queue have been processed.

# start the app 
sys.exit(App.exec()) 
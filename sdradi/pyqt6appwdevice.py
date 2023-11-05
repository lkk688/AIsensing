# importing the required libraries 
import PyQt6
from PyQt6 import QtWidgets
# importing pyqtgraph as pg
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
#from PyQt6.QtGui import *
#from PyQt6.QtCore import *

from PyQt6.QtWidgets import (
    QMainWindow, QApplication,
    QLabel, QToolBar, QStatusBar, QSlider,
    QCheckBox, QGridLayout, QLineEdit, QPushButton, QWidget
)
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtCore import QSize, Qt

# importing system
import sys
 
# importing numpy as np
import numpy as np

from myradar import RadarData, RadarDevice
from processing import rangedoppler, showspectrum

#fix the error of `np.float` was a deprecated alias for the builtin `float`
np.float = float    
np.int = int   #module 'numpy' has no attribute 'int'
np.object = object    #module 'numpy' has no attribute 'object'
np.bool = bool    #module 'numpy' has no attribute 'bool'

def create_color_heatmap(powData, PLOT_SIZE=128):
# create a colored array to represent an FFT
    MAX_DBM_DIFFERENCE = 60
    
    fftSize = len(powData)
    colorArray = np.ones((PLOT_SIZE,4), dtype = float)
    powDataAverage = np.average(powData)
    powDataMin = powDataAverage - 5
    powDataMax = powDataAverage + (-1 *  powDataMin)+ MAX_DBM_DIFFERENCE
    
    for i in range(PLOT_SIZE):
        startIndex = i * (fftSize / PLOT_SIZE)
        endIndex = (i * (fftSize / PLOT_SIZE)) + ((fftSize / PLOT_SIZE) - 1)
        fftAreaMax = np.amax(powData[int(startIndex):int(endIndex)]) 
        
        # max data is red
        if fftAreaMax > powDataMax:
            colorArray[i,0] = 1
            colorArray[i,1] = 0
            colorArray[i,2] = 0
            continue
        # min data is blue
        elif fftAreaMax < powDataMin :
            colorArray[i,0] = 0
            colorArray[i,1] = 0
            colorArray[i,2] = 1
            continue
        
        # scale everything in between    
        scaledVal = fftAreaMax + (-1 * powDataMin)
        
        # colours for lower bound (blue - green transition)
        if scaledVal < (powDataMax/2):
            colorArray[i,0] = 0
            colorArray[i,1] = 0.2 + scaledVal/powDataMax
            colorArray[i,2] = 1 - (scaledVal/powDataMax)
        
        # colours for upper bound (green - red transition)
        elif scaledVal >= (powDataMax/2):
            colorArray[i,0] = scaledVal/powDataMax
            colorArray[i,1] = 1 - (scaledVal/powDataMax)
            colorArray[i,2] = 0
       
    return colorArray

sample_rate = 0.6e6 #0.6M
fs = int(sample_rate) #0.6MHz
rxbuffersize = 1024 * 16 * 15 #fft_size
UseRadarDevice= False
if UseRadarDevice == True:
    sdrurl = "ip:pluto.local" #ip:phaser.local:50901
    phaserurl = "ip:phaser.local"
    radar=RadarDevice(sdrurl="", phaserurl="", samplerate=sample_rate, rxbuffersize=rxbuffersize)
else:
    datapath='./data/radardata5s-1101fast3move.npy'
    radar=RadarData(datapath=datapath, samplerate=sample_rate, rxbuffersize=rxbuffersize)
c, BW, num_steps, ramp_time_s, slope, N_c, N_s, freq, dist, range_resolution, signal_freq, range_x = radar.returnparameters()

ts = 1 / float(fs)
t = np.arange(0, rxbuffersize * ts, ts)

global xSize, ySize
xSize = int(N_c/2) #13 1024 #128 #26 #128
ySize = int(N_s/2) #300 1024 #150 #600 #128

class Window(QMainWindow): 
    def __init__(self): 
        super().__init__() 
  
        # set the title 
        self.setWindowTitle("RadarSensing") 
  
        # setting  the geometry of window 
        # setGeometry(left, top, width, height) 
        #self.setGeometry(100, 60, 1000, 800) 
        self.setGeometry(50, 50, 600, 700) #x-coordinate, y-coordinate, width of window, height of window
        self.minw1 = 100
        self.maxw1 = 300
        self.minw = 400
        self.graphheightrow = 10 #10

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
        widget = QtWidgets.QWidget() #QWidget()

        global layout
        layout = QGridLayout()

        # Control Panel
        control_label = QLabel("Kaikai Liu @ San Jose State University")
        font = control_label.font()
        font.setPointSize(15)
        control_label.setFont(font)
        #control_label.setAlignment(Qt.AlignHCenter)  # | Qt.AlignVCenter)
        control_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
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
            % (BW / 1e6, c / (2 * BW))
        )
        font = self.range_res_label.font()
        font.setPointSize(12)
        self.range_res_label.setFont(font)
        #self.range_res_label.setAlignment(Qt.AlignCenter) #Qt.AlignRight
        self.range_res_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.range_res_label.setMinimumWidth(self.minw1)
        layout.addWidget(self.range_res_label, 2, 0, 1, 2)

        # RF bandwidth slider
        #self.bw_slider = QSlider(Qt.Horizontal)
        #https://coderslegacy.com/python/pyqt6-qslider/
        self.bw_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.bw_slider.setMinimum(self.minw1)
        #self.bw_slider.setMaximum(self.maxw1)
        self.bw_slider.setValue(int(BW / 1e6))
        self.bw_slider.setTickInterval(50)
        self.bw_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.bw_slider.valueChanged.connect(self.get_range_res)
        layout.addWidget(self.bw_slider, 3, 0, 1, 2)

        self.set_bw = QPushButton("Set RF Bandwidth")
        #self.set_bw.pressed.connect(self.set_range_res)
        layout.addWidget(self.set_bw, 4, 0, 1, 2)

        #waterfall title
        self.water_label = QLabel("Waterfall Intensity Levels")
        self.water_label.setFont(font)
        self.water_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.water_label.setMinimumWidth(self.minw1)
        layout.addWidget(self.water_label, 5, 0, 1, 2)

        # waterfall level slider
        self.low_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.low_slider.setMinimum(-100)
        self.low_slider.setMaximum(100)
        self.low_slider.setValue(-20) #20
        self.low_slider.setTickInterval(20)
        self.low_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        #self.low_slider.valueChanged.connect(self.get_water_levels)
        layout.addWidget(self.low_slider, 6, 0)
        self.low_label = QLabel("LOW LEVEL: %0.0f" % (self.low_slider.value()))
        self.low_label.setFont(font)
        self.low_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        #self.low_label.setMinimumWidth(self.maxw1)
        layout.addWidget(self.low_label, 6, 1)

        self.high_slider = QSlider(Qt.Orientation.Horizontal, self) #QSlider(Qt.Horizontal)
        self.high_slider.setMinimum(-100)
        self.high_slider.setMaximum(100)
        self.high_slider.setValue(30) #60
        self.high_slider.setTickInterval(20)
        self.high_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        #self.high_slider.valueChanged.connect(self.get_water_levels)
        layout.addWidget(self.high_slider, 7, 0)
        self.high_label = QLabel("HIGH LEVEL: %0.0f" % (self.high_slider.value()))
        self.high_label.setFont(font)
        self.high_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        #self.high_label.setMinimumWidth(self.minw1)
        layout.addWidget(self.high_label, 7, 1)

        #Steering angle
        self.steer_title = QLabel("Receive Steering Angle")
        self.steer_title.setFont(font)
        self.steer_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.steer_title.setMinimumWidth(self.minw1)
        layout.addWidget(self.steer_title, 8, 0, 1, 2)

        self.steer_slider = QSlider(Qt.Orientation.Horizontal, self) #QSlider(Qt.Horizontal)
        self.steer_slider.setMinimum(-80)
        self.steer_slider.setMaximum(80)
        self.steer_slider.setValue(0)
        self.steer_slider.setTickInterval(20)
        self.steer_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        #self.steer_slider.valueChanged.connect(self.get_steer_angle)
        layout.addWidget(self.steer_slider, 9, 0)
        self.steer_label = QLabel("%0.0f DEG" % (self.steer_slider.value()))
        self.steer_label.setFont(font)
        self.steer_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.steer_label.setMinimumWidth(self.minw1)
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
            circle = QtWidgets.QGraphicsEllipseItem(-r, -r, r * 2, r * 2)
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
        layout.addWidget(self.plot_graph, 0, 2, self.graphheightrow, 1) #row=2, col=2m rowspan=4, colspan=1

        # FFT plot
        self.fft_plot = pg.plot()
        self.fft_plot.setMinimumWidth(self.minw)
        self.fft_plot.setBackground("w") #'k' is black https://www.pythonguis.com/tutorials/plotting-pyqtgraph/
        #self.fft_curve = self.fft_plot.plot(freq, pen="y", width=6)
        self.fft_curve = self.fft_plot.plot(freq, pen=pg.mkPen('b'), width=6)
        self.fft_plot.setLabel("bottom", text="Frequency", units="Hz", **label_style)
        self.fft_plot.setLabel("left", text="Magnitude", units="dB", **label_style)
        self.fft_plot.setTitle("Received Signal Spectrum", **title_style)
        layout.addWidget(self.fft_plot, self.graphheightrow, 2, self.graphheightrow, 1) #10, 2, 10, 1 row=2, col=2m rowspan=4, colspan=1
        self.fft_plot.setYRange(-80, 0)
        self.fft_plot.setXRange(100e3, 200e3)

        # Waterfall plot
        self.num_slices = 200
        self.img_array = np.zeros((self.num_slices, rxbuffersize))
        self.waterfall = pg.PlotWidget()
        self.waterfall.setBackground("w")
        self.waterfall.setMinimumWidth(self.minw)
        self.imageitem = pg.ImageItem() #(image=self.img_array)
        self.waterfall.addItem(self.imageitem)
        # self.imageitem.scale(0.35, sample_rate / (N))  # this is deprecated -- we have to use setTransform instead
        tr = QtGui.QTransform()
        tr.scale(0.35, sample_rate / (rxbuffersize)) ## scale horizontal and vertical axes
        self.imageitem.setTransform(tr)
        zoom_freq = 40e3
        self.waterfall.setRange(yRange=(100e3, 100e3 + zoom_freq))#starting from 100K
        self.waterfall.setTitle("Waterfall Spectrum", **title_style)
        self.waterfall.setLabel("left", "Frequency", units="Hz")
        self.waterfall.setLabel("bottom", "Time", units="sec")
        layout.addWidget(self.waterfall, 0, 3, self.graphheightrow, 1)
        
        # creating image view  object
        self.imv = pg.ImageView()
        #self.imv.setTitle("Ranger Doppler", **title_style)
        #self.imv.setBackground("w")
        self.imv.setMinimumWidth(self.minw)
        # Create random 3D data set with noisy signals
        # img = pg.gaussianFilter(np.random.normal(
        #     size=(200, 200)), (5, 5)) * 20 + 100
        # setting new axis to image
        img = np.random.normal(size=(60, 60))#[np.newaxis, :, :] #(1, 200, 200)
        # Displaying the data and assign each frame a time value from 1.0 to 3.0
        self.imv.setImage(img)
        #not used now
        #layout.addWidget(self.imv, 10, 3, 10, 1)

        self.add3Dplot()

        widget.setLayout(layout)
        # setting this widget as central widget of the main window
        self.setCentralWidget(widget)

    def add3Dplot(self):
        self.canvas3d = gl.GLViewWidget()
        layout.addWidget(self.canvas3d, self.graphheightrow, 3, self.graphheightrow, 1)
        # add grid
        # gridSize = QtGui.QVector3D(20,20,20) #(4,5,6)
        # g = gl.GLGridItem(size=gridSize)
        # g.scale(1,1,0.2)
        # self.canvas3d.addItem(g)

        self.axis = gl.GLAxisItem()
        self.canvas3d.addItem(self.axis)
        self.axis.setSize(x=150, y=150, z=150)
        self.xzgrid = gl.GLGridItem()
        self.yzgrid = gl.GLGridItem()
        self.xygrid = gl.GLGridItem()
        self.canvas3d.addItem(self.xzgrid)
        self.canvas3d.addItem(self.yzgrid)
        self.canvas3d.addItem(self.xygrid)
        self.xzgrid.setSize(x=180, y=100, z=0)
        self.xzgrid.setSpacing(x=10, y=10, z=10)
        self.yzgrid.setSize(x=100, y=180, z=0)
        self.yzgrid.setSpacing(x=10, y=10, z=10)
        self.xygrid.setSize(x=180, y=180, z=0)
        self.xygrid.setSpacing(x=10, y=10, z=10)
        # rotate x and y grids to face the correct direction
        self.xzgrid.rotate(90, 1, 0, 0)
        self.xzgrid.translate(0, -90, 50)
        self.yzgrid.rotate(90, 0, 1, 0)
        self.yzgrid.translate(-90, 0, 50)

        # the initial plot data before data is received
        z = np.random.normal(size=(xSize,ySize))
        # initialize the x-y boundries of the plot
        #x = np.linspace(-73.5, 144.3, xSize)
        #y = np.linspace(0.1, 5, ySize)

        # initialize the colors of the plots (light blue)
        self.colors = np.ones((xSize,ySize,4), dtype=float)
        self.colors[:,:,0] = 0
        self.colors[:,:,1] = 0.4
        self.colors[:,:,2] = 1

        # plot the data
        self.surface_plot = gl.GLSurfacePlotItem(z = z, shader = 'shaded',
                                     colors = self.colors.reshape(xSize*ySize,4), smooth=False)
        # # determine the size
        #self.surface_plot.scale(16./49., 20, 0.1)
        # # determine the location on the gride
        # p3.translate(-12, -50, 0)
        #self.surface_plot = gl.GLSurfacePlotItem(computeNormals=False)
        self.surface_plot.translate(-100, -100, 0) #translate(0, 0, 100)

        self.canvas3d.addItem(self.surface_plot)
        self.canvas3d.setCameraPosition(distance=300)
        #self.surface_plot.setData(x=x, y=y, z = z, colors = self.colors.reshape(xSize*ySize,4))

        
    
    def create_timer(self, timestep=300):  #300 milliseconds
        # Add a timer to simulate new temperature measurements
        self.timer = QtCore.QTimer()
        self.timer.setInterval(timestep)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
    
    def update_plot(self):
        currentdata, datalen, self.currentindex = radar.receive(self.currentindex)

        self.line.setData(t, currentdata.real)

        sepcf, s_dbfs_shift = showspectrum(currentdata, fs)
    
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
        # Ntimes=5
        # if self.currentindex>Ntimes:
        #     currentdata2=alldata[max(0, (self.currentindex-Ntimes)*N_frame):(self.currentindex+1)*N_frame]
        #     N_c = int(len(currentdata2)/N_s)-1 #number of chirps in fft_size
        #     rddata, table=rangedoppler(currentdata2, n_c=N_c, n_s=N_s, showdb=True) #81 300
        #     xSize = int(N_c/2)
        #     newsize = int(xSize * ySize)
        #     self.surface_plot.setData(z=rddata, colors = self.colors.reshape(newsize,4))
            
            #resampled_rd=ndimage.zoom(rddata, zoom=(20,5))
            #self.imv.setImage(resampled_rd)
        rddata, table = rangedoppler(currentdata, n_c=N_c, n_s=N_s, showdb=True) #Number of Chirps, Number of samples
        #print(np.max(rddata))
        #print(np.min(rddata))
        newsize = int(xSize * ySize) #204*300=61200
        #print(self.colors.shape)
        newcolorshape = self.colors.reshape(newsize,4)
        self.surface_plot.setData(z=rddata)#, colors = newcolorshape)

        #self.currentindex = self.currentindex + 1 #updated in receive()

  
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

# create pyqt6 app 
App = QApplication(sys.argv) 
  
# create the instance of our Window 
window = Window() 

# timer = QtCore.QTimer()
# timer.timeout.connect(update)
# timer.start(0) #A QTimer with a timeout interval of 0 will time out as soon as all the events in the window system's event queue have been processed.

# start the app 
#sys.exit(App.exec()) 
App.exec()
#!/usr/bin/python
#https://github.com/pyrf/pyqtgraph_examples/blob/master/plot_fft_3d.py
# import all the necessary Libraries
import sys
import numpy as np


from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl

#PLOT_SIZE = 128


#ref: https://github.com/pyrf/pyrf/blob/master/pyrf/numpy_util.py
def compute_fft(data, hide_differential_dc_offset=True, apply_window=True, convert_to_dbm=True):
    Nsamp = len(data)
    # if hide_differential_dc_offset:
    #     i_data = i_data - np.mean(i_data)
    #     q_data = q_data - np.mean(q_data)

    # if apply_window:
    #     i_data = i_data * np.hanning(len(i_data))
    #     q_data = q_data * np.hanning(len(q_data))
    
    iq = data #i_data + 1j * q_data

    if apply_window:
        iq = iq * np.hanning(Nsamp)
    
    power_spectrum = np.abs(np.fft.fftshift(np.fft.fft(iq)))/Nsamp

    if convert_to_dbm:
        power_spectrum = 20 * np.log10(power_spectrum)

    if hide_differential_dc_offset:
        median_index = len(power_spectrum) // 2
        power_spectrum[median_index] = (power_spectrum[median_index - 1]
            + power_spectrum[median_index + 1]) / 2

    return power_spectrum

def represent_fft_to_plot(powData, PLOT_SIZE=128, bias=100):
    # represent an FFT in a PLOT_SIZE integer array to be represented in a 3D plot
    
    zData = np.zeros(shape =(PLOT_SIZE,1))
    fftSize = len(powData)
    for i in range(PLOT_SIZE):
        startIndex = i * (fftSize / PLOT_SIZE)
        endIndex = (i * (fftSize / PLOT_SIZE)) + ((fftSize / PLOT_SIZE) - 1)
        zData[i] = (np.amax(powData[int(startIndex):int(endIndex)]) + bias)
    
    return zData
    
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

def loadData(N_frame):
    with open('./data/radardata5s-1101fast3move.npy', 'rb') as f:
        alldata = np.load(f)
    print(len(alldata))
    totallen=len(alldata)
    Ntotalframe=int(totallen/N_frame)-1
    return alldata, Ntotalframe
    # for i in range(Ntotalframe):
    #     data = alldata[i*N_frame:(i+1)*N_frame]

alldata, Ntotalframe = loadData(N_frame)

# hold a constant reflevel
STATIC_REFLEVEL = -35
xSize = 1024 #128 #26 #128
ySize = 150 #1024 #150 #600 #128

# define normalized 2D gaussian
def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))


# Create a GL View widget to display data
app = QtGui.QApplication([])
w = gl.GLViewWidget()

w.show()
w.setWindowTitle('PYRF pyqtgraph example: 3D FFT Plot')

w.setCameraPosition(distance=100)

# add grid
# gridSize = QtGui.QVector3D(4,5,6)
# g = gl.GLGridItem(size=gridSize)
# g.scale(5.85,5,0)
# w.addItem(g)

gridSize = QtGui.QVector3D(4,5,6)
g = gl.GLGridItem()
w.addItem(g)
g.setSize(x=150, y=150, z=150)



# the initial plot data before data is received
z = np.random.normal(size=(xSize,ySize))
#z[:,50] =100

# xgrid, yrid = np.meshgrid(np.linspace(1, xSize), np.linspace(1, ySize)) # get 2D variables instead of 1D
# z = gaus2d(xgrid, yrid)

# initialize the x-y boundries of the plot
x = np.linspace(-73.5, 144.3, xSize)
y = np.linspace(0.1, 5, ySize)

# initialize the colors of the plots (light blue)
colors = np.ones((xSize,ySize,4), dtype=float)
colors[:,:,0] = 0
colors[:,:,1] = 0.4
colors[:,:,2] = 1

# plot the data
p3 = gl.GLSurfacePlotItem(x = x, y = y, z = z, shader = 'shaded',
                            colors = colors.reshape(xSize*ySize,4), smooth=False)

# determine the size
p3.scale(16./49., 20, 0.1)

# determine the location on the gride
p3.translate(-12, -50, 0)
w.addItem(p3)

w.pan(-10,-10,-10)

global index
index=1

def update():
    # update the plot to show new data
    global p3, z, colors, index
    
    # update the plot
    #z[:,index %128] =index
    if (index+1)*N_frame >= len(alldata):
        index=1
    currentdata = alldata[index*N_frame:(index+1)*N_frame]

    # compute the fft of the complex data
    powData = compute_fft(currentdata) #16384
    # compress the FFT into a 128 array
    zData = represent_fft_to_plot(powData, PLOT_SIZE=xSize) #128,1
    # move the data stream as well as colours back to show new data
    for i in range (ySize):
        if i == 0:
            continue
        else: #shift y
            colors[:,ySize - i,:] = colors[:, ySize - i - 1,:]
            z[:,ySize - i] = z[:, ySize - i - 1]
        z[:,0] = zData[:,0]

    # grab new color
    colors[:,0,:] = create_color_heatmap(powData, PLOT_SIZE=xSize)
    #colors[:,1,:] = create_color_heatmap(powData, PLOT_SIZE=xSize)
    #https://pyqtgraph.readthedocs.io/en/latest/api_reference/3dgraphics/glsurfaceplotitem.html
    p3.setData(x=x, y=y, z = z, colors = colors.reshape(xSize*ySize,4))

    index=index+1

def updateRDgraph(currentdata):
    if (index+1)*N_frame >= len(alldata):
        index=1
    n_s = 600
    n_r = int(len(currentdata)/n_s)-1
    table = np.zeros((n_r, n_s)) #150 chirps,1000 samples/chirp
    for chirp_nr in range(n_r):
        table[chirp_nr, :] = currentdata[(chirp_nr*n_s):(n_s*(chirp_nr+1))] #26, 600
    Z_fft2 = abs(np.fft.fft2(table)) #
    s_mag = np.abs(Z_fft2)/len(Z_fft2) #power spectrum
    s_dbfs = 20 * np.log10(s_mag)
    #Data_fft2 = Z_fft2[0:int(n_r/2),0:int(n_s/2)] #get half
    #Data_fft2 = Z_fft2[0:n_r,0:int(n_s/2)] #get half 0:int(n_s/2)
    #p3.setData(z=s_dbfs)
    #p3.setData(z = s_dbfs, colors = colors.reshape(xSize*ySize,4))

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(5)
  
## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
  
  
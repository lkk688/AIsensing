import matplotlib
import platform
if platform.system() == 'Darwin':
    #matplotlib.use('MacOSX')
    matplotlib.use("TkAgg") #need to add this in Mac, otherwise, matplotlib figure stuck in debugger mode
import matplotlib.pyplot as plt
import numpy as np
# Enable interactive mode
#plt.ion() 

# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

fig.savefig("test.png")
plt.show()
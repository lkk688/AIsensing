#https://pyqtgraph.readthedocs.io/en/latest/getting_started/introduction.html

import pyqtgraph as pg
import numpy as np
plotWidget = pg.plot(title="Plot curves")
x = np.random.normal(size=1000)
y = np.random.normal(size=1000)
#pg.plot(x, y, pen=None, symbol='o')  ## setting pen=None disables line drawing
plotWidget.plot(x,y)

# importing the pyqtgraph.examples module 
import pyqtgraph.examples 
  
# run this examples 
pyqtgraph.examples.run() 

# x = np.arange(1000)
# y = np.random.normal(size=(3, 1000))
# plotWidget = pg.plot(title="Three plot curves")
# for i in range(3):
#     plotWidget.plot(x, y[i], pen=(i,3))  ## setting pen=(i,3) automaticaly creates three different-colored pens
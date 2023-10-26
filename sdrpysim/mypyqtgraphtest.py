import numpy as np
import pyqtgraph as pg

data = np.random.normal(size=1000)
pg.plot(data, title="Simplest possible plotting example")

data = np.random.normal(size=(500,500))
pg.image(data, title="Simplest possible image example")


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()
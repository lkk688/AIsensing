import os
#https://matplotlib.org/stable/api/backend_qt_api.html
print(os.environ.get('QT_API'))
#os.environ['QT_API'] = 'pyqt5'
import matplotlib
matplotlib.use('QtAgg')
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('imgs/64QAM.png',cv2.IMREAD_GRAYSCALE)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
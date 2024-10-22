import os
import time
#https://matplotlib.org/stable/api/backend_qt_api.html
print(os.environ.get('QT_API'))
#os.environ['QT_API'] = 'pyqt5'
import matplotlib
#matplotlib.use('QtAgg')
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()

import cv2 #pip install opencv-python --upgrade
import numpy as np
from matplotlib import pyplot as plt


image = cv2.imread('imgs/64QAM.png',cv2.IMREAD_GRAYSCALE)
# cv2.imshow('image',img)
# cv2.waitKey(0)
#cv2.destroyAllWindows()

# Check if the image was successfully read
if image is None:
    print('Failed to read the image')
else:
    # Display the image
    cv2.imshow('Original Image', image)
    #cv2.waitKey(0)  # Wait for a key press to close the window, the terminal may stuck
    
    # Set a timeout in seconds
    timeout = 5  # Close window after 5 seconds
    start_time = time.time()

    # Keep the window open until a key is pressed or timeout occurs
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to close the window
            break
        elif time.time() - start_time > timeout:  # Close window after timeout
            break

    # Save a copy of the image
    cv2.imwrite('imgs/64QAM_copy.png', image)

    # Close all OpenCV windows
    cv2.destroyAllWindows()
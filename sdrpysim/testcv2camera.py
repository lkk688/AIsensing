
import numpy as np
import cv2
import platform
if platform.system() == 'Darwin':
    cap = cv2.VideoCapture(1) #for Mac 
else:
    cap = cv2.VideoCapture(0)

frameWidth = 640
frameHeight = 480
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)

if(cap.isOpened()):
    print("Camera conntected")
else:
    print("Alert ! Camera disconnected")

number_itteration = 0
while cap.isOpened():
    try:
        success, frame = cap.read()
        if number_itteration == 0:
            print(len(frame))
        if success:
            cv2.imshow("Result", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print('camera already used')
        break
    number_itteration += 1
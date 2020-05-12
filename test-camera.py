# Using thresh found in: https://github.com/harshbg/Sign-Language-Interpreter-using-Deep-Learning/blob/master/Code/set_hand_histogram.py

import cv2
import numpy as np
import os
from PIL import Image

# Function to get the camera input and save it as a png
def load_and_run_webcam():
    # Get camera input
    cap = cv2.VideoCapture(0)
    # Where to put the rectangle
    left_top = (50, 100)
    right_bottom = (250, 300)
    border = (255, 255, 255)
    while(True):
        img = cap.read()[1]
        img = cv2.flip(img, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        crop = hsv[100:300, 50:250]
        crop = cv2.resize(crop, (64, 64))
        hist = cv2.calcHist([crop], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
        dst1 = dst.copy()
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
        cv2.filter2D(dst,-1,disc,dst)
        blur = cv2.GaussianBlur(dst, (11,11), 0)
        blur = cv2.medianBlur(blur, 15)
        ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thresh = cv2.merge((thresh,thresh,thresh))

        # Display the original and edited frames
        cv2.imshow("ASL", img)
        cv2.imshow("Cropped", crop)
        cv2.imshow("Thresh", thresh)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

load_and_run_webcam()

import sys
import os

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2

def video():
    # Finds contour on video
    # https://answers.opencv.org/question/77046/use-contours-in-a-video/

    cap = cv2.VideoCapture(0)
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    fgbg = cv2.createBackgroundSubtractorMOG2()
    connectivity = 4

    while(True):
      # Capture frame-by-frame
       ret, frame = cap.read()
       if not ret:
          break
       fgmask = fgbg.apply(frame)
       fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernal)
       output = cv2.connectedComponentsWithStats(fgmask, connectivity, cv2.CV_32S)
       for i in range(output[0]):
           if output[2][i][4] >= 800 and output[2][i][4] <= 10000:
               cv2.rectangle(frame, (output[2][i][0], output[2][i][1]), (
                   output[2][i][0] + output[2][i][2], output[2][i][1] + output[2][i][3]), (0, 255, 0), 2)
       cv2.imshow('detection', frame)
       if cv2.waitKey(1) & 0xFF == ord('q'):
        break

       ret2, frame2 = cap.read()

       gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
       blur = cv2.GaussianBlur(gray,(5,5),0)
       ret, thresh_img = cv2.threshold(blur,91,255,cv2.THRESH_BINARY)

       contours = cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
       for c in contours:
           cv2.drawContours(frame2, [c], -1, (0, 0, 255), 3)

         # Display the resulting frame
       cv2.imshow('frame',frame2)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img = cv2.imread("images/sudoku30.png", cv2.IMREAD_GRAYSCALE)
    cv2.imshow("Sudoku", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    video()

    blur = 3
    erode = 2
    threshold = 37
    adjustment = 11
    iterations = -3
    #process_image(blur,threshold, adjustment, erode, iterations)


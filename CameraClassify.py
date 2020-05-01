import sys
import os

import matplotlib
import numpy as np
from keras.datasets import mnist
import cv2

def prediction(pred):
    return(chr(pred+ 65))

def crop_image(image, x, y, width, height):
    return image[y:y + height, x:x + width]


def keras_predict(model, image):
    data = np.asarray(image, dtype="int32")

    pred_probab = model.predict(data)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


def keras_process_image(img):
    image_x = 28
    image_y = 28
    img = cv2.resize(img, (1, 28, 28), interpolation=cv2.INTER_AREA)

    return img
def video():

    cap = cv2.VideoCapture(0)
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgbg = cv2.createBackgroundSubtractorMOG2()
    connectivity = 4
    while(True):

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
        cv2.imshow("detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray_img, 100, 0.01, 10)
        corners = np.int0(corners)

        # draw red color circles on all corners
        for i in corners:
            x, y = i.ravel()
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

        cv2.imshow("corners", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        if ret == ord('s'):
                cv2.imwrite(filename='saved_img.jpg', img=frame)
                cap.release()
                img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
                img_new = cv2.imshow("Captured Image", img_new)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
                gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
                img_ = cv2.resize(gray, (28, 28))
                img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=img_)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    video()

    blur = 3
    erode = 2
    threshold = 37
    adjustment = 11
    iterations = -3


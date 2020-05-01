import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import time
from multiprocessing import Process, Lock, Queue, current_process




for filename in os.listdir("dataset"):
    color = (255, 0, 255)
    print(filename)
    image = cv2.imread("dataset/"+filename, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("images", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
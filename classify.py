import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import time
from multiprocessing import Process, Lock, Queue, current_process


# size of the image: 48*48 pixels
pic_size = 48

# input path for the images
base_path = "../input/VisionFinal/dataset/"

plt.figure(0, figsize=(12,20))
cpt = 0

for expression in os.listdir(base_path + "labels"):
    for i in range(1,6):
        cpt = cpt + 1
        plt.subplot(7,5,cpt)
        img = tf.keras.preprocessing.image.load_img\
            (base_path + "train/" +expression + "/" +os.listdir
            (base_path + "train/" + expression)[i], target_size=(pic_size, pic_size))
        plt.imshow(img, cmap="gray")

plt.tight_layout()
plt.show()
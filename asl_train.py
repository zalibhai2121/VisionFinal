# Train a model on the data

import numpy as np
import cv2
import tensorflow as tf
import os
import glob
import pathlib
import time
from typing import Dict, List, Tuple
from multiprocessing import Process, Lock, Queue, current_process
import random


# Remove any TF log outputs (e.g. CPU supporting stuff)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


path = glob.glob("/VisionsFinal/to/dataset/*.jpg")
images = []
for img in path:
    n = cv2.imread(img)
    images.append(n)

print(images)

# Build a model
# Takes nothing as imput, returns a model
def build_asl_model():
    # Build the model
    model = tf.keras.Sequential()


    model.add(tf.keras.layers.Embedding(input_dim=1000, output_dim=64))
    model.add(tf.keras.layers.GRU(256, return_sequences=True))
    model.add(tf.keras.layers.SimpleRNN(128))
    #model.add(tf.keras.layers.LSTM(128))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    """"
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, (5, 5), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Conv2D(24, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Conv2D(96, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(3, activation='softmax')])
    """
    print("Shape of output", model.compute_output_shape(input_shape=(None, 64, 64, 1)))

    model.summary() #Summarise the model

    #Compile the defined neural network
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    print("Done building the network topology.")

    # Read training and prediction data
    train = images
    # train, train_labels, predict, predict_labels = read_labels_and_images()

    # Save a checkpoint of our work
    checkpoint = tf.keras.callbacks.ModelCheckpoint("models/fingersteps/checkpoint_{epoch}")

    return model

def display_img(img):

    cv2.imshow('Img', img)

    cv2.waitKey(0)


for filename in os.listdir("dataset"):
    color = (255, 0, 255)
    print(filename)
    image = cv2.imread("dataset/"+filename, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("images", image)
    cv2.waitKey()
    cv2.destroyAllWindows()

def corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    print(gray)

    # find Harris corners
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    coners = cv2.cornerSubPix((img), np.float32(centroids), (10, 10), (-1, -1), criteria)
    #print(coners)
    # here u can get corners check for more information follow the link
    # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
    # Now draw them
    res = np.hstack((centroids, coners))
    res = np.int0(res)
    img[res[:, 1], res[:, 0]] = [0, 0, 255]
    img[res[:, 3], res[:, 2]] = [0, 255, 0]
    cv2.imshow('co', img)

    # (img, howmany, quality, min dist btwn corners)
    corner = cv2.goodFeaturesToTrack((img), 500, 0.01, 10)
    corner = np.int0(corner)
    print(corner)

    for c in corner:
        x, y = c.ravel()
        cv2.circle(img, (x,y), 3, 255, -1)

    cv2.imshow('corners', img)


build_asl_model()

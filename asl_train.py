# Train a model on the data

import numpy as np
import cv2
import tensorflow as tf
import os
import time
from multiprocessing import Process, Lock, Queue, current_process
import random

# Remove any TF log outputs (e.g. CPU supporting stuff)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Build a model
# Takes nothing as imput, returns a model
def build_asl_model():
    # Build the model
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

    print("Shape of output", model.compute_output_shape(input_shape=(None, 64, 64, 1)))

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    print("Done building the network topology.")

    # Read training and prediction data
    #train, train_labels, predict, predict_labels = read_labels_and_images()

    # Save a checkpoint of our work
    checkpoint = tf.keras.callbacks.ModelCheckpoint("models/fingersteps/checkpoint_{epoch}")

    return model

def read_labels_and_images():
    start_time = time.time()
    filepath = "dataset/"
    # Read the label file, mapping image filenames to (x, y) pairs for UL corner
    with open(filepath + "labels.txt") as label_file:
        label_string = label_file.readline()
        label_map = eval(label_string)
    print("Read label map with", len(label_map), "entries")

    # Read the directory and make a list of files
    filenames = []
    for filename in os.listdir(filepath):
        if random.random() < -0.7: continue # to select just some files
        if filename.find("jpg") == -1: continue # process only images
        filenames.append(filename)
    print("Read", len(filenames), "images.")

    # Extract the features from the images
    print("Extracting features")
    train, train_labels, predict, predict_labels = [], [], [], []
    processes = []     # Hold (process, queue) pairs
    num_processes = 8  # Process files in parallel

    # Launch the processes
    for start in range(num_processes):
        q = Queue()
        files_for_one_process = filenames[start:len(filenames):num_processes]
        file_process = Process(target=process_files, \
                        args=(filepath, files_for_one_process, label_map, q))
        processes.append((file_process, q))
        file_process.start()

    # Wait for processes to finish, combine their results
    for p, q in processes: # For each process and its associated queue
        result = q.get()            # Blocks until the item is ready
        train += result[0]          # Get training features from child
        train_labels += result[1]   # Get training labels from child
        predict += result[2]        # Get prediction features from child
        predict_labels += result[3] # Get prediction labels from child
        p.join()                    # Wait for child process to finish
    print("Done extracting features from images. Time =", time.time() - start_time)

    return (train, train_labels, predict, predict_labels)

# Child process, intended to be run in parallel in several copies
#  files      our portion of the files to process
#  label_map  contains the right answers for each file
#  q          a pipe for sending results back to the parent process
def process_files(filepath, files, label_map, q):
    np.random.seed(current_process().pid) # Each child gets a different RNG
    t, tl, p, pl = [], [], [], []
    checkpoint, check_interval, num_done = 0, 5, 0 # Just for showing progress
    for filename in files:
        if 100*num_done > (checkpoint + check_interval) * len(files):
            checkpoint += check_interval
            print((int)(100 * num_done / len(files)), "% done")
        num_done += 1
        img = cv2.imread(filepath + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = tf.reshape(img, [64, 64, 1])
        for contrast in [1, 1.2, 1.5, 1.7]:
            img2 = tf.image.adjust_contrast(img, contrast_factor=contrast)
            #print("*** 3")
            for bright in [0.0, 0.1, 0.2, 0.3]:
                img3 = tf.image.adjust_brightness(img2, delta=bright)
                #cv2.imshow("Augmented Image" + str(current_process().pid), img3.numpy())
                #cv2.waitKey(1)
                if np.random.random() < 0.9: # 80% of images
                    t.append(img3.numpy())
                    tl.append(label_map[filename]-1)
                else:
                    p.append(img3.numpy())
                    pl.append(label_map[filename]-1)

    q.put((t, tl, p, pl))


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

# Train a model on the data

import numpy as np
import cv2
import tensorflow as tf
import os
import time

# Remove any TF log outputs (e.g. CPU supporting stuff)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Get the dimensions of the data, all images should be the same size, so the dimensions of 1 will work for all
def image_size():
    img = cv2.imread('dataset/A/A1.jpg', 0)
    size = img.shape
    return size
# Extract the x, y lengths
img_x, img_y = image_size()

# Begin training the model
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

    # Read training and prediction data and save as a checkpoint
    train, train_labels, predict, predict_labels = read_labels_and_images()
    checkpoint = tf.keras.callbacks.ModelCheckpoint("models/fingersteps/checkpoint_{epoch}")

def read_labels_and_images():
    start_time = time.time()
    filepath = "dataset/"
    # Read the label file, mapping image filenames to (x, y) pairs for UL corner
    with open(filepath + "labels") as label_file:
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


build_asl_model()

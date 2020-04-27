# Train a model on the data

import numpy as np
import cv2
import tensorflow as tf
import os
import time

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

    # Save a checkpoint of our work
    checkpoint = tf.keras.callbacks.ModelCheckpoint("models/fingersteps/checkpoint_{epoch}")

    return model

# Train the model we made
# Takes nothing as input, returns a trained NN
def train_model():
    return

build_asl_model()

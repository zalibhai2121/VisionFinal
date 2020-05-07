import cv2
import tensorflow as tf
import os
import pickle
import classify
import numpy as np
import matplotlib.pyplot as plt
# Remove any TF log outputs (e.g. CPU supporting stuff)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train():
    # Loads data from classify.py that will be used to train the network
    (x_train, y_train), (x_test, y_test) = classify.load_data()

    # The letters we will be trainin, testing on
    labels = ['A', 'B', 'C']

    # Print the first several training images, along with the labels
    fig = plt.figure(figsize=(20, 5))
    for i in range(10):
        ax = fig.add_subplot(3, 12, i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(x_train[i]))
        ax.set_title("{}".format(labels[y_train[i]]))
    plt.show()

    # Number of each letter in the training dataset
    A_train = sum(y_train == 0)
    B_train = sum(y_train == 1)
    C_train = sum(y_train == 2)

    # Number of each letter in the test dataset
    A_test = sum(y_test == 0)
    B_test = sum(y_test == 1)
    C_test = sum(y_test == 2)

    # Print statistics about the dataset
    print("Training set:")
    print("\tA: {}, B: {}, C: {}".format(A_train, B_train, C_train))
    print("Test set:")
    print("\tA: {}, B: {}, C: {}".format(A_test, B_test, C_test))

    # One-hot encode the training and test labels
    y_train_OH = tf.keras.utils.to_categorical(y_train)
    y_test_OH = tf.keras.utils.to_categorical(y_test)

    # Begin building the model
    model = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=5, kernel_size=5, padding='same', activation='relu',input_shape=(50, 50, 3)),
                                 tf.keras.layers.MaxPooling2D(pool_size=4),
                                 tf.keras.layers.Conv2D(filters=15, kernel_size=5, padding='same', activation='relu'),
                                 tf.keras.layers.MaxPooling2D(pool_size=4),tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(3, activation='softmax')])


    # Summarize the model
    print("Summary of model:")
    model.summary()

    # Compile the model using categorical_crossentropy
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', # binary_crossentropy
                  metrics=['accuracy'])

    # Fit the model to the images
    hist = model.fit(x_train, y_train_OH,
                     validation_split=0.20,
                     epochs=10, # Increase this to improve accuracy
                     batch_size=32)
    print("Shape of output", model.compute_output_shape(input_shape=(None, 64, 64, 1)))

    # Obtain accuracy on test set
    score = model.evaluate(x=x_test,
                           y=y_test_OH,
                           verbose=0)
    print('Test accuracy:', score[1])
    y_probs = model.predict(x_test)

    # Get predicted labels for test dataset
    y_preds = np.argmax(y_probs, axis=1)

    # Indices corresponding to test images which were mislabeled
    bad_test_idxs = np.where(y_preds != y_test)[0]

    # Print mislabeled examples
    fig = plt.figure(figsize=(25, 4))
    for i, idx in enumerate(bad_test_idxs):
        ax = fig.add_subplot(2, np.ceil(len(bad_test_idxs) / 2), i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(x_test[idx]))
        ax.set_title("{} (pred: {})".format(labels[y_test[idx]], labels[y_preds[idx]]))

# Build, train and test the model
train()

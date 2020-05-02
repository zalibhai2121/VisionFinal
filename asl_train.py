# Train a model on the data


import cv2
import tensorflow as tf
import os
import pickle
import classify
import numpy as np
import matplotlib.pyplot as plt



# Remove any TF log outputs (e.g. CPU supporting stuff)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Build a model
# Takes nothing as imput, returns a model
def build_asl_model():
    # Build the model
    X = pickle.load(open("A.pickle", "rb"))
    Y = pickle.load(open("B.pickle", "rb"))




    A = A/255



    model = tf.keras.Sequential()

    """""
    model.add(tf.keras.layers.Embedding(input_dim=1000, output_dim=64))
    model.add(tf.keras.layers.GRU(256, return_sequences=True))
    model.add(tf.keras.layers.SimpleRNN(128))
    #model.add(tf.keras.layers.LSTM(128))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    """

    model.add(tf.keras.layers.Conv2D(64, (3,3), input_shape = A.shape[1:]))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64))

    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation("sigmoid"))
    model.compile(loss = "binary_crossentropy",
                  optimizer = "adam",
                  metrix = ['accuracy'])
    print("Done building the network topology.")

    model.fit(X, Y, batch_size = 32, validation_split = 0.1, )
    print("Shape of output", model.compute_output_shape(input_shape=(None, 64, 64, 1)))

    model.summary() #Summarise the model


    # Read training and prediction data
    # train, train_labels, predict, predict_labels = read_labels_and_images()

    # Save a checkpoint of our work
    checkpoint = tf.keras.callbacks.ModelCheckpoint("models/fingersteps/checkpoint_{epoch}")

    return model


def train():
    (x_train, y_train), (x_test, y_test) = classify.load_data()

    labels = ['A', 'B', 'C']

    # Print the first several training images, along with the labels
    fig = plt.figure(figsize=(20, 5))
    for i in range(10):
        ax = fig.add_subplot(3, 12, i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(x_train[i]))
        ax.set_title("{}".format(labels[y_train[i]]))
    plt.show()

    num_A_train = sum(y_train == 0)
    # Number of B's in the training dataset
    num_B_train = sum(y_train == 1)
    # Number of C's in the training dataset
    num_C_train = sum(y_train == 2)

    # Number of A's in the test dataset
    num_A_test = sum(y_test == 0)
    # Number of B's in the test dataset
    num_B_test = sum(y_test == 1)
    # Number of C's in the test dataset
    num_C_test = sum(y_test == 2)

    # Print statistics about the dataset
    print("Training set:")
    print("\tA: {}, B: {}, C: {}".format(num_A_train, num_B_train, num_C_train))
    print("Test set:")
    print("\tA: {}, B: {}, C: {}".format(num_A_test, num_B_test, num_C_test))

    # One-hot encode the training labels
    y_train_OH = tf.keras.layers.np_utils.to_categorical(y_train)

    # One-hot encode the test labels
    y_test_OH = tf.keras.layers.np_utils.to_categorical(y_test)

    model = tf.keras.layers.Sequential()
    # First convolutional layer accepts image input
    model.add(tf.keras.layers.Conv2D(filters=5, kernel_size=5, padding='same', activation='relu',
                     input_shape=(50, 50, 3)))
    # Add a max pooling layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=4))
    # Add a convolutional layer
    model.add(tf.keras.layers.Conv2D(filters=15, kernel_size=5, padding='same', activation='relu'))
    # Add another max pooling layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=4))
    # Flatten and feed to output layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    # Summarize the model
    model.summary()

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    hist = model.fit(x_train, y_train_OH,
                     validation_split=0.20,
                     epochs=2,
                     batch_size=32)

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



train()

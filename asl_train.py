# To-do list ### CURRENT ISSUE: Constantly thinks the letter is M (I think it's because of model inaccuracy)
# 1) Modify the model in the hopes of improving accuracy (add additional Conv2D layer) (Maybe another dropout layer?)
# 2) #Important# Try using Opencv to thresh, etc the dataset and train on that. We want the background black and the hands white. This can happen in a copy of the work_on_images.py file
# 3) Take photos of my own hand doing the alphabet. However, until model acuracy goes up, no photo will help...

import cv2
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from tensorflow.python.util import deprecation

# Remove any TF log outputs (e.g. CPU supporting stuff)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
deprecation._PRINT_DEPRECATION_WARNINGS = False

# (MAC ONLY) Run the following to delete DS_Store files when they cause issues:
# find . -name ".DS_Store" -delete

def train():
    """
    # Loads data from classify.py that will be used to train the network
    (x_train, y_train), (x_test, y_test) = classify.load_data()

    # The letters we will be training, testing on
    labels = ["A", "B", "C"]

    # Print the first several training images, along with the labels
    fig = plt.figure(figsize=(20, 5), num = "Some of the training images and labels")
    for i in range(20):
        ax = fig.add_subplot(3, 12, i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(x_train[i]))
        ax.set_title("{}".format(labels[y_train[i]]))

    # Number of each letter in the training
    A_train = sum(y_train == 0)
    B_train = sum(y_train == 1)
    C_train = sum(y_train == 2)

    # Number of each letter in the test
    A_test = sum(y_test == 0)
    B_test = sum(y_test == 1)
    C_test = sum(y_test == 2)

    # Print statistics about the
    print("Training set:")
    print("\tA: {}, B: {}, C: {}".format(A_train, B_train, C_train))
    print("Test set:")
    print("\tA: {}, B: {}, C: {}".format(A_test, B_test, C_test))

    # One-hot encode the training and test labels
    y_train_OH = tf.keras.utils.to_categorical(y_train)
    y_test_OH = tf.keras.utils.to_categorical(y_test)
    """

    # Build the CNN using Sequential model
    model = tf.keras.Sequential([tf.keras.layers.Convolution2D(32, 3,  3, input_shape = (64, 64, 3), activation = "relu"),
                                 tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                                 tf.keras.layers.Convolution2D(32, 3, 3, activation="relu"),
                                 tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(256, activation="relu"),
                                 tf.keras.layers.Dropout(0.5),
                                 tf.keras.layers.Dense(26, activation="softmax")])

                                 # I think the issue with accuracy occurs here, we need to add more layers to our model. I will work on this after I finish Theory

    """ # Old Model
    model = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=5, kernel_size=5, padding="same", activation="relu",input_shape=(64, 64, 3)),
                                 tf.keras.layers.MaxPooling2D(pool_size=4),
                                 tf.keras.layers.Conv2D(filters=15, kernel_size=5, padding="same", activation="relu"),
                                 tf.keras.layers.MaxPooling2D(pool_size=4),
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(3, activation="softmax")])
    """

    # Summarize the model
    print("\n Finished training model \n")
    print("\n\nSummary of model:")
    model.summary()

    # Compile the model using categorical_crossentropy
    model.compile(optimizer= tf.keras.optimizers.Adam(),     #"rmsprop", "adam"
                  loss="categorical_crossentropy",          # binary_crossentropy
                  metrics=["accuracy"])

    # Fit the NN to the images
    # Prepare the images
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)
    # Make the training and test datasets easily accessable during the process
    training_set = train_datagen.flow_from_directory(
            "dataset2",
            target_size=(64, 64),
            batch_size=32,
            class_mode="categorical")

    test_set = test_datagen.flow_from_directory(
            "test_dataset2",
            target_size=(64, 64),
            batch_size=32,
            class_mode="categorical")

    """
    # Fit the model to the images
    hist = model.fit(x_train, y_train_OH,
                     validation_split=0.20,
                     epochs=25, # Increase this to improve accuracy
                     batch_size=64)
    print("Shape of output", model.compute_output_shape(input_shape=(None, 64, 64, 1)))
    """

    # Fit the model to the images.
    fit_model = model.fit_generator(
            training_set,
            steps_per_epoch=800,
            epochs=25,
            validation_data = test_set,
            validation_steps = 20) # The larger the number the longer it takes at the end of each epoch.

    # Save the model
    model_name = "asl_5" # Change this to save as a new model, will also generate new graphs for the model
    model.save("models/" + model_name)

    """

    # Obtain accuracy on test set
    score = model.evaluate(x=x_test,
                           y=y_test_OH,
                           verbose=0)
    print("Test accuracy:", score[1])
    y_probs = model.predict(x_test)

    # Get predicted labels for test
    y_preds = np.argmax(y_probs, axis=1)

    # Indices corresponding to test images which were mislabeled
    bad_test_idxs = np.where(y_preds != y_test)[0]

    # Print mislabeled examples
    fig = plt.figure(figsize=(25, 4), num = "Incorrect predictions")
    for i, idx in enumerate(bad_test_idxs):
        ax = fig.add_subplot(2, np.ceil(len(bad_test_idxs) / 2), i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(x_test[idx]))
        ax.set_title("{} (pred: {})".format(labels[y_test[idx]], labels[y_preds[idx]]))

    model.save("models/asl_1")
    plt.show()
    """

    # Make graphs of the accuracy and loss for the model as it is trained
    # Plot for accuracy over time(epochs) while trianing the model
    plt.plot(fit_model.history["accuracy"])
    plt.plot(fit_model.history["val_accuracy"])
    plt.title("Model Accuracy over Epochs")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Training", "Testing"], loc="upper left")
    plt.savefig("graphs/" + model_name + " Accuracy.png")
    plt.show()

    # Plot for loss over time(epochs) while training the model
    plt.plot(fit_model.history["loss"])
    plt.plot(fit_model.history["val_loss"])
    plt.title("Model Loss over Epochs")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Training", "Testing"], loc="upper left")
    plt.savefig("graphs/" + model_name + " Loss.png")
    plt.show()

# Build, train, test and save the model
train()

# Train a model on the data


import cv2
import tensorflow as tf
import os
import pickle



# Remove any TF log outputs (e.g. CPU supporting stuff)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Build a model
# Takes nothing as imput, returns a model
def build_asl_model():
    # Build the model
    A = pickle.load(open("A.pickle", "rb"))
    B = pickle.load(open("B.pickle", "rb"))
    C = pickle.load(open("C.pickle", "rb"))
    D = pickle.load(open("D.pickle", "rb"))




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



build_asl_model()

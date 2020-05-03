import os
import tensorflow as tf

# Remove any TF log outputs (e.g. CPU supporting stuff)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Build the CNN
# Takes nothing as input, returns a model
def build_cnn():
    model = tf.keras.Sequential()
    # Convolutional Layer
    model.add(tf.keras.layers.Conv2D(64, (3,3), input_shape = A.shape[1:]))
    model.add(tf.keras.layers.Activation("relu"))
    # Pooling
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    # MORE LAYERS
    model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    # Flatten and Dense the layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation("sigmoid"))
    # Compile the model
    model.compile(loss = "binary_crossentropy", # Maybe categorical_crossentropy?
                  optimizer = "adam",
                  metrix = ['accuracy'])

    print("Finished building the CNN")
    return model

# Fit the model to the images
def fit_cnn():
    model

build_cnn()

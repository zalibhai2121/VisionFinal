import os
import tensorflow as tf

# Remove any TF log outputs (e.g. CPU supporting stuff)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Build the CNN
def build_cnn():
    model = tf.keras.Sequential()
    # Convolutional Layer
    model.add(tf.keras.layers.Conv2D(64, (3,3), input_shape = A.shape[1:]))
    model.add(tf.keras.layers.Activation("relu"))
    # Pooling
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    # MORE LAYERS
    model.add()

build_cnn()

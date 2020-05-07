import random
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isdir, join


def load_data(container_path='dataset/', folders = ['A', 'B', 'C'],
              size=2000, test_split=0.2, seed=0):
    """
    Loads sign language dataset.
    """

    names, labels = [], []

    for label, folder in enumerate(folders):
        folder_path = join(container_path, folder)
        images = [join(folder_path, d)
                  for d in sorted(listdir(folder_path))]
        labels.extend(len(images) * [label])
        names.extend(images)

    random.seed(seed)
    data = list(zip(names, labels))
    random.shuffle(data)
    data = data[:size]
    names, labels = zip(*data)

    # Get the images
    x = tensor_paths(names).astype('float32') / 255
    # Store the one-hot targets
    y = np.array(labels)

    x_train = np.array(x[:int(len(x) * (1 - test_split))])
    y_train = np.array(y[:int(len(x) * (1 - test_split))])
    x_test = np.array(x[int(len(x) * (1 - test_split)):])
    y_test = np.array(y[int(len(x) * (1 - test_split)):])

    return (x_train, y_train), (x_test, y_test)


def tensor_path(img_path, size):
    # loads RGB image as PIL.Image.Image type
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(size, size))
    # convert PIL.Image.Image type to 3D tensor
    x = tf.keras.preprocessing.image.img_to_array(img)
    # convert 3D tensor to 4D tensor
    return np.expand_dims(x, axis=0)


def tensor_paths(img_paths, size=50):
    tensor_list = [tensor_path(img_path, size) for img_path in img_paths]
    return np.vstack(tensor_list)

load_data()
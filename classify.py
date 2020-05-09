import random
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isdir, join


def load_data():
    """
    :return:
    """
    container_path = 'dataset2/'
    folders = ['A', 'B', 'C']
    size = 2000
    test_split = 0.2
    seed = 0
    names = []
    labels = []

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


def tensor_converter(img_path, size):
    """
    Given an RBG image, converts that image to a 3D tensor, and then turns that 3D tensor into a 4D tensor
    :param img_path:
    :param size:
    :return: a 4D tensor
    """
    # loads RGB image as PIL.Image.Image type
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(size, size))
    # convert PIL.Image.Image type to 3D tensor
    x = tf.keras.preprocessing.image.img_to_array(img)
    # convert 3D tensor to 4D tensor
    return np.expand_dims(x, axis=0)


def tensor_paths(img_paths):
    """
    Gives the list of tensor paths
    :param img_paths: the paths of the images
    :return: The list of tensor paths
    """
    size = 64
    tensor_list = [tensor_converter(img_path, size) for img_path in img_paths]

    return np.vstack(tensor_list) #np.vstack stacks arrays vertically (by row)

load_data()

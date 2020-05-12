import random
import numpy as np
import tensorflow as tf
import os
from os import listdir
from os.path import isdir, join

#https://github.com/pfoy/ASL-Recognition-with-Deep-Learning/blob/master/notebook.ipynb
def load_data():
    """
    Loads the images
    TODO: maybe switch names and labels to be numpy arrays?
    :return: Tuples of Numpy Arrays
    """
    container = 'dataset2/' #path of the container where the images are located
    folders = ['A', 'B', 'C'] #the names of the folders - each folder is given as a letter
    data_size = 2000 #the size of the data that we want
    test_split = 0.2 #fraction of the data to reserve as a test set
    names = [] #a list of the names of the images
    labels = [] # the labels for the images

    for label, folder in enumerate(folders):

        #for the path for the folder
        path = join(container, folder)
        #images = [join(folder_path, d) for d in sorted(listdir(path))]
        for pic in sorted(os.listdir(path)):
            images = [join(path, pic)]

        labels.extend(len(images) * [label]) #extend list of labels by appending elements from an iterable (label)
        names.extend(images)

    #random seed for shuffling the data before computing the test split
    random.seed(0)
    #data = list(zip(names, labels))
    #random.shuffle(data) #possibly delete later
    #data = data[:data_size]
    #names, labels = zip(*data)
    indices = np.arange(len(names))
    random.shuffle(indices) #maybe put np in front of it if it doesn't work
    names = names[indices]
    labels = labels[indices]

    x = tensor_paths(names).astype('float32') / 255 #obtain the images
    y = np.array(labels) #store the labels in a numpy array

    x_train = np.array(x[:int(len(x) * (1 - test_split))]) #training samples
    y_train = np.array(y[:int(len(x) * (1 - test_split))]) #training samples
    x_test = np.array(x[int(len(x) * (1 - test_split)):]) #testing samples
    y_test = np.array(y[int(len(x) * (1 - test_split)):]) #testing samples

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

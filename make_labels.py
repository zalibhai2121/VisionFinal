"""
This program iterates over the dataset of hands and creates a dictionary
of key value pairs consisting of the file name and the letter it represents

After making the dictionary, it saves it as a .txt file for the NN to use
"""

import os
from typing import Dict

image_directory: str = "dataset"
label_file: str = "labels.txt"

def make_labels():
    labels = dict()
    for filename in os.listdir(image_directory):
        current_filename = filename
        letter = current_filename[:1]
        labels.update({current_filename: letter})
    #os.remove("dataset/labels.txt")
    file = open("dataset/labels.txt", "w")
    file.write(str(labels))
    file.close()

make_labels()

import os
from typing import Dict

image_directory: str = "dataset/"
label_file: str = "labels.txt"

def make_labels():
    """
    This method iterates over the dataset of sign language images,
    creating a dictionary of key value pairs consisting of the file name
    and the letter it represents.

    after making the dictionary, it saves it as a .txt file for the NN to use
    """
    full_labels = dict()
    letters = ['A', 'B', 'C']
    for i in letters:
        labels = dict()
        directory = "dataset/" + i
        for filename in os.listdir(directory):
            if filename.endswith(".png"):
                current_filename = filename
                letter = current_filename[:1]
                labels.update({current_filename: letter})
                full_labels.update({current_filename: letter})
        # Write names of files in each letter
        file = open(directory + "/labels.txt", "w")
        file.write(str(labels))
        file.close()
    # Write all the files in the dataset folder
    file = open(image_directory + "/labels.txt", "w")
    file.write(str(full_labels))
    file.close()

make_labels()

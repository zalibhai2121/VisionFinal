from PIL import Image
import os, sys

# Handly little function to iterate through each of the letter folders and perform some operation on each image

path = "dataset/"
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']


def resize():
    count = 0
    for letter in os.listdir(path):
        for photo in os.listdir(path+letter):
            print("Processing %s" % photo)
            image_name = path+letter+"/"+photo
            image = Image.open(os.path.join(path+letter+"/", photo))
            # Change after this line to do the work on the images! Everything before is to get to the images
            new_dimensions = (64,64)
            output = image.resize(new_dimensions, Image.ANTIALIAS)
            output.save(image_name, "PNG", quality = 95)

resize()

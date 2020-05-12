from PIL import Image
import os, sys, shutil, random

# Handly little function to iterate through each of the letter folders and perform some operation on each image

path = "dataset/"
destination = "dataset2/"

def resize():
    for letter in os.listdir(path):
        count = 0
        while count < 200:
            photo = random.choice(os.listdir(path+letter+"/"))
            print("Processing %s" % photo)
            image_name = path+letter+"/"+photo
            # Change after this line to do the work on the images! Everything before this line is to get to the images
            shutil.copy(image_name, destination+letter+"/")
            count += 1

resize()

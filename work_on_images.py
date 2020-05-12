from PIL import Image
import os, sys
import cv2

# Handly little function to iterate through each of the letter folders and perform some operation on each image

path = "test_dataset2/"


def resize():
    for letter in os.listdir(path):
        for photo in os.listdir(path+letter):
            #print("Processing %s" % photo)
            image_name = path+letter+"/"+photo
            image = Image.open(os.path.join(path+letter+"/", photo))
            # Change after this line to do the work on the images! Everything before is to get to the images
            img = cv2.imread(path+letter+"/"+photo)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(image_name, img_gray)

resize()

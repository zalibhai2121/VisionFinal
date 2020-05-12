# To-do list
# Function to get camera input and save it as a png constantly
# Function to take that png and run it through the nn to get a letter
# Camera function above to get that letter and display it on the screen
import tensorflow as tf
import cv2
import numpy as np
import os
from PIL import Image
from keras.preprocessing import image

# Supress all non-error tf messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# Model to be used in this instance:
model_name = "models/asl_4"
model = tf.keras.models.load_model(model_name)


# Function to get the camera input and save it as a png
def load_and_run_webcam():
    # Get camera input
    cap = cv2.VideoCapture(0)
    # Where to put the rectangle
    left_top = (50, 100)
    right_bottom = (250, 300)
    border = (255, 255, 255)
    while(True):
        # Capture frame by frame
        ret, frame = cap.read()
        # Do operations on the frame here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.rectangle(frame, left_top, right_bottom, border, 2)
        # Crop the image to the rectangle to pass to the network, save it
        crop = frame[100:300, 50:250]
        crop = cv2.resize(crop, (64, 64))
        cv2.imwrite("Camera.png", crop)
        # Call the network to classify the image saved, the function returns a letter
        letter = get_letter()
        print(letter)
        image = cv2.putText(frame, letter, (260,310), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255,0,0), 5, cv2.LINE_AA)

        # Display the original and edited frame
        cv2.imshow("ASL", frame)
        cv2.imshow("Cropped", crop)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# Function to get the letter from the image sent to it
def get_letter():
    camera = image.load_img("Camera.png")
    camera = image.img_to_array(camera)
    camera = np.expand_dims(camera, axis = 0)
    letter = model.predict(camera)
    if letter[0][0] == 1:
           return 'A'
    elif letter[0][1] == 1:
           return 'B'
    elif letter[0][2] == 1:
           return 'C'
    elif letter[0][3] == 1:
           return 'D'
    elif letter[0][4] == 1:
           return 'E'
    elif letter[0][5] == 1:
           return 'F'
    elif letter[0][6] == 1:
           return 'G'
    elif letter[0][7] == 1:
           return 'H'
    elif letter[0][8] == 1:
           return 'I'
    elif letter[0][9] == 1:
           return 'J'
    elif letter[0][10] == 1:
           return 'K'
    elif letter[0][11] == 1:
           return 'L'
    elif letter[0][12] == 1:
           return 'M'
    elif letter[0][13] == 1:
           return 'N'
    elif letter[0][14] == 1:
           return 'O'
    elif letter[0][15] == 1:
           return 'P'
    elif letter[0][16] == 1:
           return 'Q'
    elif letter[0][17] == 1:
           return 'R'
    elif letter[0][18] == 1:
           return 'S'
    elif letter[0][19] == 1:
           return 'T'
    elif letter[0][20] == 1:
           return 'U'
    elif letter[0][21] == 1:
           return 'V'
    elif letter[0][22] == 1:
           return 'W'
    elif letter[0][23] == 1:
           return 'X'
    elif letter[0][24] == 1:
           return 'Y'
    elif letter[0][25] == 1:
           return 'Z'


load_and_run_webcam()

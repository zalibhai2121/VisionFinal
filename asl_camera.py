import tensorflow as tf
import cv2
import numpy as np

# Function to load a model and predict the camera input
def load_and_run_webcam(model):
    model = tf.keras.models.load_model(model)
    cap = cv2.VideoCapture(0)
    # Where to put the rectangle
    left_top = (50, 100)
    right_bottom = (250, 300)
    border = (255, 0, 0)
    while(True):
        # Capture frame by frame
        ret, frame = cap.read()
        # Do operations on the frame here including getting a prediction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.rectangle(gray, left_top, right_bottom, border, 2)
        h, w = gray.shape
        # Crop the image to the rectangle to pass to the network
        crop = gray[100:300, 50:250]
        tf_img = tf.reshape(crop, [50, 50, 16])
        # Predict the letter
        p = model.predict_classes(np.asarray([tf_img], dtype=np.float32), batch_size=1)[0]


        # Display the edited frame
        cv2.imshow("ASL", gray)
        cv2.imshow("Cropped", crop)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

load_and_run_webcam("models/asl_1")

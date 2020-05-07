import tensorflow as tf
import cv2

# Function to load a model and predict the camera input
def load_and_run_webcam(model):
    model = tf.keras.models.load_model(model)
    cap = cv2.VideoCapture(0)
    w, h = 200, 200
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    width = int(width)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    height = int(height)
    # Where to put the rectangle
    left_top = (width-w, 0)
    right_bottom = (width, h)
    color = (255, 0, 0)
    while(True):
        # Capture frame by frame
        ret, frame = cap.read()

        # Do operations on the frame here
        cv2.rectangle(frame, left_top, right_bottom, color, 2)
        cv2.flip(frame, 1)

        # Display the edited frame
        cv2.imshow("ASL", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

load_and_run_webcam("models/asl_1")

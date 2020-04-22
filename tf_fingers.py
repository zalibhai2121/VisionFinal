import tensorflow as tf
import numpy as np
import cv2
import os
import random
import time
from multiprocessing import Process, Lock, Queue, current_process
#2020visons class file
def read_labels_and_images():
    start_time = time.time()
    filepath = "fingers/"
    # Read the label file, mapping image filenames to (x, y) pairs for UL corner
    with open(filepath + "labels") as label_file:
        label_string = label_file.readline()
        label_map = eval(label_string)
    print("Read label map with", len(label_map), "entries")

    # Read the directory and make a list of files
    filenames = []
    for filename in os.listdir(filepath):
        if random.random() < -0.95: continue # to select just some files
        if filename.find("png") == -1: continue # process only images
        filenames.append(filename)
    print("Read", len(filenames), "images.")

    # Extract the features from the images
    print("Extracting features")
    train, train_labels, predict, predict_labels = [], [], [], []
    processes = []     # Hold (process, queue) pairs
    num_processes = 8  # Process files in parallel
    lock = Lock()      # Protect access to the shared lists
    queues = {}        # For getting results from child processes

    # Launch the processes
    for start in range(num_processes):
        q = Queue()
        files_for_one_process = filenames[start:len(filenames):num_processes]
        file_process = Process(target=process_files, \
                        args=(filepath, files_for_one_process, label_map, q))
        processes.append((file_process, q))
        file_process.start()

    # Wait for processes to finish, combine their results
    for p, q in processes: # For each process and its associated queue
        result = q.get()            # Blocks until the item is ready
        train += result[0]          # Get training features from child
        train_labels += result[1]   # Get training labels from child
        predict += result[2]        # Get prediction features from child
        predict_labels += result[3] # Get prediction labels from child
        p.join()                    # Wait for child process to finish
    print("Done extracting features from images. Time =", time.time() - start_time)

    return (train, train_labels, predict, predict_labels)


# Child process, intended to be run in parallel in several copies
#  files      our portion of the files to process
#  label_map  contains the right answers for each file
#  q          a pipe for sending results back to the parent process
def process_files(filepath, files, label_map, q):
    np.random.seed(current_process().pid) # Each child gets a different RNG
    t, tl, p, pl = [], [], [], []
    checkpoint, check_interval, num_done = 0, 5, 0
    for filename in files:
        if 100*num_done > (checkpoint + check_interval) * len(files):
            checkpoint += check_interval
            print((int)(100 * num_done / len(files)), "% done")
        num_done += 1
        img = cv2.imread(filepath + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = tf.reshape(img, [64, 64, 1])
        for contrast in [1, 1.2, 1.5]:
            img2 = tf.image.adjust_contrast(img, contrast_factor=contrast)
            #print("*** 3")
            for bright in [0.0, 0.1, 0.2]:
                img3 = tf.image.adjust_brightness(img2, delta=bright)
                #cv2.imshow("Augmented Image" + str(current_process().pid), img3.numpy())
                #cv2.waitKey(1)
                if np.random.random() < 0.8: # 80% of images 
                    t.append(img3.numpy()) 
                    tl.append(label_map[filename]-1)
                else:
                    p.append(img3.numpy())
                    pl.append(label_map[filename]-1)
                        
    q.put((t, tl, p, pl))


def build_finger_model():
    # Build the model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(3, activation='softmax')])
    """
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(3, activation='softmax')])
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (5, 5), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(3, activation='softmax')])
    print("Shape of output", model.compute_output_shape(input_shape=(None, 64, 64, 1)))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    print("Done building the network topology.")

    # Read training and prediction data
    train, train_labels, predict, predict_labels = read_labels_and_images()

    # Train the network
    print("Starting to train the network, on", len(train), "samples.")
    model.fit(np.asarray(train, dtype=np.float32), np.asarray(train_labels), epochs=40, batch_size=32, verbose=2)
    model.save("models/fingers_4")
    print("Done training network.")

    # Predict
    p = model.predict_classes(np.asarray(predict))
    for pr, pl in zip(p, predict_labels):
        print("Predict", pr, "\tActual", pl, "***" if (pr != pl) else ".") 
        
    return model


def load_and_run_model(filename):
    model = tf.keras.models.load_model(filename)
    cap = cv2.VideoCapture(1)
    keep_going = True
    while keep_going:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        crop = gray[:,(w-h)//2:(w+h//2)]
        crop = cv2.resize(crop, (64, 64))
        tf_img = tf.reshape(crop, [64, 64, 1])

        p = model.predict_classes(np.asarray([tf_img], dtype=np.float32), batch_size=1)[0]
        #p = model.predict(np.asarray([tf_img], dtype=np.float32))[0] # See the raw data
        
        color = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR),(256, 256))
        cv2.putText(color, str(p+1), (20, 30) , cv2.FONT_HERSHEY_SIMPLEX,\
                        1.0,(255,0, 0),2, lineType=cv2.LINE_AA)
        cv2.imshow("Number of Fingers", color)
        if cv2.waitKey(100) & 0xFF == ord(' '):
            keep_going = False

    cap.release()
    cv2.destroyAllWindows()


# Run this file if you'd like to test your function from a live webcam feed
def capture_and_save_images():
    #Video capture from webcam
    num_each_digit = 1000
    labels = {}
    count = 0
    filepath = "fingers/"
    for label in [1, 2, 3]:
        cap = cv2.VideoCapture(1)
        for i in range(num_each_digit):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            crop = gray[:,(w-h)//2:(w+h//2)]
            crop = cv2.resize(crop, (64, 64))
            name = "finger_" + str(label) + "_" + str(i) + ".png"
            filename = filepath + name            
            cv2.imwrite(filename, crop)
            labels[name] = label
            
            cv2.imshow("Visualizing the Cropped Image", cv2.pyrUp(cv2.pyrUp(crop)))
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
            count += 1
            # Save every 100 images
            if count % 100 == 0:
                with open(filepath + "labels", "w") as label_output_file:
                    label_output_file.write(str(labels))

        # Save after each label
        with open(filepath + "labels", "w") as label_output_file:
            label_output_file.write(str(labels))
        cap.release()
        cv2.waitKey()

    cap.release()
    cv2.destroyAllWindows()


#capture_and_save_images()
#build_finger_model()
load_and_run_model("models/fingers_3")


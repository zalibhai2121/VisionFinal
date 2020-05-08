import cv2
import classify
import matplotlib as plt
import numpy as np
import tensorflow as tf
# To save our own images
def capture_and_save_images():
    #Video capture from webcam


    num_each_digit = 100
    labels = {}
    count = 0
    filepath = "dataset2/"
    for label in ['A', 'B', 'C']:
        cap = cv2.VideoCapture(0)
        for i in range(num_each_digit):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            crop = gray[:,(w-h)//2:(w+h//2)]
            crop = cv2.resize(crop, (64, 64))
            name = label + str(i) + ".png"
            filename = filepath + label + "/" + name
            cv2.imwrite(filename, crop)
            labels[name] = label

            cv2.imshow("Visualizing the Cropped Image", cv2.pyrUp(cv2.pyrUp(crop)))
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            count += 1
        cap.release()
        cv2.waitKey()

    cap.release()
    cv2.destroyAllWindows()


capture_and_save_images()
classify.load_data(container_path='dataset2/', folders = ['A', 'B', 'C'],
              size=2000, test_split=0.2, seed=0)
classify()
def classify():
# Loads data from classify.py that will be used to train the network
    (x_train, y_train), (x_test, y_test) = classify.load_data()

    # The letters we will be training, testing on
    labels = ['A', 'B', 'C']

    # Print the first several training images, along with the labels
    fig = plt.figure(figsize=(20, 5), num = "Some of the training images and labels")
    for i in range(50):
        ax = fig.add_subplot(3, 12, i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(x_train[i]))
        ax.set_title("{}".format(labels[y_train[i]]))

    # Number of each letter in the training dataset
    A_train = sum(y_train == 0)
    B_train = sum(y_train == 1)
    C_train = sum(y_train == 2)

    # Number of each letter in the test dataset
    A_test = sum(y_test == 0)
    B_test = sum(y_test == 1)
    C_test = sum(y_test == 2)

    # Print statistics about the dataset
    print("Training set:")
    print("\tA: {}, B: {}, C: {}".format(A_train, B_train, C_train))
    print("Test set:")
    print("\tA: {}, B: {}, C: {}".format(A_test, B_test, C_test))

    # One-hot encode the training and test labels
    y_train_OH = tf.keras.utils.to_categorical(y_train)
    y_test_OH = tf.keras.utils.to_categorical(y_test)

    # Begin building the model
    model = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=5, kernel_size=5, padding='same', activation='relu',input_shape=(50, 50, 3)),
                                 tf.keras.layers.MaxPooling2D(pool_size=4),
                                 tf.keras.layers.Conv2D(filters=15, kernel_size=5, padding='same', activation='relu'),
                                 tf.keras.layers.MaxPooling2D(pool_size=4),tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(3, activation='softmax')])


    # Summarize the model
    print("Summary of model:")
    model.summary()

    # Compile the model using categorical_crossentropy
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', # binary_crossentropy
                  metrics=['accuracy'])

    # Fit the model to the images
    hist = model.fit(x_train, y_train_OH,
                     validation_split=0.20,
                     epochs=15, # Increase this to improve accuracy
                     batch_size=32)
    print("Shape of output", model.compute_output_shape(input_shape=(None, 64, 64, 1)))

    # Obtain accuracy on test set
    score = model.evaluate(x=x_test,
                           y=y_test_OH,
                           verbose=0)
    print('Test accuracy:', score[1])
    y_probs = model.predict(x_test)

    # Get predicted labels for test dataset
    y_preds = np.argmax(y_probs, axis=1)

    # Indices corresponding to test images which were mislabeled
    bad_test_idxs = np.where(y_preds != y_test)[0]

    # Print mislabeled examples
    fig = plt.figure(figsize=(25, 4), num = "Incorrect predictions")
    for i, idx in enumerate(bad_test_idxs):
        ax = fig.add_subplot(2, np.ceil(len(bad_test_idxs) / 2), i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(x_test[idx]))
        ax.set_title("{} (pred: {})".format(labels[y_test[idx]], labels[y_preds[idx]]))

    model.save('models/asl_1')
    plt.show()

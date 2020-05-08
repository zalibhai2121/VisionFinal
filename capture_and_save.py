import cv2
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

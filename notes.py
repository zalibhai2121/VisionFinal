"""
Model types:
asl_1: Crappy model that I used for Testing
*asl_2: First legit model, trained on color image dataset, 42% accuracy
        Steps = 800
        Epochs = 25
        Validation steps = 1000
asl_3: Second model, trained on color image dataset, 46% accuracy
        Steps = 800
        Epochs = 25
        Validation steps = 10
asl_4: Third model, trained on color image dataset, 50% accuracy
        Steps = 800
        Epochs = 50
        Validation steps = 10
asl_5: Fourth model, trained on greyscale dataset, 37% accuracy (I think it could be better tho)
        Steps = 800
        Epochs = 25
        Validation steps = 20

"""
#####################################################################################################################

"""
Files:
asl_train: train a nn on the dataset, saves a model that can be loaded in asl_camera
asl_camera: loads a trained model and runs it on camera input and returns a predicted letter
capture_and_save: takes photos vis camera and saves them to the dataset
copy_images: copies images from one dataset to another (useful for making duplicates to save progress)
sources: sites we used to help us with issues
work_on_images: handly little function to modify every image in a dataset however you want

Older files:
CameraClassify: from Gabriella's sudoku
classify: same thing
label_keypoints: same thing
make_pics_black_and_white: To turn the dataset into black and white images
make_labels: I forget
black_and_white_for_video: Same as earlier but for video feed
test-camera: tester file while trying new things
visualize_model: old function to visualize the model that was created

Folders:
dataset: Contains color photos of all the images
dataset2: Same as dataset but greyscale instead of color
test_dataset: Used for testing the model, has ~200 photos of each letter pulled from dataset
test_dataset2: Used for testing the model, has ~200 photos of each letter pulled from dataset2
models: All our saved models
graphs: Contains graphs of the models
Unused letters: WHen we were still hopeful we would be able to do things like have spaces and delete letters to make words
"""

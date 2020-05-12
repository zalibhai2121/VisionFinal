# VisionFinal

Building a neural network that recognizes the American Sign Language (ASL) alphabet using Python, Open-CV, and Tensor-Flow.

Note, this project is harder than expected, especially when taking into account similar letters - even for letters that do not require motion. We have enjoyed learning bits of sign language though. Also, even though certain people made certain commits, a lot (and I mean a lot) of ideas are discussed and worked out in our group chat.

Tonight and tomorrow, I am going to look into how other ASL reading programs deal with "j" and "z" and also get some pictures in so we can start classifying the other alphabets

4/24 - the cameraClassify.py can detect a hand when put on the screen, kept the contours may help in classification, but do not know as of yet -- next step is to train data in classify.py file -- z

4/27 - created label maker for the nn, and started training data -- M
--- can use corner detection to help detect what alphabet we are using -- Z

5/02 - pair/mob programming commiting through zainabalibhai

5/7 - Working on adding camera input -- M

5/10 - I just remembered we were supposed to update this as we go.... Guess who forgot??? Whoops --G

5/11 - "Improved" the model  (still has a long way to go), got the camera to work, need some sleep --M


# How it works:
## Two ways to go about this:
### 1. Train the network yourself before classifying
  - If you want to train the nn, please go to line 112 and change the number to save a new model when done. Run asl_train.py and it will begin trining.
  - Depending on the number of layers, epochs, steps, and validation steps, it can take a while
  - When done, 2 graphs will be generated, displayed and saved for you. Proceed to the next step
### 2. Begin classifying your camera input using an already trained nn
  - If you did not train your own nn, move to the next step. Otherwise, please go to line 16 of asl_camera and change the number to the model you just trained earlier
  - Run asl_camera.py. 2 windows will pop up. 1 will be your camera with a box to put your hand in, the other will be the cropped box. Letters will begin displaying with your predicted letter.

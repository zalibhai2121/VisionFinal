"""letter = current_filename[:1]
Moves 5 of each letter from extra_dataset to dataset
Calls the make_labels at the end to make an updated label.txt file
"""

import os
import shutil
from make_labels import make_labels
import os.path

# Set source, destination folders
source: str = "extra_dataset"
destination: str = "dataset" #why is this still here
# Set alphabet as a list
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']


number = 1

def find_lowest():
    """
    Get the lowest number to work from
    ToDo: define number in a smarter way
    """
    global number
    
    if len(os.listdir(source)) == 0:
        print("The extra dataset folder is empty")
    else:
        file = 'A' + str(number) + '.jpg'
        yoohoo = source + "/" + file
        
        if os.path.isfile(yoohoo):
            move_five()
        else:
            number += 5
            find_lowest()
            
    make_labels()


def move_five():
    """
    move 5 files of each letter to the destination folder
    """
    global number
    
    for i in alphabet:
        digit = number
        letter = i
        counter = 0
        
        while counter != 5:
            file = i + str(digit) + '.jpg'
            #print(file)
            shutil.move(source + "/" + file, destination + "/" + file)
            #print("Number: " + str(digit), "Counter: " + str(counter))
            digit += 1
            counter += 1

find_lowest()

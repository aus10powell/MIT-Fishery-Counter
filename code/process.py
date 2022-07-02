"""
Modified from Nils Tijtgat's script, found at: https://timebutt.github.io/static/how-to-train-yolov2-to-detect-custom-objects/
Purpose: Generates train.txt and test.txt files, which contain the paths to training/test images and their annotations.
Modifications: In its current state, this script has been altered to add all images in the given folder to a single train.txt
file, as opposed to a certain percentage being set aside for testing. This was done in order to specifically select a test
set that contained representative numbers of each class, rather than a random set of images that may not sufficient numbers
of all classes. If a random selection is desired, uncomment the appropriate lines.
Directory Constraints: Run from the directory containing both the image files and txt files.
Usage: python process.py
"""

import glob, os

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Directory where the data will reside.
path_data = os.getcwd + "/"

# Percentage of images to be used for the test set. Uncomment if desired.
# percentage_test = 10;

# Create and/or truncate train.txt and, if desired, test.txt.
file_train = open("train.txt", "w")
# file_test = open('test.txt', 'w')

# Populate train.txt and test.txt, if desired.
counter = 1
# index_test = round(100 / percentage_test)
# Change the extension to appropriate image type. Note, however, that, although training works with other image types,
# certain functions (recall, valid) only work with .jpg files. If these functions are desired, you must convert the images.
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))

    # Uncomment if populating test.txt
    # if counter == index_test:
    # counter = 1
    # file_test.write(path_data + title + '.jpg' + "\n")
    # else:
    file_train.write(
        path_data + title + ".jpg" + "\n"
    )  # re-indent if populating test.txt
    # counter = counter + 1

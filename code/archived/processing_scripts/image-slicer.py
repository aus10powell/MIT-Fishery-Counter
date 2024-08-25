"""
Authors: Tzofi Klinghoffer & Caleb Perez
Date: 6/20/2017
Description: Given a set of stereo images saved in the specified directory, each image is
sliced, saving the left and right halves as separate new images. "_01_01" is appended to the filename
for the left image, and "_01_02" is appended to the filename of the right image. Split copies are
saved in both the image directory and the current directory.
Usage: python image-slicer.py
"""

import os
import sys


def main():
    imageDirName = sys.argv[1]

    extensions = [".jpg", ".png"]

    currDir = os.getcwd()
    imageDirPath = currDir + "/" + imageDirName
    for filename in os.listdir(imageDirPath):
        if os.path.splitext(filename)[1] in extensions:
            imagePath = imageDirName + "/" + filename
            os.system("slice-image " + imagePath + " 2")


main()

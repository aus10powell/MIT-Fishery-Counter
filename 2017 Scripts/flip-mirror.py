'''
Authors: Tzofi Klinghoffer & Caleb Perez
Date: 6/20/2017
Description: Assuming image-slicer.py has already been run to split stereo images into their respective
halves, this script flips the image vertically and horizontally, allowing for additional semi-unique
training data. The left half is left untouched.
Usage: python flip-mirror.py
Script must be run from the same directory as the images.
'''

import sys
import os
from PIL import Image, ImageOps

def main():
    #imageDirName = sys.argv[1]

    currDir = os.getcwd()
    #imageDirPath = currDir + "/" + imageDirName

    count = 0
    for filename in os.listdir(currDir):
        if filename[-6:] == "02.png":
            count += 1
            print(filename + " flipped and mirrored. " + count)
            image = Image.open(filename)
            image = ImageOps.flip(image)
            image = ImageOps.mirror(image)
            image.save(filename, "PNG")

main()

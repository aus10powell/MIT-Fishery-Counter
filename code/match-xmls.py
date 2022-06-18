'''
Authors:  Tzofi Klinghoffer, Caleb Perez
Date:     July 14, 2017
Purpose:  Checks that all images have a matching xml annotation file, moving those that do not to
a Removed directory, created in the current directory if it does not already exist. There are assumed
to be two distinct image and xml directories in the current directory, the names of which are taken
as arguments.
Usage:    python match-xmls.py imageDirName xmlDirName
'''

import os
import sys

cwd = os.getcwd()
imageDir = cwd + "/" + sys.argv[1]
xmlDir = cwd + "/" + sys.argv[2]

# Checks for Removed directory, creating one if it doesn't exist.
if not os.path.isdir(cwd + "/Removed"):
    os.system("mkdir Removed")

mvPath = cwd + "/Removed"

# Checks all images for matching xmls.
for filename in os.listdir(imageDir):
    xmlFile = xmlDir + "/" + filename[:-4] + ".xml"
    imageFile = imageDir + "/" + filename
    if not os.path.isfile(xmlFile):
        os.system("mv " + imageFile + " " + mvPath)
        print("No matching xml found for " + filename + ". Moved to different directory.")

# Checks all xmls for matching images.
for filename in os.listdir(xmlDir):
    xmlFile = xmlDir + "/" + filename
    imageFile = imageDir + "/" + filename[:-4] + ".png"
    if not os.path.isfile(imageFile):
        os.system("mv " + xmlFile + " " + mvPath)
        print("No matching image found for " + filename + ". Moved to different directory.")

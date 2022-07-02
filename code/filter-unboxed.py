# Authors:      Tzofi Klinghoffer & Caleb Perez
# Date:         6/28/2017
# Description:  Filter out images from a directory that are not in the list of annotations in a specified directory

import sys
import os


def main():
    labelPath = sys.argv[1]

    currDir = os.getcwd()

    for filename in os.listdir(labelPath):
        print(filename)

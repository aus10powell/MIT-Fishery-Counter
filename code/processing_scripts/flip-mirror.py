"""
Authors: Tzofi Klinghoffer & Caleb Perez
Date: 6/20/2017
Description: Assuming image-slicer.py has already been run to split stereo images into their respective
halves, this script flips the image vertically and horizontally, allowing for additional semi-unique
training data. The left half is left untouched.

Usage: 
    python python code/flip-mirror.py data/rv_boxed_herring/Images  
"""
# Utility
import sys
import os
import glob
from tqdm import tqdm

# Image
from PIL import Image, ImageOps


def main():
    try:
        image_dir_path = sys.argv[1]
        print("Image path:", image_dir_path)
    except:
        print("Default image path is current directory")

    for filename in tqdm(os.listdir(image_dir_path)):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            in_filepath = os.path.join(image_dir_path, filename)
            image = Image.open(in_filepath)
            image = ImageOps.flip(image)
            image = ImageOps.mirror(image)
            image.save(filename, "PNG")


if __name__ == "__main__":
    main()

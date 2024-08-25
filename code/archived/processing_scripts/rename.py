# Author:       Tzofi Klinghoffer
# Date:         5/31/2017
# Description:  This script updates all .xml files in the same directory with a new path and folder name. Run this script from the same directory as the .xml files.
# Syntax:       python rename.py

import os
import sys
import xml.etree.ElementTree as ET


def main():
    

    currDir = os.getcwd()

	for filename in os.listdir(currDir):
        	if not filename.endswith('.xml'): continue
        	fullname = os.path.join(currDir, filename)
		#replacement = os.path.join(path, filename)
        	tree = ET.parse(fullname)
        	root = tree.getroot()
        	#print(root[2].text)
        	#newPath = fullname[:-4] + ".png" #path + "/" + root[1].text + ".png"
            
            # UPDATE THIS PATH TO BE THE CORRECT PATH
            newPath = "C:/Users/sguser/Desktop/images/" +  filename[:-4] + ".png"
        	#print(newPath)
		if root[2].text != newPath:
            # UPDATE THIS STRING TO BE THE CORRECT FOLDER NAME (should be the same as the end of the path)
            root[0].text = "images"
        	root[2].text = newPath
           	print(root[1].text + ".png PATH HAS BEEN UPDATED.")
        	tree.write(fullname)
main()

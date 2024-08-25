# Authors:      Tzofi Klinghoffer, Caleb Perez
# Date:         7/12/2017
# Description:  Remove all sea urchin objects from xml in directory

import os
import sys
import xml.etree.ElementTree as ET


def filterOut(root, filename):
    modified = 0
    for child in root:
        if child.tag == "object":
            if child[0].text == "sea urchin":
                modified = 1
                root.remove(child)
                print("sea urchin object removed from " + filename)
    return modified


def main():
    xmlPath = "C:/Users/sguser/Desktop/test"
    modified = 0

    for filename in os.listdir(xmlPath):
        if os.path.splitext(filename)[1] != ".xml":
            continue
        fullXMLPath = xmlPath + "/" + filename
        print(fullXMLPath)
        tree = ET.parse(fullXMLPath)
        root = tree.getroot()
        modified = filterOut(root, filename)
        while modified == 1:
            modified = filterOut(root, filename)
        tree.write(fullXMLPath)
        modified = 0


main()

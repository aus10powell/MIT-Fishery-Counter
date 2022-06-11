# Author:       Tzofi Klinghoffer
# Date:         5/31/2017
# Description:  Detect overlap in VOC XML files

import sys
import os
import xml.etree.ElementTree as ET

def overlap(x1_min, x1_max, y1_min, y1_max, x2_min, x2_max, y2_min, y2_max):
    #if x1_min == x2_min and x1_max == x2_max and y1_min == y2_min and y1_max == y2_max:
    #    return False
    if max(x1_min, x2_min) <= min(x1_max, x2_max) and max(y1_min, y2_min) <= min(y1_max, y2_max):
        return False
    else:
        return True

def main():
    path = sys.argv[2] #'/Users/Tzofi/Dropbox (MIT)/Vincent/Hollings Scholars/Tzofi Klinghoffer/test'
    object_name = sys.argv[1] #'herring'
    for filename in os.listdir(path):
        if not filename.endswith('.xml'): continue
        fullname = os.path.join(path, filename)
        tree = ET.parse(fullname)
        root = tree.getroot()

        xmins = []
        ymins = []
        xmaxs = []
        ymaxs = []
        for o in root.iter('object'):
            if o.find('name').text == object_name:
                xmins.append(o[4][0].text) 
                ymins.append(o[4][1].text)    
                xmaxs.append(o[4][2].text) 
                ymaxs.append(o[4][3].text)

        result = False
        i = 0
        #print(len(xmins))
        while i < len(xmins)-1:
            z = 1
            while z < len(xmins)-1:
                #print(xmins[i] + " " + xmaxs[i] + " " + ymins[i] + " " + ymaxs[i] + " " + xmins[z] + " " + xmaxs[z] + " " + ymins[z] + " " + ymaxs[z])
                result = overlap(xmins[i], xmaxs[i], ymins[i], ymaxs[i], xmins[z], xmaxs[z], ymins[z], ymaxs[z])
                if result == True:
                    print("OVERLAP: " + fullname)
                    break;
                z += 1
            
            if result == True: break;
            i += 1
main()

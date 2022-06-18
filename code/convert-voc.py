'''
Adapted from voc_label.py, available from https://github.com/pjreddie/darknet
Description: Converts VOC XML annotation files to text files compatible with Darknet. Examines
each XML file in the specified XML directory, extracts the class and bounding box information,
and writes a text file compatible with Darknet for each XML. The text files are saved with the
same filename in the current directory.
Usage: python convert-voc.py XMLDirectoryPath
'''

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import sys

# sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Currently set as default classes - change to match classes in xmls 
classes = ["scallop", "dead scallop", "roundfish", "flounder", "skate"]


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

#def convert_annotation(year, image_id):

# Converts a single annotation from an XML file (in_fileName) to a single annotation in Darknet format, writing to out_fileName.
def convert_annotation(in_fileName, out_fileName):
    in_file = open(in_fileName)
    out_file = open(out_fileName, 'w')
    #in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    #out_file = open('VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

xmlDir = sys.argv[1] 
currDir = os.getcwd()
for filename in os.listdir(xmlDir):
    # print(currDir + filename[:-4] + ".txt")
    convert_annotation(xmlDir + filename, currDir + "/" + filename[:-4] + ".txt")

'''
for year, image_set in sets:
    if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):
        os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
        convert_annotation(year, image_id)
    list_file.close()
'''

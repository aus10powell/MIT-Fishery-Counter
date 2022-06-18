# Authors:	    Tzofi Klinghoffer & Caleb Perez
# Date: 	    6/16/2017
# Description:	Runs YOLO Darknet on a directory of images and counts fish. Pass in a data file, cfg file, weights file, and directory of images,
#               each predictions image is saved to darknet/predictions folder, yolo-log.txt file has complete log of predicted classes and
#               confidences for each image
# Usage:        python count-fish.py image_folder_path data_file_path cfg_file_path weights_path threshold

import sys
import os

# Returns integer value for confidence
def cleanConfidence(confidence):
    return int(confidence.replace("%", "").replace(" ", ""))

# Parses YOLO output into two arrays: one for labels and one for confidence
def parseOutput(path, labels, newLabels, confidences, newConfidences):
    count = 0
    with open(path, 'rU') as f:
        for line in f:
            if count == 0:
                count += 1
                continue
            temp = line.split(":")
            labels.append(temp[0])
            newLabels.append(temp[0])
            confidences.append( str(cleanConfidence(temp[1])) )            
            newConfidences.append( str(cleanConfidence(temp[1])) )            
    return labels, newLabels, confidences, newConfidences

# Increases the count whenever a label is found that is expected
def countLabels(labels, expected):
    count = 0
    for label in labels:
        if label in expected:
            count += 1
    return count

# Appends information to log file
def appendToLog(message, log):
    f = open(log, 'a')
    f.write(message)
    f.close()

# Main Program: runs YOLO Darknet on all images and increments count
def main():
    # USER VARIABLES - modify in EXCEPT clause
    #try:
    #    expected = sys.argv[3]
    #    expected = expected.split(",")
    #except IndexError:
    #    expected = ["dog", "feline"]

    # Modify with classes you wish to count
    expected = ["scallop", "dead scallop", "roundfish", "flatfish", "skate"]

    # initialization

    imageDirName = sys.argv[1]
    dataPath = sys.argv[2]
    cfgPath = sys.argv[3]
    weightsPath = sys.argv[4]
    threshold = sys.argv[5]
    outputFile = "yolo-output.txt"
    labels = []
    newLabels = []
    confidences = []
    newConfidences = []
    log = "yolo-log.txt"

	# obtain the directory containing images
    currDir = os.getcwd()
    imageDirPath = currDir + "/" + imageDirName
    log = currDir + "/" + log

    # file extensions that will be included
    extensions = [".jpg", ".png"]
	
	# create predictions picture folder
    if not os.path.isdir(currDir + "/predictions"):
        os.system("mkdir predictions")

    # create log file
    f = open(log, 'w')
    f.close()

    # iterate directory
    for filename in os.listdir(imageDirPath):
        if os.path.splitext(filename)[1] in extensions:
            newConfidences = []
            newLabels = []
            imagePath = imageDirName + "/" + filename
            
            ### IF RUNNING ON WINDOWS, CHANGE ./darknet TO darknet.exe:
            os.system("./darknet detector test " + dataPath + " " + cfgPath + " " + weightsPath + " " + imagePath + " -thresh " + threshold + " > " + outputFile)
            
            labels, newLabels, confidences, newConfidences = parseOutput(outputFile, labels, newLabels, confidences, newConfidences)
            count = countLabels(labels, expected)
            os.system("cp predictions.png ./predictions/predictions_"+ filename)
            appendToLog(filename + ": " + "\n" + str(newLabels) + "\n" + str(newConfidences) + "\n" + str(count) + "\n\n", log)
            
            # UNCOMMENT FOLLOWING COMMENTS FOR DEBUGGING: 
            #print(labels)
            #print(confidences)
            #print(count)

main()

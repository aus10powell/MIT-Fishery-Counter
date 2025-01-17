"""
Authors: Tzofi Klinghoffer, Caleb Perez
Date: 7/20/2017
Description: Calculate recall and precision values by class using the results files generated by the
"detector valid" function. Assumes the results files are placed in the same directory as this script.
Values are generated across 100 thresholds, from 0.01 to 1. Threshold, precision, and recall values
are saved into separate csv files for each class.
Usage: python calculate-recalls.py txtDirectoryPath
"""

import os
import sys


# Defines ground truth detections of a single class for each image based on each Darknet text file in the specified txtDirectoryPath.
# Returns a dictionary pairing the ground truth number of the specified className with each image.
def generateTrueDict(className, txtDir):
    # Associates each class with its class number -- currently set to default classes and class numbers
    classDict = {
        "scallop": "0",
        "dead scallop": "1",
        "roundfish": "2",
        "flatfish": "3",
        "skate": "4",
    }
    classNum = classDict[className]
    trueDict = dict()
    objectCount = 0

    for filename in os.listdir(txtDir):
        if not filename.endswith(".txt"):
            continue
        f = open(txtDir + "/" + filename, "r")
        prefix = filename[:-4]
        count = 0
        for line in f:
            if line[0] == classNum:
                count += 1
        trueDict[prefix] = count
        objectCount += count

    return trueDict, objectCount


# Calculates precision and recall for the given className, threshold, using the files in the given txtDirectoryPath.
# Returns precision and recall values. If no detections are made at a given threshold, "ERROR" is returned as precision.
def calculateResults(className, thresh, txtDir):
    fileDict = dict()
    lineContents = []

    almostTrueDict, objectCount = generateTrueDict(className, txtDir)
    trueDict = dict()

    notCounted = 0
    allPositives = 0
    with open("comp4_det_test_" + className + ".txt", "r") as resultsFile:
        for line in resultsFile:
            lineContents = line.split(" ")
            if lineContents[0] in fileDict and float(lineContents[1]) >= thresh:
                fileDict[lineContents[0]] += 1
                allPositives += 1
                trueDict[lineContents[0]] = almostTrueDict[lineContents[0]]
            elif float(lineContents[1]) >= thresh:
                fileDict[lineContents[0]] = 1
                allPositives += 1
                trueDict[lineContents[0]] = almostTrueDict[lineContents[0]]
            else:
                notCounted += 1
                # fileDict[lineContents[0]] = 0

    truePositives = 0
    for key, basePositives in trueDict.iteritems():
        positivesDetected = fileDict[key]
        truePositives += min(basePositives, positivesDetected)

    recall = float(truePositives * 100) / float(objectCount)
    precision = 0
    if allPositives == 0:
        precision = "ERROR"
    else:
        precision = float(truePositives * 100) / float(allPositives)

    # print("RECALL for " + className + " is: " + str(float(truePositives * 100) / float(objectCount)) + "%")
    # print("PRECISION for " + className + " is: " + str(float(truePositives * 100)/ float(allPositives)) + "%")
    return recall, precision


def main():
    txtDir = sys.argv[1]

    # Assumes default classes
    classList = ["skate", "roundfish", "flatfish", "scallop", "dead scallop"]

    for className in classList:
        classFile = open(className.replace(" ", "_") + "_results.csv", "w")
        thresh = 0.01
        while thresh < 1.01:
            recall, precision = calculateResults(className, thresh, txtDir)
            classFile.write(
                str(thresh) + "," + str(recall) + "," + str(precision) + "\n"
            )
            thresh += 0.01
        classFile.close()


main()

# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python3 object_detection_yolo.py --video=run.mp4 --device 'cpu'
#                 python3 object_detection_yolo.py --video=run.mp4 --device 'gpu'
#                 python3 object_detection_yolo.py --image=bird.jpg --device 'cpu'
#                 python3 object_detection_yolo.py --image=bird.jpg --device 'gpu'
# python3 test2.py --image=/Users/apowell/Downloads/HerringInTrap.JPG --device 'cpu'
# python3 test2.py --video=/Users/apowell/Downloads/sampleFull.avi --device 'cpu'
# sampleFull.avi

import argparse
import os.path
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Initialize the parameters
confThreshold = 0.1  # Confidence threshold
nmsThreshold = 0.1  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

default_image = "/Users/apowell/Downloads/HerringInTrap.JPG"
default_video = "/Users/apowell/Downloads/2_2018-04-27_15-50-53.mp4"

parser = argparse.ArgumentParser(description="Object Detection using YOLO in OPENCV")
parser.add_argument(
    "--device", default="cpu", help="Device to perform inference on 'cpu' or 'gpu'."
)
parser.add_argument("--image", help="Path to image file.")
parser.add_argument("--video", default=default_video, help="Path to video file.")
args = parser.parse_args()

# Load class name
classes = "Herring"

# Give the configuration and weight files for the model and load the network using them.
config_path = "../../weights/weights_and_config_files/Herring/herring.cfg"
weights_path = "../../weights/weights_and_config_files/Herring/herring_final.weights"
modelConfiguration = config_path
modelWeights = weights_path

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

if args.device == "cpu":
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    print("Using CPU device.")
elif args.device == "gpu":
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    print("Using GPU device.")


def getOutputsNames(net):
    """Get the names of the output layers"""
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]


def drawPred(classId, conf, left, top, right, bottom):
    """Draw the predicted bounding box"""
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = "%.2f" % conf

    # Get the label for the class name and its confidence
    if classes:
        assert classId < len(classes)
        label = "%s:%s" % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(
        frame,
        (left, top - round(1.5 * labelSize[1])),
        (left + round(1.5 * labelSize[0]), top + baseLine),
        (255, 255, 255),
        cv.FILLED,
    )
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


def postprocess(frame, outs):
    """Remove the bounding boxes with low confidence using non-maxima suppression"""
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                print("confidence", confidence, "confThreshold", confThreshold)
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    # print('indices:',indices)
    # print('boxes:',boxes)
    for i in indices:
        i = i
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)


# Process inputs
winName = "Deep learning object detection in OpenCV"
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"
# args.image = file = "/Users/apowell/Downloads/Herring in trap Soules Pond (5).JPG"
if args.image:
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    # cap = cv.VideoCapture(args.image)
    cap = cv.imread(args.image)
    outputFile = args.image[:-4] + "_yolo_out_py.jpg"
elif args.video:
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4] + "_yolo_out_py.avi"
else:
    # Webcam input
    cap = cv.VideoCapture(0)
    print("Capture video from built-in camera")


if __name__ == "__main__":

    # Get the video writer initialized to save the output video
    if not args.image:
        vid_writer = cv.VideoWriter(
            outputFile,
            cv.VideoWriter_fourcc("M", "J", "P", "G"),
            30,
            (
                round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                round(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
            ),
        )

    while cv.waitKey(1) < 0:

        # get frame from the video
        hasFrame, frame = cap.read()
        if hasFrame:
            cv.namedWindow("frame", cv.WINDOW_AUTOSIZE)
            cv.imshow("Frame", frame)
            # print()

        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            print("Output file is stored as ", outputFile)
            cv.waitKey(3000)
            # Release device
            cap.release()
            break

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(
            frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False
        )

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))
        # Remove the bounding boxes with low confidence
        postprocess(frame, outs)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = "Inference time: %.2f ms" % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # Write the frame with the detection boxes
        if args.image:
            cv.imwrite(outputFile, frame.astype(np.uint8))
        else:
            vid_writer.write(frame.astype(np.uint8))

        cv.imshow(winName, frame)

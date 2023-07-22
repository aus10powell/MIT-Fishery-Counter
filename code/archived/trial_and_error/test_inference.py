# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python3 object_detection_yolo.py --video=run.mp4 --device 'cpu'
#                 python3 object_detection_yolo.py --video=run.mp4 --device 'gpu'
#                 python3 object_detection_yolo.py --image=bird.jpg --device 'cpu'
#                 python3 object_detection_yolo.py --image=bird.jpg --device 'gpu'
# python3 test2.py --image=/Users/apowell/Downloads/HerringInTrap.JPG --device 'cpu'
# python3 test2.py --video=/Users/apowell/Downloads/sampleFull.avi --device 'cpu'
# sampleFull.avi

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import matplotlib.pyplot as plt
from test_utils import *


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
modelConfiguration = "../../weights/weights_and_config_files/Herring/herring.cfg"
modelWeights = "../../weights/weights_and_config_files/Herring/herring_final.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

outputFile = "yolo_out_py.avi"
if args.device == "cpu":
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    print("Using CPU device.")
elif args.device == "gpu":
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    print("Using GPU device.")

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
        outs = net.forward(getOutputsNames(net=net, frame=frame))
        # Remove the bounding boxes with low confidence
        postprocess(
            frame=frame,
            outs=outs,
            confThreshold=confThreshold,
            nmsThreshold=nmsThreshold,
            classes=classes,
        )

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

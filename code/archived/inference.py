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
import os
import matplotlib
import streamlit as st

matplotlib.use("Agg")
from inference_utils import *
from PIL import Image, ImageOps
import logging

# Custom
from centroidtracker import CentroidTracker

# Set default static images for testing while working locally
DEFAULT_IMAGE = "/Users/apowell/Downloads/HerringInTrap.JPG"
DEFAULT_VIDEO = "/Users/apowell/Downloads/sampleFull.avi"
YOUTUBE = "https://www.youtube.com/watch?v=CbB7vl_HUbU&ab_channel=AustinPowell"


def main(input_file=None, is_image=False, device="cpu"):
    """
    Run main inference script. Returns annotated frames from inference and counts of fish.

    Args:
        - input_file: image or video file input from OpenCV
        - is_image: Binary denoting single image
        - device: CPU or GPU processing
    """
    ## Initialize the parameters
    # Confidence threshold
    conf_threshold = 0.5
    # Non-maximum suppression threshold (maximum bounding box)
    nms_threshold = 0.05
    input_width = 416  # Width of network's input image
    input_height = 416  # Height of network's input image

    # Generic name assignment for output file
    outputFile = "yolo2_out_py.mp4"
    # Load class name
    classes = "Herring"
    # Give the configuration and weight files for the model and load the network using them.
    modelConfiguration = "herring.cfg"
    modelWeights = "herring_final.weights"

    # Centroid tracker to Id specific objects (NOTE: This is temporary and not fully tested)
    tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)

    # Process inputs
    if (
        type(input_file) == cv.VideoCapture
    ):  # Video objects passed from something like Streamlit
        cap = input_file
    elif type(input_file) == str:  # For local uploads
        cap = cv.VideoCapture(input_file)
        logging.info("INFO: Loading file locally: {}".format(input_file))
    else:
        sys.exit(
            "Input file is of type {} and not solved for.".format(type(input_file))
        )

    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

    # Get the video writer initialized to save the output video
    vid_writer = cv.VideoWriter(
        outputFile,
        cv.VideoWriter_fourcc("M", "J", "P", "G"),
        30,
        (
            round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            round(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
        ),
    )

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv.CAP_PROP_FPS)
    logging.info(
        "INFO: Starting inference process on video frames. Total: {}, fps: {}".format(
            total_frames, video_fps
        )
    )
    timestamps = [cap.get(cv.CAP_PROP_POS_MSEC)]  # Timestamp for frame
    calc_timestamps = [0.0]  # Relative timestamps to first timestamp
    saved_frames = []  # Save CV2 frames
    count_list = []
    while cap.isOpened():

        # Get frame from the video
        hasFrame, frame = cap.read()

        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            print("Output file is stored as ", outputFile)
            # Release device
            cap.release()
            break

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(
            frame, 1 / 255, (input_width, input_height), [0, 0, 0], 1, crop=False
        )

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(get_outputs_names(net=net))

        # Remove the bounding boxes with low confidence
        counts = postprocess(
            frame=frame,
            outs=outs,
            tracker=tracker,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            classes=classes,
        )
        count_list.append(counts)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = "Inference time: %.2f ms" % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # Save frame
        saved_frames.append(frame.astype(np.uint8))  # )

        # Write the frame with the detection boxes
        if is_image:
            cv.imwrite(outputFile, frame.astype(np.uint8))
        else:
            vid_writer.write(frame.astype(np.uint8))

        timestamps.append(cap.get(cv.CAP_PROP_POS_MSEC))
        calc_timestamps.append(calc_timestamps[-1] + 1000 / video_fps)
    # Calculate time difference for different timestamps
    time_diffs = [
        abs(ts - cts) for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps))
    ]

    with open("your_file.csv", "w") as f:
        for i in range(len(count_list)):
            f.write(
                f"{count_list[i]}, {time_diffs[i+1]}, {timestamps[i]}, {calc_timestamps[i]}\n"
            )

    return saved_frames, count_list, timestamps


if __name__ == "__main__":

    # Script below to enable running pure inference from command line
    file_path = "/Users/apowell/Downloads/2_2018-04-27_15-50-53.mp4"
    saved_frames, counts, timestamps = main(input_file=file_path)

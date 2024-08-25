"""
Description: 

"""
import argparse
import os.path
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def get_outputs_names(net):
    """Get the names of the output layers.

    Args:
        net: convolution object detection NN
    """
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]


def draw_pred_box(class_id, conf, left, top, right, bottom, frame, classes, count=None):
    """Draw the predicted bounding box. Used in postProcess script.

    Args:
        - class_id: Indices for each of the object classes (e.g. herring)
        - conf: confidence score for box
        - left, top, right, bottom: frame indexes for corners of box
        - frame: frame object
        - classes: different class names
        - count: current count of objects detected
    """
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = "%.2f" % conf
    count_label = "{}".format(count)

    # Get the label for the class name and its confidence
    if classes:
        assert class_id < len(classes)
        label = "%s:%s" % (classes[class_id], label)

    # Display the label at the top of the bounding box
    label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, label_size[1])
    cv.rectangle(
        frame,
        (left, top - round(1.5 * label_size[1])),
        (left + round(1.5 * label_size[0]), top + base_line),
        (255, 255, 255),
        cv.FILLED,
    )
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
    # Draw count in box
    cv.putText(
        img=frame,
        text=count_label,
        org=(right + 20, top + 50),
        fontFace=cv.FONT_HERSHEY_SIMPLEX,
        fontScale=1.75,
        color=(255, 255, 255),
        thickness=1,
    )


def postprocess(frame, outs, conf_threshold, tracker, nms_threshold, classes):
    """Remove the bounding boxes with low confidence using non-maxima suppression.

    Args:
        - frame (frame object from cv2.VideoCapture): Image from video
        - outs (nn layers): Output of the output layers from forward pass

    """
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    class_ids = []
    confidences = []
    boxes = []
    counts = 0
    for out in outs:  # Scan through bounding boxes
        for detection in out:  # Scan through
            scores = detection[5:]

            class_id = np.argmax(scores)  # class with highest score
            confidence = scores[class_id]  # confidence score for class
            if confidence > conf_threshold:
                print(detection)
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                counts += 1

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    print("BOXES", boxes, indices, class_ids)

    max_counts = 0
    for i in indices:
        left, top, width, height = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        # temporary logic for counting all fish from detected boxes
        if counts > max_counts:
            max_counts = counts

        draw_pred_box(
            class_ids[i],
            confidences[i],
            left,
            top,
            left + width,
            top + height,
            frame=frame,
            classes=classes,
            count=counts,
        )

    print("counts", max_counts)
    return max_counts, boxes

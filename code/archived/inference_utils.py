"""
Description: 

Classes and functions needed to make inference
"""
import os.path
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np


class FishCounter(object):
    """Counts fish from single frame image."""

    def __init__(self, model_config, model_weights):
        """model_config and model_weights are paths."""

        self.net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
        self.input_width = 416
        self.input_height = 416
        self.conf_threshold = 0.5  # Confidence threshold
        self.nms_threshold = (
            0.05  # Non-maximum suppression threshold (maximum bounding box)
        )

    def load_image(self, image):
        """Create a 4D blob from a single frame.
        Args:
            - image: path to jpg image
        """
        self.frame = cv2.imread(image)

        blob = cv2.dnn.blobFromImage(
            self.frame,
            1 / 255,
            (self.input_width, self.input_height),
            [0, 0, 0],
            1,
            crop=False,
        )

        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        self.outs = self.net.forward(self._get_outputs_names(net=self.net))

    def _get_outputs_names(self, net):
        """Get the names of the output layers."""
        # Get the names of all the layers in the network
        layersNames = self.net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def process_frame(self, time_inference=False):
        """Runs inference on loaded frame.

        Returns:
            indices:
            boxes: 1D array length 4 of coordinates of bounding box (left,top,width,height)
            class_ids:
        """
        frame_height = self.frame.shape[0]
        frame_width = self.frame.shape[1]

        t0 = time.time()  # Time inference
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        class_ids, confidences, boxes = [], [], []
        for out in self.outs:  # Scan through bounding boxes
            for detection in out:  # Scan through
                scores = detection[5:]

                class_id = np.argmax(scores)  # class with highest score
                confidence = scores[class_id]  # confidence score for class
                if confidence > self.conf_threshold:
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

        # Get counts of detected fish (technically boxes)
        counts = len([b for b in boxes if len(b) > 1])
        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.conf_threshold, self.nms_threshold
        )
        t1 = time.time()
        if time_inference:
            print(f"Inference time on image: {(t1-t0)*100:.2f}")

        self.boxes = boxes

        return boxes, counts, indices, class_ids

    def get_annotated_frame(self):
        """Returns main frame with fish highlighted."""
        frame = self.frame.copy()

        for box in self.boxes:
            left, top, width, height = box[0], box[1], box[2], box[3]
            xmin, ymin, xmax, ymax = left, top, left + width, top + height
            predicted_box = xmin, ymin, xmax, ymax
            cv2.rectangle(
                frame,
                (predicted_box[0], predicted_box[1]),
                (predicted_box[2], predicted_box[3]),
                (178, 255, 255),
                2,
            )
        plt.imshow(frame)

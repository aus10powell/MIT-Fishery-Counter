# inference_modules.py

import cv2
import time
from ultralytics import YOLO
import supervision as sv
import numpy as np
import os
import resource
import logging
from .video_utils import set_logging_level, get_processesor_type

class InferenceCounter:
    def __init__(self, device='cpu', tracker="botsort.yaml", model_path="yolov8s.pt"):
        self.device = device
        self.tracker = tracker
        self.model = self.load_model(model_path)
        self.logger = self.set_logging_level()

    def load_model(self, model_path):
        model = YOLO(model_path)
        return model

    def set_logging_level(self, level=None):
        """
        Set logging level based on the class name.

        Args:
            class_name (str): The name of the class.
        """
        logging_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }

        # Check if the extracted level is valid
        if level in logging_levels:
            logging.basicConfig(level=logging_levels[level])
        else:
            logging.basicConfig(level=logging.INFO)  # Default to INFO level if level is not recognized

        return logging.getLogger(self.__class__.__name__)

    def run_inference(self, video_path, conf_threshold=0.525, stream=True, show=False):
        """
        Run inference on a video file using the YOLO model.

        Args:
            video_path (str): Path to the video file.
            stream (bool, optional): Flag to enable streaming of the processed video. Defaults to False. Needs to be set to True for the video to be displayed.
            show (bool, optional): Flag to display the video window. Defaults to True.

        Returns:
            Tuple: A tuple containing:
                - float: Frame rate of the video.
                - List: List of annotated frames.
                - int: Total count of fish moving out (right to left).
                - int: Total count of fish moving in (left to right).
                - float: Duration of the video in seconds.
                - List: List of relative frame times.
                - List: List of detection counts for each frame.
        """
        assert os.path.exists(video_path), f"Video file not found at '{video_path}'"
        assert os.path.exists(self.tracker), f"Tracker file not found at '{self.tracker}'"
        video = cv2.VideoCapture(video_path)

        _, frame = video.read()

        frame_rate = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / frame_rate
        time_interval = 1.0 / frame_rate

        frame_height, frame_width, _ = frame.shape
        LINE_START = sv.Point(int(frame_width / 3), 0)
        LINE_END = sv.Point(int(frame_width / 3), frame_height)
        line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
        line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.2)
        box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.3)

        parameters = {
            "video_path": video_path,
            "tracker": self.tracker,
            "line_start": LINE_START,
            "line_end": LINE_END,
            "frame_rate": frame_rate,
            "confidence_threshold": conf_threshold,
        }
        self.logger.info("********** Running inference with the following parameters: **********")
        self.logger.info(parameters)
        self.logger.info("-" * 100)

        annotated_frames = []
        frame_detections = []
        relative_frame_times = []

        for frame_index, result in enumerate(
            self.model.track(
                imgsz=frame_width,
                source=video_path,
                device=self.device,
                show=show,
                stream=stream,
                conf=conf_threshold,
                tracker=self.tracker,
            )
        ):
            frame = result.orig_img

            current_frame_time = frame_index * time_interval
            relative_frame_times.append(current_frame_time)

            detections = sv.Detections.from_ultralytics(result)
            if result.boxes.id is not None:
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
            labels = [
                f"tracker_id: {tracker_id} {self.model.model.names[class_id]} {confidence:0.2f}"
                for idx, (bbox, _, confidence, class_id, tracker_id) in enumerate(detections)
            ]
            if len(detections) > 0:
                frame_detections.append(len(detections))
            else:
                frame_detections.append(0)

            frame = box_annotator.annotate(
                scene=frame, detections=detections, labels=labels
            )

            line_counter.trigger(detections=detections)

            line_annotator.annotate(frame=frame, line_counter=line_counter)

            if show:
                cv2.imshow("yolo:", frame)

            annotated_frames.append(frame)

            if cv2.waitKey(30) == 27 and show:
                break

        self.logger.info("*" * 100)
        self.logger.info(
            f"TOTAL FISH OUT: {line_counter.out_count} \t TOTAL FISH IN: {line_counter.in_count} \t NET (moving right to left): {line_counter.out_count - line_counter.in_count}"
        )
        self.logger.info(f"total frame_detections: {sum(frame_detections)}")
        self.logger.info("*" * 100)

        return (
            frame_rate,
            annotated_frames,
            line_counter.out_count,
            line_counter.in_count,
            duration_seconds,
            relative_frame_times,
            frame_detections
        )

if __name__ == "__main__":
    # Define the parameters
    t0 = time.time()
    # Measure memory usage before running the script
    start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Set output dir to write results
    OUTPUT_DIR = os.path.join("/", "Users", "aus10powell", "Downloads")

    logger = set_logging_level(__file__ + ":" + __name__)

    # Get annotated frames and input video frame rate
    video_path =  "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/data/gold_dataset/videos/2_2018-04-14_10-06-19.mp4"
    tracker = "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/code/utils/tracking_configs/botsort.yaml"
    model_path = "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/code/notebooks/runs/detect/train133/weights/best.pt"
    device = get_processesor_type()
    logger.info(f"Running inference on {video_path} using {device} processor.")
    herring_counter = InferenceCounter(device=device, tracker=tracker, model_path=model_path)
    frame_rate, annotated_frames, out_count, in_count, duration_seconds, relative_frame_times, frame_detections = herring_counter.run_inference(video_path=video_path)

    logger.info(f"incount: {in_count} outcount: {out_count} duration: {duration_seconds} frame_rate: {frame_rate}")

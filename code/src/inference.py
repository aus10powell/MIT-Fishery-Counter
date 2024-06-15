import cv2
import logging
import time
from ultralytics import YOLO
import supervision as sv
import numpy as np
import json
import io
import time
import logging
import os
from tqdm import tqdm
import resource
import re
from datetime import timedelta
import inspect


def frames_to_video(frames=None, fps=12):
    """
    Convert frames to video for Streamlit without writing_to_file

    Args:
        frames: frame from cv2.VideoCapture as numpy. E.g. frame.astype(np.uint8)
        fps: Frames per second. Useful if the inference video is compressed to slow down for analysis
    """
    # Grab information from the first frame
    height, width, layers = frames[0].shape

    # Create a BytesIO "in memory file"
    output_memory_file = io.BytesIO()

    # Open "in memory file" as MP4 video output
    output = av.open(output_memory_file, "w", format="mp4")

    # Add H.264 video stream to the MP4 container, with framerate = fps
    stream = output.add_stream("h264", str(fps))

    # Set frame width and height
    stream.width = width
    stream.height = height

    # Set pixel format (yuv420p for better compatibility)
    stream.pix_fmt = "yuv420p"

    # Select low crf for high quality (the price is larger file size)
    stream.options = {"crf": "17"}

    # Iterate through the frames, encode, and write to MP4 memory file
    logger.info("Encoding frames and writing to MP4 format.")
    for frame in tqdm(frames):
        # Convert frame to av.VideoFrame format
        frame = av.VideoFrame.from_ndarray(frame.astype(np.uint8), format="bgr24")

        # Encode the video frame
        packet = stream.encode(frame)

        # "Mux" the encoded frame (add the encoded frame to MP4 file)
        output.mux(packet)

    # Flush the encoder
    packet = stream.encode(None)
    output.mux(packet)

    # Close the output video file
    output.close()

    # Reset the file pointer to the beginning of the memory file
    output_memory_file.seek(0)

    # Return the output memory file
    return output_memory_file


def write_frames_to_file(
    annotated_frames=None, output_video_path="annotated_video.mp4", fps=30
):

    # Get the frame dimensions from the first annotated frame
    height, width, _ = annotated_frames[0].shape
    total_frames = len(annotated_frames)

    # Define the video writer object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Specify the codec

    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write each annotated frame to the video
    for frame in annotated_frames:
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    logger.info(
        f"INFO: Wrote {output_video_path} @ fps = {fps:.3}. Total frames = {total_frames}"
    )


def extract_datetime_from_filename(filename):
  """
  Extracts date and time from a filename with specific format.

  Args:
      filename (str): The filename to extract datetime from.

  Returns:
      datetime.datetime: The extracted datetime object or None if not found.

  Raises:
      ValueError: If the filename format doesn't match the expected pattern.
  """
  pattern = r"(\d+)_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\.(.+)?"
  match = re.search(pattern, filename)

  if match:
    _, date_str, time_str, _ = match.groups()
    datetime_str = f"{date_str} {time_str}"
    try:
      from datetime import datetime
      return datetime.strptime(datetime_str, "%Y-%m-%d %H-%M-%S")
    except ValueError:
      raise ValueError("Invalid date or time format in filename")
  else:
    raise ValueError("Filename format doesn't match expected pattern")


def create_timestamps(relative_frame_times, reference_datetime, format_string="%Y-%m-%d %H:%M:%S.%f"):
  """
  Creates a list of formatted timestamps from relative frame times and a reference datetime.

  Args:
      relative_frame_times (list): A list of frame times in seconds (floats).
      reference_datetime (datetime.datetime): The reference datetime for the video.
      format_string (str, optional): The format string for timestamps (default "%Y-%m-%d %H:%M:%S.%f").

  Returns:
      list: A list of formatted timestamp strings suitable for CSV.
  """
  timestamps = []
  for frame_time in relative_frame_times:
    time_delta = timedelta(seconds=frame_time)
    timestamp = reference_datetime + time_delta
    formatted_timestamp = timestamp.strftime(format_string)
    timestamps.append(formatted_timestamp)
  return timestamps



def write_frame_data_to_csv(frame_detections, relative_frame_times, video_fname, output_dir):
    """
    Write frame detections and relative frame times to a CSV file.

    Args:
        frame_detections (list): List of frame detections.
        relative_frame_times (list): List of relative frame times.
        video_fname (str): Name of the video file.
        output_dir (str): Directory where the CSV file will be saved.
    """
    import csv
    import json 
    output_csv_path = os.path.join(output_dir, f"{video_fname}_detections.csv")

    with open(output_csv_path, mode='w', newline='') as csvfile:
        fieldnames = ['Frame', 'Detection', 'Relative Time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for frame, detection, time in zip(range(len(frame_detections)), frame_detections, relative_frame_times):
            writer.writerow({'Frame': frame, 'Detection': detection, 'Time': time})

    print(f"Frame detections and relative frame times written to: {output_csv_path}")

def write_counts_to_json(data, output_dir):
    """
    Write the fish counts to a JSON file.

    Args:
        data (dict): Dictionary containing the fish counts.
        output_dir (str): Directory where the JSON file will be saved.
    """
    with open(os.path.join(output_dir, "video_counts.json"), "w") as file:
        json_dumps_str = json.dumps(data, indent=4)
        print(json_dumps_str, file=file)

def get_annotated_video_name(video_path):
    """
    Generate annotated video file name from the given video path.

    Args:
        video_path (str): Path to the video file.

    Returns:
        str: Annotated video file name.
    """
    video_fname = video_path.split("/")[-1].split(".")[0] + "_annotated"
    return video_fname

class InferenceCounter:
    def __init__(self, device='cpu', tracker="../botsort.yaml", model_path="yolov8s.pt"):
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

    def run_inference(self, video_path, conf_threshold=0.525, stream=True, show=True):
            """
            Run inference on a video file using the YOLO model.

            Args:
                video_path (str): Path to the video file.
                stream (bool, optional): Flag to enable streaming of the processed video. Defaults to True.
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

            assert os.path.exists(video_path)
            assert os.path.exists(self.tracker)
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
                    for _, _, confidence, class_id, tracker_id in detections
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

                cv2.imshow("yolo:", frame)

                annotated_frames.append(frame)

                if cv2.waitKey(30) == 27:
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

    # Get annotated frames and input video frame rate
    video_path =  "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/data/gold_dataset/videos/2_2018-04-14_10-06-19.mp4"
    tracker = "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/code/botsort.yaml"
    model_path = "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/code/notebooks/runs/detect/train133/weights/best.pt"
    herring_counter = InferenceCounter(device="cpu", tracker=tracker, model_path=model_path)
    frame_rate, annotated_frames, out_count, in_count, duration_seconds, relative_frame_times, frame_detections = herring_counter.run_inference(video_path=video_path)

    video_fname = get_annotated_video_name(video_path)
    output_video_path = os.path.join(
        OUTPUT_DIR, f"{video_fname}.mp4"
    ) 
    write_frames_to_file(
        annotated_frames=annotated_frames,
        output_video_path=output_video_path,
        fps=frame_rate,
    )
    # Extract reference datetime (assuming this logic exists elsewhere)
    reference_datetime = extract_datetime_from_filename(filename=video_path)
    # Create formatted timestamps based on reference datetime and relative times
    formatted_timestamps = create_timestamps(relative_frame_times, reference_datetime)
    write_frame_data_to_csv(frame_detections, formatted_timestamps, video_fname, OUTPUT_DIR)
    # Write number of counted fish to file
    data = {
        "out_count": out_count,
        "in_count": in_count,
        "net_out_count": out_count - in_count,
        "video_fname": video_fname,
        "location": "IRWA",
    }
    write_counts_to_json(data, OUTPUT_DIR)
    t1 = time.time()
    # Measure memory usage after running the script
    end_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Calculate the memory usage
    memory_usage = end_memory - start_memory
    print(f"Memory usage: {memory_usage} bytes")
    runtime = (t1 - t0) / 60
    print(f"Total time: {runtime:.2}")

import inspect
import io
import json
import logging
import os
import re
import resource
import time
from datetime import timedelta

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO


def set_logging_level(filename):
    """
    Set logging level based on the filename.

    Args:
        filename (str): The name of the file being processed.
    """
    # Extract logging level from filename
    logging_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    level = filename.split("_")[-1].split(".")[0].upper()

    # Check if the extracted level is valid
    if level in logging_levels:
        logging.basicConfig(level=logging_levels[level])
    else:
        logging.basicConfig(level=logging.INFO)  # Default to INFO level if level is not recognized

    return logging
# Set logging level based on filename
# logger = set_logging_level(__file__)



## Gold Standard
# One fish
video_path1 = "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/data/gold_dataset/videos/2_2017-04-13_14-10-29.mp4"
video_path1a = "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/data/gold_dataset/videos/2_2018-04-14_10-06-19.mp4"  # Currently detected not counted
video_path2c = "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/data/gold_dataset/videos/2_2018-04-14_10-06-19.mp4"


# Multiple fish swimming sequentally
video_path2b = "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/data/gold_dataset/videos/2_2018-05-10_06-39-30.mp4"

# 3 fish
video_path3a = "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/data/gold_dataset/videos/2_2018-04-27_15-23-03.mp4"


# 1 fish out
# Works with model 105 best
video_path1b = "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/data/gold_dataset/videos/2_2018-04-14_10-06-19.mp4"

# Two fish swimming concurrently
# Works with model 105 best
video_path2 = "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/data/gold_dataset/videos/2_2017-04-13_14-10-29.mp4"  # Current:

# Multiple fish swimming sequentally
video_path2a = "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/data/videos/2_2018-05-10_06-39-30.mp4"


# One fish reversing direction
video_path3 = "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/data/gold_dataset/videos/2_2017-06-04_06-09-56.mp4"

# UNSOLVED

# 2018
video_path4 = "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/data/gold_dataset/videos/2_2017-04-13_13-10-00.mp4"
video_path5 = "/Users/aus10powell/Downloads/RiverHerring/IRWA 2017 Videos/2018 Fish Sightings/2_2018-04-26_15-03-18.mp4"

# Non-herring species
video_path6 = "/Users/aus10powell/Downloads/RiverHerring/IRWA 2017 Videos/2018 Fish Sightings/2_2018-05-26_17-35-05.mp4"

video_path7 = "/Users/aus10powell/Downloads/RiverHerring/IRWA 2017 Videos/2018 Fish Sightings/UNKNOWN FISH/2_2018-05-26_17-35-05.mp4"
video_path8 = "/Users/aus10powell/Downloads/1_2024-04-24_12-00-01_726.mp4"

# Gold standard santuit 2024
video_path = "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/data/gold_dataset/videos/santuit_2024/1_2024-04-24_12-00-01_726.mp4"

# Gold standard coonameset 2024
video_path = "/Users/aus10powell/Downloads/1_2024-05-27_09-00-01_762.mp4" #/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/data/gold_dataset/videos/coonameset_2024/1_2024-05-14_09-00-00_987.mp4"



# IWRA Current best
# "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/code/notebooks/runs/colab_runs/best_m_1.pt"
# "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/code/notebooks/runs/detect/train179/weights/best.pt" #MAP50-95 .72 on small
# "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/code/notebooks/runs/detect/train184/weights/best.pt" MAP50-95 .719 on nano
# 1)  "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/code/notebooks/runs/detect/train133/weights/best.pt"
# 2) "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/code/notebooks/runs/detect/train105/weights/best.pt"
# Testing
# "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/code/notebooks/runs/detect/train38/weights/best.pt"
# Testing Larg model (0.69 MAP50-95): "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/code/notebooks/runs/detect/train176/weights/best.pt"
# "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/runs/detect/train79/weights/last.pt"  # yolov8s: seems very stable but not able to get tracking working
#"/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/code/notebooks/runs/colab_runs/best12.pt"  # best12.pt
# 
#"/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/code/notebooks/runs/detect/train262/weights/best.pt"

# Best  
# yolov9c (05/23/24):
#"/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/runs/yolov9c_runs/runs/detect/train2/weights/last.pt"
# yolov8m (04/23/24): 
#"/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/code/notebooks/runs/colab_runs/best_m_1.pt"
# "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/code/notebooks/runs/colab_runs/last_m_18.pt"

# Santuit best
#"/Users/aus10powell/Downloads/best-3.pt"

def load_yolo_model(model_path):
    model = YOLO(model_path)
    return model
model = load_yolo_model(model_path="/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/code/notebooks/runs/colab_runs/best_m_1.pt")


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


def main(video_path, device="cpu", stream=True, show=True, tracker="/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/code/src/utils/tracking_configs/botsort.yaml"):
    """
    Process a video file with object detection and line counting.
    
    Args:
        video_path (str): Path to the video file for processing.
        device (str, optional): Device for inference. Defaults to "mps".
        stream (bool, optional): Flag to stream the processed video. Defaults to True.
        show (bool, optional): Flag to display the video window. Defaults to True.
        tracker (str, optional): Path to the tracker configuration file. Defaults to "../botsort.yaml".
        
    Returns:
        Tuple[float, List[sv.Frame], int, int, float, List[float]]: 
        - Frame rate of the video.
        - List of annotated frames.
        - Total fish count moving out (right to left).
        - Total fish count moving in (left to right).
        - Duration of the video in seconds.
        - List of relative frame times.
    """
    # Open the video file
    assert os.path.exists(video_path), f"no video found at {video_path}"
    video = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, frame = video.read()

    # Get the frame rate of the video
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # Calculate the duration in seconds
    duration_seconds = total_frames / frame_rate

    # Calculate the time interval between frames
    time_interval = 1.0 / frame_rate

    # Get the frame size (width and height)
    frame_height, frame_width, _ = frame.shape

    # Define the start and end points of a line
    LINE_START = sv.Point(int(frame_width / 3), 0)
    LINE_END = sv.Point(int(frame_width / 3), frame_height)

    # Create a line zone for counting
    line_counter = sv.LineZone(start=LINE_START, end=LINE_END)

    # Create annotators for line zone and bounding box
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.2)
    box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.3)

    # frame_rate = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)

    annotated_frames = []
    frame_detections = []
    # Initialize the list to store relative frame times
    relative_frame_times = []

    # Iterate over the results of the YOLO model's track method
    # Arguments can be found here: https://docs.ultralytics.com/modes/predict/#inference-arguments
    assert os.path.exists(tracker), f"No tracker found at {tracker}"
    for frame_index, result in enumerate(
        model.track(
            imgsz=frame_width, # default 640
            source=video_path,
            device=device,  # Uncomment for faster inference on M2 macbook
            show=show,
            stream=stream,
            conf=0.525,  # 0.651
            # conf=0.1, # default 0.25. Doesn't register as a detection
            # iou=0.2, # default 0.7
            tracker=tracker,
        )
    ):
        frame = result.orig_img

        # Calculate the relative time for the current frame
        current_frame_time = frame_index * time_interval
        relative_frame_times.append(current_frame_time)

        # Convert the YOLO detection results to custom Detections format
        #detections = sv.Detections.from_yolov8(result)
        detections = sv.Detections.from_ultralytics(result)
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
        # Generate labels for each detection
        labels = [
            f"tracker_id: {tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, tracker_id in detections
        ]
        if len(detections) > 0:
            print("*" * 50)
            print("labels::::", labels)
            print("detections:::", detections)
            # time.sleep(0.5)
            print("*" * 50)
            print()
            frame_detections.append(len(detections))
        else:
            # No fish
            frame_detections.append(0)

        # Annotate the frame with bounding boxes and labels
        frame = box_annotator.annotate(
            scene=frame, detections=detections, labels=labels
        )

        # Update the line counter with the current detections
        line_counter.trigger(detections=detections)

        # Annotate the frame with line counter information
        line_annotator.annotate(frame=frame, line_counter=line_counter)

        # Display the frame with annotated information
        cv2.imshow("yolov8", frame)

        # Save frame to list
        annotated_frames.append(frame)
        # time.sleep(0.2)

        if cv2.waitKey(30) == 27:
            break
    # video.release()
    # Print the total fish count from the line counter
    logger.info("-" * 100)
    logger.info(
        f"TOTAL FISH OUT: {line_counter.out_count} \t TOTAL FISH IN: {line_counter.in_count} \t NET (moving right to left): {line_counter.out_count - line_counter.in_count}"
    )
    logger.info(f"total frame_detections: {sum(frame_detections)}")
    logger.info("-" * 100)
    return (
        frame_rate,
        annotated_frames,
        line_counter.out_count,
        line_counter.in_count,
        duration_seconds,
        relative_frame_times,
        frame_detections
    )


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

if __name__ == "__main__":
    # Write annotated frames to local disk
    OUTPUT_DIR = os.path.join("/", "Users", "aus10powell", "Downloads")

    t0 = time.time()
    # Measure memory usage before running the script
    start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Get annotated frames and input video frame rate
    frame_rate, annotated_frames, out_count, in_count, duration_seconds, relative_frame_times, frame_detections = main(
        video_path=video_path
    )
    video_fname = video_path.split("/")[-1].split(".")[0] + "_annotated"
    output_video_path = os.path.join(
        OUTPUT_DIR, f"{video_fname}.mp4"
    )  #  "/Users/aus10powell/Downloads/annotated_video.mp4"
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

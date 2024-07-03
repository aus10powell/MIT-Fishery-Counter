# video_utils.py

import logging
import cv2
import csv
import json
import os
import re
from datetime import timedelta


def set_logging_level(filename):
    """
    Set logging level based on the filename.

    Args:
        filename (str): The name of the file being processed.
    """
    # Extract logging level from filename
    import logging

    logging_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    level = filename.split("_")[-1].split(".")[0].upper()

    # Check if the extracted level is valid
    if level in logging_levels:
        logging.basicConfig(level=logging_levels[level])
    else:
        logging.basicConfig(
            level=logging.INFO
        )  # Default to INFO level if level is not recognized

    return logging.getLogger(filename)


logger = set_logging_level(__file__ + ":" + __name__)


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


def write_frame_data_to_csv(
    frame_detections, relative_frame_times, video_fname, output_dir
):
    """
    Write frame detections and relative frame times to a CSV file.

    Args:
        frame_detections (list): List of frame detections.
        relative_frame_times (list): List of relative frame times.
        video_fname (str): Name of the video file.
        output_dir (str): Directory where the CSV file will be saved.
    """
    output_csv_path = os.path.join(output_dir, f"{video_fname}_detections.csv")

    with open(output_csv_path, mode="w", newline="") as csvfile:
        fieldnames = ["Frame", "Detection", "Relative Time"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for frame_index, (detection, time) in enumerate(
            zip(frame_detections, relative_frame_times)
        ):
            writer.writerow(
                {"Frame": frame_index, "Detection": detection, "Relative Time": time}
            )

    print(f"Frame detections and relative frame times written to: {output_csv_path}")


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
    logger.info(f"Fish counts written to: {output_dir}/video_counts.json")


def create_timestamps(
    relative_frame_times, reference_datetime, format_string="%Y-%m-%d %H:%M:%S.%f"
):
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


def get_processesor_type():
    """
    Get the available processor types.
    """
    processor_type = {"cpu": True, "gpu": False, "mps": False}
    if torch.cuda.is_available():
        processor_type["gpu"] = True
    if torch.backends.mps.is_built():
        processor_type["mps"] = True
    return processor_type

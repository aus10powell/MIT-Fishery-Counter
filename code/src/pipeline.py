# pipeline.py

import os
import resource
import time

from inference_modules import InferenceCounter
from utils import video_utils

logger = video_utils.set_logging_level(__file__ + ":" + __name__)


def process_video_analysis(
    video_path, OUTPUT_DIR, tracker, model_path, write_to_local=True
):
    """
    Process video analysis including inference, frame writing, timestamp creation,
    CSV and JSON writing, and memory and time logging.

    Args:
        video_path (str): Path to the input video file.
        OUTPUT_DIR (str): Directory where output files will be saved.
        tracker (str): Path to the tracker configuration file.
        model_path (str): Path to the model weights file.

    Returns:
        dict: Dictionary containing processed data including counts, timestamps, filenames, etc.
    """
    t0 = time.time()
    # Measure memory usage before running the script
    start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Perform inference and get results
    herring_counter = InferenceCounter(
        device="cpu", tracker=tracker, model_path=model_path
    )
    (
        frame_rate,
        annotated_frames,
        out_count,
        in_count,
        duration_seconds,
        relative_frame_times,
        frame_object_detections,
    ) = herring_counter.run_inference(video_path=video_path, show=True)

    # Get video filename
    video_fname = video_utils.get_annotated_video_name(video_path)
    output_video_path = os.path.join(OUTPUT_DIR, f"{video_fname}.mp4")

    # Write annotated frames to video file
    video_utils.write_frames_to_file(
        annotated_frames=annotated_frames,
        output_video_path=output_video_path,
        fps=frame_rate,
    )

    # Extract reference datetime from filename
    video_reference_datetime = video_utils.extract_datetime_from_filename(
        filename=video_path
    )

    # Create formatted timestamps based on reference datetime and relative times
    formatted_timestamps = video_utils.create_timestamps(
        relative_frame_times, video_reference_datetime
    )

    # Write frame data to CSV
    # video_utils.write_frame_data_to_csv(frame_object_detections, formatted_timestamps, video_fname, OUTPUT_DIR)

    # Prepare data for JSON writing
    data = {
        "out_count": out_count,
        "in_count": in_count,
        "reference_datetime": formatted_timestamps,
        "frame_object_detections": frame_object_detections,
        "net_out_count": out_count - in_count,
        "video_fname": video_fname,
        "location": "IRWA",
    }

    # Write counts to JSON file if write_to_local is True
    if write_to_local:
        video_utils.write_counts_to_json(data, OUTPUT_DIR)

    t1 = time.time()
    # Measure memory usage after running the script
    end_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Calculate memory usage and runtime
    memory_usage = end_memory - start_memory
    runtime = (t1 - t0) / 60  # in seconds

    # Logging memory usage and runtime
    logger.info(f"Memory usage: {memory_usage/1e6:.2f} max MB")
    logger.info(f"Total time: {runtime:.2f} seconds")

    # Return processed data dictionary
    return data


def parse_args(params):
    """
    Parse arguments from a dictionary.

    Args:
        params (dict): Dictionary containing arguments.

    Returns:
        Namespace: A namespace containing the parsed arguments.
    """
    import argparse

    parser = argparse.ArgumentParser()
    for key, value in params.items():
        parser.add_argument(f"--{key}", default=value)
    return parser.parse_args()


def main(params):
    # Set logging level based on filename
    logger = video_utils.set_logging_level(__file__ + ":" + __name__)
    args = parse_args(params)
    video_path = args.video_path
    OUTPUT_DIR = args.OUTPUT_DIR
    tracker = args.tracker
    model_path = args.model_path
    logger.info(f"Starting the main function with the following arguments: {args}")
    processed_data = process_video_analysis(
        video_path, OUTPUT_DIR, tracker, model_path, write_to_local=True
    )

if __name__ == "__main__":
    # Set output dir to write results
    OUTPUT_DIR = os.path.join("/", "Users", "aus10powell", "Downloads")

    # Get annotated frames and input video frame rate
    # video_path =  "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/data/gold_dataset/videos/2_2018-04-14_10-06-19.mp4"
    # video_path = "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/data/gold_dataset/videos/1_2016-04-13_13-57-11.mp4"
    # video_path = "/Users/aus10powell/Downloads/1_2024-05-27_09-00-01_762.mp4"
    # Short 10 second video IRWA
    video_path = "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/data/gold_dataset/videos/irwa/1_2016-04-22_12-36-58.mp4"
    tracker = "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/code/src/utils/tracking_configs/botsort.yaml"
    model_path = "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/code/notebooks/runs/detect/train133/weights/best.pt"
    params = {
        "video_path": video_path,
        "OUTPUT_DIR": OUTPUT_DIR,
        "tracker": tracker,
        "model_path": model_path,
        "write_to_local": True
    }
    main(params)

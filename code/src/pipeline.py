from video_utils import set_logging_level
from inference_module import InferenceCounter
import time
import resource
import os
from video_utils import get_annotated_video_name, write_frame_data_to_csv, set_logging_level, get_annotated_video_name, get_annotated_video_name, write_frames_to_file

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

def main():
    # Set logging level based on filename
    logger = set_logging_level( __file__  + ":" + __name__)
    # Define the parameters. Set at defaults.
    params = {
        "data_config": "data.yaml",
        "model": "yolov8s",
        "imgsize": 640,
        "epochs": 100,
        "dropout": 0.5,
        "batch": 16}
    args = parse_args(params)
    logger.info("Starting the main function with the following arguments: %s", args)

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


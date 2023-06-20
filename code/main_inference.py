import cv2
import time
from ultralytics import YOLO
import supervision as sv
import numpy as np
import io
import time
import logging
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)

# One fish
video_path1 = "/Users/aus10powell/Downloads/RiverHerring/IRWA 2017 Videos/Fish Sightings 2017/2_2017-04-13_13-10-00.mp4"
video_path1a = "/Users/aus10powell/Downloads/RiverHerring/IRWA 2017 Videos/2018 Fish Sightings/2_2018-04-14_13-18-51.mp4"  # Currently detected not counted
video_path1b = "/Users/aus10powell/Downloads/RiverHerring/IRWA 2017 Videos/2018 Fish Sightings/2_2018-04-14_10-06-19.mp4"  # Currently detected w/ id but not counted

# Two fish swimming concurrently
video_path2 = "/Users/aus10powell/Downloads/RiverHerring/IRWA 2017 Videos/Fish Sightings 2017/2_2017-04-13_14-10-29.mp4"  # Current:


video_path = video_path1a

model = YOLO(
    # Current best
    #"/Users/aus10powell/Downloads/RiverHerring/runs/detect/train38/weights/best.pt"
    # Testing
    "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/code/notebooks/runs/detect/train38/weights/best.pt"

)


def frames_to_video(frames=None, fps=12):
    """
    Convert frames to video for Streamlit

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
    logging.info("INFO: Encoding frames and writing to MP4 format.")
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


def frames_to_file(
    annotated_frames=None, output_video_path="annotated_video.mp4", fps=30
):
    import cv2
    import logging


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
    logging.info(f"INFO: Wrote {output_video_path} @ fps = {fps:.3}. Total frames = {total_frames}")


def main(video_path):


    # # Can statically set
    # LINE_START = sv.Point(90, 0)
    # LINE_END = sv.Point(90, 500)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, frame = video.read()

    # Get the frame rate of the video
    frame_rate = video.get(cv2.CAP_PROP_FPS)

    # Get the frame size (width and height)
    frame_height, frame_width, _ = frame.shape

    # Define the start and end points of a line
    LINE_START = sv.Point(90, 0)
    LINE_END = sv.Point(90, frame_height)

    # Create a line zone for counting
    line_counter = sv.LineZone(start=LINE_START, end=LINE_END)

    # Create annotators for line zone and bounding box
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.2)
    box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.3)

    # frame_rate = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)

    annotated_frames = []
    # Iterate over the results of the YOLO model's track method
    for result in model.track(
        source=video_path,
        show=True,
        stream=True,
        agnostic_nms=True,
       tracker="bytetrack.yaml",
    ):
        frame = result.orig_img

        # Convert the YOLO detection results to custom Detections format
        detections = sv.Detections.from_yolov8(result)

        # Set tracker IDs if available
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        # Generate labels for each detection
        labels = [
            f"tracker_id: {tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, tracker_id in detections
        ]

        # Count the number of fish detected
        if len(result) > 0:
            num_fish = len(result[0].boxes.data)
        else:
            num_fish = 0

        # Print information about fish detections and line counters
        if num_fish > 0 and result.boxes.id is not None:
            print("*" * 50)
            print(f"num_fish: {num_fish}")
            print(labels)
            print(detections)
            print(".boxes.xyxy.cpu().numpy()", result.boxes.xyxy.cpu().numpy())
            print(
                "result.boxes.id.cpu().numpy().astype(int)-----",
                result.boxes.id.cpu().numpy().astype(int),
            )
            print("line_counter::::")
            print("\t in_count:", line_counter.in_count)
            print("\t out_count:", line_counter.out_count)
            print("\t tracker_state:", line_counter.tracker_state)
            print("\t tracker_state:", line_counter.tracker_state.keys())
            print("*" * 50)
            time.sleep(1.5)

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

        if cv2.waitKey(30) == 27:
            break

        # Print the total fish count from the line counter
        logging.info("-" * 100)
        logging.info(
            f"TOTAL FISH OUT: {line_counter.out_count} \t TOTAL FISH IN: {line_counter.in_count}"
        )
    return frame_rate, annotated_frames


if __name__ == "__main__":
    t0 = time.time()

    # Get annotated frames and input video frame rate
    frame_rate, annotated_frames = main(video_path=video_path)

    # Write annotated frames to local disk
    output_video_path = "/Users/aus10powell/Downloads/annotated_video.mp4"
    frames_to_file(
        annotated_frames=annotated_frames,
        output_video_path=output_video_path,
        fps=frame_rate,
    )
    t1 = time.time()
    runtime = (t1 - t0) / 60
    print(f"Total time: {runtime:.2}")

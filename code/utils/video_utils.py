from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
import os

def split_video(video_path, output_dir, parts = 4):
    """
    Split a video file into 'parts' equal parts.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where the split video parts will be saved.
        parts (int): Number of parts to split the video into.

    Returns:
        None
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    clip = VideoFileClip(video_path)
    duration = clip.duration
    part_duration = duration / parts

    for i in range(4):
        start_time = i * part_duration
        end_time = (i + 1) * part_duration
        output_path = os.path.join(output_dir, f"part_{i+1}.mp4")
        ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=output_path)


if __name__ == "__main__":
    split_video("/Users/aus10powell/Downloads/1_2024-05-27_09-00-01_762.mp4", "/Users/aus10powell/Downloads/1_2024-05-27_09-00-01_762_parts")

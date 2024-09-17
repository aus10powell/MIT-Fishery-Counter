import csv
import os
import tempfile
from datetime import datetime

import cv2
import numpy as np
import pytest
from src.utils.video_utils import (extract_datetime_from_filename,
                                   write_frame_data_to_csv,
                                   write_frames_to_file)


def test_extract_datetime_from_filename_valid():
    filename = "1234_2023-10-05_14-30-00.mp4"
    expected_datetime = datetime(2023, 10, 5, 14, 30, 0)
    result = extract_datetime_from_filename(filename)
    assert result == expected_datetime


def test_extract_datetime_from_filename_invalid_format():
    filename = "invalid_filename.mp4"
    with pytest.raises(
        ValueError, match="Filename format doesn't match expected pattern"
    ):
        extract_datetime_from_filename(filename)


def test_extract_datetime_from_filename_invalid_date():
    filename = "1234_2023-13-05_14-30-00.mp4"  # Invalid month
    with pytest.raises(ValueError, match="Invalid date or time format in filename"):
        extract_datetime_from_filename(filename)


def test_write_frame_data_to_csv():
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define test data
        frame_detections = ["fish", "no_fish", "fish"]
        relative_frame_times = [0.0, 1.0, 2.0]
        video_fname = "test_video"

        # Call the function
        write_frame_data_to_csv(
            frame_detections, relative_frame_times, video_fname, temp_dir
        )

        # Define the expected CSV file path
        expected_csv_path = os.path.join(temp_dir, f"{video_fname}_detections.csv")

        # Check if the CSV file is created
        assert os.path.exists(expected_csv_path), "CSV file was not created"

        # Read the CSV file and verify its contents
        with open(expected_csv_path, mode="r") as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

            # Verify the header
            assert reader.fieldnames == [
                "Frame",
                "Detection",
                "Relative Time",
            ], "CSV header does not match"

            # Verify the rows
            expected_rows = [
                {"Frame": "0", "Detection": "fish", "Relative Time": "0.0"},
                {"Frame": "1", "Detection": "no_fish", "Relative Time": "1.0"},
                {"Frame": "2", "Detection": "fish", "Relative Time": "2.0"},
            ]
            assert rows == expected_rows, "CSV rows do not match expected data"

def generate_dummy_frames(num_frames, height, width, color=(0, 255, 0)):
    """
    Generate a list of dummy frames for testing.

    Args:
        num_frames (int): Number of frames to generate.
        height (int): Height of each frame.
        width (int): Width of each frame.
        color (tuple): Color of the frames (default is green).

    Returns:
        list: List of dummy frames.
    """
    frames = []
    for _ in range(num_frames):
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        frames.append(frame)
    return frames

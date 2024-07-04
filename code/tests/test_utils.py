import pytest

from src.utils import video_utils

def test_extract_datetime_from_filename():
    filename = "2_2018-04-14_10-06-19.mp4"
    datetime = video_utils.extract_datetime_from_filename(filename)
    assert str(datetime) == "2018-04-14 10:06:19"

def test_write_counts_to_json():
    """Expected behavior for output inference
    """
    data = {
        "out_count": 10,
        "in_count": 5,
        "net_count": 5,
        "frame_rate": 30,
        "duration_seconds": 30,
        "relative_frame_times": [0, 1, 2, 3, 4],
        "frame_detections": [0, 1, 2, 3, 4],
    }
    video_utils.write_counts_to_json(data, "/tmp")
    assert True
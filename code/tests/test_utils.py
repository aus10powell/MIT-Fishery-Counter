import pytest

from src import video_utils

def test_extract_datetime_from_filename():
    filename = "2_2018-04-14_10-06-19.mp4"
    datetime = video_utils.extract_datetime_from_filename(filename)
    assert str(datetime) == "2018-04-14 10:06:19"
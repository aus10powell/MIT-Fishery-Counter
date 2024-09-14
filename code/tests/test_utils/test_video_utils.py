import pytest
from datetime import datetime
from src.utils.video_utils import extract_datetime_from_filename

def test_extract_datetime_from_filename_valid():
    filename = "1234_2023-10-05_14-30-00.mp4"
    expected_datetime = datetime(2023, 10, 5, 14, 30, 0)
    result = extract_datetime_from_filename(filename)
    assert result == expected_datetime

def test_extract_datetime_from_filename_invalid_format():
    filename = "invalid_filename.mp4"
    with pytest.raises(ValueError, match="Filename format doesn't match expected pattern"):
        extract_datetime_from_filename(filename)

def test_extract_datetime_from_filename_invalid_date():
    filename = "1234_2023-13-05_14-30-00.mp4"  # Invalid month
    with pytest.raises(ValueError, match="Invalid date or time format in filename"):
        extract_datetime_from_filename(filename)
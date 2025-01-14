import os
import pytest
from unittest import mock
from unittest.mock import Mock, patch, MagicMock
from src.pipeline import parse_args, process_video_analysis, main

def test_parse_args_valid_params():
    params = {
        "video_path": "/path/to/video.mp4",
        "OUTPUT_DIR": "/path/to/output",
        "tracker": "/path/to/tracker.yaml",
        "model_path": "/path/to/model.pt",
        "write_to_local": 'true'
    }
    args = parse_args(params)
    
    assert args.video_path == "/path/to/video.mp4"
    assert args.OUTPUT_DIR == "/path/to/output"
    assert args.tracker == "/path/to/tracker.yaml"
    assert args.model_path == "/path/to/model.pt"
    assert args.write_to_local == 'true'

def test_parse_args_additional_params():
    params = {
        "video_path": "/path/to/video.mp4",
        "OUTPUT_DIR": "/path/to/output",
        "tracker": "/path/to/tracker.yaml",
        "model_path": "/path/to/model.pt",
        "write_to_local": 'true',
        "extra_param": "extra_value"
    }
    args = parse_args(params)
    
    assert args.video_path == "/path/to/video.mp4"
    assert args.OUTPUT_DIR == "/path/to/output"
    assert args.tracker == "/path/to/tracker.yaml"
    assert args.model_path == "/path/to/model.pt"
    assert args.write_to_local == 'true'
    assert args.extra_param == "extra_value"

def test_parse_args_different_data_types():
    params = {
        "video_path": "/path/to/video.mp4",
        "OUTPUT_DIR": "/path/to/output",
        "tracker": "/path/to/tracker.yaml",
        "model_path": "/path/to/model.pt",
        "write_to_local": True
    }
    args = parse_args(params)
    
    assert args.video_path == "/path/to/video.mp4"
    assert args.OUTPUT_DIR == "/path/to/output"
    assert args.tracker == "/path/to/tracker.yaml"
    assert args.model_path == "/path/to/model.pt"
    assert args.write_to_local == 'True'  # argparse converts True to 'True'

@mock.patch('src.pipeline.InferenceCounter')
@mock.patch('src.pipeline.video_utils')
@mock.patch('src.pipeline.resource')
@mock.patch('src.pipeline.time')
def test_process_video_analysis_success(mock_time, mock_resource, mock_video_utils, mock_InferenceCounter):
    # Mocking time and resource usage
    mock_time.time.side_effect = [0, 60]
    mock_resource.getrusage.return_value.ru_maxrss = 1000000

    # Mocking InferenceCounter
    mock_counter = mock_InferenceCounter.return_value
    mock_counter.run_inference.return_value = (
        30,  # frame_rate
        ['frame1', 'frame2'],  # annotated_frames
        10,  # out_count
        5,  # in_count
        120,  # duration_seconds
        [0.1, 0.2],  # relative_frame_times
        ['detection1', 'detection2']  # frame_object_detections
    )

    # Mocking video_utils functions
    mock_video_utils.get_annotated_video_name.return_value = 'video_annotated'
    mock_video_utils.extract_datetime_from_filename.return_value = '2023-01-01 00:00:00'
    mock_video_utils.create_timestamps.return_value = ['2023-01-01 00:00:01', '2023-01-01 00:00:02']

    params = {
        "video_path": "/path/to/video.mp4",
        "OUTPUT_DIR": "/path/to/output",
        "tracker": "/path/to/tracker.yaml",
        "model_path": "/path/to/model.pt",
        "write_to_local": True
    }

    result = process_video_analysis(**params)

    assert result['out_count'] == 10
    assert result['in_count'] == 5
    assert result['reference_datetime'] == ['2023-01-01 00:00:01', '2023-01-01 00:00:02']
    assert result['frame_object_detections'] == ['detection1', 'detection2']
    assert result['net_out_count'] == 5
    assert result['video_fname'] == 'video_annotated'
    assert result['location'] == 'IRWA'

    mock_video_utils.write_frames_to_file.assert_called_once()
    mock_video_utils.write_counts_to_json.assert_called_once()


@patch('src.pipeline.InferenceCounter')
def test_process_video_analysis_invalid_video_path(mock_inference_counter):
    valid_params = {
        "video_path": "test_video.mp4",
        "OUTPUT_DIR": "/tmp/output",
        "tracker": "tracker.yaml",
        "model_path": "model.pt"
    }

    mock_counter = Mock()
    mock_counter.run_inference.side_effect = AssertionError("Invalid video path")
    mock_inference_counter.return_value = mock_counter
    
    with pytest.raises(AssertionError):
        process_video_analysis(**valid_params)


def test_process_video_analysis_missing_params():
    params = {
        "video_path": "/path/to/video.mp4",
        "OUTPUT_DIR": "/path/to/output",
        "tracker": "/path/to/tracker.yaml",
        # Missing model_path
        "write_to_local": True
    }

    with pytest.raises(TypeError):
        process_video_analysis(**params)

def test_process_video_analysis_invalid_params():
    params = {
        "video_path": 123,  # Invalid type
        "OUTPUT_DIR": "/path/to/output",
        "tracker": "/path/to/tracker.yaml",
        "model_path": "/path/to/model.pt",
        "write_to_local": True
    }

    with pytest.raises(Exception):
        process_video_analysis(**params)

import pytest
from unittest import mock
from src.pipeline import process_video_analysis

@mock.patch('src.pipeline.InferenceCounter')
@mock.patch('src.pipeline.video_utils')
@mock.patch('src.pipeline.resource')
@mock.patch('src.pipeline.time')
def test_process_video_analysis_success(mock_time, mock_resource, mock_video_utils, mock_InferenceCounter):
    # Mocking time and resource usage
    mock_time.time.side_effect = [0, 60]
    mock_resource.getrusage.return_value.ru_maxrss = 1000000

    # Mocking InferenceCounter
    mock_counter = mock_InferenceCounter.return_value
    mock_counter.run_inference.return_value = (
        30,  # frame_rate
        ['frame1', 'frame2'],  # annotated_frames
        10,  # out_count
        5,  # in_count
        120,  # duration_seconds
        [0.1, 0.2],  # relative_frame_times
        ['detection1', 'detection2']  # frame_object_detections
    )

    # Mocking video_utils functions
    mock_video_utils.get_annotated_video_name.return_value = 'video_annotated'
    mock_video_utils.extract_datetime_from_filename.return_value = '2023-01-01 00:00:00'
    mock_video_utils.create_timestamps.return_value = ['2023-01-01 00:00:01', '2023-01-01 00:00:02']

    params = {
        "video_path": "/path/to/video.mp4",
        "OUTPUT_DIR": "/path/to/output",
        "tracker": "/path/to/tracker.yaml",
        "model_path": "/path/to/model.pt",
        "write_to_local": True
    }

    result = process_video_analysis(**params)

    assert result['out_count'] == 10
    assert result['in_count'] == 5
    assert result['reference_datetime'] == ['2023-01-01 00:00:01', '2023-01-01 00:00:02']
    assert result['frame_object_detections'] == ['detection1', 'detection2']
    assert result['net_out_count'] == 5
    assert result['video_fname'] == 'video_annotated'
    assert result['location'] == 'IRWA'

    mock_video_utils.write_frames_to_file.assert_called_once()
    mock_video_utils.write_counts_to_json.assert_called_once()

def test_process_video_analysis_missing_params():
    params = {
        "video_path": "/path/to/video.mp4",
        "OUTPUT_DIR": "/path/to/output",
        "tracker": "/path/to/tracker.yaml",
        # Missing model_path
        "write_to_local": True
    }

    with pytest.raises(TypeError):
        process_video_analysis(**params)

def test_process_video_analysis_invalid_params():
    params = {
        "video_path": 123,  # Invalid type
        "OUTPUT_DIR": "/path/to/output",
        "tracker": "/path/to/tracker.yaml",
        "model_path": "/path/to/model.pt",
        "write_to_local": True
    }

    with pytest.raises(Exception):
        process_video_analysis(**params)

import pytest
from unittest import mock
from src.pipeline import main

@mock.patch('src.pipeline.process_video_analysis')
@mock.patch('src.pipeline.video_utils')
def test_main_valid_params(mock_video_utils, mock_process_video_analysis):
    params = {
        "video_path": "/path/to/video.mp4",
        "OUTPUT_DIR": "/path/to/output",
        "tracker": "/path/to/tracker.yaml",
        "model_path": "/path/to/model.pt",
        "write_to_local": True
    }
    
    mock_process_video_analysis.return_value = {
        "out_count": 10,
        "in_count": 5,
        "reference_datetime": ['2023-01-01 00:00:01', '2023-01-01 00:00:02'],
        "frame_object_detections": ['detection1', 'detection2'],
        "net_out_count": 5,
        "video_fname": 'video_annotated',
        "location": 'IRWA'
    }
    
    result = main(params)
    
    assert result['out_count'] == 10
    assert result['in_count'] == 5
    assert result['reference_datetime'] == ['2023-01-01 00:00:01', '2023-01-01 00:00:02']
    assert result['frame_object_detections'] == ['detection1', 'detection2']
    assert result['net_out_count'] == 5
    assert result['video_fname'] == 'video_annotated'
    assert result['location'] == 'IRWA'

def test_main_missing_params():
    params = {
        "video_path": "/path/to/video.mp4",
        "OUTPUT_DIR": "/path/to/output",
        "tracker": "/path/to/tracker.yaml",
        # Missing model_path
        "write_to_local": True
    }

    with pytest.raises(TypeError):
        main(params)

def test_main_invalid_params():
    params = {
        "video_path": 123,  # Invalid type
        "OUTPUT_DIR": "/path/to/output",
        "tracker": "/path/to/tracker.yaml",
        "model_path": "/path/to/model.pt",
        "write_to_local": True
    }

    with pytest.raises(Exception):
        main(params)
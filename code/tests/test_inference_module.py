import logging
import os
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from src.inference_modules import InferenceCounter

TEST_MODEL_YOLO_8_PATH = "code/tests/test_models/yolov8n.pt"

def test_inference_module():
    inference_module = InferenceCounter()
    assert inference_module is not None


def test_inference_module_load_model():
    inference_module = InferenceCounter()
    model = inference_module.load_model("yolov8s.pt")
    assert model is not None


@mock.patch("cv2.VideoCapture")
@mock.patch("os.path.exists")
@mock.patch("ultralytics.YOLO")
def test_inference_module_initialization(mock_yolo, mock_exists, mock_videocapture):
    mock_exists.return_value = True
    mock_yolo.return_value = mock.Mock()
    mock_videocapture.return_value = mock.Mock()

    inference_module = InferenceCounter()
    assert inference_module is not None


def test_inference_module():
    inference_module = InferenceCounter()
    assert inference_module is not None


def test_inference_module_load_model():
    inference_module = InferenceCounter()
    model = inference_module.load_model("yolov8s.pt")
    assert model is not None


@mock.patch("cv2.VideoCapture")
@mock.patch("os.path.exists")
@mock.patch("ultralytics.YOLO")
def test_inference_module_initialization(mock_yolo, mock_exists, mock_videocapture):
    mock_exists.return_value = True
    mock_yolo.return_value = mock.Mock()

    inference_module = InferenceCounter()
    assert inference_module is not None


@mock.patch("cv2.VideoCapture")
@mock.patch("os.path.exists")
@mock.patch("ultralytics.YOLO")
def test_run_inference_invalid_video(mock_yolo, mock_exists, mock_videocapture):
    mock_exists.side_effect = (
        lambda path: False if path == "invalid_video_path.mp4" else True
    )
    mock_yolo.return_value = mock.Mock()

    inference_module = InferenceCounter()

    with pytest.raises(AssertionError):
        inference_module.run_inference("invalid_video_path.mp4")

def test_inference_counter_initialization_with_custom_params():
    """Test InferenceCounter initialization with custom parameters"""
    device = "cuda"
    tracker = "custom_tracker.yaml"

    
    with mock.patch('ultralytics.YOLO') as mock_yolo:
        mock_yolo.return_value = mock.Mock()
        counter = InferenceCounter(device=device, tracker=tracker, model_path=TEST_MODEL_YOLO_8_PATH)
        
        assert counter.device == device
        assert counter.tracker == tracker
        assert isinstance(counter.logger, logging.Logger)


def test_set_logging_level_with_valid_level():
    """Test set_logging_level with valid logging level"""
    counter = InferenceCounter()

    # Test each valid logging level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    for level in valid_levels:
        logger = counter.set_logging_level(level)
        assert logger.getEffectiveLevel() == getattr(logging, level)

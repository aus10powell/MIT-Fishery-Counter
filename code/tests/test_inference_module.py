import logging
import os
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from src.inference_modules import InferenceCounter


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

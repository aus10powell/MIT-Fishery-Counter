import pytest
from src.inference_modules import InferenceCounter


def test_inference_module():
    inference_module = InferenceCounter()
    assert inference_module is not None


def test_inference_module_load_model():
    inference_module = InferenceCounter()
    model = inference_module.load_model("yolov8s.pt")
    assert model is not None

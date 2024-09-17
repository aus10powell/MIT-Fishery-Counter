# import os
# import pytest
# from unittest import mock
# from src.pipeline import parse_args

# def test_parse_args_valid_params():
#     params = {
#         "video_path": "/path/to/video.mp4",
#         "OUTPUT_DIR": "/path/to/output",
#         "tracker": "/path/to/tracker.yaml",
#         "model_path": "/path/to/model.pt",
#         "write_to_local": True
#     }
#     args = parse_args(params)
    
#     assert args.video_path == "/path/to/video.mp4"
#     assert args.OUTPUT_DIR == "/path/to/output"
#     assert args.tracker == "/path/to/tracker.yaml"
#     assert args.model_path == "/path/to/model.pt"
#     assert args.write_to_local == True

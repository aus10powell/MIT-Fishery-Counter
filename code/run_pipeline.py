import sys, os
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).resolve().parent / 'src'
sys.path.insert(0, str(src_path))

from pipeline import main  # Assuming you have a main function in pipeline.py

if __name__ == "__main__":
    OUTPUT_DIR = os.path.join("/", "Users", "aus10powell", "Downloads") 
    video_path = "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/data/gold_dataset/videos/irwa/1_2016-04-22_12-36-58.mp4"
    tracker = "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/code/src/utils/tracking_configs/botsort.yaml"
    model_path = "/Users/aus10powell/Documents/Projects/MIT-Fishery-Counter/code/notebooks/runs/detect/train133/weights/best.pt"
    params = {
        "video_path": video_path,
        "OUTPUT_DIR": OUTPUT_DIR,
        "tracker": tracker,
        "model_path": model_path,
        "write_to_local": True
    }
    main(params)
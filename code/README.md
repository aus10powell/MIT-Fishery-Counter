# Running inference code

## Running `run_pipeline.py`

To run the `run_pipeline.py` script directly, use the following command:

```bash
python code/src/run_inference_pipeline.py
```

Make sure you have all the necessary dependencies installed. You can install them using:

```pip install -r requirements.txt```

Running Tests
To run the tests, you can use pytest. Run the following command:

To run the tests with coverage, use:

```pytest code/tests```

This will execute all the tests and provide a coverage report.

```pytest code/tests --cov=code/src```

## Example: Using pipeline.py in a new script

To use the `pipeline.py` in another script, follow these steps:

1. Create a new Python file (e.g., `new_script.py`) in your project directory.

2. Add the following code to `new_script.py`:

```python
from src.pipeline import main as pipeline_main

# Define your parameters
params = {
    "video_path": "/path/to/your/video.mp4",
    "OUTPUT_DIR": "/path/to/output/directory",
    "tracker": "/path/to/tracker/config.yaml",
    "model_path": "/path/to/model/weights.pt",
    "write_to_local": True
}

# Run the pipeline and get the processed data
processed_data = pipeline_main(params)

# Now you can use processed_data as needed
print(processed_data)
```
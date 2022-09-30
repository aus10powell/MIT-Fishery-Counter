# Libraries
import streamlit as st
from PIL import Image, ImageOps
import io
import numpy
import numpy as np
import pandas as pd
import time

# Options
st.set_option("deprecation.showfileUploaderEncoding", False)

# App documentation
st.title("MIT Count Fish Demo")
st.text("Upload an image or video file to detect and count fish")

# Initialize
img_types = ["jpg", "png", "jpeg"]
video_types = ["mp4", "avi"]
uploaded_file = st.file_uploader(
    "Select an image or video file...", type=img_types + video_types
)

# Temp data
df = pd.DataFrame(data={"file": "example.mp4", "counts": 1}, index=[0])


def process_file(file):
    """Script processes uploaded file and returns processed file"""
    processed_file = file
    return processed_file


# Display uploaded file
if uploaded_file is not None:
    if str(uploaded_file.type).split("/")[-1] in img_types:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_column_width=True)
    elif str(uploaded_file.type).split("/")[-1] in video_types:

        ## Stopgap until inference code is ready
        processed_file = process_file(uploaded_file)
        time.sleep(3)

        # Display processed file
        st.video(processed_file)

    # Show count information
    st.dataframe(df)
else:
    st.write("No file uploaded")

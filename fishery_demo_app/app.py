# Libraries
import streamlit as st
from PIL import Image, ImageOps
import io
import utils


st.set_option('deprecation.showfileUploaderEncoding', False)

# App documentation
st.title("MIT Count Fish Demo")
st.text("Upload an image or video file to detect and count fish")

# Initialize pp
print('Starting Streamlit app')
img_types = ["jpg","png","jpeg"]
video_types = ["mp4","avi"]
uploaded_file = st.file_uploader("Select an image or video file...", type=img_types+video_types)

        
# Display uploaded file
if uploaded_file is not None:
    if str(uploaded_file.type).split("/")[-1] in img_types:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded image', use_column_width=True)
    elif str(uploaded_file.type).split("/")[-1] in video_types:
        st.video(uploaded_file)
else:
    st.write('No file uploaded')
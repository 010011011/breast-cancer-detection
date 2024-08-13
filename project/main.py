from PIL import Image
import numpy as np
import streamlit as st
from ultralytics import YOLO

from util import visualize, set_background

# Load the YOLO model
model_path = "C:\\Users\\DELL\\Desktop\\project\\model\\best.pt"
model = YOLO(model_path)

# Set the background image
set_background('./bg.png')

# Set the title
st.title('Breast Cancer Tumor Detection')

# Set the header
st.header('Please upload an image')

# Upload the image file
file = st.file_uploader('', type=['png', 'jpg', 'jpeg'])

# Process the uploaded image
if file:
    image = Image.open(file).convert('RGB')
    image_array = np.asarray(image)

    # Detect objects using the YOLO model
    results = model.predict(image_array)

    # Extract bounding boxes and scores
    bboxes_ = []
    threshold = 0.5
    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.conf > threshold:  # box.conf is the confidence score
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # get bounding box coordinates
                bboxes_.append([int(x1), int(y1), int(x2), int(y2)])

    # Visualize the results
    visualize(image, bboxes_)

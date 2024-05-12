import streamlit as st
import tifffile
import numpy as np
from keras.models import load_model
import pickle
import matplotlib.pyplot as plt

best_model = load_model('/content/drive/MyDrive/Landcover_recognition/vgg19_model.h5')
with open('/content/drive/MyDrive/Landcover_recognition/class_labels.pkl', 'rb') as file:
  class_labels = pickle.load(file)

# Function to classify the image
def classify_image(image):
  image = image.astype('float32') / 1500
  predictions = best_model.predict(image[np.newaxis,:,:,:])
  label = class_labels[np.argmax(predictions)]
  return label
  
def convert_to_rgb(image_data):
    # Read bands 4 (red), 3 (green), and 2 (blue)
    bands = np.transpose(image_data, (2, 0, 1))[[3,2,1],:,:] 
    # Scale the pixel values to the range [0, 255]
    scaled_bands = [(band - band.min()) / (band.max() - band.min()) for band in bands]
    # Stack the bands into an RGB image
    rgb_image = np.dstack(scaled_bands)
    return rgb_image

# Streamlit app

st.title('Land Cover Prediction')
uploaded_file = st.file_uploader("Select an image...", type=["tif"])

st.write("""
<style>
div.stButton > button:first-child {
    background-color: green;
    color: white;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

if uploaded_file is not None:
  image = tifffile.imread(uploaded_file)
  rgb_image = convert_to_rgb(image)
  image_width  = 256
  image_height = 256
  st.image(rgb_image)
  st.markdown(
    f"""
    <style>
    /* Set the image size */
    div[data-testid="stImage"] img {{
        width: {image_width}px;
        height: {image_height}px;
    }}
    </style>
    """,
    unsafe_allow_html=True)

  if st.button('Classify'):
    with st.spinner('Classifying...'):
      predicted_label = classify_image(image)
    st.write(f"Predicted class label: {predicted_label}")


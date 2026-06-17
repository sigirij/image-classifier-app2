import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Set up the page
st.set_page_config(page_title="AI Image Classifier", page_icon="🖼️")
st.title("🖼️ Image Classification Web App")
st.write("Upload an image, and the MobileNetV2 model will predict its class!")

# Load class names
@st.cache_data
def load_classes():
    with open('class_names.json', 'r') as f:
        return json.load(f)

class_names = load_classes()

# Load the trained model
@st.cache_resource
def load_model():
    # If you renamed your model to .keras, change the filename here!
    return tf.keras.models.load_model('model.h5') 

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Analyzing image...")
    
    # Preprocess the image to match training data format
    img_resized = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) 
    # Note: We don't need manual rescaling here because we included the 
    # layers.Rescaling inside our model architecture during training!
    
    # Make a prediction
    predictions = model.predict(img_array)
    
    # Get the highest probability class
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = np.max(predictions[0]) * 100
    
    # Display results
    st.success(f"**Predicted Class:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")
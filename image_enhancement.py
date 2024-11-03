import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf

# Define the SSIM-based loss function
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

enhanced_autoencoder_v3 = load_model('/content/drive/MyDrive/SI_backup/enhanced_autoencoder_v3_copy.h5', custom_objects={'ssim_loss': ssim_loss})
enhanced_autoencoder_v4 = load_model('/content/drive/MyDrive/SI_backup/enhanced_autoencoder_v4_copy.h5', custom_objects={'ssim_loss': ssim_loss})

def preprocess_image(image):
    # Convert to grayscale and resize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255.0  # Normalize to [0, 1]
    return image.reshape(1, 128, 128, 1)  # Reshape for the model

def postprocess_image(image):
    # Rescale and convert to 8-bit
    image_8bit = cv2.convertScaleAbs(image[0], alpha=(255.0 / image.max()))
    return image_8bit

def apply_post_processing(enhanced_image):
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(enhanced_image)

    # Unsharp Masking
    blurred = cv2.GaussianBlur(enhanced_image, (9, 9), 10.0)
    sharp_image = cv2.addWeighted(enhanced_image, 1.5, blurred, -0.5, 0)

    # Denoising
    denoised_image = cv2.fastNlMeansDenoising(sharp_image, None, 30, 7, 21)

    # Smoothing
    smoothed_image = cv2.GaussianBlur(denoised_image, (3, 3), 0)

    return enhanced_image, sharp_image, denoised_image, smoothed_image

# Streamlit UI
st.set_page_config(page_title="Image Enhancement", layout="wide")
st.title("Image Enhancement using Autoencoders")
st.write("Upload an image to enhance it using two different autoencoder models.")

# File uploader with custom styling
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    # Read and display the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for processing
    st.image(original_image, caption="Original Image", use_column_width=300, channels="GRAY")

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Process the image using both versions
    enhanced_image_v3 = enhanced_autoencoder_v3.predict(preprocessed_image)
    enhanced_image_v4 = enhanced_autoencoder_v4.predict(preprocessed_image)

    # Post-process images
    enhanced_image_v3_post = postprocess_image(enhanced_image_v3)
    enhanced_image_v4_post = postprocess_image(enhanced_image_v4)

    # Apply additional processing
    enhanced_image_v3_final, sharp_v3, denoised_v3, smoothed_v3 = apply_post_processing(enhanced_image_v3_post)
    enhanced_image_v4_final, sharp_v4, denoised_v4, smoothed_v4 = apply_post_processing(enhanced_image_v4_post)

    # Display enhanced images in a side-by-side format
    st.subheader("Enhanced Image - Version 3")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image(enhanced_image_v3_final, caption="CLAHE Enhanced", use_column_width=True, channels="GRAY")

    with col2:
        st.image(sharp_v3, caption="Unsharp Masking", use_column_width=True, channels="GRAY")

    with col3:
        st.image(denoised_v3, caption="Denoised", use_column_width=True, channels="GRAY")

    with col4:
        st.image(smoothed_v3, caption="Smoothed", use_column_width=True, channels="GRAY")

    # New row for Version 4 images
    st.subheader("Enhanced Image - Version 4")
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.image(enhanced_image_v4_final, caption="CLAHE Enhanced", use_column_width=True, channels="GRAY")

    with col6:
        st.image(sharp_v4, caption="Unsharp Masking", use_column_width=True, channels="GRAY")

    with col7:
        st.image(denoised_v4, caption="Denoised", use_column_width=True, channels="GRAY")

    with col8:
        st.image(smoothed_v4, caption="Smoothed", use_column_width=True, channels="GRAY")


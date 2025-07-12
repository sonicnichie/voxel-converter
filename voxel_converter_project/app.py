import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import io
import cv2

st.set_page_config(page_title="3D Voxel Style Image Converter", layout="wide")
st.title("🧊 3D Voxel Style Image Converter")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

block_size = st.slider("Block Size (Smaller = More Voxels)", min_value=1, max_value=16, value=4)
z_scale = st.slider("Z Height Scale", min_value=1, max_value=50, value=15)
contrast_boost = st.slider("Contrast Boost (0.5 - 3.0)", min_value=0.5, max_value=3.0, value=2.0, step=0.1)
mode = st.selectbox("Rendering Mode", ["Voxel Full Color", "Voxel Edge Only", "Voxel Silhouette"])

def remove_background(image_np):
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    lower_bg = np.array([0, 0, 120])
    upper_bg = np.array([255, 60, 255])
    mask = cv2.inRange(hsv, lower_bg, upper_bg)
    fg_mask = cv2.bitwise_not(mask)
    return cv2.bitwise_and(image_np, image_np, mask=fg_mask)

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((128, 128))
    img_array = np.array(image).astype(np.uint8)

    # Auto Remove Background
    img_array = remove_background(img_array)

    img_array = img_array.astype(np.float32) * contrast_boost
    img_array = np.clip(img_array, 0, 255)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor((1, 1, 1))
    ax.set_axis_off()

    _x, _y, _z, _dx, _dy, _dz, _colors = [], [], [], [], [], [], []
    h, w, _ = img_array.shape

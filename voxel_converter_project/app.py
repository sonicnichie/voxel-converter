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
z_scale = st.slider("Z Height Scale", min_value=1, max_value=50, value=5)
contrast_boost = st.slider("Contrast Boost (0.5 - 3.0)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
mode = st.selectbox("Rendering Mode", ["Voxel Full Color", "Voxel Edge Only"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((64, 64))
    img_array = np.array(image).astype(np.float32) * contrast_boost
    img_array = np.clip(img_array, 0, 255)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor((1, 1, 1))
    ax.set_axis_off()

    _x, _y, _z, _dx, _dy, _dz, _colors = [], [], [], [], [], [], []
    h, w, _ = img_array.shape

    if mode == "Voxel Edge Only":
        gray = np.mean(img_array, axis=2).astype(np.uint8)
        edges = cv2.Canny(gray, 50, 150)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = img_array[y:y+block_size, x:x+block_size]
            r, g, b = np.mean(block[:, :, 0]), np.mean(block[:, :, 1]), np.mean(block[:, :, 2])
            brightness = 0.299*r + 0.587*g + 0.114*b
            height = (brightness / 255.0) * z_scale

            include = True
            if mode == "Voxel Edge Only":
                edge_block = edges[y:y+block_size, x:x+block_size]
                if np.mean(edge_block) < 10:
                    include = False

            if include and height > 0.5:
                _x.append(x)
                _y.append(h - y)
                _z.append(0)
                _dx.append(block_size)
                _dy.append(block_size)
                _dz.append(height)
                _colors.append((r/255.0, g/255.0, b/255.0))

    ax.bar3d(_x, _y, _z, _dx, _dy, _dz, color=_colors, shade=True, edgecolor='k', linewidth=0.05)
    ax.view_init(elev=45, azim=45)
    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button("💾 Download Voxel Image as PNG", data=buf.getvalue(), file_name="voxel_output.png", mime="image/png")

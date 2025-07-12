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
contrast_boost = st.slider("Contrast Boost (0.5 - 3.0)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
mode = st.selectbox("Rendering Mode", ["Voxel Full Color", "Voxel Edge Only", "Voxel Silhouette"])

# Fungsi sederhana untuk hapus background terang
def remove_background(image_np):
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    lower_bg = np.array([0, 0, 120])
    upper_bg = np.array([255, 60, 255])
    mask = cv2.inRange(hsv, lower_bg, upper_bg)
    fg_mask = cv2.bitwise_not(mask)
    return cv2.bitwise_and(image_np, image_np, mask=fg_mask)

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((128, 128))  # Ukuran tetap
    img_array = np.array(image).astype(np.uint8)

    # Hapus background terang
    img_array = remove_background(img_array)

    # Tingkatkan kontras
    img_array = img_array.astype(np.float32) * contrast_boost
    img_array = np.clip(img_array, 0, 255)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor((1, 1, 1))
    ax.set_axis_off()

    _x, _y, _z, _dx, _dy, _dz, _colors = [], [], [], [], [], [], []
    h, w, _ = img_array.shape

    # Mode deteksi pinggiran
    if mode in ["Voxel Edge Only", "Voxel Silhouette"]:
        gray = np.mean(img_array, axis=2).astype(np.uint8)
        if mode == "Voxel Edge Only":
            mask = cv2.Canny(gray, 50, 150)
        elif mode == "Voxel Silhouette":
            mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 11, 10)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = img_array[y:y+block_size, x:x+block_size]
            r, g, b = np.mean(block[:, :, 0]), np.mean(block[:, :, 1]), np.mean(block[:, :, 2])
            brightness = 0.299*r + 0.587*g + 0.114*b
            height = (brightness / 255.0) * z_scale
            height = max(height, 0.5)  # jangan nol biar tetap muncul

            include = True
            color = (r/255.0, g/255.0, b/255.0)

            if mode in ["Voxel Edge Only", "Voxel Silhouette"]:
                mask_block = mask[y:y+block_size, x:x+block_size]
                if np.mean(mask_block) < 10:
                    include = False
                if mode == "Voxel Silhouette":
                    height = z_scale
                    color = (0.2, 0.2, 0.2)

            if include:
                if (r + g + b) > 15:  # skip blok gelap/hitam
                    _x.append(x)
                    _y.append(h - y)
                    _z.append(0)
                    _dx.append(block_size)
                    _dy.append(block_size)
                    _dz.append(height)
                    _colors.append(color)

    if len(_x) == 0:
        st.warning("⚠️ Tidak ada voxel yang dihasilkan. Coba ubah block size, contrast, atau rendering mode.")
    else:
        ax.bar3d(_x, _y, _z, _dx, _dy, _dz, color=_colors, shade=True, edgecolor='k', linewidth=0.05)
        ax.view_init(elev=45, azim=45)
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.download_button("💾 Download Voxel Image as PNG", data=buf.getvalue(), file_name="voxel_output.png", mime="image/png")

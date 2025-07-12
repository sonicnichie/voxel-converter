import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import io

st.set_page_config(page_title="3D Voxel Style Image Converter", layout="wide")
st.title("🧊 3D Voxel Style Image Converter")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

block_size = st.slider("Block Size (Smaller = More Voxels)", min_value=1, max_value=10, value=4)
z_scale = st.slider("Z Height Scale", min_value=1, max_value=50, value=10)
brightness_threshold = st.slider("Brightness Threshold", 0, 255, 30)

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((64, 64))
    img_array = np.array(image)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()

    _x, _y, _z, _dx, _dy, _dz, _colors = [], [], [], [], [], [], []

    for y in range(0, img_array.shape[0], block_size):
        for x in range(0, img_array.shape[1], block_size):
            r, g, b = img_array[y, x]
            brightness = 0.299*r + 0.587*g + 0.114*b

            if brightness > brightness_threshold:
                height = (brightness / 255) * z_scale
                _x.append(x)
                _y.append(img_array.shape[0] - y)  # flip y for upright
                _z.append(0)
                _dx.append(block_size)
                _dy.append(block_size)
                _dz.append(height)
                _colors.append((r / 255, g / 255, b / 255))

    ax.bar3d(_x, _y, _z, _dx, _dy, _dz, color=_colors, shade=True)
    ax.view_init(elev=60, azim=135)
    st.pyplot(fig)

    # Tombol download gambar
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button("💾 Download Voxel Image as PNG", data=buf.getvalue(), file_name="voxel_output.png", mime="image/png")

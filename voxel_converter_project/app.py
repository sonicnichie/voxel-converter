import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import io

st.set_page_config(page_title="3D Voxel Style Image Converter", layout="wide")
st.title("🧊 3D Voxel Style Image Converter")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

block_size = st.slider("Block Size (Smaller = More Voxels)", min_value=1, max_value=20, value=6)
z_scale = st.slider("Z Height Scale", min_value=1, max_value=100, value=20)
brightness_threshold = st.slider("Brightness Threshold", 0, 255, 30)

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((64, 64))
    img_array = np.array(image)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()

    x_size, y_size, _ = img_array.shape
    
    for x in range(0, x_size, block_size):
        for y in range(0, y_size, block_size):
            r, g, b = img_array[x, y]
            brightness = 0.299*r + 0.587*g + 0.114*b
            
            if brightness > brightness_threshold:
                height = (brightness / 255) * z_scale
                color = (r / 255, g / 255, b / 255)
                ax.bar3d(x, y, 0, block_size, block_size, height, color=color, shade=True)

    ax.view_init(elev=45, azim=135)
    st.pyplot(fig)

    # Tombol download gambar
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button("💾 Download Voxel Image as PNG", data=buf.getvalue(), file_name="voxel_output.png", mime="image/png")

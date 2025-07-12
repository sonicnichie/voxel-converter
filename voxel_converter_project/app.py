import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def image_to_voxel(image_path, block_size=10, brightness_threshold=100):
    # Load image and resize for faster processing
    img = Image.open(image_path).convert('L')  # convert to grayscale
    img = img.resize((int(img.width / block_size), int(img.height / block_size)))

    data = np.asarray(img)
    x, y = data.shape

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(x):
        for j in range(y):
            brightness = data[i, j]
            if brightness < brightness_threshold:
                height = (255 - brightness) / 20
                ax.bar3d(j, i, 0, 1, 1, height, shade=True, color=plt.cm.gray(brightness/255))

    ax.set_axis_off()
    st.pyplot(fig)

# Streamlit UI
st.title('3D Voxel Style Image Converter')
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    image_to_voxel(uploaded_image)

import streamlit as st
import numpy as np
import cv2
from skimage import transform, io
from skimage.util import img_as_ubyte
from skimage.transform import swirl

# Title
st.title("🌀 Image Warping and Distortion Effects")

# Image uploader
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = io.imread(uploaded_file)
    st.image(image, caption="Original Image", use_container_width=True)

    # Effect selector
    effect = st.selectbox("🎨 Choose an effect", ["None", "Swirl", "Fisheye", "Perspective Warp"])

    if effect == "Swirl":
        strength = st.slider("Swirl Strength", 0, 20, 5)
        radius = st.slider("Swirl Radius", 50, 500, 200)
        swirled = swirl(image, rotation=0, strength=strength, radius=radius)
        st.image(img_as_ubyte(swirled), caption="🌀 Swirled Image", use_container_width=True)

    elif effect == "Fisheye":
        def fisheye(img):
            rows, cols = img.shape[:2]
            K = np.array([[cols, 0, cols/2],
                          [0, rows, rows/2],
                          [0, 0, 1]])
            D = np.array([0.3, 0.0, 0.0, 0.0])  # Fisheye distortion coefficients
            map1, map2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), K, (cols, rows), cv2.CV_16SC2)
            return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

        fish = fisheye(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        st.image(cv2.cvtColor(fish, cv2.COLOR_BGR2RGB), caption="🐟 Fisheye Effect", use_container_width=True)

    elif effect == "Perspective Warp":
        rows, cols = image.shape[:2]
        margin = 60
        src_points = np.float32([[margin, margin], [cols - margin, margin],
                                 [margin, rows - margin], [cols - margin, rows - margin]])
        dst_points = np.float32([[margin + 40, margin + 20], [cols - margin - 20, margin],
                                 [margin, rows - margin], [cols - margin - 40, rows - margin - 20]])
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped = cv2.warpPerspective(image, matrix, (cols, rows))
        st.image(warped, caption="📐 Perspective Warp", use_container_width=True)

    elif effect == "None":
        st.info("👈 Select a distortion effect from the dropdown above.")

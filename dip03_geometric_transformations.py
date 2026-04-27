import cv2
import numpy as np
import streamlit as st
from PIL import Image

def geometric_lab():
    st.set_page_config(page_title="Geometric Transformation Lab", layout="wide")
    st.title("Geometric Transformations & ROI Dashboard")
    st.write("Eksperimen transformasi spasial untuk kalibrasi dan pre-processing citra.")

    # --- SIDEBAR: CONTROLS ---
    st.sidebar.header("Transformation Tools")
    uploaded_file = st.sidebar.file_uploader("Upload Citra Riset", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Load image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        h, w = img.shape[:2]

        # 1. Scaling & Rotation
        st.sidebar.subheader("Scaling & Rotation")
        scale = st.sidebar.slider("Scale (Zoom)", 0.1, 3.0, 1.0, 0.1)
        angle = st.sidebar.slider("Rotation Angle", -180, 180, 0)

        # 2. Translation
        st.sidebar.subheader("Translation")
        tx = st.sidebar.slider("Translate X (px)", -w, w, 0)
        ty = st.sidebar.slider("Translate Y (px)", -h, h, 0)

        # 3. Cropping (ROI)
        st.sidebar.subheader("Cropping / ROI")
        col_c1, col_c2 = st.sidebar.columns(2)
        crop_x = col_c1.slider("X Start", 0, w, 0)
        crop_y = col_c2.slider("Y Start", 0, h, 0)
        crop_w = col_c1.slider("Crop Width", 10, w - crop_x, w - crop_x)
        crop_h = col_c2.slider("Crop Height", 10, h - crop_y, h - crop_y)

        # 4. Affine / Perspective Toggle
        st.sidebar.subheader("Advanced Mapping")
        mode = st.sidebar.selectbox("Mode", ["None", "Affine (Shear)", "Perspective (Warp)"])

        # --- PROCESSING PIPELINE ---

        # A. Cropping (ROI) - Dilakukan pertama untuk efisiensi
        processed = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        curr_h, curr_w = processed.shape[:2]

        # B. Translation Matrix
        M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
        processed = cv2.warpAffine(processed, M_trans, (curr_w, curr_h))

        # C. Rotation & Scaling Matrix
        center = (curr_w // 2, curr_h // 2)
        M_rot_scale = cv2.getRotationMatrix2D(center, angle, scale)
        processed = cv2.warpAffine(processed, M_rot_scale, (curr_w, curr_h))

        # D. Advanced Mapping
        if mode == "Affine (Shear)":
            pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
            pts2 = np.float32([[50, 100], [200, 50], [100, 250]])
            M_affine = cv2.getAffineTransform(pts1, pts2)
            processed = cv2.warpAffine(processed, M_affine, (curr_w, curr_h))
        elif mode == "Perspective (Warp)":
            pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
            pts2 = np.float32([[0, 0], [curr_w, 0], [0, curr_h], [curr_w, curr_h]])
            M_persp = cv2.getPerspectiveTransform(pts1, pts2)
            processed = cv2.warpPerspective(processed, M_persp, (curr_w, curr_h))

        # --- DISPLAY ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Dimensi Asli: {w}x{h}")

        with col2:
            st.subheader("Transformed Image")
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Download Button
            _, buffer = cv2.imencode(".jpg", processed)
            st.download_button("Save Transformation", data=buffer.tobytes(), file_name="transformed.jpg")

    else:
        st.info("Silakan unggah citra untuk memulai transformasi.")

if __name__ == '__main__':
    geometric_lab()

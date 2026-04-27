import cv2
import numpy as np
import streamlit as st
from PIL import Image

def spatial_filtering_full():
    st.set_page_config(page_title="Spatial Filtering Full Lab", layout="wide")
    st.title("Spatial Filtering & Neighborhood Operations (Full Version)")
    st.write("Eksperimen Smoothing, Sharpening, dan Edge Detection untuk Riset Computer Vision")

    st.sidebar.header("Filter Configuration")
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        category = st.sidebar.selectbox("Pilih Kategori Operasi", 
                                        ["Smoothing (Low Pass)", "Sharpening (High Pass)", "Edge Detection"])
        
        processed = None
        desc = ""

        # --- 1. SMOOTHING ---
        if category == "Smoothing (Low Pass)":
            algo = st.sidebar.radio("Algoritma", ["Box Filter", "Gaussian Blur", "Median Blur", "Bilateral Filter"])
            k_size = st.sidebar.slider("Kernel Size", 3, 25, 5, step=2)
            if algo == "Box Filter":
                processed = cv2.blur(img, (k_size, k_size))
                desc = "Smoothing sederhana menggunakan rata-rata tetangga."
            elif algo == "Gaussian Blur":
                processed = cv2.GaussianBlur(img, (k_size, k_size), 0)
                desc = "Gold Standard: Menghaluskan noise dengan distribusi normal."
            elif algo == "Median Blur":
                processed = cv2.medianBlur(img, k_size)
                desc = "Sangat efektif untuk menghapus noise bintik (salt-and-pepper)."
            elif algo == "Bilateral Filter":
                processed = cv2.bilateralFilter(img, k_size, 75, 75)
                desc = "Edge-Preserving: Menghaluskan permukaan tanpa mengaburkan tepi objek."

        # --- 2. SHARPENING ---
        elif category == "Sharpening (High Pass)":
            algo = st.sidebar.radio("Metode", ["Laplacian", "Unsharp Masking", "Standard Sharpen"])
            if algo == "Laplacian":
                lap = cv2.Laplacian(img_gray, cv2.CV_64F)
                processed = cv2.convertScaleAbs(lap)
                desc = "Mendeteksi perubahan intensitas cepat (turunan kedua)."
            elif algo == "Unsharp Masking":
                blur = cv2.GaussianBlur(img, (5, 5), 1.0)
                processed = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
                desc = "Mempertegas detail dengan mengurangi versi blur dari gambar asli."
            elif algo == "Standard Sharpen":
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                processed = cv2.filter2D(img, -1, kernel)
                desc = "Kernel sharpening klasik untuk mempertegas kontras lokal."

        # --- 3. EDGE DETECTION (FULL LOGIC) ---
        elif category == "Edge Detection":
            algo = st.sidebar.selectbox("Operator", ["Canny", "Sobel", "Prewitt", "Scharr", "Roberts Cross", "Kirsch Compass"])
            
            # --- Canny Logic ---
            if algo == "Canny":
                t1 = st.sidebar.slider("Threshold Low", 0, 255, 100)
                t2 = st.sidebar.slider("Threshold High", 0, 255, 200)
                processed = cv2.Canny(img_gray, t1, t2)
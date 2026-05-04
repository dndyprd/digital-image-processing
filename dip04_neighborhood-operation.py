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

        # --- 3. EDGE DETECTION ---
        elif category == "Edge Detection":
            algo = st.sidebar.selectbox("Operator", ["Canny", "Sobel", "Prewitt", "Scharr", "Roberts Cross", "Kirsch Compass"])
            
            if algo == "Canny":
                t1 = st.sidebar.slider("Threshold Low", 0, 255, 100)
                t2 = st.sidebar.slider("Threshold High", 0, 255, 200)
                processed = cv2.Canny(img_gray, t1, t2)
                desc = "Algoritma multi-tahap paling robust untuk deteksi tepi presisi."
            
            elif algo == "Sobel":
                direction = st.sidebar.radio("Arah", ["Horizontal (Dx)", "Vertical (Dy)", "Combined"])
                dx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
                dy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
                if direction == "Horizontal (Dx)": processed = cv2.convertScaleAbs(dx)
                elif direction == "Vertical (Dy)": processed = cv2.convertScaleAbs(dy)
                else: processed = cv2.convertScaleAbs(cv2.magnitude(dx, dy))
                desc = "Menggunakan turunan pertama dengan pembobotan lebih pada pusat."

            elif algo == "Prewitt":
                direction = st.sidebar.radio("Arah", ["Horizontal", "Vertical", "Diagonal", "Combined"])
                Kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
                Ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
                Kd = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]])
                if direction == "Horizontal": processed = cv2.filter2D(img_gray, -1, Kx)
                elif direction == "Vertical": processed = cv2.filter2D(img_gray, -1, Ky)
                elif direction == "Diagonal": processed = cv2.filter2D(img_gray, -1, Kd)
                else:
                    px = cv2.filter2D(img_gray, cv2.CV_64F, Kx)
                    py = cv2.filter2D(img_gray, cv2.CV_64F, Ky)
                    processed = cv2.convertScaleAbs(cv2.magnitude(px, py))
                desc = "Serupa Sobel namun menggunakan pembobotan seragam (1)."

            elif algo == "Scharr":
                direction = st.sidebar.radio("Arah", ["Horizontal", "Vertical", "Combined"])
                sx = cv2.Scharr(img_gray, cv2.CV_64F, 1, 0)
                sy = cv2.Scharr(img_gray, cv2.CV_64F, 0, 1)
                if direction == "Horizontal": processed = cv2.convertScaleAbs(sx)
                elif direction == "Vertical": processed = cv2.convertScaleAbs(sy)
                else: processed = cv2.convertScaleAbs(cv2.magnitude(sx, sy))
                desc = "Versi Sobel yang lebih akurat untuk mendeteksi tepi pada sudut miring."

            elif algo == "Roberts Cross":
                direction = st.sidebar.radio("Arah", ["Diagonal 1", "Diagonal 2", "Combined"])
                K1 = np.array([[1, 0], [0, -1]])
                K2 = np.array([[0, 1], [-1, 0]])
                if direction == "Diagonal 1": processed = cv2.filter2D(img_gray, -1, K1)
                elif direction == "Diagonal 2": processed = cv2.filter2D(img_gray, -1, K2)
                else:
                    r1 = cv2.filter2D(img_gray, cv2.CV_64F, K1)
                    r2 = cv2.filter2D(img_gray, cv2.CV_64F, K2)
                    processed = cv2.convertScaleAbs(cv2.magnitude(r1, r2))
                desc = "Operator 2x2 yang sangat cepat untuk deteksi tepi diagonal tajam."

            elif algo == "Kirsch Compass":
                direction = st.sidebar.selectbox("Arah Mata Angin", 
                                                ["North", "Northwest", "West", "Southwest", "South", "Southeast", "East", "Northeast"])
                k_map = {
                    "North": [[5, 5, 5], [-3, 0, -3], [-3, -3, -3]],
                    "Northwest": [[5, 5, -3], [5, 0, -3], [-3, -3, -3]],
                    "West": [[5, -3, -3], [5, 0, -3], [5, -3, -3]],
                    "Southwest": [[-3, -3, -3], [5, 0, -3], [5, 5, -3]],
                    "South": [[-3, -3, -3], [-3, 0, -3], [5, 5, 5]],
                    "Southeast": [[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]],
                    "East": [[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]],
                    "Northeast": [[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]
                }
                kernel = np.array(k_map[direction])
                processed = cv2.filter2D(img_gray, -1, kernel)
                desc = f"Mendeteksi tepi secara spesifik dari arah {direction}."

        # --- OUTPUT DISPLAY ---
        if processed is not None:
            st.info(f"**Info:** {desc}")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Input")
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            with col2:
                st.subheader("Result")
                # Cek jika output grayscale atau berwarna
                disp = processed if len(processed.shape) == 2 else cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                st.image(disp, use_container_width=True)
            
            buffer = cv2.imencode(".png", processed)[1]
            st.download_button("Save Result", data=buffer.tobytes(), file_name="output_filter.png")
        else:
            st.info("Silakan unggah citra untuk mengaktifkan lab.")

if __name__ == "__main__":
    spatial_filtering_full()
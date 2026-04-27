import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Fungsi caching untuk performa loading citra yang lebih cepat
@st.cache_data
def load_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, 1)

def point_operations_app():
    st.set_page_config(page_title="CV Research Lab - Point Ops", layout="wide")
    st.title("Point Operations Dashboard")
    st.write("Operasi berbasis piksel tunggal untuk manipulasi intensitas dan segmentasi awal.")

    # --- SIDEBAR: CONTROLS ---
    st.sidebar.header("Data Input")
    uploaded_file = st.sidebar.file_uploader("Pilih Citra Riset", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None:
        # Load image (menggunakan cache)
        img_bgr = load_image(uploaded_file)

        # Menu Utama Operasi Titik
        st.sidebar.subheader("Select Operation")
        operation = st.sidebar.selectbox(
            "Jenis Operasi",
            ["Brightness & Contrast", "Histogram Equalization", "Thresholding", "Image Negative"]
        )

        processed = img_bgr.copy()
        desc = ""

        # --- LOGIC PER OPERASI ---
        if operation == "Brightness & Contrast":
            alpha = st.sidebar.slider("Contrast (Alpha)", 0.0, 3.0, 1.0, 0.1)
            beta = st.sidebar.slider("Brightness (Beta)", -100, 100, 0)
            r_off = st.sidebar.slider("Red Offset", -255, 255, 0)
            g_off = st.sidebar.slider("Green Offset", -255, 255, 0)
            b_off = st.sidebar.slider("Blue Offset", -255, 255, 0)

            # Linear Transformation
            adjusted = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
            b_chan, g_chan, r_chan = cv2.split(adjusted)
            r_chan = cv2.add(r_chan, r_off)
            g_chan = cv2.add(g_chan, g_off)
            b_chan = cv2.add(b_chan, b_off)
            processed = cv2.merge([b_chan, g_chan, r_chan])
            desc = "Penyesuaian intensitas linier dan pergeseran saluran warna."

        elif operation == "Histogram Equalization":
            # Pemerataan pada Y-channel (Luminance) untuk mempertahankan warna aslinya
            img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            processed = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            desc = "Pemerataan histogram untuk meningkatkan kontras (pada channel luminance)."

        elif operation == "Thresholding":
            thresh_val = st.sidebar.slider("Threshold Value", 0, 255, 127)
            thresh_type = st.sidebar.selectbox("Method", ["Binary", "Binary Inv", "Otsu (Auto)"])

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            if thresh_type == "Binary":
                _, processed = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            elif thresh_type == "Binary Inv":
                _, processed = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
            else:
                _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                desc = "Otsu: Menentukan ambang optimal berdasarkan variansi antar-kelas secara otomatis"

            if not desc: desc = f"Segmentasi Biner pada nilai ambang {thresh_val}."

        elif operation == "Image Negative":
            processed = cv2.bitwise_not(img_bgr)
            desc = "Membalikkan nilai intensitas: s = (L - 1) - r"

        # --- LAYOUT: PREVIEW & HISTOGRAM ---
        st.info(f"**Analisis:** {desc}")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Visual Preview")
            # Konversi untuk display
            is_gray = len(processed.shape) == 2
            display_img = processed if is_gray else cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            st.image(display_img, use_container_width=True)

            # Download Button
            _, buffer = cv2.imencode(".png", processed)
            st.download_button("Export Citra", buffer.tobytes(), "result.png", "image/png")

        with col2:
            st.subheader("Intensity Histogram")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            if is_gray:
                hist = cv2.calcHist([processed], [0], None, [256], [0, 256])
                ax.plot(hist, color='black', lw=1.5)
                ax.fill_between(range(256), hist.flatten(), color='gray', alpha=0.3)
            else:
                for i, col in enumerate(['b', 'g', 'r']):
                    hist = cv2.calcHist([processed], [i], None, [256], [0, 256])
                    ax.plot(hist, color=col, lw=1.5)
            
            ax.set_xlim([0, 256])
            ax.set_title("Pixel Distribution")
            st.pyplot(fig)

    else:
        st.info("Pusat Riset Citra Digital: Silakan unggah gambar untuk memulai.")

if __name__ == "__main__":
    point_operations_app()

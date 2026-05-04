import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Image Segmentation Lab", 
    page_icon="🔬",
    layout="wide"
)

# WATERSHED PIPELINE
def run_watershed(image_array, noise_kernel, dist_threshold, iteration_morph):
    """
    Marker-Controlled Watershed Pipeline
    """
    # 1. Pre-processing
    img = np.copy(image_array)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Binarization menggunakan Otsu's Thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Noise Removal
    kernel = np.ones((noise_kernel, noise_kernel), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=iteration_morph)

    # 3. Mencari 'Sure Background'
    sure_bg = cv2.dilate(opening, kernel, iterations=iteration_morph)

    # 4. Mencari 'Sure Foreground'
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, dist_threshold * dist_transform.max(), 255, 0)

    # 5. Mencari 'Unknown Region'
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6. Marker Labeling
    _, markers = cv2.connectedComponents(sure_fg)

    # Tambah 1 ke semua label agar background bernilai 1 (bukan 0)
    markers = markers + 1

    # Tandai area yang tidak diketahui dengan 0
    markers[unknown == 255] = 0

    # 7. Apply watershed
    markers = cv2.watershed(img, markers)

    # 8. Post-processing : Visualisasi
    # Tandai boundary dengan warna merah
    img[markers == -1] = [255, 0, 0]
    
    return img, markers, dist_transform

# UI COMPONENTS
def main():
    st.title("Interactive Marker-Controlled Watershed")
    st.markdown("""
    Algoritma ini sangat efektif untuk segmentasi objek yang saling bersentuhan.
    Gunakan kontrol di sidebar untuk menyesuaikan deteksi *foreground* dan *background*.
    """)

    # SIDEBAR
    st.sidebar.header("Parameter Segmentasi")
    
    k_size = st.sidebar.slider("Morphology Kernel Size", 3, 15, 3, step=2)
    iters = st.sidebar.slider("Morphology Iterations", 1, 5, 2)
    dist_val = st.sidebar.slider("Distance Transform Threshold", 0.1, 0.9, 0.5,
                                 help="Semakin tinggi nilai, semakin selektif penentuan pusat objek.")
    
    st.sidebar.divider()
    out_dir = st.sidebar.text_input("Artifact Path", r"D:\Python\DigitalImageProcessing\digital-image-processing")

    # --- UPLOAD SECTION ---
    uploaded_file = st.sidebar.file_uploader("Upload Citra untuk Segmentasi", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        raw_img = Image.open(uploaded_file).convert("RGB")
        img_arr = np.array(raw_img)

        # Prosedur Pemrosesan
        with st.spinner("Menghitung Watershed..."):
            result_img, markers, dist_map = run_watershed(img_arr, k_size, dist_val, iters)

        # --- DISPLAY TABS ---
        tab1, tab2, tab3 = st.tabs(["Segmentation Result", "Distance Transform", "Metrics"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(img_arr, use_container_width=True)
            with col2:
                st.subheader("Segmented Result")
                st.image(result_img, use_container_width=True)
            
            # Tombol Simpan
            if st.button("Simpan Artifact ke Server"):
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime('%H%M%S')
                filename = f"watershed_{timestamp}.png"
                path = os.path.join(out_dir, filename)
                
                # Convert back to BGR for OpenCV saving
                save_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(path, save_img)
                st.success(f"Tersimpan di: `{path}`")

        with tab2:
            st.subheader("Topographic Map (Distance Transform)")
            st.write("Area paling terang adalah 'puncak' atau pusat massa dari objek.")
            # Normalisasi untuk display
            dist_disp = cv2.normalize(dist_map, None, 0, 255, cv2.NORM_MINMAX)
            st.image(dist_disp.astype(np.uint8), use_container_width=True, clamp=True)

        with tab3:
            # Hitung jumlah objek (dikurangi label background dan boundary)
            unique_labels = np.unique(markers)
            # -1 untuk boundary, 1 untuk background, sisanya objek
            obj_count = len(unique_labels) - 2 
            
            st.metric("Total Objek Terdeteksi", f"{max(0, obj_count)} unit")
            
            st.info("""
            **Catatan:** 
            * Warna merah pada hasil segmentasi menunjukkan 'watershed line' atau batas antar objek.
            * Jika objek terlalu banyak (over-segmentation), naikkan nilai *Distance Transform Threshold*.
            """)

if __name__ == "__main__":
    main()
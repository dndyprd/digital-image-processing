import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="DIP Lab - Research Nexus", layout="wide")

def process_image_input(uploaded_file):
    """Pastikan citra dalam format RGB 3-channel (Mengabaikan Alpha)"""
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img)

def apply_arithmetic(img1, img2, op, alpha=0.5):
    #Sinkronisasi ukuran gambar kedua ke gambar pertama
    img2_res = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    if op == "Add":
        return cv2.add(img1, img2_res)
    elif op == "Subtract":
        return cv2.subtract(img1, img2_res)
    elif op == "Multiply":
        res = cv2.multiply(img1.astype(np.float32), img2_res.astype(np.float32))
        return np.clip(res / 255.0, 0, 255).astype(np.uint8)
    elif op == "Divide":
        img2_res[img2_res == 0] = 1
        res = cv2.divide(img1.astype(np.float32), img2_res.astype(np.float32))
        return np.clip(res * 255.0, 0, 255).astype(np.uint8)
    elif op == "Blend (Linear)":
        return cv2.addWeighted(img1, alpha, img2_res, 1 - alpha, 0)
    return img1

def apply_logic(img1, img2, op):
    # NOT hanya butuh img1, selain itu butuh resize img2
    if op != "NOT (Image 1)":
        img2_res = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    if op == "AND":
        return cv2.bitwise_and(img1, img2_res)
    elif op == "OR":
        return cv2.bitwise_or(img1, img2_res)
    elif op == "XOR":
        return cv2.bitwise_xor(img1, img2_res)
    elif op == "NOT (Image 1)":
        return cv2.bitwise_not(img1)
    return img1

st.title("Modul Praktikum : Operasi Aritmatika dan Logika Citra")
st.caption("Environment: Research-Nexus (Stable 2026)")
st.markdown("---")

# Sidebar
st.sidebar.header("Control Panel")
category = st.sidebar.radio("Pilih Kategori:", ["Aritmatika", "Logika"])

if category == "Aritmatika":
    operation = st.sidebar.selectbox("Operasi:", ["Add", "Subtract", "Multiply", "Divide", "Blend"])
    alpha = st.sidebar.slider("Alpha", 0.0, 1.0, 0.5) if "Blend" in operation else 0.5
else:
    operation = st.sidebar.selectbox("Operasi:", ["AND", "OR", "XOR", "NOT (Image 1)"])

#Layout Kolom
col1, col2 = st.columns(2)
img1, img2 = None, None

with col1:
    # Upload 1
    file1 = st.file_uploader("Pilih Citra 1", type=["jpg", "jpeg", "png", "webp"], key="u1")
    if file1:
        img1 = process_image_input(file1)
        st.image(img1, caption="Citra 1", width='stretch')

with col2:
    # Upload 2 dinonaktifkan jika operasi adalah NOT
    disable_upload2 = (category == "Logika" and operation == "NOT (Image 1)")
    file2 = st.file_uploader("Pilih Citra 2", type=["jpg", "jpeg", "png", "webp"], key="u2", disabled=disable_upload2)
    if file2 and not disable_upload2:
        img2 = process_image_input(file2)
        st.image(img2, caption="Citra 2", width='stretch')
    elif disable_upload2:
        st.info("Operasi NOT hanya memerlukan Citra 1.")

# Area Hasil
st.markdown("---")
st.subheader("Hasil Pemrosesan")

# Logika eksekusi yang diperbaiki :
# Jalankan jika ada img1 DAN (img2 ada ATAU operasi NOT)
can_process = img1 is not None and (img2 is not None or (category == "Logika" and operation == "NOT (Image 1)"))

if can_process:
    with st.spinner('Memproses Citra...'):
        if category == "Aritmatika":
            result = apply_arithmetic(img1, img2 if img2 is not None else img1, operation, alpha)
        else:
            result = apply_logic(img1, img2 if img2 is not None else img1, operation)
        
    st.image(result, caption=f"Output: {operation}", width='stretch')
    
    # Download Button
    result_pil = Image.fromarray(result)
    buf = io.BytesIO()
    result_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    
    st.download_button(
        label="Simpan Hasil Pemrosesan",
        data=byte_im,
        file_name=f"hasil_{operation.lower()}.png",
        mime="image/png"
    )
else:
    st.warning("Menunggu input gambar yang diperlukan untuk oeprasi ini!")
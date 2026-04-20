import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Utility Functions
def resize_for_display(image, max_size=400):
    '''
    Resize image dan sesuaikan dengan dimensi GUI Windows
    '''
    h, w = image.shape[:2]
    scale = min(max_size/h, max_size/w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size)

def cv_to_tk(image):
    '''
    Convert OpenCV image (BGR) to Tkinter
    '''
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tk = Image.fromarray(image_rgb)
    return ImageTk.PhotoImage(image_tk)

def add_images(img1, img2): 
    '''
    Operasi penjumlahan citra (Saturasi)
    '''
    return cv2.add(img1, img2)

def subtract_images(img1, img2):
    '''
    Operasi subtraksi Citra (Saturasi)
    '''
    return cv2.subtract(img1, img2)

def multiply_images(img1, img2):
    '''
    Operasi multiplikasi citra, yang sudah di scaled untuk membatasi overflow
    ''' 
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    result = cv2.multiply(img1, img2)
    return np.clip(result / 255.0, 0, 255).astype(np.uint8)

def divide_images(img1, img2):
    '''
    Operasi pembagian citra dengan pembatasan agar pembagian 0 tidak terjadi
    '''
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    result = cv2.divide(img1, img2)
    return np.clip(result, 0, 255).astype(np.uint8)

# Logical Operations
def and_images(img1, img2):
    return cv2.bitwise_and(img1, img2)

def or_images(img1, img2):
    return cv2.bitwise_or(img1, img2)

def xor_images(img1, img2):
    return cv2.bitwise_xor(img1, img2)

def not_images(img1):
    return cv2.bitwise_not(img1)

# GUI Class

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digital Image Processing App") 
        self.root.geometry("1200x700")
        self.root.configure(bg="#121212")

        # Define Colors
        self.bg_color = "#121212"
        self.sidebar_color = "#1e1e1e"
        self.accent_blue = "#0078d4"
        self.accent_green = "#28a745"
        self.accent_orange = "#fd7e14"
        self.text_color = "#e0e0e0"

        self.img1 = None
        self.img2 = None
        self.result = None

        # Sidebar Frame
        self.sidebar = tk.Frame(root, bg=self.sidebar_color, width=200, padx=10, pady=10)
        self.sidebar.pack(side="left", fill="y")

        tk.Label(self.sidebar, text="PENGATURAN", bg=self.sidebar_color, fg=self.accent_blue, font=("Segoe UI", 12, "bold")).pack(pady=(0, 10))

        # Image Load Buttons
        self.btn_style = {"bg": "#333333", "fg": "white", "relief": "flat", "padx": 10, "pady": 5, "font": ("Segoe UI", 9)}
        
        tk.Button(self.sidebar, text="📁 Pilih Gambar 1", command=self.load_img1, **self.btn_style).pack(fill="x", pady=5)
        tk.Button(self.sidebar, text="📁 Pilih Gambar 2", command=self.load_img2, **self.btn_style).pack(fill="x", pady=5)

        tk.Label(self.sidebar, text="OPERASI ARITMATIKA", bg=self.sidebar_color, fg=self.accent_green, font=("Segoe UI", 10, "bold")).pack(pady=(15, 5))
        
        arith_ops = [
            ("Tambah (Add)", lambda: self.apply(add_images)),
            ("Kurang (Subtract)", lambda: self.apply(subtract_images)),
            ("Kali (Multiply)", lambda: self.apply(multiply_images)),
            ("Bagi (Divide)", lambda: self.apply(divide_images)),
        ]
        for text, cmd in arith_ops:
            tk.Button(self.sidebar, text=text, command=cmd, **self.btn_style).pack(fill="x", pady=2)

        tk.Label(self.sidebar, text="OPERASI LOGIKA", bg=self.sidebar_color, fg=self.accent_orange, font=("Segoe UI", 10, "bold")).pack(pady=(15, 5))
        
        logic_ops = [
            ("AND", lambda: self.apply(and_images)),
            ("OR", lambda: self.apply(or_images)),
            ("XOR", lambda: self.apply(xor_images)),
            ("NOT (Img 1)", self.apply_not),
        ]
        for text, cmd in logic_ops:
            tk.Button(self.sidebar, text=text, command=cmd, **self.btn_style).pack(fill="x", pady=2)

        # Content Frame
        self.content = tk.Frame(root, bg=self.bg_color, padx=20, pady=20)
        self.content.pack(side="right", expand=True, fill="both")

        # Image Displays
        self.display_frame = tk.Frame(self.content, bg=self.bg_color)
        self.display_frame.pack(expand=True)

        def create_panel(parent, title):
            frame = tk.Frame(parent, bg="#1e1e1e", padx=5, pady=5)
            frame.pack(side="left", padx=10)
            tk.Label(frame, text=title, bg="#1e1e1e", fg=self.text_color, font=("Segoe UI", 9)).pack()
            panel = tk.Label(frame, bg="#252525", width=40, height=20)
            panel.pack()
            return panel

        self.panel1 = create_panel(self.display_frame, "Gambar 1")
        self.panel2 = create_panel(self.display_frame, "Gambar 2")
        self.panel3 = create_panel(self.display_frame, "Hasil (Result)")

    def load_img1(self):
        path = filedialog.askopenfilename()
        if path:
            self.img1 = cv2.imread(path)
            self.update_display()

    def load_img2(self):
        path = filedialog.askopenfilename()
        if path:
            self.img2 = cv2.imread(path)
            self.update_display()

    def update_display(self):
        if self.img1 is not None:
            self.show_image(self.img1, self.panel1)
        if self.img2 is not None:
            self.show_image(self.img2, self.panel2)
        if self.result is not None:
            self.show_image(self.result, self.panel3)

    def apply(self, func):
        if self.img1 is None or self.img2 is None:
            return
        
        # Penyesuaian Ukuran Otomatis (Fix Size Mismatch)
        img2_ready = self.img2
        if self.img1.shape[:2] != self.img2.shape[:2]:
            img2_ready = cv2.resize(self.img2, (self.img1.shape[1], self.img1.shape[0]))
            
        self.result = func(self.img1, img2_ready)
        self.update_display()

    def apply_not(self):
        if self.img1 is not None:
            self.result = not_images(self.img1)
            self.update_display()
    
    def show_image(self, img, panel):
        img_resized = resize_for_display(img)
        img_tk = cv_to_tk(img_resized)
        panel.configure(image=img_tk, width=img_tk.width(), height=img_tk.height())
        panel.image = img_tk

# MAIN
if __name__ == "__main__" :
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
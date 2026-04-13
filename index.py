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

        self.img1 = None
        self.img2 = None
        self.result = None

        #Button
        tk.Button(root, text="Pilih Gambar 1", command=self.load_img1).pack()
        tk.Button(root, text="Pilih Gambar 2", command=self.load_img2).pack()

        tk.Button(root, text="Add", command=lambda: self.apply(add_images)).pack()
        tk.Button(root, text="Subtract", command=lambda: self.apply(subtract_images)).pack()
        tk.Button(root, text="Multiply", command=lambda: self.apply(multiply_images)).pack()
        tk.Button(root, text="Divide", command=lambda: self.apply(divide_images)).pack()
        tk.Button(root, text="AND", command=lambda: self.apply(and_images)).pack()
        tk.Button(root, text="OR", command=lambda: self.apply(or_images)).pack()
        tk.Button(root, text="XOR", command=lambda: self.apply(xor_images)).pack()
        tk.Button(root, text="NOT", command=self.apply_not).pack()

        #Panels
        self.panel1 = tk.Label(root)
        self.panel1.pack(side="left")

        self.panel2 = tk.Label(root)
        self.panel2.pack(side="left")

        self.panel3 = tk.Label(root)
        self.panel3.pack(side="left")

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
        self.result = func(self.img1, self.img2)
        self.update_display()

    def apply_not(self):
        if self.img1 is not None:
            self.result = not_images(self.img1)
            self.show_image(self.result, self.panel3)
    
    def show_image(self, img, panel):
        img_resized =  resize_for_display(img)
        img_tk = cv_to_tk(img_resized)
        panel.configure(image=img_tk)
        panel.image = img_tk

# MAIN
if __name__ == "__main__" :
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
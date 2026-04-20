import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def process_image(input_path=None, output_path="output_result.jpg"):
    if not input_path:
        root_init = tk.Tk()
        root_init.withdraw()
        input_path = filedialog.askopenfilename(
            title="Pilih citra input",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        root_init.destroy()
        if not input_path:
            return

    original_img = cv2.imread(input_path)
    if original_img is None:
        print("Gagal memuat gambar.")
        return

    root = tk.Tk()
    root.title("Image Processing Lab - Histogram & RGB Control")

    ctrl_frame = tk.Frame(root, padx=10, pady=10)
    ctrl_frame.pack(side=tk.LEFT, fill=tk.Y)

    view_frame = tk.Frame(root, padx=10, pady=10)
    view_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    var_contrast = tk.DoubleVar(value=1.0)
    var_brightness = tk.IntVar(value=0)
    var_r = tk.IntVar(value=0)
    var_g = tk.IntVar(value=0)
    var_b = tk.IntVar(value=0)
    var_he = tk.BooleanVar(value=False)
    current_result = [original_img.copy()]

    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=view_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def draw_histogram(img):
        ax.clear()
        colors = ("b", "g", "r")
        for i, col in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            ax.plot(hist, color=col)
        ax.set_xlim([0, 256])
        ax.set_title("Real-time RGB Histogram")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")
        canvas.draw()

    def update_image(*args):
        alpha = var_contrast.get()
        beta = var_brightness.get()
        r_off = var_r.get()
        g_off = var_g.get()
        b_off = var_b.get()

        adjusted = cv2.convertScaleAbs(original_img, alpha=alpha, beta=beta)
        b_chan, g_chan, r_chan = cv2.split(adjusted)

        r_chan = cv2.add(r_chan, r_off)
        g_chan = cv2.add(g_chan, g_off)
        b_chan = cv2.add(b_chan, b_off)

        processed_img = cv2.merge([b_chan, g_chan, r_chan])

        if var_he.get():
            img_yuv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            processed_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        draw_histogram(processed_img)

        cv2.imshow("Processed Image", processed_img)

        current_result[0] = processed_img

    def save_and_close():
        cv2.imwrite(output_path, current_result[0])
        messagebox.showinfo("Saved", f"Processed image saved as {output_path}")
        root.destroy()
        cv2.destroyAllWindows()

    tk.Label(ctrl_frame, text="Contrast (0.0 - 3.0)").pack()
    tk.Scale(
        ctrl_frame,
        from_=0,
        to=3.0,
        resolution=0.1,
        orient=tk.HORIZONTAL,
        variable=var_contrast,
        command=update_image,
        length=200,
    ).pack()

    tk.Label(ctrl_frame, text="Brightness (-100 to 100)").pack()
    tk.Scale(
        ctrl_frame,
        from_=-100,
        to=100,
        orient=tk.HORIZONTAL,
        variable=var_brightness,
        command=update_image,
        length=200,
    ).pack()

    tk.Label(ctrl_frame, text="Red Tone Offset", fg="red").pack()
    tk.Scale(
        ctrl_frame,
        from_=-255,
        to=255,
        orient=tk.HORIZONTAL,
        variable=var_r,
        command=update_image,
        length=200,
    ).pack()

    tk.Label(ctrl_frame, text="Green Tone Offset", fg="green").pack()
    tk.Scale(
        ctrl_frame,
        from_=-255,
        to=255,
        orient=tk.HORIZONTAL,
        variable=var_g,
        command=update_image,
        length=200,
    ).pack()

    tk.Label(ctrl_frame, text="Blue Tone Offset", fg="blue").pack()
    tk.Scale(
        ctrl_frame,
        from_=-255,
        to=255,
        orient=tk.HORIZONTAL,
        variable=var_b,
        command=update_image,
        length=200,
    ).pack()

    tk.Checkbutton(
        ctrl_frame,
        text="Enable Histogram Equalization",
        variable=var_he,
        command=update_image,
    ).pack(pady=10)

    tk.Button(ctrl_frame, text="Save", command=save_and_close, bg="lightblue").pack(pady=10)

    root.protocol("WM_DELETE_WINDOW", lambda: (root.destroy(), cv2.destroyAllWindows()))
    update_image()
    root.mainloop()


if __name__ == "__main__":
    process_image(None, "hasil_pengolahan.jpg")
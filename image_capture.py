#!/usr/bin/python3

import sys
import threading
import time
import PIL.Image
import PIL.ImageTk
import tkinter as tk
import scipy.misc
import numpy as np
import pyscreenshot
import functions

if len(sys.argv) < 4:
    sys.stderr.write("\nUsage:\n\t");
    sys.stderr.write("captureimage.py <dir.model> <input tensor> <output tensor>\n\n")
    sys.exit(1)

def image_to_tk(image):
    image = PIL.Image.fromarray(image)
    return PIL.ImageTk.PhotoImage(image)

class MainWindow:
    def __init__(self, in_shape, out_shape, process_img):
        self.in_size = in_shape[0:2]
        self.out_size = out_shape[0:2]
        self.in_channels = in_shape[2] if len(in_shape) >= 3 else 1
        self.process_img = process_img

        self.window = tk.Tk()
        self.window.attributes('-topmost', True)
        self.window.resizable(width=False, height=False)
        self.window.minsize(width = 200, height = 10)
        self.window.configure(background = 'green')

        self.capture_size = self.scale_on_screen(in_shape[0:2])
        self.display_size = self.scale_on_screen(out_shape[0:2])
        self.borders = [RedBorder(self.window) for i in range(4)]

        self.frame = tk.Frame(
            self.window,
            highlightbackground = "green",
            highlightcolor = "green",
            highlightthickness = 5
        )
        self.frame.pack()

        self.canvas = tk.Canvas(
            self.frame,
            width = self.display_size[1],
            height = self.display_size[0],
            highlightthickness = 0
        )
        self.canvas.pack()

        self.window.bind("<Configure>", self.on_configure)
        self.on_configure(None)

        self.prev_capture = None
        self.start_update_thread()

    def on_configure(self, event):
        self.last_moved = time.time()
        self.draw_borders()

    def scale_on_screen(self, dims):
        factor = 256.0 / max(dims)
        if factor <= 1.0:
            return dims
        return int(dims[0] * factor + 0.5), int(dims[1] * factor + 0.5)

    def draw_borders(self):
        x, y = self.window.winfo_x(), self.window.winfo_y()
        w, h = self.capture_size
        line = 5
        self.borders[0].set_pos(x-w-line*2, y, w+line*2, line)
        self.borders[1].set_pos(x-w-line*2, y+h+line, w+line*2, line)
        self.borders[2].set_pos(x-w-line*2, y+line, line, h)
        self.borders[3].set_pos(x-line, y+line, line, h)
        return x-w-line, y+line

    def start_update_thread(self):
        t = threading.Thread(target = self.update)
        t.daemon = True
        t.start()

    def capture_image(self):
        x, y = self.draw_borders()
        w, h = self.capture_size
        im = pyscreenshot.grab(bbox = (x, y, x+w, y+h))
        im = im.convert(
            "RGBA" if self.in_channels == 4
            else "RGB" if self.in_channels == 3
            else "L"
        )
        return np.asarray(im)

    def update(self):
        while True:
            if time.time() < self.last_moved + 0.3:
                time.sleep(0.1)
                continue

            img = self.capture_image()
            if self.prev_capture is not None:
                diff = np.absolute(self.prev_capture - img)
                if np.sum(diff) < 10.0:
                    time.sleep(0.3)
                    continue
            self.prev_capture = img

            if self.capture_size != self.in_size:
                img = scipy.misc.imresize(img, self.in_size)

            img = self.process_img(img)

            if self.display_size != self.out_size:
                img = scipy.misc.imresize(
                    img, self.display_size,
                    interp = "nearest"
                )

            self.display = image_to_tk(img)
            self.window.after(0, self.update_display)

    def update_display(self):
        self.canvas.image = self.display
        self.canvas.create_image(
            0, 0,
            image = self.canvas.image,
            anchor = tk.NW
        )

class RedBorder:
    def __init__(self, master):
        self.window = tk.Toplevel(master)
        self.window.transient(master)
        self.window.configure(background = 'red')
        self.window.overrideredirect(True)
        self.window.resizable(width=False, height=False)

    def hide(self):
        self.window.withdraw()

    def set_pos(self, x, y, w, h):
        self.window.geometry("%dx%d+%d+%d" % (w, h, x, y))
        self.window.deiconify()

with functions.load_session(sys.argv[1]) as sess:
    x = sess.graph.get_tensor_by_name(sys.argv[2])
    y = sess.graph.get_tensor_by_name(sys.argv[3])

    x_shape = functions.tensor_image_shape(x)
    y_shape = functions.tensor_image_shape(y)

    def out_img(x_img):
        x_img = np.reshape(
            x_img.astype(np.float32) / 255.99,
            [1 if v == None else v for v in x.shape.as_list()]
        )
        y_img = sess.run(y, feed_dict = {x: x_img})
        y_img = np.reshape(y_img, y_shape)
        y_img = np.clip(y_img, 0.0, 1.0) * 255.99
        return y_img.astype(np.uint8)

    app = MainWindow(x_shape, y_shape, out_img)
    app.window.title(sys.argv[1])
    app.window.mainloop()

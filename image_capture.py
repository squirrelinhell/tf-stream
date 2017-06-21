#!/usr/bin/python3

import sys
import threading
import time
import tkinter as tk
import scipy.misc
import numpy as np
import pyscreenshot

class RedBorder:
    def __init__(self, master):
        self.window = master
        self.window.overrideredirect(True)
        self.window.resizable(width=False, height=False)
        self.window.configure(background = 'red')

    def set_pos(self, x, y, w, h):
        self.window.geometry("%dx%d+%d+%d" % (w, h, x, y))
        self.window.deiconify()

    def get_pos(self):
        return self.window.winfo_x(), self.window.winfo_y()

class CaptureWindow(RedBorder):
    def __init__(self, master, shape):
        RedBorder.__init__(self, master)
        self.shape = shape
        self.pos = 20, 20
        self.mouse_pos = None

        self.borders = [self] + [self.add_border() for x in range(3)]
        self.draw_borders()
        for b in self.borders:
            self.make_draggable(b.window)

        self.prev_capture = None
        self.start_capture_thread()

    def add_border(self):
        return RedBorder(tk.Toplevel(self.window))

    def draw_borders(self):
        self.last_moved = time.time()
        x, y = self.pos
        w, h = self.shape[0:2]
        line = 10
        self.borders[0].set_pos(x-line, y-line, w+line*2, line)
        self.borders[1].set_pos(x-line, y+h, w+line*2, line)
        self.borders[2].set_pos(x-line, y, line, h)
        self.borders[3].set_pos(x+w, y, line, h)

    def make_draggable(self, w):
        w.bind("<ButtonPress-1>", self.on_button_press)
        w.bind("<ButtonRelease-1>", self.on_button_release)
        w.bind("<B1-Motion>", self.on_mouse_move)

    def on_button_press(self, event):
        self.mouse_pos = event.x, event.y
        for b in self.borders:
            b.window.config(cursor = "fleur")

    def on_button_release(self, event):
        self.on_mouse_move(event)
        self.mouse_pos = None
        for b in self.borders:
            b.window.config(cursor = "arrow")

    def on_mouse_move(self, event):
        if self.mouse_pos is None:
            return
        diff = event.x - self.mouse_pos[0], event.y - self.mouse_pos[1]
        self.pos = self.pos[0] + diff[0], self.pos[1] + diff[1]
        self.draw_borders()

    def start_capture_thread(self):
        t = threading.Thread(target = self.capture_thread)
        t.start()

    def capture_image(self):
        x, y = self.pos
        w, h = self.shape[0:2]
        im = pyscreenshot.grab(bbox = (x, y, x+w, y+h))
        if self.shape[2] == 3:
            im = im.convert("RGB")
        elif self.shape[2] == 1:
            im = im.convert("L")
        else:
            raise ValueError("Invalid number of channels")
        return np.asarray(im, dtype="uint8")

    def capture_thread(self):
        while True:
            if time.time() < self.last_moved + 0.3:
                time.sleep(self.last_moved + 0.3 - time.time())
                continue

            img = self.capture_image()
            if img.shape[0:2] != self.shape[1::-1]:
                img = scipy.misc.imresize(img, self.shape[1::-1])

            if self.prev_capture is not None:
                diff = np.absolute(self.prev_capture - img)
                if np.sum(diff) < 10.0:
                    time.sleep(0.3)
                    continue
            self.prev_capture = img

            sys.stdout.buffer.write(img.tobytes())

if __name__ == "__main__":

    if len(sys.argv) < 2:
        sys.stderr.write("\nUsage:\n");
        sys.stderr.write("\timage_capture.py WxH\n")
        sys.stderr.write("\timage_capture.py WxHx3\n\n")
        sys.exit(1)

    dims = [int(x) for x in sys.argv[1].split("x")]
    if len(dims) == 2:
        dims += [1]
    if len(dims) != 3 or not dims[2] in (1,3) or min(dims[0:2]) < 2:
        sys.stderr.write("Error: Invalid image shape: %s\n" % dims)
        sys.exit(1)

    if sys.stdout.isatty():
        sys.stderr.write("Refusing to write binary data to a terminal\n")
        sys.exit(1)

    app = CaptureWindow(tk.Tk(), dims)
    app.window.mainloop()

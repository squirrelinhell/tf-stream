#!/usr/bin/python3

import sys
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import functions

class ImageView():
    def __init__(self, shape):
        self.shape = shape
        self.img = np.zeros(shape, dtype="uint8")
        self.aximg = plt.imshow(
            self.img, vmin=0.0, vmax=255.0, cmap="gray"
        )
        self.start_update_thread()

    def start_update_thread(self):
        t = threading.Thread(target=self.update_thread)
        t.daemon = True
        t.start()

    def update_thread(self):
        buf = bytes()
        while True:
            buf += sys.stdin.buffer.read(self.img.size - len(buf))
            if len(buf) < self.img.size:
                time.sleep(0.1)
            else:
                self.img = np.frombuffer(buf, dtype="uint8")
                self.img = self.img.reshape(self.shape)
                buf = bytes()
                self.aximg.set_data(self.img)
                plt.draw()

if __name__ == "__main__":

    if len(sys.argv) < 2:
        sys.stderr.write("\nUsage:\n\n");
        sys.stderr.write("\timage_show.py H,W\n")
        sys.stderr.write("\timage_show.py H,W,3\n\n")
        sys.exit(1)

    shape = functions.str_to_image_shape(sys.argv[1])
    app = ImageView(shape)
    plt.show()

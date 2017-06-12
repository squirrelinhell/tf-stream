#!/usr/bin/python3

import os
import sys

if len(sys.argv) < 2:
    sys.stderr.write("\nUsage:\n\tinputimage.py <some.model>\n" +
        "\tinputimage.py <some.model> <image>\n\n")
    sys.exit(1)

modeldir = sys.argv[1]

if not os.path.isdir(modeldir):
    sys.stderr.write("Error: '%s' is not a directory\n" % modeldir)
    sys.exit(1)

import loadsave
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.ndimage

sess, x, y = loadsave.load(modeldir)

if len(sys.argv) >= 3:
    img = scipy.ndimage.imread(sys.argv[2])
else:
    img = np.zeros((28, 28))

implot = plt.imshow(img, vmin=0, vmax=255, cmap="gray")

def update():
    global sess, x, y, img
    v = sess.run(y, feed_dict={x: img.reshape((1, 784))})
    print(v, np.argmax(v))

update()

mouse_x, mouse_y = -1, -1
def on_mouse_move(event):
    global mouse_x, mouse_y
    if event.xdata != None and event.ydata != None:
        if event.button == 1:
            nx, ny = int(event.xdata + 0.5), int(event.ydata + 0.5)
            if (mouse_x, mouse_y) != (nx, ny):
                mouse_x, mouse_y = nx, ny
                img[mouse_y][mouse_x] = 255
                implot.set_data(img)
                plt.draw()
                update()
        else:
            mouse_x, mouse_y = -1, -1

implot.figure.canvas.mpl_connect('motion_notify_event', on_mouse_move)
plt.show()

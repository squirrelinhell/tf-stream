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
import matplotlib.widgets
import scipy.ndimage
import scipy.misc

class ImageEditor():
    def __init__(self, img):
        self.brush_radius = 2
        self.brush_color = 1.0
        self.brush_pos = None
        self.callback = lambda: None
        self.buttons = []
        self.next_btn_x = 0.02
        self.img = img
        self.aximg = plt.imshow(img, vmin=0.0, vmax=1.0, cmap="gray")
        canvas = self.aximg.figure.canvas
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("button_release_event", self._on_release)
        canvas.mpl_connect("motion_notify_event", self._on_move)
        plt.subplots_adjust(bottom=0.15)
        self._add_btn(self._on_btn_clear, "Clear")
        for v in [0.0, 0.5, 1.0]:
            self._add_btn(self._on_btn_color, "")
            self.buttons[-1].color = (v, v, v)
        self._add_btn(self._on_btn_bigger, "Bigger")
        self._add_btn(self._on_btn_smaller, "Smaller")

    def get_image(self):
        return self.img

    def on_changed(self, callback):
        self.callback = callback

    def _on_img_changed(self):
        self.aximg.set_data(self.img)
        self.callback()
        plt.draw()

    def _on_press(self, event):
        if event.xdata == None or event.ydata == None \
                or event.button != 1 \
                or self.aximg.axes != event.inaxes:
            return
        self.brush_pos = np.array([event.xdata, event.ydata])
        self._draw_brush()
        self._on_img_changed()

    def _on_release(self, event):
        self.brush_pos = None

    def _on_move(self, event):
        if event.xdata == None or event.ydata == None \
                or self.brush_pos is None \
                or self.aximg.axes != event.inaxes:
            return
        mouse_pos = np.array([event.xdata, event.ydata])
        dist = np.linalg.norm(mouse_pos - self.brush_pos) \
            * (1.4 / self.brush_radius)
        if dist < 1.0:
            return
        for i in range(int(dist)):
            self.brush_pos += (mouse_pos - self.brush_pos) / dist
            self._draw_brush()
        self._on_img_changed()

    def _on_btn_clear(self, event):
        self.img = np.zeros(self.img.shape)
        self._on_img_changed()

    def _on_btn_color(self, event):
        for b in self.buttons:
            if b.ax == event.inaxes:
                self.brush_color = b.color[0]

    def _on_btn_bigger(self, event):
        self.brush_radius *= 1.5

    def _on_btn_smaller(self, event):
        self.brush_radius = max(1.0, self.brush_radius * 0.7)

    def _brush_range(self, axis):
        v_min = max(0, int(self.brush_pos[axis] - self.brush_radius))
        v_max = min(self.img.shape[axis], \
            int(self.brush_pos[axis] + self.brush_radius + 1))
        return range(v_min, v_max)

    def _draw_brush(self):
        for y in self._brush_range(1):
            for x in self._brush_range(0):
                point = np.array([x, y])
                dist = np.linalg.norm(point - self.brush_pos) \
                    / self.brush_radius
                if dist > 1.0:
                    continue
                dist = max(0.0, 2.0 * dist - 1.0)
                self.img[y][x] *= dist
                self.img[y][x] += (1.0 - dist) * self.brush_color

    def _add_btn(self, action, label):
        axes = plt.axes([self.next_btn_x, 0.02, 0.09, 0.07])
        self.next_btn_x += 0.1
        btn = matplotlib.widgets.Button(axes, label)
        btn.on_clicked(action)
        self.buttons.append(btn)

def tensor_image_shape(v):
    nontrivial = []
    for s in v.shape:
        if s.value != None and s.value >= 2:
            nontrivial.append(s.value)
    if len(nontrivial) == 1:
        return nontrivial + [1]
    if len(nontrivial) == 2:
        return nontrivial
    if len(nontrivial) == 3 and nontrivial[2] in [3, 4]:
        return nontrivial
    raise ValueError("Invalid tensor shape: " + str(v.shape))

def load_image(shape):
    if len(sys.argv) >= 3:
        img = scipy.ndimage.imread(sys.argv[2])
        if img.shape[0:2] != shape[0:2]:
            img = scipy.misc.imresize(img, shape[0:2])
        if len(img.shape) >= 3 and len(shape) < 3:
            img = np.mean(img, 2)
        img = img / 255.0
        return np.reshape(img, shape)
    return np.zeros(shape)

sess, x, y = loadsave.load(modeldir)
fig = plt.figure()

fig.add_subplot(1,2,1).set_title('Input')
image_editor = ImageEditor(load_image(tensor_image_shape(x)))

def out_img():
    x_val = image_editor.get_image()
    x_shape = [1 if v == None else v for v in x.shape.as_list()]
    x_val = np.reshape(x_val, x_shape)
    y_val = sess.run(y, feed_dict={x: x_val})
    y_val = np.reshape(y_val, tensor_image_shape(y))
    return y_val

fig.add_subplot(1,2,2).set_title('Output')
out_plt = plt.imshow(out_img(), vmin=0.0, vmax=1.0, cmap="gray")
image_editor.on_changed(lambda: out_plt.set_data(out_img()))

plt.show()

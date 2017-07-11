#!/usr/bin/env python3

import os
import sys
import hashlib
import threading
import queue
import numpy as np

def print_info(*args):
    data = " ".join([str(x) for x in args]) + "\n"
    sys.stderr.write(data)
    sys.stderr.flush()

def save_result(header, *args):
    data = " ".join([str(x) for x in args]) + "\n"
    header_hash = hashlib.md5(bytes(header, "UTF-8")).hexdigest()
    file_name = "results-%s.log" % header_hash[0:12]
    if not os.path.isfile(file_name):
        data = header + "\n---\n" + data
    with open(file_name, "a") as f:
        f.write(data)

class dotmap(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def str_to_image_shape(s):
    try:
        dims = [int(x) for x in s.split(",")]
        if len(dims) == 2:
            dims += [1]
        if len(dims) != 3 or not dims[2] in (1,3) or min(dims[0:2]) < 1:
            raise ValueError()
        if dims[2] == 1:
            dims = dims[0:2]
        return dims
    except ValueError:
        raise ValueError("Invalid image shape: %s" % s) from None

def dict_sum(a, b):
    return { k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b) }

def crop_zeros(array, epsilon = 0.001):
    array[np.abs(array) < epsilon] = 0.0
    coord = np.argwhere(array)
    coord = np.array([coord.min(axis = 0), coord.max(axis = 0)]).T
    coord = [slice(a[0], a[1] + 1) for a in coord]
    return array[coord]

def random_pad(array, shape):
    pad = np.maximum(0, np.asarray(shape) - np.asarray(array.shape))
    shift = np.floor(np.random.rand(len(pad)) * pad)
    pad = np.array([shift, pad - shift], dtype = "int32").T
    return np.pad(array, pad, mode = "constant")

def random_batches(data, batch_size = 128):
    x, t = data
    x, t = np.asarray(x), np.asarray(t)
    p = np.random.shuffle(np.arange(len(x)))
    x, t = x[p].reshape(x.shape), t[p].reshape(t.shape)
    start = 0
    while start + batch_size <= len(x):
        yield((x[start:start+batch_size], t[start:start+batch_size]))
        start += batch_size

def __put_all(items, q):
    for i in items:
        q.put(i)
        if i is None:
            return
    q.put(None)

def iterate_in_thread(items):
    q = queue.Queue()
    thread = threading.Thread(target = __put_all, args = (items, q))
    thread.start()
    while True:
        i = q.get(block = True)
        if i is None:
            break
        yield(i)
    thread.join()

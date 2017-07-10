#!/usr/bin/env python3

import os
import sys
import hashlib

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

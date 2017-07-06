#!/usr/bin/python3

import sys

def print_info(*args):
    sys.stderr.write(" ".join([str(x) for x in args]))
    sys.stderr.write("\n")
    sys.stderr.flush()

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

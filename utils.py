#!/usr/bin/python3

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

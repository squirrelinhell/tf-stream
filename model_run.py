#!/usr/bin/python3

import sys
import numpy as np
import functions

if __name__ == "__main__":

    if len(sys.argv) < 2:
        sys.stderr.write("\nUsage:\n\n");
        sys.stderr.write("\tmodel_run.py <dir.model> <input tensor> <output tensor>\n\n")
        sys.exit(1)

    with functions.load_session(sys.argv[1]) as sess:
        x = sess.graph.get_tensor_by_name(sys.argv[2])
        y = sess.graph.get_tensor_by_name(sys.argv[3])

        x_shape = [1 if v == None else v for v in x.shape.as_list()]
        x_size = np.prod(x_shape)

        buf = bytes()
        while True:
            buf += sys.stdin.buffer.read(x_size - len(buf))
            if len(buf) < x_size:
                time.sleep(0.1)
            else:
                x_val = np.frombuffer(buf, dtype="uint8")
                buf = bytes()
                x_val = np.reshape(x_val, x_shape)
                x_val = x_val.astype("float32") / 256.0
                y_val = sess.run(y, feed_dict = {x: x_val})
                y_val = np.clip(y_val * 256.0, 0.0, 255.1)
                y_val = y_val.astype("uint8")
                sys.stdout.buffer.write(y_val.tobytes())
                sys.stdout.buffer.flush()

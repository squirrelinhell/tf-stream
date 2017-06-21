#!/usr/bin/python3

import sys
import functions

def args_str(args):
    return "[" + ", ".join(
        [t.name + " " + str(t.shape) + " " + t.dtype.name for t in args]
    ) + "]"

if __name__ == "__main__":

    if len(sys.argv) < 2:
        sys.stderr.write("\nUsage:\n");
        sys.stderr.write("\tmodel_print.py <dir.model>\n\n")
        sys.exit(1)

    with functions.load_session(sys.argv[1]) as sess:
        for op in sess.graph.get_operations():
            print(
                op.name,
                "<" + op.type + ">",
                args_str(op.inputs),
                "->",
                args_str(op.outputs)
            )
#!/usr/bin/python3

import os
import tensorflow as tf

def load_session(path):
    saver = tf.train.import_meta_graph(
        os.path.join(path, "graph.meta")
    )
    sess = tf.train.SessionManager().prepare_session(
        "",
        init_op=tf.global_variables_initializer(),
        saver=saver,
        checkpoint_dir=path
    )
    return sess

def save_session(sess, path):
    saver = tf.train.Saver()
    if os.path.exists(path):
        prev_state = tf.train.get_checkpoint_state(path)
        prev_step = 0
        if prev_state is not None:
            name = os.path.basename(prev_state.model_checkpoint_path)
            name = name.split('-')
            if len(name) >= 2:
                prev_step = int(name[1])
        saver.save(
            sess,
            os.path.join(path, "checkpoint"),
            global_step=(prev_step+1)
        )
    else:
        os.makedirs(path)
        saver.export_meta_graph(
            filename=os.path.join(path, "graph.meta")
        )
        saver.save(
            sess,
            os.path.join(path, "checkpoint"),
            global_step=1
        )
    return path

def str_to_image_shape(s):
    try:
        dims = [int(x) for x in s.split(",")]
        if len(dims) == 2:
            dims += [1]
        if len(dims) != 3 or not dims[2] in (1,3) or min(dims[0:2]) < 2:
            raise ValueError()
        if dims[2] == 1:
            dims = dims[0:2]
        return dims
    except ValueError:
        raise ValueError("Invalid image shape: %s" % s) from None

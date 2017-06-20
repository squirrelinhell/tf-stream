#!/bin/false

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

def tensor_image_shape(v):
    nontrivial = []
    for s in v.shape:
        if s.value != None and s.value >= 2:
            nontrivial.append(s.value)
    if len(nontrivial) == 1:
        return [1] + nontrivial
    if len(nontrivial) == 2:
        return nontrivial
    if len(nontrivial) == 3 and nontrivial[2] in [3, 4]:
        return nontrivial
    raise ValueError("Invalid tensor shape: " + str(v.shape))

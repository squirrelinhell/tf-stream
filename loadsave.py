#!/usr/bin/python3

import os
import sys
import tensorflow as tf

from utils import *

def load(path, create_model = None):
    if os.path.exists(path):
        print_info("Loading from '" + path + "'...")
        saver = tf.train.import_meta_graph(
            os.path.join(path, "graph.meta")
        )
        sess = tf.train.SessionManager().prepare_session(
            "",
            init_op=tf.global_variables_initializer(),
            saver=saver,
            checkpoint_dir=path
        )
    else:
        if create_model is None:
            raise ValueError("Could not open '%s'" % path)
        print_info("Creating a new model...")
        create_model()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

    sess.names = dotmap()
    for op in sess.graph.get_operations():
        sess.names[op.name] = op
        for t in op.outputs:
            sess.names[t.name.split(":")[0]] = t
    tf.Session.__getattr__ = sess.names.get

    return sess

def get_checkpoint_step(path):
    prev_state = tf.train.get_checkpoint_state(path)
    if prev_state is None:
        return 0
    name = os.path.basename(prev_state.model_checkpoint_path)
    name = name.split('-')
    if len(name) < 2:
        return 0
    return int(name[1])

def save(sess, path):
    saver = tf.train.Saver()
    global_step = tf.contrib.framework.get_global_step()
    if global_step is None:
        step_value = 0
    else:
        step_value = tf.train.global_step(sess, global_step)

    if os.path.exists(path):
        print_info("Adding checkpoint to '" + path + "'...")
        prev_value = get_checkpoint_step(path)
        if step_value <= prev_value:
            if global_step is None:
                step_value = prev_value + 1
            else:
                raise ValueError(
                    "The value of global_step has not been" +
                    " incremented: %d" % step_value
                )
        saver.save(
            sess,
            os.path.join(path, "checkpoint"),
            global_step = step_value
        )
    else:
        print_info("Saving to '" + path + "'...")
        os.makedirs(path)
        saver.export_meta_graph(
            filename = os.path.join(path, "graph.meta")
        )
        saver.save(
            sess,
            os.path.join(path, "checkpoint"),
            global_step = step_value
        )
    return path

#!/usr/bin/python3

import os
import sys
import tensorflow as tf

def load(path, create_model = None):
    if os.path.exists(path):
        sys.stderr.write("Loading from '" + path + "'...\n")
        saver = tf.train.import_meta_graph(
            os.path.join(path, "graph.meta")
        )
        return tf.train.SessionManager().prepare_session(
            "",
            init_op=tf.global_variables_initializer(),
            saver=saver,
            checkpoint_dir=path
        )
    else:
        if create_model is None:
            raise ValueError("Could not open '%s'" % path)
        sys.stderr.write("Creating a new model...\n")
        session = tf.Session()
        create_model()
        session.run(tf.global_variables_initializer())
        return session

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
        sys.stderr.write("Adding checkpoint to '" + path + "'...\n")
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
        sys.stderr.write("Saving to '" + path + "'...\n")
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

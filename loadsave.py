#!/usr/bin/python3

import os
import tensorflow as tf

def load(d):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(os.path.join(d, "export.meta"))
    saver.restore(sess, os.path.join(d, "export"))
    in_name = tf.get_collection("input")[0].decode("utf-8")
    out_name = tf.get_collection("output")[0].decode("utf-8")
    in_var = sess.graph.get_tensor_by_name(in_name)
    out_var = sess.graph.get_tensor_by_name(out_name)
    return sess, in_var, out_var

def save(d, sess, in_var, out_var):
    os.mkdir(d)
    tf.add_to_collection("input", in_var.name.encode("utf-8"))
    tf.add_to_collection("output", out_var.name.encode("utf-8"))
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(d, "export"), write_meta_graph=True)
    return d

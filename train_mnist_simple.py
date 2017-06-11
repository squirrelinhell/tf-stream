#!/usr/bin/python3

import os
import sys

if len(sys.argv) < 2:
    sys.stderr.write("\nUsage:\n\ttrain_mnist_simple.py <out.model>\n\n")
    sys.exit(1)

outdir = sys.argv[1]

if os.path.exists(outdir):
    sys.stderr.write("Error: file '%s' exists\n" % outdir)
    sys.exit(1)

import loadsave
import tensorflow as tf
import tensorflow.examples.tutorials.mnist as tf_mnist

def dense(x, o):
    i = x.shape[1].value
    sd = 2. / (i + o)
    b = tf.Variable(tf.constant(0.1, shape=[o]))
    w = tf.Variable(tf.random_normal([i, o], stddev=sd))
    return tf.matmul(x, w) + b

x = tf.placeholder(tf.float32, [None, 784])

y = x
y = dense(y, 100)
y = tf.nn.relu(y)
y = dense(y, 100)
y = tf.nn.relu(y)
y = dense(y, 10)

t = tf.placeholder(tf.float32, [None, 10])

loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y)
)

accuracy = tf.reduce_mean(tf.cast(
    tf.equal(tf.argmax(y, 1), tf.argmax(t, 1)),
    tf.float32
))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

def stat(d):
    v = sess.run(
        [accuracy, loss],
        feed_dict={x: d.images, t: d.labels}
    )
    return "accuracy=%f, loss=%f" % (v[0], v[1])

mnist = tf_mnist.input_data.read_data_sets("/tmp/mnist", one_hot=True)

for _ in range(20):
    for _ in range(300):
        batch_xs, batch_ts = mnist.train.next_batch(200)
        sess.run(train_step, feed_dict={x: batch_xs, t: batch_ts})
    print("train: ", stat(mnist.train), "   test: ", stat(mnist.test))

print("model saved to: " + loadsave.save(outdir, sess, x, y))

import tensorflow as tf
import numpy as np

def get_accuracy(y, y_pred):
    y_hat = tf.argmax(tf.nn.softmax(y_pred), axis=-1)
    acc = tf.equal(y_pl, tf.cast(y_hat, tf.int32))
    acc = tf.cast(acc, tf.float32)
    return tf.reduce_mean(acc)

y_pl = tf.placeholder(tf.int32, [None])
y_pred_pl = tf.placeholder(tf.float32, [None, 3])
acc = get_accuracy(y_pl, y_pred_pl)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    y = np.array([1,0,1,2])
    y_pred = np.array([[1,5,3],[5,2,4],[5,6,1],[1,2,3]])
    acc_val = sess.run(acc, feed_dict = {y_pl: y, y_pred_pl: y_pred})
    print(acc_val)


import os
import matplotlib
matplotlib.use('Agg')
from scipy.misc import imsave

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.framework.ops import reset_default_graph
from src import utils, models, datagen

class Engine:
    def __init__(self, image_size, model_name, loss_name, model_params):
        reset_default_graph()

        model_lut = {
                'inception_v3': models.InceptionV3
                }
       
        [h, w, ch] = image_size
        self.x = tf.placeholder(tf.float32, [None, h, w, ch])
        self.y = tf.placeholder(tf.float32, [None])
        self.y_pred = model_lut[model_name](self.x, **model_params)
        
        self.loss = self.loss_fn(loss_name, self.y, self.y_pred)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    def init_session(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        return sess

    def loss(self, name, y, y_pred):
        if name == 'categorical_crossentropy':
            loss = tf.reduce_mean(\
                    tf.nn.sparse_softmax_cross_entropy_with_logits(\
                    labels = y, logits = y_pred))
        return loss

    def train(self):
        pass

    def evaluate(self):
        pass
    

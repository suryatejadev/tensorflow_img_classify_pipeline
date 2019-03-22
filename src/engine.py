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

        model_lut = {
                'inception_v3': models.InceptionV3
                }
       
        [h, w, ch] = image_size
        self.x = tf.placeholder(tf.float32, [None, h, w, ch])
        self.y = tf.placeholder(tf.int32, [None])
        self.y_pred = model_lut[model_name](self.x, **model_params)

        self.loss = self.loss_fn(loss_name, self.y, self.y_pred)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        # summaries
        self.summary_loss_train = tf.summary.scalar('Loss', self.loss)
        self.summary_loss_val = tf.summary.scalar('Loss', self.loss)

    #  def init_session(self):
    #      #  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #      #  os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    #      config = tf.ConfigProto()
    #      config.gpu_options.allow_growth = True
    #      sess = tf.Session(config=config)
    #      sess.run(tf.global_variables_initializer())
    #      sess.run(tf.local_variables_initializer())
    #      return sess

    def loss_fn(self, name, y, y_pred):
        if name == 'categorical_crossentropy':
            loss = tf.reduce_mean(\
                    tf.nn.sparse_softmax_cross_entropy_with_logits(\
                    labels = y, logits = y_pred))
        return loss

    def train(self, train_iter, val_iter, logs_dir, 
            num_epochs, batch_size, checkpoint_freq):     
        self.batch_size = batch_size

        #  sess = self.init_session()
        with tf.Session() as sess:

            writer_train = tf.summary.FileWriter(logs_dir+'/train', sess.graph)
            writer_val = tf.summary.FileWriter(logs_dir+'/val', sess.graph)
            sess.run(tf.global_variables_initializer())
            
            itr_summary_train = 0
            itr_summary_val = 0
            for epoch in range(num_epochs):
                # Training
                sess.run(train_iter.initializer)
                itr = 0
                while 1:
                    try: 
                        # Train the model
                        x_train, y_train = sess.run(train_iter.get_next())
                        _, loss_train, summary = sess.run(
                                [self.optimizer, self.loss, self.summary_loss_train], 
                                feed_dict = {self.x: x_train, self.y: y_train}
                                )
                        writer_train.add_summary(summary, itr_summary_train)
                        
                        # Get the summaries
                        if itr % checkpoint_freq == 0:
                            print('Epoch = {}, Iteration = {}, Train Loss = {}'.\
                                    format(epoch, itr, loss_train))
                        itr += 1; itr_summary_train += 1
                        if itr == 4:
                            break
                    except tf.errors.OutOfRangeError:
                        break

                # Validation
                sess.run(val_iter.initializer)
                print('Validation')
                loss_val = 0
                itr_val = 0
                while 1:
                    try:
                        x_val, y_val = sess.run(val_iter.get_next())
                        loss_val, summary = sess.run(
                                [self.loss, self.summary_loss_val], 
                                feed_dict = {self.x: x_val, self.y: y_val}
                                )
                        writer_val.add_summary(summary, itr_summary_val)
                        itr_val += 1; itr_summary_val += 1
                        if itr_val == 4:
                            break
                    except tf.errors.OutOfRangeError:
                        break
                print('Validation Loss = ', loss_val)

    def evaluate(self):
        pass
    

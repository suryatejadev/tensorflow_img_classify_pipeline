import os
import matplotlib
matplotlib.use('Agg')
from scipy.misc import imsave
from time import time

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

        self.loss = self.get_loss(loss_name, self.y, self.y_pred)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        self.accuracy = self.get_accuracy(self.y, self.y_pred)

    def get_loss(self, name, y, y_pred, reduce_mean=True):
        if name == 'categorical_crossentropy':
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels = y, logits = y_pred)
        if reduce_mean:
            loss = tf.reduce_mean(loss)
        return loss

    def get_accuracy(self, y, y_pred):
        y_hat = tf.argmax(tf.nn.softmax(y_pred), axis=-1)
        acc = tf.equal(y, tf.cast(y_hat, tf.int32))
        acc = tf.cast(acc, tf.float32)
        return tf.reduce_mean(acc)

    def train(self, train_iter, val_iter, logs_dir, 
            num_epochs, batch_size, checkpoint_freq):     

        # summaries
        summary_loss_train = tf.summary.scalar('Train Loss', self.loss)
        loss_gap, acc_val = tf.Variable(0.0), tf.Variable(0.0)
        summary_loss_gap = tf.summary.scalar('Train-Val Gap', loss_gap)
        summary_acc_val = tf.summary.scalar('Validation Accuracy', acc_val)

        #  sess = self.init_session()
        with tf.Session() as sess:

            writer_train = tf.summary.FileWriter(logs_dir+'/train', sess.graph)
            writer_gap = tf.summary.FileWriter(logs_dir+'/train_val_gap', sess.graph)
            writer_val = tf.summary.FileWriter(logs_dir+'/val', sess.graph)
            sess.run(tf.global_variables_initializer())
            
            itr_summary_train = 0
            itr_summary_val = 0
            for epoch in range(num_epochs):
                # Training
                t=time()
                sess.run(train_iter.initializer)
                iteration = 0
                while True:
                    try: 
                        # Train the model
                        x_train, y_train = sess.run(train_iter.get_next())
                        _, loss_train, summary = sess.run(
                                [self.optimizer, self.loss, summary_loss_train], 
                                feed_dict = {self.x: x_train, self.y: y_train}
                                )
                        writer_train.add_summary(summary, itr_summary_train)
                        itr_summary_train += 1
                        
                        # Get validation results at checkpoint
                        # Summary : Train-val gap
                        if iteration % checkpoint_freq == 0:
                            sess.run(val_iter.initializer)
                            loss_val_i = 0; acc_val_i = 0; num_val = 0
                            print('Train Time for {} iterations = {}'.\
                                    format(checkpoint_freq, time()-t))
                            t = time()
                            while True:
                                try:
                                    x_val, y_val = sess.run(val_iter.get_next())
                                    n = x_val.shape[0]
                                    num_val += x_val.shape[0]
                                    loss_val_temp, acc_val_temp = \
                                            sess.run(\
                                            [self.loss, self.accuracy],\
                                                    feed_dict = {self.x: x_val, \
                                                    self.y: y_val})
                                    loss_val_i += n * loss_val_temp
                                    acc_val_i += n * acc_val_temp
                                except tf.errors.OutOfRangeError:
                                    break
                            print('Validation Time = ', time()-t)
                            acc_val_i /= num_val
                            summary = sess.run(summary_acc_val, \
                                    feed_dict = {acc_val: acc_val_i})
                            writer_val.add_summary(summary, itr_summary_val)
                            loss_val_i /= num_val
                            loss_train = sess.run(self.loss, 
                                    feed_dict = {self.x: x_train, self.y: y_train}
                                    )
                            loss_gap_i = loss_val_i - loss_train
                            summary = sess.run(summary_loss_gap, 
                                    feed_dict = {loss_gap: loss_gap_i})
                            writer_gap.add_summary(summary, itr_summary_val)
                            itr_summary_val += 1

                            # Print output
                            print('Epoch = {}, Iteration = {}, \
                                    Train Loss = {}. Gap = {}, \
                                    Validation Accuracy = {}'.\
                                    format(epoch, iteration, loss_train, \
                                    loss_gap_i, acc_val_i))
                        
                        iteration += 1

                        # Get the summaries
                        #  if itr % checkpoint_freq == 0:
                        #      print('Epoch = {}, Iteration = {}, Train Loss = {}'.\
                        #              format(epoch, itr, loss_train))
                        #  itr += 1; itr_summary_train += 1
                        #  if itr == 4:
                        #      break
                    except tf.errors.OutOfRangeError:
                        break
                #  t=time()
                #  # Validation
                #  sess.run(val_iter.initializer)
                #  print('Validation')
                #  loss_val = 0
                #  itr_val = 0
                #  while 1:
                #      try:
                #          x_val, y_val = sess.run(val_iter.get_next())
                #          loss_val, summary = sess.run(
                #                  [self.loss, summary_loss_val],
                #                  feed_dict = {self.x: x_val, self.y: y_val}
                #                  )
                #          writer_val.add_summary(summary, itr_summary_val)
                #          itr_val += 1; itr_summary_val += 1
                #          if itr_val == 4:
                #              break
                #      except tf.errors.OutOfRangeError:
                #          break
                #  print('Validation Loss = ', loss_val)
                #  print(time()-t)


    def evaluate(self):
        pass
    
    #  def init_session(self):
    #      #  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #      #  os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    #      config = tf.ConfigProto()
    #      config.gpu_options.allow_growth = True
    #      sess = tf.Session(config=config)
    #      sess.run(tf.global_variables_initializer())
    #      sess.run(tf.local_variables_initializer())
    #      return sess



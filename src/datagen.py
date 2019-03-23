import os
import pandas as pd
import tensorflow as tf
import numpy as np
from glob import glob

class Data:
    def __init__(self, path, num_classes, display_stats=False):
        # Get train and validation data paths and labels
        (self.train_paths, self.train_labels), \
                (self.val_paths, self.val_labels) = \
                self.get_data(path)
        self.num_classes = num_classes
        if display_stats:
            self.display_stats()

    # Preprocess data
    def preprocess_data(self, path, label, image_size):
        img = tf.read_file(path)
        img = tf.image.decode_jpeg(img, channels = 3)
        [h, w, ch] = image_size
        img = tf.image.resize_images(img, [h, w])
        img = img/255.0
        return img, label

    # Augment data (translate and rotate)
    def augment_data(self, img, label):
        img_tx = tf.contrib.image.rotate(img, \
                angles = tf.random_uniform(shape=[], minval=-10, maxval=10))
        img_tx = tf.contrib.image.translate(img_tx, \
                translations = [tf.random_uniform(shape=[], minval=-10, maxval=10), \
                               tf.random_uniform(shape=[], minval=-10, maxval=10)])
        return img_tx, label
    
    def datagen(self, logs_dir, batch_size, image_size):
        self.batch_size = batch_size

        # Get train datagen and iterator
        train_dataset = tf.data.Dataset.from_tensor_slices(\
                (self.train_paths, self.train_labels))
        train_dataset = train_dataset.shuffle(\
                buffer_size = len(self.train_paths))
        train_dataset = train_dataset.map(\
                lambda x, y: self.preprocess_data(x, y, image_size)) #, \
                              #  num_parallel_calls = \
                              #  tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.map(self.augment_data) #, \
                              #  num_parallel_calls = \
                              #  tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.batch(batch_size)
        train_iter = train_dataset.make_initializable_iterator()

        # Create validation dataset and iterator 
        # (no shuffling and data augmentation)
        val_dataset = tf.data.Dataset.from_tensor_slices(\
                (self.val_paths, self.val_labels))
        val_dataset = val_dataset.map(\
                lambda x, y: self.preprocess_data(x, y, image_size)) #, \
                              #  num_parallel_calls = \
                              #  tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size)
        val_iter = val_dataset.make_initializable_iterator()

        # Get summaries of datagen
        train_image_summary = tf.summary.image('Train Images',
                train_iter.get_next()[0], max_outputs = 3)
        val_image_summary = tf.summary.image('Validation Images',
                val_iter.get_next()[0], max_outputs = 3)
        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(logs_dir, sess.graph)
            sess.run(tf.global_variables_initializer())
            sess.run([train_iter.initializer, val_iter.initializer])
            summary = sess.run(merged)
            writer.add_summary(summary)

        return train_iter, val_iter

    # get_data() for CIFAR-10 dataset
    def get_data(self, src_path):
        img_paths = glob(src_path + '*.jpg')
        N = len(img_paths)

        idx = np.arange(N)
        np.random.shuffle(idx)
        num_train = int(0.8*N)
        train_paths = [img_paths[idx[i]] for i in range(num_train)]
        val_paths = [img_paths[idx[num_train + i]] \
                for i in range(N - num_train)]

        train_labels = []
        for path in train_paths:
            name = os.path.basename(path)
            train_labels.append(int(name[name.find('_')+1:-4]))

        val_labels = []
        for path in val_paths:
            name = os.path.basename(path)
            val_labels.append(int(name[name.find('_')+1:-4]))

        return (train_paths, train_labels), \
                (val_paths, val_labels)


    # get_data() for Tiny Imagenet-200 dataset
    # This function changes for each dataset and task
    #  def get_data(self, path):
    #      # Get train data paths and labels
    #      train_dir = os.path.join(path, 'train')
    #      train_folders = os.listdir(train_dir)
    #      train_paths, train_labels = [], []
    #      label_dict = {}
    #      for i, folder in enumerate(train_folders):
    #          label_dict[folder] = i
    #          paths = glob(os.path.join(train_dir, folder, 'images/*.JPEG'))
    #          train_paths += paths
    #          train_labels += [i]*len(paths)
    #
    #      # Get validation data paths and labels
    #      val_dir = os.path.join(path, 'val')
    #      val_paths = glob(os.path.join(val_dir, 'images/*.JPEG'))
    #      df_annot = pd.read_csv(os.path.join(val_dir, 'val_annotations.txt'), \
    #                            header=None, sep='\t')
    #      df_annot = df_annot.set_index(0)
    #      val_labels = []
    #      for path in val_paths:
    #          name = os.path.basename(path)
    #          folder = df_annot.loc[name].iloc[0]
    #          val_labels.append(label_dict[folder])
    #
    #      return (train_paths, train_labels), \
    #              (val_paths, val_labels)

    # Display data stats in console
    def display_stats(self):
        print('Data Stats...')
        print('\n')
        
        print('Train Data: ')
        print('Number of samples = ', len(self.train_paths))
        print('Number of classes = ', self.num_classes)
        print('Range of labels = {} - {}'.\
                format(min(self.train_labels), max(self.train_labels)))
        print('Example path = ', self.train_paths[0])
        print('\n')

        print('Validation Data: ')
        print('Number of samples = ', len(self.val_paths))
        print('Range of labels = {} - {}'.\
                format(min(self.val_labels), max(self.val_labels)))
        print('Example path = ', self.val_paths[0])
        print('\n')






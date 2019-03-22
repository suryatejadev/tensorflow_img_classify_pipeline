import os
import pandas as pd
import tensorflow as tf

class Data:
    def __init__(self, path):
        # Get train and validation data paths and labels
        (self.train_paths, self.train_labels), \
                (self.val_paths, self.val_labels) = \
                self.get_data(path)

    # This function changes for each dataset and task
    def get_data(self, path):
        # Get train data paths and labels
        train_dir = os.path.join(path, 'train')
        train_folders = os.listdir(train_dir)
        train_paths, train_labels = [], []
        label_dict = {}
        for i, folder in enumerate(train_folders):
            label_dict[folder] = i
            paths = glob(os.path.join(train_dir, folder, 'images/*.JPEG'))
            train_paths += paths
            train_labels += [i]*len(paths)

        # Get validation data paths and labels
        val_dir = os.path.join(path, 'val')
        val_paths = glob(os.path.join(val_dir, 'images/*.JPEG'))
        df_annot = pd.read_csv(os.path.join(val_dir, 'val_annotations.txt'), \
                              header=None, sep='\t')
        df_annot = df_annot.set_index(0)
        val_labels = []
        for path in val_paths:
            name = os.path.basename(path)
            folder = df_annot.loc[name].iloc[0]
            val_labels.append(label_dict[folder])
        
        return (train_paths, train_labels), \
                (val_paths, val_labels)

    # Preprocess data
    def preprocess_data(self, path, label):
        img = tf.read_file(path)
        img = tf.image.decode_jpeg(img, channels = 3)
        img = tf.image.resize_images(img, [224, 224])
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
    
    def datagen(self, batch_size):
        # Get train datagen
        train_dataset = tf.data.Dataset.from_tensor_slices(\
                (self.train_paths, self.train_labels))
        train_dataset = train_dataset.shuffle(\
                buffer_size = len(self.train_paths))
        train_dataset = train_dataset.map(self.preprocess_data, \
                              num_parallel_calls = \
                              tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.map(self.augment_data, \
                              num_parallel_calls = \
                              tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.batch(batch_size)
        train_iter = train_dataset.make_initializable_iterator()

        # Create validation dataset and iterator 
        # (no shuffling and data augmentation)
        val_dataset = tf.data.Dataset.from_tensor_slices(\
                (self.val_paths, self.val_labels))
        val_dataset = val_dataset.map(self.preprocess_data, \
                              num_parallel_calls = \
                              tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size)
        val_iter = val_dataset.make_initializable_iterator()

        return train_iter, val_iter


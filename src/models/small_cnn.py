import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as layers

def conv2d_block(x, filters, kernel_size = (3,3), \
        padding='same', activation = 'relu', \
        batchnorm=True, pool = True, dropout = 0.25):
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, \
            padding=padding, activation=activation)(x)
    x = layers.BatchNormalization()(x) if batchnorm else x
    x = layers.MaxPool2D(2, 2)(x) if pool else x
    x = layers.Dropout(rate = dropout)(x)
    return x

def fc_block(x, units, activation = 'relu', dropout=0.25):
    x = layers.Dense(units, activation)(x)
    x = layers.Dropout(rate = dropout)(x)
    return x

def smallCNN(x, num_classes=10):
    
    x = conv2d_block(x, 32, pool=False, dropout=0)
    x = conv2d_block(x, 32, pool=True, dropout=0.25)
    x = conv2d_block(x, 64, pool=False, dropout=0)
    x = conv2d_block(x, 64, pool=True, dropout=0.25)
    x = layers.Flatten()(x)
    x = fc_block(x, 512, dropout=0.5)
    x = fc_block(x, num_classes, 'linear')
    return x


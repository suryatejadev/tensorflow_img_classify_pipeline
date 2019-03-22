import tensorflow as tf
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

def InceptionV3(x, num_classes=1000):
    base_model = inception_v3.InceptionV3(
            input_shape = (224,224,3),
            include_top = False,
            weights = 'imagenet',
            )
    global_average_layer = GlobalAveragePooling2D()
    prediction_layer = Dense(num_classes)
    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
    ])
    return model(x)

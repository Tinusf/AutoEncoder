import tensorflow.keras
import tensorflow.keras.layers as layers
import config
import numpy as np


class DecoderModel(tensorflow.keras.Model):
    def __init__(self, image_shape):
        super(DecoderModel, self).__init__()
        self.rgb = image_shape[-1] == 3
        # How many nodes in a flattened network
        if self.rgb and config.AUTO_SPLIT_RGB:
            flatten_nodes = image_shape[0] * image_shape[1]
        else:
            flatten_nodes = np.product(image_shape)
        self.layer1 = layers.Dense(512, activation="relu")
        self.layer2 = layers.Dense(512, activation="relu")
        self.layer3 = layers.Dense(flatten_nodes, activation="relu")
        if self.rgb and config.AUTO_SPLIT_RGB:
            image_shape = (28, 28, 1)
        self.output_layer = layers.Reshape(image_shape, input_shape=(flatten_nodes,))
        self.batchnorm0 = layers.BatchNormalization()
        self.batchnorm1 = layers.BatchNormalization()
        self.batchnorm2 = layers.BatchNormalization()
        self.batchnorm3 = layers.BatchNormalization()

    def call(self, x):
        if config.AUTO_BATCH_NORM:
            x = self.batchnorm0(x)
        x = self.layer1(x)
        if config.AUTO_BATCH_NORM:
            x = self.batchnorm1(x)
        x = self.layer2(x)
        if config.AUTO_BATCH_NORM:
            x = self.batchnorm3(x)
        x = self.layer3(x)
        x = self.output_layer(x)
        if self.rgb and config.AUTO_SPLIT_RGB:
            # x is now for example (96, 28, 28, 1) and we need to get this back into the shape
            # (32, 28, 28, 3)
            # Batch size / 3
            batch_size = tensorflow.keras.backend.shape(x)[0] // 3
            r = x[0:batch_size]
            g = x[batch_size: 2 * batch_size]
            b = x[2 * batch_size:]
            # Concatenate the rgb back together.
            x = tensorflow.keras.backend.concatenate([r, g, b])
        return x

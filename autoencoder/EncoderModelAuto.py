import tensorflow.keras
import tensorflow.keras.layers as layers
import config


class EncoderModel(tensorflow.keras.Model):
    def __init__(self, image_shape):
        super(EncoderModel, self).__init__()
        self.rgb = image_shape[-1] == 3
        self.conv1 = layers.Conv2D(
            filters=28, kernel_size=(5, 5), activation="relu"
        )
        # self.conv2 = layers.Conv2D(56, (3, 3), activation="relu")
        # self.conv3 = layers.Conv2D(112, (3, 3), activation="relu")
        self.maxpool = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.layer1 = layers.Dense(512, activation="relu")
        self.layer2 = layers.Dense(512, activation="relu")
        self.latent = layers.Dense(config.AUTO_LATENT_SIZE, activation="relu")
        self.batchnorm0 = layers.BatchNormalization()
        self.batchnorm1 = layers.BatchNormalization()
        self.batchnorm2 = layers.BatchNormalization()
        self.batchnorm3 = layers.BatchNormalization()
        self.batchnorm4 = layers.BatchNormalization()

    def call(self, x):
        if config.AUTO_BATCH_NORM:
            x = self.batchnorm0(x)
        if self.rgb and config.AUTO_SPLIT_RGB:
            #  Split the (32, 28, 28, 3) into (96, 28, 28, 1).
            r = x[:, :, :, 0:1]
            g = x[:, :, :, 1:2]
            b = x[:, :, :, 2:3]
            x = tensorflow.keras.backend.concatenate([r, g, b], axis=0)
        x = self.conv1(x)
        x = self.maxpool(x)
        if config.AUTO_BATCH_NORM:
            x = self.batchnorm1(x)
        x = self.flatten(x)
        x = self.layer1(x)
        if config.AUTO_BATCH_NORM:
            x = self.batchnorm2(x)
        x = self.layer2(x)
        if config.AUTO_BATCH_NORM:
            x = self.batchnorm3(x)
        return self.latent(x)

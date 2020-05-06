import tensorflow.keras
from autoencoder.DecoderModelAuto import DecoderModel
from autoencoder.EncoderModelAuto import EncoderModel


class AutoencoderModel(tensorflow.keras.Model):
    def __init__(self, image_shape):
        super(AutoencoderModel, self).__init__()
        # Uses the encoder and decoder model.
        self.encoder_model = EncoderModel(image_shape)
        self.decoder_model = DecoderModel(image_shape)
        optimizer = tensorflow.optimizers.Adam(lr=0.001)
        self.compile(optimizer=optimizer, loss="mse", metrics=["accuracy"])

    def call(self, inputs):
        # First call the encoder and then put that output in the decoder.
        x = self.encoder_model.call(inputs)
        return self.decoder_model.call(x)

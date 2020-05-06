from stacked_mnist import DataMode

#
# Autoencoder Config
#


# The number of nodes in the latent.
AUTO_LATENT_SIZE = 48
# If the autoencoder should be loaded from file or retrained.
LOAD_AUTOENCODER = True
# How many epochs to train the autoencoder for
AUTO_EPOCHS = 20
# Which dataset to use for the autoencoder for the generative and reconstruction purposes.
AUTO_GEN_DATAMODE = DataMode.COLOR_BINARY_COMPLETE
# Which dataset to use for the autoencoder for the anomaly detector.
AUTO_ANOM_DATAMODE = DataMode.COLOR_BINARY_MISSING
# If the RGB Images should be split up into 3 (28, 28, 1) greyscale images before putting them
# into the model and learning the weights.
AUTO_SPLIT_RGB = True
# Should the autoencoder use batch normalization or not
AUTO_BATCH_NORM = False # MONO is saved with batch norm off,
                       # auto split rgb is with batch norm off,
                       # COLOR without split is with batch norm on.

#
# VAE (Variational Autoencoder Config
#

# The number of nodes in the latent.
VAE_LATENT_SIZE = 48
# If the VAE should be loaded from file or retrained.
LOAD_VAE = True
# Which dataset to use for the VAE for the generative and reconstruction purposes.
VAE_GEN_DATAMODE = DataMode.COLOR_BINARY_COMPLETE
# Which dataset to use for the VAE for the anomaly detector.
VAE_ANOM_DATAMODE = DataMode.COLOR_BINARY_MISSING
# Batch size for the VAE
VAE_BATCH_SIZE = 128
# How many epochs should the VAE learn.
VAE_EPOCHS = 20
# Should the VAE use batch normalization between each layer
VAE_BATCH_NORM = True

#
# GAN config
#

# Should the GAN be loaded from file
LOAD_GAN = True
# Which dataset to use for the GAN
GAN_DATAMODE = DataMode.MONO_BINARY_COMPLETE
# Should the GAN use batch normalization on the generator.
GAN_BATCHNORM_GEN = True
# Should the GAN use batch normalization on the disciminator.
GAN_BATCHNORM_DISC = False
# The dim size for the generator.
GAN_INPUT_DIM_SIZE = 128
# How many epochs
GAN_EPOCHS = 25
# Batch size
GAN_BATCH_SIZE = 512
# In order to reduce mode collapse you can use a classifier which adds loss when the generator
# creates images that predicts to the same class.
GAN_USE_CLASSIFIER = True
# Which folder to save all the figures in, both graph of loss and reconstructions.
GAN_FOLDER_GRAPHS = f"gan_graphs_{GAN_BATCH_SIZE}_{int(GAN_USE_CLASSIFIER)}" \
                    f"_{GAN_INPUT_DIM_SIZE}_{int(GAN_BATCHNORM_GEN)}_{GAN_DATAMODE.name}"
#
# Util
#
# If the verification net should be loaded from file or retrained.
LOAD_VERIFICATION_NET_MODEL = True

import tensorflow.keras.losses as losses
from stacked_mnist import StackedMNISTData
import numpy as np
import draw
import config
from autoencoder.AutoencoderModel import AutoencoderModel
import os
import util
import tensorflow as tf

# This is a wierd fix for my GPU to get it to always work.
gpu = tf.config.experimental.list_physical_devices('GPU')
if gpu:
    print("Setting memory growth on GPU 0")
    tf.config.experimental.set_memory_growth(gpu[0], True)


def get_autoencoder(datamode, generator):
    # This method either loads the model from file or creates a new autoencoder model and trains
    # it and then returns it.
    x_train, y_train = generator.get_full_data_set(training=True)
    x_train = x_train.astype(np.float64)

    # The shape of the image.
    image_shape = x_train.shape[1:]

    autoencoder_model = AutoencoderModel(image_shape)
    extra_file_name = "_split" if config.AUTO_SPLIT_RGB and datamode.name.startswith("COLOR") else ""
    autoencoder_file_name = f"./models/autoencoder_{datamode.name}{extra_file_name}.h5"

    if config.LOAD_AUTOENCODER and os.path.exists(autoencoder_file_name):
        # A little hack in order to be able to load the weights of a custom function. You need
        # to have the same amount of weights before loading the data, so i Fit on one example
        # before overriding the weights.
        autoencoder_model.fit(np.array([x_train[0]]), np.array([x_train[0]]), epochs=1, verbose=0)
        autoencoder_model.load_weights(autoencoder_file_name)
    else:
        autoencoder_model.fit(x_train, x_train, epochs=config.AUTO_EPOCHS)
        autoencoder_model.save_weights(autoencoder_file_name)

    return autoencoder_model


def main():
    # Which data to use.
    datamode = config.AUTO_GEN_DATAMODE
    # Create a generator for that type of data.
    generator = StackedMNISTData(mode=datamode, default_batch_size=2048)

    # Take out the testing dataset.
    x_test, y_test = generator.get_full_data_set(training=False)
    x_test = x_test.astype(np.float64)

    # Create a verification model.
    net = util.get_verification_model(datamode, generator)

    autoencoder_model = get_autoencoder(datamode, generator)

    draw.predict_and_draw(autoencoder_model,
                          np.array(x_test[0:16]),
                          np.array(y_test[0:16]),
                          mult_255=False)
    batch_size = 16

    # Reconstruct the images of the test set.
    reconstructed_images = autoencoder_model.predict(x_test, batch_size=batch_size)
    # Check the mode collapse. If coverage is high then we don't have mode collapse.
    cov = net.check_class_coverage(data=reconstructed_images, tolerance=.8)
    pred, acc = net.check_predictability(data=reconstructed_images, correct_labels=y_test)
    print(f"Autoencoder - Reconstructed images - Coverage: {100 * cov:.2f}%")
    print(f"Autoencoder - Reconstructed images - Predictability : {100 * pred:.2f}%")
    # This one should be over 80%
    print(f"Autoencoder - Reconstructed images - Accuracy: {100 * acc:.2f}%")
    print("---------------------------------------------")

    # Random latents.
    if datamode.name.startswith("COLOR") and config.AUTO_SPLIT_RGB:
        # if color Then we need 3 times as many batches
        batch_size *= 3
    latents = np.random.randn(batch_size, config.AUTO_LATENT_SIZE)
    # Generate images by using the random data in the decoder model.
    generated_images = autoencoder_model.decoder_model(latents)
    # Convert them to numpy arrays.
    generated_images = tf.keras.backend.eval(generated_images)
    # Draw the generated images.
    draw.draw_images(generated_images, mult_255=False)

    cov = net.check_class_coverage(data=generated_images, tolerance=.8)
    pred, _ = net.check_predictability(data=generated_images)
    print(f"Autoencoder - Generated images - Coverage: {100 * cov:.2f}%")
    print(f"Autoencoder - Generated images - Predictability: {100 * pred:.2f}%")
    print("---------------------------------------------")

    #
    # Anomaly detector
    #
    datamode = config.AUTO_ANOM_DATAMODE
    generator = StackedMNISTData(mode=datamode, default_batch_size=2048)
    x_test, y_test = generator.get_full_data_set(training=False)
    x_test = x_test.astype(np.float64)

    autoencoder_model = get_autoencoder(datamode, generator)

    prediction = autoencoder_model.predict(x_test)
    # Flatten in order to simplify the loss calculation.
    x_test_flatten = x_test.reshape(x_test.shape[0], np.product(x_test.shape[1:]))
    pred_flatten = prediction.reshape(prediction.shape[0], np.product(prediction.shape[1:]))

    loss = losses.mse(x_test_flatten, pred_flatten)
    # get the top 16 with most loss.
    top_loss = loss.numpy().argsort()[-16:][::-1]

    top_16 = []
    top_16_labels = []
    for i in top_loss:
        top_16.append(x_test[i])
        top_16_labels.append(str(y_test[i]))

    # Conclusion: Autoencoders work well with finding anomalies, however quite bad for being a
    # generator.
    draw.draw_images(np.array(top_16), labels=top_16_labels, mult_255=False)


if __name__ == '__main__':
    main()

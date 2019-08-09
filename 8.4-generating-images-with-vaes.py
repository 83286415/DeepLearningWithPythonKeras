import keras
from keras import layers
from keras import backend as K
from keras.models import Model
import numpy as np


# VAE encoder network:

img_shape = (28, 28, 1)  # MNIST image shape: (28, 28, 1)  that is (width, height, pixel grey) and grey in (0, 255) P22
batch_size = 16
latent_dim = 2  # Dimensionality of the latent space: a plane

input_img = keras.Input(shape=img_shape)

x = layers.Conv2D(32, 3,
                  padding='same', activation='relu')(input_img)  # padding same: cloud note search "padding"
# padding same: https://github.com/keras-team/keras/pull/9473#issuecomment-372166860
# padding=valid: do NOT padding the void space.    padding=same: padding the void space

x = layers.Conv2D(64, 3,
                  padding='same', activation='relu',
                  strides=(2, 2))(x)  # strides: steps on (x axis, y axis)
x = layers.Conv2D(64, 3,
                  padding='same', activation='relu')(x)
x = layers.Conv2D(64, 3,
                  padding='same', activation='relu')(x)
shape_before_flattening = K.int_shape(x)

x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)

# instantiation of Dense layer into two vectors below:
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)


# sample func in DIY layer
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=1.)  # epsilon: a very tiny vector in 01 standard normal distribution
    return z_mean + K.exp(z_log_var) * epsilon  # return a random sample point


z = layers.Lambda(sampling)([z_mean, z_log_var])  # layers.Lambda(func)(para): DIY layer


# VAE decoder network:

# This is the input where we will feed `z`.
decoder_input = layers.Input(K.int_shape(z)[1:])  # int_shape: return a shape of a tensor as the tuple

# Up-sample to the correct number of units
x = layers.Dense(np.prod(shape_before_flattening[1:]),
                 activation='relu')(decoder_input)  # prod: product, product=a*b

# Reshape into an image of the same shape as before our last `Flatten` layer
x = layers.Reshape(shape_before_flattening[1:])(x)  # just reshape the input tensor and output a reshaped tensor

# We then apply then reverse operation to the initial
# stack of convolution layers: a `Conv2DTranspose` layers
# with corresponding parameters.
x = layers.Conv2DTranspose(32, 3,
                           padding='same', activation='relu',
                           strides=(2, 2))(x)  # 32: filters    3: conv kernel size (3, 3)
x = layers.Conv2D(1, 3,
                  padding='same', activation='sigmoid')(x)  # sigmoid: output in (0, 1)
# We end up with a feature map of the same size as the original input.

# This is our decoder model.
decoder = Model(decoder_input, x)  # Model(input_layer_output_tensor, output_layer_output_tensor) P208

# We then apply it to `z` to recover the decoded `z`.
z_decoded = decoder(z)  # input the sample point into the decoder instance And z ONLY used as input tensor


# VAE losses

# make layer class to add loss to VAE
class CustomVariationalLayer(keras.layers.Layer):

    def vae_loss(self, x, z_decoded):  # define the VAE loss
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs, **kwargs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)  # loss: define VAE loss in the def above
        self.add_loss(loss, inputs=inputs)  # add the loss to VAE layer
        return x  # We don't use this output. And only use the loss added to this layer


# We call our custom layer on the input and the decoded output,
# to obtain the final model output.
y = CustomVariationalLayer()([input_img, z_decoded])


# build the final model and load data into it

from keras.datasets import mnist

vae = Model(input_img, y)
vae.compile(optimizer='rmsprop', loss=None)  # loss has been defined in vae output_tensor' model CustomVariationalLayer.
print(vae.summary())

# Train the VAE on MNIST digits
(x_train, _), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape(x_test.shape + (1,))

vae.fit(x=x_train, y=None,
        shuffle=True,
        epochs=10,
        batch_size=batch_size,
        validation_data=(x_test, None))  # Also validation does NOT need the target label data set either. So NONE.


# plot
import matplotlib.pyplot as plt
from scipy.stats import norm

# Display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# Linearly spaced coordinates on the unit square were transformed
# through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z,
# since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = decoder.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
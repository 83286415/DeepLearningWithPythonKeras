import keras
print(keras.__version__)  # 2.2.4

from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical


# 5.1 py
# Import codes from 5.1 py for data preparation and model building

#  build the network model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())  # 3D->1D
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # multi-classification


# prepare mnist data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()  # total 70000 images in MNIST

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255  # astype: keep the int part

test_images = test_images.reshape((10000, 28, 28, 1))  # reshape 255 -> 1
test_images = test_images.astype('float32') / 255  # make values in [0, 255] to values in [0, 1]

train_labels = to_categorical(train_labels)  # one_hot labels
test_labels = to_categorical(test_labels)

# commented out below codes for adding callbacks in fit process
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['acc'])
# model.fit(train_images, train_labels, epochs=5, batch_size=64)
#
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print(test_loss, test_acc)  # 0.0261321800082751    0.9926 better than 0.978 which in chapter 2 using dense layers

print('data preparation and model building done')
print('---------------------------------------------------------------------------------------------------------------')


# 7.2.1.3 define my own callbacks

import keras


class ActivationLogger(keras.callbacks.Callback):

    def set_model(self, model):
        self.model = model
        layer_outputs = [layer.output for layer in model.layers]  # can read model's property
        self.activations_model = keras.models.Model(model.input, layer_outputs)  # can make Model instance

    def on_epoch_end(self, epoch, logs=None):  # other 5 defs could be recognized by fit(): refer to book P212
        if self.validation_data is None:  # can read validation_data in fit()
            raise RuntimeError('Requires validation_data.')
        f = open('activations_at_epoch_' + str(epoch) + '.txt', 'w')
        f.write('the first epoch activation saved by my own callbacks')
        f.close()


print('define my own callbacks done')
print('---------------------------------------------------------------------------------------------------------------')


# 7.2.1.1 keras callbacks in training process
import os

base_dir = 'D:/AI/deep-learning-with-python-notebooks-master'
h5_path = os.path.join(base_dir, '7.2.1_model_checkpoint.h5')


# callback (list) should be defined in front of the fit()
callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=1,),
                  keras.callbacks.ModelCheckpoint(filepath=h5_path, monitor='val_loss', save_best_only=True),
                  keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3),  # keras callbacks
                  ActivationLogger()]  # my own callback

# import compile() and fit() from 5.1 py and add callbacks into the fit() as below
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])
history = model.fit(train_images, train_labels, epochs=20, batch_size=64, callbacks=callbacks_list, validation_split=0.2)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_loss, test_acc)  # 0.03851035969056033 0.9921

# plot
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

print('keras callbacks in training process done')
print('---------------------------------------------------------------------------------------------------------------')
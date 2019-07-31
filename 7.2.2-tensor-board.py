# import code from 6.4.4 py for Tensor Board

import keras
from keras import layers
from keras.layers import Embedding
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential

max_features = 2000
max_len = 500

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# data input
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

# build model
model = Sequential()
model.add(Embedding(max_features, 128, input_length=max_len, name='embed'))  # name

model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPool1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1))

print(model.summary())

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# define callbacks before model.fit()
callbacks = [keras.callbacks.TensorBoard(log_dir='my_log_dir', histogram_freq=1, embeddings_freq=1)]

# add callbacks in fit()
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2, callbacks=callbacks)


# new method to show layer picture via plot_model: it's not good. so use TensorBoard first.
# from keras.utils import plot_model
#
# plot_model(model, to_file='D:/AI/deep-learning-with-python-notebooks-master/my_log_dir/model.png', show_shapes=True)


import keras
print(keras.__version__)  # 2.1.6


# data pre-processing
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000  # number of words to consider as features
maxlen = 500  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)  # refer to pad_sequence's definition
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)


# build network model
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Embedding, Dense

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))  # other params: default values are all right.
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)  # 20% train data is used to validation. refer to P171 or cloud note
                    # if para shuffle is used in fit() with validation_split. validation_split first, shuffle later.P171


# build network model with bio-direction RNN
from keras import layers
bio_direction_model = Sequential()
bio_direction_model.add(layers.Embedding(max_features, 32))
bio_direction_model.add(layers.Bidirectional(layers.LSTM(32)))
bio_direction_model.add(layers.Dense(1, activation='sigmoid'))

bio_direction_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
bio_direction_history = bio_direction_model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)


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


# plot bio_direction results
bio_direction_acc = bio_direction_history.history['acc']
bio_direction_val_acc = bio_direction_history.history['val_acc']
bio_direction_loss = bio_direction_history.history['loss']
bio_direction_val_loss = bio_direction_history.history['val_loss']

bio_direction_epochs = range(len(bio_direction_acc))
plt.figure()

plt.plot(bio_direction_epochs, bio_direction_acc, 'bo', label='Training acc')
plt.plot(bio_direction_epochs, bio_direction_val_acc, 'b', label='Validation acc')
plt.title('BI Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(bio_direction_epochs, bio_direction_loss, 'bo', label='Training loss')
plt.plot(bio_direction_epochs, bio_direction_val_loss, 'b', label='Validation loss')
plt.title('BI Training and validation loss')
plt.legend()

plt.show()
import keras
print(keras.__version__)  # 2.2.4


# 6.2 RNN introduction

# RNN pseudo code
input_sequence = []


def f(_input, state):
    new_state = _input + state  # made up for def below
    return new_state


def rnn_pseudo_code():
    state_t = 0
    for input_t in input_sequence:
        output_t = f(input_t, state_t)
        state_t = output_t


# detailed RNN pseudo code
# W, U, b = 1, 2, 3


def dot(y, z):
    pass


def activation(x):
    return x


def rnn_pseudo_code_detailed():
    state_t = 0
    for input_t in input_sequence:
        output_t = activation(dot(W, input_t) + dot(U, state_t) + b)
        state_t = output_t


# RNN Numpy
import numpy as np

timesteps = 100
input_features = 32
output_features = 64

inputs = np.random.random((timesteps, input_features))
state_t = np.zeros((output_features, ))

W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features, ))

successive_outputs = []
for input_t in inputs:  # input_t.shape: (input_features, ) and it loops in inputs' timesteps(timesteps, input_features)
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)  # tanh: tan    dot: no matter dimensions of params
    successive_outputs.append(output_t)
    state_t = output_t

final_output_sequence = np.stack(successive_outputs, axis=0)  # refer to cloud note's math concept: dimension


print('6.2 RNN introduction done')
print('---------------------------------------------------------------------------------------------------------------')



# 6.2.1 RNN in Keras introduction

from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN

# RNN layer shows the final output of each sequence only
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))
print(model.summary())
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, None, 32)          320000    
_________________________________________________________________
simple_rnn_1 (SimpleRNN)     (None, 32)                2080      # rnn layer output: (batch, output_features)
=================================================================
Total params: 322,080
Trainable params: 322,080
Non-trainable params: 0
_________________________________________________________________
'''

# RNN layer shows all output at each time step of each sequence including the final output
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))  # not only the final output, the rnn layer will show the whole sequence
print(model.summary())
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_2 (Embedding)      (None, None, 32)          320000    
_________________________________________________________________
simple_rnn_2 (SimpleRNN)     (None, None, 32)          2080      # rnn layer output: (batch, timesteps, output_features)
=================================================================
Total params: 322,080
Trainable params: 322,080
Non-trainable params: 0
_________________________________________________________________
"""

# multi-RNN layers stack: need to show the whole output of each sequence in each layer, not necessary for the last RNN.
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))  # This last layer only returns the last outputs.
print(model.summary())
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_3 (Embedding)      (None, None, 32)          320000    
_________________________________________________________________
simple_rnn_3 (SimpleRNN)     (None, None, 32)          2080      
_________________________________________________________________
simple_rnn_4 (SimpleRNN)     (None, None, 32)          2080      
_________________________________________________________________
simple_rnn_5 (SimpleRNN)     (None, None, 32)          2080      
_________________________________________________________________
simple_rnn_6 (SimpleRNN)     (None, 32)                2080      # the last RNN may show the whole sequence or not.
=================================================================
Total params: 328,320
Trainable params: 328,320
Non-trainable params: 0
_________________________________________________________________
'''

print('6.2.1 RNN in Keras introduction done')
print('---------------------------------------------------------------------------------------------------------------')


# IMDB reviews classification by RNN in Keras


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
from keras.layers import Dense

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))  # only show the final output
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)
# loss: 0.0186 - acc: 0.9947 - val_loss: 0.6927 - val_acc: 0.8062 not good because review is cut for 500 words.


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

print('IMDB reviews classification by RNN in Keras done')
print('---------------------------------------------------------------------------------------------------------------')
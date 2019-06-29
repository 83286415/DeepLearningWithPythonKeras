import keras
print(keras.__version__)  # 2.1.6


# define embedding layer
from keras.layers import Embedding

# The Embedding layer takes at least two arguments:
# the number of possible tokens, here 1000 (1 + maximum word index),
# and the dimensionality of the embeddings, here 64.
embedding_layer = Embedding(1000, 64)


# load IMDB data
from keras.datasets import imdb
from keras import preprocessing

# Number of words to consider as features
max_features = 10000  # 1000 tokens: the most 1000 common words in comments
# Cut texts after this number of words
# (among top max_features most common words)
maxlen = 20  # most words in a comment: 20

# Load the data as lists of integers.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# This turns our lists of integers
# into a 2D integer tensor of shape `(samples, maxlen)`
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)


# build network model
from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
# We specify the maximum input length to our Embedding layer
# so we can later flatten the embedded inputs
model.add(Embedding(10000, 8, input_length=maxlen))
# After the Embedding layer,
# our activations have shape `(samples, maxlen, 8)`.

# We flatten the 3D tensor of embeddings
# into a 2D tensor of shape `(samples, maxlen * 8)`
model.add(Flatten())

# We add the classifier on top
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)

history_dict = history.history  # history is a dict with loss, validation loss, acc and validation acc items
print(history_dict.items())
# dict_items([('val_loss', [0.6398002065658569, 0.546724056148529, 0.5113224940299987, 0.5008151711463928,
# 0.49810957527160643, 0.5013553310394288, 0.5051595081329345, 0.5131720875740051, 0.5213448650360107,
# 0.5302545790672302]), ('val_acc', [0.6814, 0.7206, 0.7384, 0.7452, 0.7536, 0.753, 0.752, 0.7486, 0.749, 0.7466]),
# ('loss', [0.6758890947341919, 0.5657497178077697, 0.4751638753890991, 0.4263368104696274, 0.3930284927129746,
# 0.36679954481124877, 0.34346731429100036, 0.32227740573883057, 0.3022508435368538, 0.28389928830862043]),
# ('acc', [0.605, 0.74255, 0.78075, 0.80775, 0.8258, 0.8395, 0.85335, 0.8657, 0.8766, 0.886])])

results = model.evaluate(x_test, y_test)  # evaluate returns loss value and metrics value in a list
print(results)  # [0.5214700441932678, 0.75584] the first is loss, the second is acc


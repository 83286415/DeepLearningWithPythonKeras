# 3.5.1 reuters data set
from keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
print(len(train_data))  # 8982
print(len(train_data[0]))  # 87
print(len(test_data))  # 2246

words_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in words_index.items()])
print(reverse_word_index)  # {10996: 'mdbl', 16260: 'fawc', 12089: 'degussa', 8803: 'woods', 13796: 'hanging'...}
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(train_data[10])  # [1, 245, 273, 207, 156, 53, 74, 160, 26, 14, 46, 296, 26, 39, 74, 2979, ..., 12]
print(train_labels[10])  # output: 3.   labels are 0 ~ 45， all 46 new topics
print(reverse_word_index.get(3))  # to

print('3.5.1 done')
print('---------------------------------------------------------------------------------------------------------------')

# 3.5.2 data preparing
import numpy as np


def vectorize_sequences(sequences, dimension=10000):  # 10000 columns in this array but only 87 used for train_data[0]
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # explanation refer to 3.4-binary-classification.py line 22
    return results


# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)


# do not use this method. use def to_categorical below
def to_one_hot(labels, dimension=46):  # make one hot label array, only 46 news topics, so 46 columns in this array
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.  # make it 1 at the label's number in each row
    return results


# Our vectorized training labels
one_hot_train_labels = to_one_hot(train_labels)  # one hot array
# Our vectorized test labels
one_hot_test_labels = to_one_hot(test_labels)


from keras.utils.np_utils import to_categorical  # to_categorical is the one hot function in Keras

one_hot_train_labels = to_categorical(train_labels)  # it can replace the to_one_hot function above
one_hot_test_labels = to_categorical(test_labels)

print('3.5.2 done')
print('---------------------------------------------------------------------------------------------------------------')

# 3.5.3 network building
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))  # 16 is too small for 46 output, so make it 64
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
# softmax is used in multi-classification problem's last layer  # softmax refer to my cloud note

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])  # loss function is categorical_crossentropy for one hot lable.
# And for int label the loss function is sparse_categorical_crossentropy  refer to my cloud note.

print('3.5.3 done')
print('---------------------------------------------------------------------------------------------------------------')

# 3.5.4 validation
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# re-create a 9 epochs network model
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=9,
          batch_size=512,
          validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)

print(results)  # 9 epochs: [1.0231352194228134, 0.7751558326443496]  10 epochs: [1.040993108137951, 0.7827248442204849]

# re-create a 8 epochs network model
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=8,
          batch_size=512,
          validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)

print(results)  # 8 epochs[0.9895354098543352, 0.7791629563934126] is the best results

# the accuracy of random classification is about 18% as below. So 78% above is much better than this random one.
import copy

test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
print(float(np.sum(np.array(test_labels) == np.array(test_labels_copy))) / len(test_labels))  # 0.18477292965271594

print('3.5.4 done')
print('---------------------------------------------------------------------------------------------------------------')


# 3.5.5 prediction
predictions = model.predict(x_test)
print(predictions)
''' prediction:
[[1.2422490e-04 8.8846609e-05 3.8576080e-05 ... 6.9606205e-05
  4.2313910e-05 7.5345665e-06]
 [1.6841191e-03 1.8730082e-01 6.8207250e-05 ... 8.7879237e-04
  4.1628376e-05 9.0748147e-04]
 [2.9311755e-03 7.6827568e-01 1.2176393e-03 ... 9.4559242e-04
  2.4842826e-04 3.6094635e-04]
 ...
 [4.5932658e-05 3.3827784e-04 3.2794735e-05 ... 4.7543181e-05
  1.6395859e-05 9.5352607e-06]
 [1.7312149e-03 4.9540058e-02 6.3819289e-03 ... 1.0937958e-03
  2.6582938e-04 4.6970788e-04]
 [1.1999221e-04 8.1502205e-01 2.8892888e-03 ... 4.9808586e-04
  1.3027200e-05 8.0291204e-05]]
'''

print(predictions[0].shape)  # 46: each array has 46 elements
print(np.sum(predictions[0]))  # 1: the sum of each array's elements is 1
print(np.argmax(predictions[0]))  # 4: the max value is the fourth because it's the fourth kind of news

print('3.5.5 done')
print('---------------------------------------------------------------------------------------------------------------')


# 3.5.6 another loss function
# loss function: sparse_categorical_crossentropy.
# refer to my cloud note

print('3.5.6 done')
print('---------------------------------------------------------------------------------------------------------------')


# 3.5.7 limited Dense dimension Flask
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))  # make it 4 instead of 64
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=20,
          batch_size=128,
          validation_data=(x_val, y_val))

results = model.evaluate(x_test, one_hot_test_labels)
print(results)  # [1.9896976053980133, 0.6665182546749777] the accuracy 66.7% is much lower than 78% above
# So the 4 dimension in layer 2 is the flask for the information storage in this network model.

print('3.5.7 done')
print('---------------------------------------------------------------------------------------------------------------')


# 3.5.8 further test
# problem 1: Dense(128)
# epochs = 4, 5, 6
# 4: [1.0079761674546388, 0.7698130009435482]
# 5: [0.9829678087497755, 0.7827248442204849]
# 6: [0.9697435776887997, 0.7911843276936776] this is the best one and even better than (Dense(64) and 8 epochs)'s

# problem 2: only 1 layer or 3 layers in network model
# only 1 layer
# epochs = 10
# 10 [0.9018554657777292, 0.7956366874974218] this is the best one ever!!!

# 3 layers
# epochs = 8, 11
# 8: [1.0612086290137002, 0.7720391807658059]
# 11: [1.1325280534722182, 0.7813891362422084]

print('3.5.8 done')
print('---------------------------------------------------------------------------------------------------------------')
# 3.4.1 IMDB data

from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)  # the first 10000 most used word
print(len(train_data), len(test_data))  # 25000 25000
print(len(train_data[0]), train_data[0])  # output: 218, [1, 14, 22, ... , 178, 32]; each int means a word in English
print(train_labels[0])  # output: 1  means positive; and 0 means negative

word_index = imdb.get_word_index()  # word_index {'word': int}
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])  # dict((a, b)) makes {a: b}
print(reverse_word_index)  # {int: word, int: word, ...}
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])  # the first third are not words
print(decoded_review)  # output: ? this film was just brilliant casting location scenery ...
print('3.4.1 done')
print('---------------------------------------------------------------------------------------------------------------')


# 3.4.2 prepare data with one hot encode

import numpy as np

def vectorize_squences(sequences, dimension=10000):
    # 10000: 10000 columns in this array though only 218 used for train data
    results = np.zeros((len(sequences), dimension))  # zeros(x, y) makes a x row y columns matrix with all 0 elements
    for i, sequence in enumerate(sequences):
        # print('sequence:', sequence)  # output: sequence: [1, 445, 32, ..., 434]
        results[i, sequence] = 1.  # the elements in the i's row and all columns in sequence list are 1. see e.g. below:
        '''
        results = np.zeros((3, 4))
        print(results)
        results[1, [2, 3]] = 1
        print(results)
        output:
        [[0. 0. 0. 0.]
         [0. 0. 0. 0.]
         [0. 0. 0. 0.]]
        [[0. 0. 0. 0.]
         [0. 0. 1. 1.]
        [0. 0. 0. 0.]]
        '''
    return results

x_train = vectorize_squences(train_data)
x_test = vectorize_squences(test_data)
print(x_train[0])  # output: [0. 1. 1. ... 0. 0. 0.]

y_train = np.asarray(train_labels).astype('float32')  # make a float array(y_train) from the list train_labels
y_test = np.asarray(test_labels).astype('float32')
print(y_train[0])  # output: 1.0
print('3.4.2 done')
print('---------------------------------------------------------------------------------------------------------------')


# 3.4.3 building network

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# codes commented below are used for self-defined parameters optimizers, losses and metrics
# from keras import optimizers
# from keras import losses
# from keras import metrics
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#               loss=losses.binary_crossentropy,
#               metrics=[metrics.binary_accuracy])

print('3.4.3 done')
print('---------------------------------------------------------------------------------------------------------------')


# 3.4.4 validation this approach

x_val = x_train[:10000]  # the first 10000 used for validation
partial_x_train = x_train[10000:]  # the last 15000 used for training
# Train on 15000 samples, validate on 10000 samples

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,  # iteration 20 times
                    batch_size=512,  # 512 batches data input in each fit process
                    validation_data=(x_val, y_val))

history_dict = history.history  # history is a dict with loss, validation loss, acc and validation acc items
print(history_dict.items())  # loss: 0.0098 - acc: 0.9979 - val_loss: 0.6984 - val_acc: 0.8664

# pictures
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# start a new network model with only 4 iteration because the min loss is in the fourth iteration
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)  # evaluate returns loss value and metrics value in a list
print(results)  # [0.32347040958404544, 0.87304]

print('3.4.4 done')
print('---------------------------------------------------------------------------------------------------------------')


# 3.4.5 prediction on x_test data set

prediction = model.predict(x_test)
print(prediction)
'''
value > 0.99 is positive and < 0.01 is negative; else is not confirmed
[[0.13736041]
 [0.9996959 ]
 [0.29713306]
 ...
 [0.07059101]
 [0.0433808 ]
 [0.47282436]]
'''

print('3.4.5 done')
print('---------------------------------------------------------------------------------------------------------------')
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

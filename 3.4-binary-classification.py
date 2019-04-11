# 3.4.1 IMDB data
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
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
    results = np.zeros((len(sequences), dimension))  # zeros(x, y) makes a x row y columns matrix with all 0 elements
    for i, sequence in enumerate(sequences):
        # print('sequence:', sequence)  # outpu: senquence: [1, 445, 32, ..., 434]
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


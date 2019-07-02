import keras
print(keras.__version__)  # 2.2.4
import os


# prepare the imdb data
base_dir = 'D:/AI/deep-learning-with-python-notebooks-master'
imdb_dir = os.path.join(base_dir, 'aclImdb')
train_dir = os.path.join(imdb_dir, 'train')
print(train_dir)

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding='utf-8')  # # to fix the crash due to gbk encoding in files
            # print(os.path.join(dir_name, fname))  # to fix the crash due to gbk encoding in files
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
print(len(texts), len(labels))  # 25000, 25000: 25000 reviews but 88583 different words so 88583 unique tokens


print('prepare the imdb data done')
print('---------------------------------------------------------------------------------------------------------------')


#  Tokenize the data

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 100  # We will cut reviews after 100 words
training_samples = 200  # We will be training on 200 samples as we use pre-trained model
validation_samples = 10000  # We will be validating on 10000 samples
max_words = 10000  # We will only consider the top 10,000 words in the dataset

# P150 review -> word -> word index -> index sequence
tokenizer = Tokenizer(num_words=max_words)  # keep the most frequently used words input
# print(texts[:5])  # texts list is not changed.
tokenizer.fit_on_texts(texts)  # texts: sentences list input.
# print(texts[:5])  # texts is not changed. The words sequence is kept in instance tokenizer.
sequences = tokenizer.texts_to_sequences(texts)
# print(sequences[:5])  # reviews in texts are changed into int list sequence. Each elements in list is a int review.
print(len(sequences))  # 25000. So still 25000 reivew in sequence list. but in int form.

word_index = tokenizer.word_index  # it's a word index dict {word: vector array}
print('Found %s unique tokens.' % len(word_index))  # Found 88583 unique tokens. But only first 10000 taken by us.

data = pad_sequences(sequences, maxlen=maxlen)  # data is a 2 dimension tensor (reviews count, maxlen): (25000, 100)
print(sequences[0])  # the first review sentence in int list form
print(data[0])  # the first review sentence's first 100 words in Numpy array form

labels = np.asarray(labels)  # list -> array
print('Shape of data tensor:', data.shape)  # Shape of data tensor: (25000, 100)
print('Shape of label tensor:', labels.shape)  # Shape of label tensor: (25000,)

# Split the data into a training set and a validation set
# But first, shuffle the data, since we started from data
# where sample are ordered (all negative first, then all positive).
indices = np.arange(data.shape[0])  # data.shape[0] is the int 25000; np.arange make this int to a array range as below
print("indices: ", indices)  # indices:  [    0     1     2 ... 24997 24998 24999]
np.random.shuffle(indices)
# To re-shuffle indices actually re-shuffle data set and labels array
print("indices after re-shuffle: ", indices)  # indices after re-shuffle:  [ 1627 15466 12511 ... 13539   917 10234]
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]  # 0-199: total 200
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]  # 200-10200: total 10000
y_val = labels[training_samples: training_samples + validation_samples]


print('Tokenize the data done')
print('---------------------------------------------------------------------------------------------------------------')


# pre-processing the embedding words
glove_dir = os.path.join(base_dir, "glove")

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf-8')

for line in f:  # each line is like this below:
# of -0.1529 -0.24279 0.89837 0.16996 0.53516 0.48784 -0.58826 -0.17982 -1.3581 0.42541 0.15377 0.24215 0.13474 0.41193
#  0.67043 -0.56418 0.42985 -0.012183 -0.11677 0.31781 0.054177 -0.054273 0.35516 -0.30241 0.31434 -0.33846 0.71715
# -0.26855 -0.15837 -0.47467 0.051581 -0.33252 0.15003 -0.1299 -0.54617 -0.37843 0.64261 0.82187 -0.080006 0.078479
# -0.96976 -0.57741 0.56491 -0.39873 -0.057099 0.19743 0.065706 -0.48092 -0.20125 -0.40834 0.39456 -0.02642 -0.11838
# 1.012 -0.53171 -2.7474 -0.042981 -0.74849 1.7574 0.59085 0.04885 0.78267 0.38497 0.42097 0.67882 0.10337 0.6328
# -0.026595 0.58647 -0.44332 0.33057 -0.12022 -0.55645 0.073611 0.20915 0.43395 -0.012761 0.089874 -1.7991 0.084808
# 0.77112 0.63105 -0.90685 0.60326 -1.7515 0.18596 -0.50687 -0.70203 0.66578 -0.81304 0.18712 -0.018488 -0.26757
# 0.727 -0.59363 -0.34839 -0.56094 -0.591 1.0039 0.20664

    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')  # make this list into a array
    embeddings_index[word] = coefs  # make the dict
f.close()

print('Found %s word vectors.' % len(embeddings_index))  # Found 400000 word vectors. 40000 words total in glove.txt.

embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))  # (10000, 100) we only take the first 10000 words in glove
# emvedding_matrix: the weight matrix used in embedding layer
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        # print(word, embedding_vector)
        """
        flowers [-0.44986   1.2823   -1.1963   -0.32365   0.68291   0.52996  -0.21791
                    -0.17259  -0.23735  -0.81014  -0.40655   0.26745   0.75077   1.275
                    0.59501  -0.024502  0.51581   0.772     0.34644  -0.30634  -0.70957
                     -0.42754   0.15067   0.535     0.84944   0.87777  -0.052852  0.14286
                      0.13412  -0.56484  -0.194    -0.33699   1.3296   -0.54434   0.018516
                      0.68235   0.6706    0.23561  -0.31935   0.041871  0.83853  -0.93614
                     -0.023684 -0.15435   0.50661  -0.22877  -0.46391   0.88397  -0.10311
                     -0.27301   0.90356  -1.183     0.5856    0.7186   -0.1454   -0.90433
                     -0.62158   0.42393   0.873    -0.045632  0.077794  1.9711   -0.29125
                     -0.015154  0.34863  -0.3784    0.88986  -0.23513   0.24497  -1.0246
                     -0.39326   0.060931 -0.56803  -0.93476   0.27972   1.2796    0.12686
                     -0.1511   -0.18075   0.40589   0.64355   0.10271  -0.63923  -0.27259
                     -0.26398  -0.60602   0.66943  -0.21665   0.65733   0.43637  -0.16395
                     -0.61197   0.3774   -0.39222  -1.0735   -0.069955 -1.7182    0.033848
                      0.48937   0.25335 ]
        """
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector


print('pre-processing the embedding words done')
print('---------------------------------------------------------------------------------------------------------------')


# build network model

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))  # maxlen: 100 the first 100 words in review
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 100, 100)          1000000   
_________________________________________________________________
flatten_1 (Flatten)          (None, 10000)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 32)                320032    
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 33        
=================================================================
Total params: 1,320,065
Trainable params: 1,320,065
Non-trainable params: 0
_________________________________________________________________
'''

print('build network model done')
print('---------------------------------------------------------------------------------------------------------------')


# embedding glove and train the model
model.layers[0].set_weights([embedding_matrix])  # embedding words weight matrix is used here
model.layers[0].trainable = False  # embedding layer is frozen here

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')

print('build network model done')
print('---------------------------------------------------------------------------------------------------------------')


# plot
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

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

print('plot done')
print('---------------------------------------------------------------------------------------------------------------')


# re-build network model without freezing Embedding layer or setting_weight
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

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

print('re-build network model without freezing Embedding layer or setting_weight done')
print('---------------------------------------------------------------------------------------------------------------')


# evaluate network model with test data set
test_dir = os.path.join(imdb_dir, 'test')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding='utf-8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)


model.load_weights('pre_trained_glove_model.h5')
print(model.evaluate(x_test, y_test))  # [0.7617308892059326, 0.56712] that is [loss, acc]

print('evaluate network model with test data set done')
print('---------------------------------------------------------------------------------------------------------------')
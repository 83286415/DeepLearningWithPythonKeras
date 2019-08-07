import keras
import numpy as np


# data prepare
path = keras.utils.get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')  # download the file from url and return its path

text = open(path).read().lower()
print('Corpus length:', len(text))  # Corpus length: 600893

# Length of extracted character sequences
maxlen = 60

# We sample a new sequence every `step` characters
step = 3

# This holds our extracted sequences
sentences = []

# This holds the targets (the follow-up characters)
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])  # each element in sentences list is a sub-list of 60 length
    next_chars.append(text[i + maxlen])  # the character following the sentences' list is appended into the next_char
print('Number of sequences:', len(sentences))  # Number of sequences: 200278

# List of unique characters in the corpus
chars = sorted(list(set(text)))  # set: unique    list: can be sorted     sorted: list all elements in ascending order
print('Unique characters:', len(chars))  # Unique characters: 58

# Dictionary mapping unique characters to their index in `chars`
char_indices = dict((char, chars.index(char)) for char in chars)  # {char: index}

# Next, one-hot encode the characters into binary arrays.
print('Vectorization...')  # make train data set and its label set filled with 0s
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1  # one-hot code change the char's 0 into 1 in the x array
    y[i, char_indices[next_chars[i]]] = 1  # make the next character's 0 into 1 in label set y


print('data prepare done')
print('---------------------------------------------------------------------------------------------------------------')


# build network model

from keras import layers

model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))  # input shape is each sentence one-hot array
# not only LSTM layer but also Conv1D layer could be used here instead
model.add(layers.Dense(len(chars), activation='softmax'))  # softmax: multi-classifier

optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)  # the loss function is due to one-hot code


print('build network model done')
print('---------------------------------------------------------------------------------------------------------------')


# text generation

# re-weighting the probability array with temperature and return the max probability's index for prediction. P230
def sample(preds, temperature=1.0):  # preds: [p1, p2, p3, p4 ... p58] a probability list
    preds = np.asarray(preds).astype('float64')  # make this list into a array
    preds = np.log(preds) / temperature  # log function divides temp
    exp_preds = np.exp(preds)  # Calculate the exponential of all elements in the input array.
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)  # 1: pick up 1 sample; preds: pick up from this; 1: output shape
    return int(np.argmax(probas))  # Returns the indices of the maximum values along an axis.


import random
import sys

for epoch in range(1, 60):  # epochs = 59
    print('epoch', epoch)
    # Fit the model for 1 epoch on the available training data
    model.fit(x, y,
              batch_size=128,
              epochs=1)

    # Select a text seed at random
    start_index = random.randint(0, len(text) - maxlen - 1)  # random pick a starting point
    generated_text = text[start_index: start_index + maxlen]
    print('--- Generating with seed: "' + generated_text + '"')

    for temperature in [0.2, 0.5, 1.0, 1.2]:  # higher temperature, more randomness
        print('------ temperature:', temperature)
        sys.stdout.write(generated_text)

        # We generate 400 characters
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):  # one-hot code the randomly picked generated_text
                sampled[0, t, char_indices[char]] = 1.

            predses = model.predict(sampled, verbose=0)  # return a y label list like: [[p1, p2, p3...]]
            preds = predses[0]                          # remove the [] outside

            next_index = sample(preds, temperature)  # sample() returns the max value's index (int)
            next_char = chars[next_index]  # next_index is a int. i change it manually

            generated_text += next_char
            generated_text = generated_text[1:]  # rebuild the generated text list with new added next_char predicted

            sys.stdout.write(next_char)  # output the next_char one by one
            sys.stdout.flush()  # clean the caches in case of showing old next_char
        print()

print('text generation done')
print('---------------------------------------------------------------------------------------------------------------')
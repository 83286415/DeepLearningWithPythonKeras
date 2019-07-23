# do not run this code
from keras.models import Model
from keras import layers
from keras import Input

# build network model with multi-input

text_vocabulary_size = 10000  # the first 10000 words used most frequently; 10000 as the input dim in Embedding layer.
question_vocabulary_size = 10000  # question text input. input one
answer_vocabulary_size = 500       # answer text input. input two

# input 1:
text_input = Input(shape=(None,), dtype='int32', name='text')
# shape(None,): None dimension specified, The length of array input is uncertain.
# name: the unique layer name in the model.

embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)  # out:a word embedded into a 64 dimension vector
encoded_text = layers.LSTM(32)(embedded_text)

# input 2:
question_input = Input(shape=(None,), dtype='int32', name='question')  # the name should be unique!
embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

# connect two input tensors which could be different size. input 1 is 32 but input 2 is 16.
concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)  # -1: the last axis
# concatenat: https://blog.csdn.net/leviopku/article/details/82380710
answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)  # multi-classification

# make model instance
model = Model([text_input, question_input], answer)  # if multi-input, use list or tuple of tensors
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

print('build network model done')
print('---------------------------------------------------------------------------------------------------------------')

# prepare input data and model fit ways
import numpy as np
from keras.utils import to_categorical  # one_hot code

num_samples = 1000
max_length = 100

text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))  # low:1; high:10000; size:array size
question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))
answers = np.random.randint(answer_vocabulary_size, size=(num_samples,))  # low: 500; no high limit
answers = to_categorical(answers, answer_vocabulary_size)  # one hot:map answer array into a 500 dimension binary matrix

# two ways to fit this model. pick one of them. P202

# fit way 1:
model.fit([text, question], answers, epochs=10, batch_size=128)

# fit way 2: need to give a unique name to each input layer
model.fit({'text': text, 'question': question}, answers, epochs=10, batch_size=128)  # dict (name: input tensor)

print('prepare input data done')
print('---------------------------------------------------------------------------------------------------------------')
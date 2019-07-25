# do not run this code
from keras.models import Model
from keras import layers
from keras import Input


left_data, right_data, targets = None, None, None  # define these training data and label to ignore the error red signs
lstm = layers.LSTM(32)  # make a LSTM layer instance to share this layer's weight

left_input = Input(shape=(None, 128))  # None: batch count is uncertain; 128: the length of sequence of input P208
left_output = lstm(left_input)  # use the shared lstm layer weight: the 1st time

right_input = Input(shape=(None, 128))
right_output = lstm(right_input)  # use the shared lstm layer weight: the 2nd time

merged = layers.concatenate([left_output, right_output], axis=-1)  # connect two tensors and put it into a classifier
predictions = layers.Dense(1, activation='sigmoid')(merged)  # Dense layer to make the Model's output

model = Model([left_input, right_input], predictions)  # make the model instance for fitting later
model.fit([left_data, right_data], targets)  # input training data and label to fit the model

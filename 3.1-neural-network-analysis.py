from keras import layers
print(len((784,)))
layer = layers.Dense(units=32, input_shape=(784,))

# do not run this code
from keras import applications
from keras import layers
from keras import Input

xception_base = applications.Xception(weights=None, include_top=False)  # make the model class a instance

left_input = Input(shape=(250, 250, 3))
right_input = Input(shape=(250, 250, 3))

left_features = xception_base(left_input)  # use the model instance as a layer with input and output P209
right_features = xception_base(right_input)

merged_features = layers.concatenate([left_features, right_features], axis=-1)  # connect


import keras
print(keras.__version__)  # 2.2.4


from keras.applications import VGG16  # import the network model trained
from keras import backend as K  # backend: tensorflow. refer to https://keras.io/zh/backend/#keras

model = VGG16(weights='imagenet',
              include_top=False)  # no Dense layer included

layer_name = 'block3_conv1'
filter_index = 0

layer_output = model.get_layer(layer_name).output  # Tensor("block3_conv1/Relu:0", shape=(?, ?, ?, 256), dtype=float32)

loss = K.mean(layer_output[:, :, :, filter_index])  # return the tensor of the first filter's mean
# loss: Tensor("Mean:0", shape=(), dtype=float32)

# The call to `gradients` returns a list of tensors (of size 1 in this case)
# hence we only keep the first element -- which is a tensor.
grads = K.gradients(loss, model.input)[0]





print('5.3.1.1 done')
print('---------------------------------------------------------------------------------------------------------------')

print(loss)
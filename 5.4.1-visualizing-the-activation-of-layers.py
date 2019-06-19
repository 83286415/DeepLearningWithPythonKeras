import keras
print(keras.__version__)  # 2.2.4

from keras.models import load_model

model = load_model('cats_and_dogs_small_2.h5')  # model: data augmentation with dropout between Flatten and Dense layers
print(model.summary())  # As a reminder.
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_5 (Conv2D)            (None, 148, 148, 32)      896         # the first 8 layers are Convnets layers
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 74, 74, 32)        0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 72, 72, 64)        18496     
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 36, 36, 64)        0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 34, 34, 128)       73856     
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 17, 17, 128)       0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 15, 15, 128)       147584    
_________________________________________________________________
max_pooling2d_8 (MaxPooling2 (None, 7, 7, 128)         0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 6272)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 6272)              0           # dropout layer: make 50% output 0 due to data aug
_________________________________________________________________
dense_3 (Dense)              (None, 512)               3211776   
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 513       
=================================================================
Total params: 3,453,121
Trainable params: 3,453,121
Non-trainable params: 0
_________________________________________________________________
'''

# load a picture
img_path = 'D:/AI/deep-learning-with-python-notebooks-master/cats_and_dogs_small/test/cats/cat.1700.jpg'

# We pre-process the image into a 4D tensor
from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path, target_size=(150, 150))  # translate image into tensor array below
img_tensor = image.img_to_array(img)  # img_tensor.shape: (150, 150, 3)
img_tensor = np.expand_dims(img_tensor, axis=0)  # add a dimension at 0 index, so the  new shape: (1, 150, 150, 3)
print(img_tensor.shape)  # (1, 150, 150, 3)

print('load the image done')
print('---------------------------------------------------------------------------------------------------------------')


# Remember that the model was trained on inputs that were preprocessed in the following way:
img_tensor /= 255.  # rescale all elements in the tensor: new elements value = old value / 255


# display this picture
import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])  # shape: (1, 150, 150, 3) so img_tensor[0] is the 1st dimension array (150, 150, 3), the pic.
plt.show()

print('show the image done')
print('---------------------------------------------------------------------------------------------------------------')

# display the third tunnel's (filter's) picture of the first layer in this model
from keras import models

# Extracts the outputs of the top 8 layers, which are Convnets layers defined as above.
layer_outputs = [layer.output for layer in model.layers[:8]]  # model from load_model() above
# make sure the output list contains 8 sub-list, each sub-list is like [1, 150, 150, 32].
# the last is 32, not 3(RGB). Because the first layer Convnet2D's 32 tunnels

# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)  # make a instance of this class Model()

# This will return a list of 8 Numpy arrays, one array per layer activation
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]  # the first layer
print(first_layer_activation.shape)  # (1, 148, 148, 32)

# show the 3rd tunnel's picture that is a diagonal edge detector
plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')  # the first element is 0 for there is only one output
# matshow: Display an array as a matrix in a new figure window. cmap == colour, viridis == green
# matplotlib: https://www.meiwen.com.cn/subject/tiilsqtx.html

plt.show()

print('show the third tunnel"s picture done')
print('---------------------------------------------------------------------------------------------------------------')

# To show all channels' picture of 8 layers

# These are the names of the layers, so can have them as part of our plot
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

# Now let's display our feature maps
for layer_name, layer_activation in zip(layer_names, activations):
    # This is the number of features in the feature map
    n_features = layer_activation.shape[-1]
    print('n_features:', n_features)  # 32, 32, 64, 64, 128, 128, 128, 128

    # The feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]

    # We will tile the activation channels in this matrix
    n_cols = n_features // images_per_row  # the first layer: n_cols = 32 / 16 = 2
    display_grid = np.zeros((size * n_cols, images_per_row * size))  # ( 150*2, 16*150 )

    # We'll tile each filter into this big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                            :, :,
                            col * images_per_row + row]  # the last element: from 0 to 1*16+15

            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')  # clip(input, min, max) set limits to array
            display_grid[col * size: (col + 1) * size,
            row * size: (row + 1) * size] = channel_image

    # Display the grid
    scale = 1. / size  # down size of pictures
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)  # False: no grid line in tables
    plt.imshow(display_grid, aspect='auto', cmap='viridis')  # aspect == direction

plt.show()

print('show all channels: picture of 8 layers done')
print('---------------------------------------------------------------------------------------------------------------')
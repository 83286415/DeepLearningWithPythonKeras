import keras
print(keras.__version__)  # 2.2.4

from keras.applications import VGG16  # import the network model trained
from keras import backend as K  # backend: tensorflow. refer to https://keras.io/zh/backend/#keras


K.clear_session()

# show heat map of the class activation


# load model
model = VGG16(weights='imagenet')  # Note: include the Dense layer. Downloading new model which size over 500M
              #include_top=False)  # no Dense layer included
print(model.summary())  # As a reminder.
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 224, 224, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808     # the last layer above Dense layer
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544   # not only convnet. Dense layers are included.
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312  
_________________________________________________________________
predictions (Dense)          (None, 1000)              4097000   
=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
_________________________________________________________________
'''

# pre-process the picture
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

# The local path to our target image
img_path = 'D:\AI\deep-learning-with-python-notebooks-master\cats_and_dogs_small\\temp\\dog4.jpg'

# `img` is a PIL image of size 224x224
img = image.load_img(img_path, target_size=(224, 224))

# `x` is a float32 Numpy array of shape (224, 224, 3)
x = image.img_to_array(img)

# We add a dimension to transform our array into a "batch"
# of size (1, 224, 224, 3)
x = np.expand_dims(x, axis=0)

# Finally we preprocess the batch
# (this does channel-wise color normalization)
x = preprocess_input(x)  # refer to preprocess_input's definition => mode

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])  # index: 259  Pomeranian  99.9%
# decode_predictions return a top list of tuples like (class_name, class_description, score)

print('class: ', np.argmax(preds[0]))  # the index of the class predicted: 259


print('5.4.3 pre-process the picture done')
print('---------------------------------------------------------------------------------------------------------------')


# Grad-CAM algorithm to show the heat map

# This is the "african elephant" entry in the prediction vector
pomeranian_output = model.output[:, 259]  # output: the output tensors of the layer
# Tensor("predictions/Softmax:0", shape=(?, 1000), dtype=float32)

# The is the output feature map of the `block5_conv3` layer,
# the last convolutional layer in VGG16
last_conv_layer = model.get_layer('block5_conv3')  # # the last layer above Dense layer

# This is the gradient of the "african elephant" class with regard to
# the output feature map of `block5_conv3`
grads = K.gradients(pomeranian_output, last_conv_layer.output)[0]
# Tensor("gradients/block5_pool/MaxPool_grad/MaxPoolGrad:0", shape=(?, 14, 14, 512), dtype=float32)

# This is a vector of shape (512,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `block5_conv3`,
# given a sample image
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

# These are the values of these two quantities, as Numpy arrays,
# given our sample image of two elephants
pooled_grads_value, conv_layer_output_value = iterate([x])

# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the elephant class
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
heatmap = np.mean(conv_layer_output_value, axis=-1)
print('heat map: ', heatmap)  # a array


# show the heam map picture
import matplotlib.pyplot as plt

# max, maximum: https://blog.csdn.net/CSDN5529/article/details/79038544
heatmap = np.maximum(heatmap, 0)  # maximum: return bigger one between maximum and 0 bit by bit
heatmap /= np.max(heatmap)  # max: return the biggest value in the array
print('heat map: ', heatmap)

plt.matshow(heatmap)
plt.show()


print('5.4.3 show the heat map picture done')
print('---------------------------------------------------------------------------------------------------------------')


# make the heat map and the original map into one picture

import cv2

# We use cv2 to load the original image
img = cv2.imread(img_path)

# We resize the heatmap to have the same size as the original image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# We convert the heatmap to RGB
heatmap = np.uint8(255 * heatmap)  # change heat map from [0, 1] to [0, 255]

# We apply the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # (140,121) -> (140,121,3) RGB

# 0.4 here is a heatmap intensity factor
superimposed_img = heatmap * 0.4 + img

# Save the image to disk
cv2.imwrite('D:\AI\deep-learning-with-python-notebooks-master\cats_and_dogs_small\\temp\\dog4_n2.jpg', superimposed_img)


print('5.4.3 done')
print('---------------------------------------------------------------------------------------------------------------')
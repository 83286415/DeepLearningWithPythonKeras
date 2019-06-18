import keras
print(keras.__version__)  # 2.2.4

from keras.models import load_model

model = load_model('cats_and_dogs_small_2.h5')  # model: data augmentation with dropout between Flatten and Dense layers
print(model.summary())  # As a reminder.
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_5 (Conv2D)            (None, 148, 148, 32)      896       
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

# Remember that the model was trained on inputs that were preprocessed in the following way:
img_tensor /= 255.  # rescale all elements in the tensor: new elements value = old value / 255


# display this picture
import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])  # TODO: why [0]
plt.show()
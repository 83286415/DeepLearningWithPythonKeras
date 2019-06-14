import keras
print(keras.__version__)  # 2.2.4

from keras.applications import VGG16

# pre-trained network
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

print(conv_base.summary())
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 150, 150, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 150, 150, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 150, 150, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 75, 75, 64)        0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 75, 75, 128)       73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 75, 75, 128)       147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 37, 37, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 37, 37, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 37, 37, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 37, 37, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 18, 18, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 18, 18, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 18, 18, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 18, 18, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 9, 9, 512)         0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 9, 9, 512)         2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0         
=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
_________________________________________________________________
'''


# 5.3.1.1 feature extraction without data augmentation

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

base_dir = 'D:/AI/deep-learning-with-python-notebooks-master/cats_and_dogs_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
# All images will be rescaled by 1./255 (the max pixel value is 255. here its max is 1)
batch_size = 20


def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))  # make zero matrix for features extracted below
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,  # 20, defined outside this def
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:  # loops in images dir and never stops, so need a break below
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch  # map the feature's map to features matrix
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:  # the sample images count in dirs
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels


train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

# Dense layer needs 3D -> 1D, so reshape feature matrix to (sample_count, 8129), and this 8129=4*4*512
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))  # 4, 4, 512: the last MaxPooling2D layer output
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))


# build network with pre-trained network
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))


# plot
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

# plt.show()  # commented it out for running code below

print('5.3.1.1 done')
print('---------------------------------------------------------------------------------------------------------------')


# 5.3.1.2 feature extraction with data augmentation

# if no GPU supports, do not try this way


model = models.Sequential()
model.add(conv_base)  # add pre-trained network model into this model
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())
'''
Layer (type)                 Output Shape              Param #   
=================================================================
vgg16 (Model)                (None, 4, 4, 512)         14714688  
_________________________________________________________________
flatten_1 (Flatten)          (None, 8192)              0         
_________________________________________________________________
dense_3 (Dense)              (None, 256)               2097408   
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 257       
=================================================================
Total params: 16,812,353
Trainable params: 16,812,353
Non-trainable params: 0
_________________________________________________________________
'''


# freezing con_base's weights
print('This is the number of trainable weights '
      'before freezing the conv base:', len(model.trainable_weights))  # output: 30
conv_base.trainable = False  # then compile the model make this modification work
print('This is the number of trainable weights '
      'after freezing the conv base:', len(model.trainable_weights))  # output: 4


# make train data generator (only) with data augmentation
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)

model.save('cats_and_dogs_small_3_feature_extraction_with_data_augmentation.h5')

# plot
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy_2')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss_2')
plt.legend()

plt.show()
'''it costs 8 hours to run this model fit
Epoch 1/30
 - 814s - loss: 0.5846 - acc: 0.7055 - val_loss: 0.4523 - val_acc: 0.8140
Epoch 2/30
 - 844s - loss: 0.4821 - acc: 0.7785 - val_loss: 0.3697 - val_acc: 0.8550
Epoch 3/30
 - 882s - loss: 0.4210 - acc: 0.8185 - val_loss: 0.3524 - val_acc: 0.8580
Epoch 4/30
 - 914s - loss: 0.3949 - acc: 0.8325 - val_loss: 0.3277 - val_acc: 0.8530
Epoch 5/30
 - 878s - loss: 0.3815 - acc: 0.8355 - val_loss: 0.2894 - val_acc: 0.8850
Epoch 6/30
 - 865s - loss: 0.3618 - acc: 0.8390 - val_loss: 0.2870 - val_acc: 0.8900
Epoch 7/30
 - 863s - loss: 0.3670 - acc: 0.8340 - val_loss: 0.2894 - val_acc: 0.8760
Epoch 8/30
 - 937s - loss: 0.3493 - acc: 0.8455 - val_loss: 0.2716 - val_acc: 0.8980
Epoch 9/30
 - 879s - loss: 0.3399 - acc: 0.8515 - val_loss: 0.2646 - val_acc: 0.8980
Epoch 10/30
 - 895s - loss: 0.3327 - acc: 0.8545 - val_loss: 0.2598 - val_acc: 0.8960
Epoch 11/30
 - 854s - loss: 0.3266 - acc: 0.8540 - val_loss: 0.2555 - val_acc: 0.9020
Epoch 12/30
 - 856s - loss: 0.3341 - acc: 0.8510 - val_loss: 0.2535 - val_acc: 0.8980
Epoch 13/30
 - 854s - loss: 0.3189 - acc: 0.8600 - val_loss: 0.2554 - val_acc: 0.8970
Epoch 14/30
 - 849s - loss: 0.3167 - acc: 0.8620 - val_loss: 0.2493 - val_acc: 0.9020
Epoch 15/30
 - 839s - loss: 0.3256 - acc: 0.8490 - val_loss: 0.2465 - val_acc: 0.9010
Epoch 16/30
 - 840s - loss: 0.3118 - acc: 0.8645 - val_loss: 0.2456 - val_acc: 0.9040
Epoch 17/30
 - 841s - loss: 0.3114 - acc: 0.8635 - val_loss: 0.2464 - val_acc: 0.9030
Epoch 18/30
 - 841s - loss: 0.3045 - acc: 0.8690 - val_loss: 0.2478 - val_acc: 0.9010
Epoch 19/30
 - 845s - loss: 0.3112 - acc: 0.8605 - val_loss: 0.2468 - val_acc: 0.8990
Epoch 20/30
 - 843s - loss: 0.2951 - acc: 0.8735 - val_loss: 0.2412 - val_acc: 0.9030
Epoch 21/30
 - 840s - loss: 0.3061 - acc: 0.8715 - val_loss: 0.2403 - val_acc: 0.9060
Epoch 22/30
 - 866s - loss: 0.2909 - acc: 0.8750 - val_loss: 0.2405 - val_acc: 0.9050
Epoch 23/30
 - 912s - loss: 0.2976 - acc: 0.8690 - val_loss: 0.2448 - val_acc: 0.9030
Epoch 24/30
 - 838s - loss: 0.3001 - acc: 0.8730 - val_loss: 0.2451 - val_acc: 0.8990
Epoch 25/30
 - 833s - loss: 0.2876 - acc: 0.8720 - val_loss: 0.2382 - val_acc: 0.9030
Epoch 26/30
 - 845s - loss: 0.2913 - acc: 0.8775 - val_loss: 0.2415 - val_acc: 0.9020
Epoch 27/30
 - 915s - loss: 0.2986 - acc: 0.8725 - val_loss: 0.2370 - val_acc: 0.9060
Epoch 28/30
 - 968s - loss: 0.2830 - acc: 0.8735 - val_loss: 0.2423 - val_acc: 0.9000
Epoch 29/30
 - 1004s - loss: 0.2853 - acc: 0.8840 - val_loss: 0.2483 - val_acc: 0.9020
Epoch 30/30
 - 1046s - loss: 0.2865 - acc: 0.8770 - val_loss: 0.2416 - val_acc: 0.9030
 '''

print('5.3.1.2 done')
print('---------------------------------------------------------------------------------------------------------------')
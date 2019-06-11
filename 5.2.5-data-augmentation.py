import keras
print(keras.__version__)  # 2.2.4
import os

# 5.2.5 data augmentation

# The path to the directory where the original
# dataset was uncompressed
original_dataset_dir = 'D:/AI/deep-learning-with-python-notebooks-master/kaggle_original_data/train'

# The directory where we will
# store our smaller dataset
base_dir = 'D:/AI/deep-learning-with-python-notebooks-master/cats_and_dogs_small'

# make train dir
train_dir = os.path.join(base_dir, 'train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)

from keras.preprocessing.image import ImageDataGenerator  # used for image processing

# data augmentation instance
datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# This is module with image preprocessing utilities
from keras.preprocessing import image
import matplotlib.pyplot as plt


# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
if not os.path.exists(train_cats_dir):
    os.mkdir(train_cats_dir)

fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

# We pick one image to "augment"
img_path = fnames[5]

# Read the image and resize it
img = image.load_img(img_path, target_size=(150, 150))

# Convert it to a Numpy array with shape (150, 150, 3)
x = image.img_to_array(img)

# Reshape it to (1, 150, 150, 3)
x = x.reshape((1,) + x.shape)

# The .flow() command below generates batches of randomly transformed images.
# It will loop indefinitely, so we need to `break` the loop at some point!
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 5 == 0:
        break

plt.show()
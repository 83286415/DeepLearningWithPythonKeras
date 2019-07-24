from keras import layers
import numpy as np

# Inception:

x = np.random.randint(1, 10000, size=(1000, 1000, 1000, 1000))  # x is a 4D tensor

# the following branches a, b, c, d: refer to the graph on P205
# make sure all branches have the same strides (steps)
branch_a = layers.Conv2D(128, 1, activation='relu', strides=2)(x)  # strides: step

branch_b = layers.Conv2D(128, 1, activation='relu')(x)  # 1: 1x1 convnet P205
branch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_b)

branch_c = layers.AveragePooling2D(3, strides=2)(x)
branch_c = layers.Conv2D(128, 3, activation='relu')(branch_c)

branch_d = layers.Conv2D(128, 1, activation='relu')(x)
branch_d = layers.Conv2D(128, 3, activation='relu')(branch_d)
branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)

# connect four branches
output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)

print('Inception done')
print('---------------------------------------------------------------------------------------------------------------')
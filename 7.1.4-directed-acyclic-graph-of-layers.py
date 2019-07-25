from keras import layers
import numpy as np

# Inception: parallel branches are connected

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


# Residual connection: the output of some layer is added to the other's activation

# 1: identity residual connection: the feature map size of some upper level layer is the same as later's
z = np.random.randint(1, 10000, size=(1000, 1000, 1000, 1000))  # z is a 4D tensor
y = layers.Conv2D(128, 3, activation='relu', padding='same')(z)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)  # y is the same size of z

y = layers.add([y, z])  # add z to y


# 2: linear residual connection: the feature map size is not the same
i = np.random.randint(1, 10000, size=(1000, 1000, 1000, 1000))  # i is a 4D tensor
j = layers.Conv2D(128, 3, activation='relu', padding='same')(i)
j = layers.Conv2D(128, 3, activation='relu', padding='same')(j)  # the size of j's feature map is not changed yet
j = layers.MaxPooling2D(2, strides=2)(j)                         # size changed into half

# linear transformation: change i's feature map size into half by strides 2
residual = layers.Conv2D(128, 1, strides=2, padding='same')(i)  # 1x1 convnet: change i's size into half linearly

j = layers.add([j, residual])  # add i's linear transformation to j

print('Residual connection done')
print('---------------------------------------------------------------------------------------------------------------')
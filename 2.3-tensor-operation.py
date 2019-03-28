__author__ = 'Dan'

import numpy as np

# broadcast
x = np.random.random((64, 3, 32, 10))
y = np.random.random((32, 10))

z = np.maximum(x, y)  # max is a element-wise operation. So it broadcasts y to x's shape before max()
print(z.ndim)  # output: 4


# tensor reshaping
x = np.zeros((300, 20))
x = np.transpose(x)  # transpose makes x(300, 20) -> x(20, 300). It's a special tensor reshaping.
print(x.shape)  # output: (20, 300)
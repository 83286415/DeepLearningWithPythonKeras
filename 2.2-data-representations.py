__author__ = 'Dan'

import numpy as np

x = np.array(12)  # 12 is a int
print(x, x.ndim)  # ndim: number of dimensions  output: 12  0

y = np.array([[[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]],
             [[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]],
             [[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]]])
print(y.ndim)  # output: 3
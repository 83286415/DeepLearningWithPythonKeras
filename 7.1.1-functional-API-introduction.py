from keras.layers import Input, Dense
from keras.models import Model


# 7.1.1 Keras functional API introduction

inputs = Input(shape=(784,))  # Input() return a tensor

x = Dense(64, activation='relu')(inputs)  # layer Dense(output channels, activation)(input para)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
# class Model(input, output) change the input and output tensor into a network model instance, like using Sequential.

print(model.summary())
'''______________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 784)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 64)                50240     
_________________________________________________________________
dense_2 (Dense)              (None, 64)                4160      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                650       
=================================================================
Total params: 55,050
Trainable params: 55,050
Non-trainable params: 0
_________________________________________________________________
'''

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# model.fit(data, labels)  # fit training. No training data, label or validation data set available so commented out.



# un-related input and output tensor occurs a run time error: P200
unrelated_input = Input(shape=(32,))
bad_model = Model(inputs=unrelated_input, outputs=predictions)  # input and output are not related
# run time error:
'''Traceback (most recent call last):
  File "D:/AI/deep-learning-with-python-notebooks-master/7.1.1-functional-API-introduction.py", line 43, in <module>
    bad_model = Model(inputs=unrelated_input, outputs=predictions)
  File "C:\\Users\zhang.d\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\legacy\interfaces.py", line 91, in wrapper
    return func(*args, **kwargs)
  File "C:\\Users\zhang.d\AppData\Local\Programs\Python\Python36\lib\site-packages\keras\engine\topology.py", line 1811, in __init__
    str(layers_with_complete_input))
RuntimeError: Graph disconnected: cannot obtain value for tensor Tensor("input_1:0", shape=(?, 784), dtype=float32) at layer "input_1". The following previous layers were accessed without issue: []
'''
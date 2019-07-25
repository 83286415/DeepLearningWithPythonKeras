# do not run this code
from keras.models import Model
from keras import layers
from keras import Input
# from keras.utils import plot_model  # to plot the model structure


# build network model with multi-output


vocabulary_size = 50000  # Embedding layer output dim
num_income_groups = 10  # Dense layer output channel count
posts = None  # train data input
age_targets, income_targets, gender_targets = None, None, None  # train labels

# Input and Embedding layer
posts_input = Input(shape=(None,), dtype='int32', name='posts')  # name: unique name of this input layer
embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)  # 256: input dim      50000: output dim

# build 1D convnet and max pooling network model: faster than LSTM and GRU
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)  # 2 convnet with 1 max pooling layer
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)              # global max pooling layer + Dense classification
x = layers.Dense(128, activation='relu')(x)

# multi-output: multi Dense classification. Each layer has its own unique name!
age_prediction = layers.Dense(1, name='age')(x)  # regression problem, no activation but only 1 linear output (-∞, ∞)
income_prediction = layers.Dense(num_income_groups, activation='softmax', name='income')(x)  # 10: multi-classification
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)  # binary classification

# model instance: multi-output in a list
model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])
# plot_model(model,show_shapes=True,to_file='model.png')

# Two compile styles as below: 2 is better.

# 1: multi-loss functions in a list whose order is the order output layers defined above.
model.compile(optimizer='rmsprop', loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'])

# 2: multi-loss function in a dict and the keys are defined in output layers above
model.compile(optimizer='rmsprop', loss={'age': 'mse', 'income': 'categorical_crossentropy',
                                         'gender': 'binary_crossentropy'})

# 1: compile function with loss weight list. the order is the same as the order in loss function list
model.compile(optimizer='rmsprop', loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],
              loss_weights=[0.25, 1., 10.])  # loss weight values refer to cloud note 7.1.3 and book P204

# 2: compile function with loss weight dict. The keys are defined in output layers above
model.compile(optimizer='rmsprop', loss={'age': 'mse', 'income': 'categorical_crossentropy',
                                         'gender': 'binary_crossentropy'},
              loss_weights={'age': 0.25, 'income': 1., 'gender': 10.})


# Two fit styles as below:

# 1: x labels in a list: order is as the order in compile function's loss list
model.fit(posts, [age_targets, income_targets, gender_targets], epochs=10, batch_size=64)

# 2: x labels in a dict: keys are the same as the names defined in output layers
model.fit(posts, {'age': age_targets, 'income': income_targets, 'gender': gender_targets}, epochs=10, batch_size=64)



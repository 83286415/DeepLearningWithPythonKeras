import keras
print(keras.__version__)  # 2.2.4

# 3.6.1 boston housing data set
from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()
print(train_data.shape)  # (404, 13)  404 rows 13 columns. totally 404 + 102 pieces housing data
print(test_data.shape)  # (102, 13)
'''
13 columns: 13 features as below:
Per capita crime rate.
Proportion of residential land zoned for lots over 25,000 square feet.
Proportion of non-retail business acres per town.
Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
Nitric oxides concentration (parts per 10 million).
Average number of rooms per dwelling.
Proportion of owner-occupied units built prior to 1940.
Weighted distances to five Boston employment centres.
Index of accessibility to radial highways.
Full-value property-tax rate per $10,000.
Pupil-teacher ratio by town.
1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
% lower status of the population.
'''
print(train_targets[0])  # 15.2 the median values of housing is the target

print('3.6.1 done')
print('---------------------------------------------------------------------------------------------------------------')


# 3.6.2 prepare data
# standardization refer to cloud note
mean = train_data.mean(axis=0)  # mean value
train_data -= mean  # train_data = train_data - mean
std = train_data.std(axis=0)  # standard error
train_data /= std

test_data -= mean  # mean and std value come from train_data not test_data
test_data /= std
print(test_data[:5])
print(train_data[:5])

print('3.6.2 done')
print('---------------------------------------------------------------------------------------------------------------')


# 3.6.3 network model
from keras import models
from keras import layers

def build_model():
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))  # input_shape=(13, )
    model.add(layers.Dense(64, activation='relu'))
    # only two hidden layers added in case of over-fit for only hundreds of train data

    model.add(layers.Dense(1))  # only one output dimension: the predicted price
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    # mse: mean squared error. refer to my cloud note 3.6 data prepare
    # mae: mean absolute error. refer to my cloud note
    # unit: thousand dollar
    return model

print('3.6.3 done')
print('---------------------------------------------------------------------------------------------------------------')


# 3.6.4 validation using K-fold validation
import numpy as np

k = 4  # total 404, so k=4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):  # range(start, stop, step) default start: 0; end at stop-step;
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)  # along x axis(row) join train data arrays(not validation above) to one array sequence
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # Build the Keras model (already compiled)
    model = build_model()
    # Train the model (in silent mode, verbose=0)
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=0)  # verbose: refer to fit's definition
    # batch_size: https://www.cnblogs.com/gengyi/p/9853664.html

    # Evaluate the model on the validation data
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
print(all_scores)  # [2.0603285496777826, 2.218196408583386, 2.874764798891426, 2.32039079630729]
print(np.mean(all_scores))  # 2.3684201383649715 that means 2368 dollar error. This error is too significant.


# a 500 epochs training to make the mean mae value smaller
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # Build the Keras model (already compiled)
    model = build_model()
    # Train the model (in silent mode, verbose=0)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']  # why: refer to my cloud note 3.6 K-fold
    # dict_keys(['val_loss', 'val_mean_absolute_error', 'loss', 'mean_absolute_error'])
    # mae_history is a list of 500 epochs mae values!

    all_mae_histories.append(mae_history)  # all_mae_histories: [[500 epochs mae values], [], [], []]

average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
# the everage mae value of all validation data in 500 epochs
print(average_mae_history)  # len: 500


# draw a picture
import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
plt.clf()   # clear figure


# Omit the first 10 data points, which are on a different scale from the rest of the curve.
# Replace each point with an exponential moving average of the previous points, to obtain a smooth curve.
def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))  # EMA value refer to my cloud note
    else:
      smoothed_points.append(point)
  return smoothed_points


smooth_mae_history = smooth_curve(average_mae_history[10:])  # Omit the first 10 data points

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()  # over-fit at the 45th epochs


# build a new model according to the picture shown above
model = build_model()
# Train it on the entirety of the data.
model.fit(train_data, train_targets,
          epochs=45, batch_size=16, verbose=0)  # 45 epochs
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mae_score)  # 2.877347833970014 that is 2877 dollar... not good at all...

print('3.6.4 done')
print('---------------------------------------------------------------------------------------------------------------')
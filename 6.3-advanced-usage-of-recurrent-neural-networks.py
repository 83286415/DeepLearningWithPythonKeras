import keras
print(keras.__version__)  # 2.1.6
import os


# 6.3.1 prepare the climate data
base_dir = 'D:/AI/deep-learning-with-python-notebooks-master'
climate_dir = os.path.join(base_dir, 'jena_climate')
fname = os.path.join(climate_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]  # title is not included

print(header)  # ['"Date Time"', '"p (mbar)"', '"T (degC)"', '"Tpot (K)"', '"Tdew (degC)"', '"rh (%)"', '"VPmax (mbar)"'
# , '"VPact (mbar)"', '"VPdef (mbar)"', '"sh (g/kg)"', '"H2OC (mmol/mol)"', '"rho (g/m**3)"', '"wv (m/s)"',
# '"max. wv (m/s)"', '"wd (deg)"']

print(len(lines))  # 420551: there are 420551 rows of information in this csv file totally


# data analysis
import numpy as np

float_data = np.zeros((len(lines), len(header) - 1))  # len(header)-1: data time is not included
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values  # put values of climate data into float_date list without data time


# plot the temperature time sequence
from matplotlib import pyplot as plt

temp = float_data[:, 1]  # temperature (in degrees Celsius)
plt.plot(range(len(temp)), temp)  # plot(x range, y value)
plt.title('7 years temperature data')
# plt.legend()  # add picture explanation on the top right corner

plt.figure()  # create a new figure to plot as below
plt.plot(range(1440), temp[:1440])  # the first 10 days' temperature
plt.title('first 10 days temperature data')

plt.show()  # show the two figures plotted above

print('prepare the climate data done')
print('---------------------------------------------------------------------------------------------------------------')


# 6.3.2 data pre-precessing

# standardization: the first 200000 rows data
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std


# generator(data=all data, lookback=the count of lookback's time steps data must included in return samples array,
# min_index=low limit of data rows range,
# batch_size=rows total count in each loop(it's a yield generator), step=rows interval)
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1  # max limit < delay's beginning if max is None
    i = min_index + lookback  # the beginning index of the data picked by steps
    while 1:  # each loop in this yield generator
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
            # non-sense: it means (low, low+lookback, size) also needs to modify the indices base on this
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback  # reset i to the original value outside this loop
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))  # (samples cout: 128 or less, 1440/6, total columns count each row)
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            # indices: range(200001, 201441, 6)
            # indices: range(200002, 201442, 6)
            # indices: range(200003, 201443, 6)
            samples[j] = data[indices]  # all data in lookback time steps
            targets[j] = data[rows[j] + delay][1]  # the temperature after delay time steps
        yield samples, targets


lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

# This is how many steps to draw from `val_gen`
# in order to see the whole validation set:
val_steps = (300000 - 200001 - lookback) // batch_size

# This is how many steps to draw from `test_gen`
# in order to see the whole test set:
test_steps = (len(float_data) - 300001 - lookback) // batch_size


print('data pre-precessing done')
print('---------------------------------------------------------------------------------------------------------------')


def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))


evaluate_naive_method()
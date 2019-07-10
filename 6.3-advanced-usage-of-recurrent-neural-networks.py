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

#  CSV file read
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


# plot the temperature time sequence P174
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


# def of my own
def generator1(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1  # max limit < delay's beginning if max is None
    i = min_index  # the beginning index of the data picked by steps
    # i = min_index + lookback  # delete the lookback from i to make this def more simple
    while 1:  # each loop in this yield generator
        if shuffle:
            rows = np.random.randint(
                # min_index + lookback, max_index, size=batch_size)  # delete the lookback
                min_index, max_index, size = batch_size)
            # non-sense: it means (low, low+lookback, size) also needs to modify the indices base on this
        else:
            if i + batch_size >= max_index:
                i = min_index  # reset i to the original value outside this loop (delete the lookback)
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))  # (samples cout: 128 or less, 1440/6, total columns count each row)
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            # indices = range(rows[j] - lookback, rows[j], step)  # delete the lookback and the indices results are same
            indices = range(rows[j], rows[j] + lookback, step)
            # print('indices: ', indices)
            # indices: range(200001, 201441, 6)
            # indices: range(200002, 201442, 6)
            # indices: range(200003, 201443, 6)

            # indices: range(200001, 201441, 6)
            # target
            # j: 0.4385937817045248
            # indices: range(200002, 201442, 6)
            # target
            # j: 0.42729753653450014
            # indices: range(200003, 201443, 6)
            # target
            # j: 0.4340752836365149

            # indices: range(298430, 299870, 6)
            # target
            # j: 0.27931672480717973
            # indices: range(298431, 299871, 6)
            # target
            # j: 0.28609447190919424
            # indices: range(298432, 299872, 6)
            # target
            # j: 0.2917425944942066

            samples[j] = data[indices]  # all lookback time period's data
            targets[j] = data[rows[j] + lookback + delay][1]  # the temperature after 10 days and delay time steps later
            # 10 days data(lookback) is used to calculate the mae which will compare with the temperature after delay

            # print('target j: ', targets[j])
        # print('rows:', rows)
        # with lookback added in data index to target j, will get a lower mae. With lookback added, rows:
        # rows: [201441 201442 201443 201444 201445 201446 201447 201448 201449 201450
        #        201451 201452 201453 201454 201455 201456 201457 201458 201459 201460
        #        201461 201462 201463 201464 201465 201466 201467 201468 201469 201470
        #        201471 201472 201473 201474 201475 201476 201477 201478 201479 201480
        #        201481 201482 201483 201484 201485 201486 201487 201488 201489 201490
        #        201491 201492 201493 201494 201495 201496 201497 201498 201499 201500
        #        201501 201502 201503 201504 201505 201506 201507 201508 201509 201510
        #        201511 201512 201513 201514 201515 201516 201517 201518 201519 201520
        #        201521 201522 201523 201524 201525 201526 201527 201528 201529 201530
        #        201531 201532 201533 201534 201535 201536 201537 201538 201539 201540
        #        201541 201542 201543 201544 201545 201546 201547 201548 201549 201550
        #        201551 201552 201553 201554 201555 201556 201557 201558 201559 201560
        #        201561 201562 201563 201564 201565 201566 201567 201568]

        # without lookback in rows:
        # rows: [298305 298306 298307 298308 298309 298310 298311 298312 298313 298314
        #        298315 298316 298317 298318 298319 298320 298321 298322 298323 298324
        #        298325 298326 298327 298328 298329 298330 298331 298332 298333 298334
        #        298335 298336 298337 298338 298339 298340 298341 298342 298343 298344
        #        298345 298346 298347 298348 298349 298350 298351 298352 298353 298354
        #        298355 298356 298357 298358 298359 298360 298361 298362 298363 298364
        #        298365 298366 298367 298368 298369 298370 298371 298372 298373 298374
        #        298375 298376 298377 298378 298379 298380 298381 298382 298383 298384
        #        298385 298386 298387 298388 298389 298390 298391 298392 298393 298394
        #        298395 298396 298397 298398 298399 298400 298401 298402 298403 298404
        #        298405 298406 298407 298408 298409 298410 298411 298412 298413 298414
        #        298415 298416 298417 298418 298419 298420 298421 298422 298423 298424
        #        298425 298426 298427 298428 298429 298430 298431 298432]
        yield samples, targets


# def in book P176
# generator(data=all data, lookback=the count of lookback's time steps data must included in return samples array,
# min_index=low limit of data rows range,
# batch_size=rows total count in each loop(it's a yield generator), step=rows interval)
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            # print('indices: ', indices)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
            # print('target j: ', targets[j])
            # indices: range(200001, 201441, 6)
            # target
            # j: -0.722660221773991
            # indices: range(200002, 201442, 6)
            # target
            # j: -0.7181417237059811
            # indices: range(200003, 201443, 6)
            # target
            # j: -0.7170120991889787

            # indices: range(298430, 299870, 6)
            # target
            # j: 0.6159448308739088
            # indices: range(298431, 299871, 6)
            # target
            # j: 0.6102967082888967
            # indices: range(298432, 299872, 6)
            # target
            # j: 0.6091670837718942
        # print('rows:', rows)
        # rows: [299745 299746 299747 299748 299749 299750 299751 299752 299753 299754
        #        299755 299756 299757 299758 299759 299760 299761 299762 299763 299764
        #        299765 299766 299767 299768 299769 299770 299771 299772 299773 299774
        #        299775 299776 299777 299778 299779 299780 299781 299782 299783 299784
        #        299785 299786 299787 299788 299789 299790 299791 299792 299793 299794
        #        299795 299796 299797 299798 299799 299800 299801 299802 299803 299804
        #        299805 299806 299807 299808 299809 299810 299811 299812 299813 299814
        #        299815 299816 299817 299818 299819 299820 299821 299822 299823 299824
        #        299825 299826 299827 299828 299829 299830 299831 299832 299833 299834
        #        299835 299836 299837 299838 299839 299840 299841 299842 299843 299844
        #        299845 299846 299847 299848 299849 299850 299851 299852 299853 299854
        #        299855 299856 299857 299858 299859 299860 299861 299862 299863 299864
        #        299865 299866 299867 299868 299869 299870 299871 299872]
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


# evaluate the method
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))  # 0.2897359729905486


evaluate_naive_method()

print('data pre-precessing done')
print('---------------------------------------------------------------------------------------------------------------')


# build a simple Dense network model

# a basic dense network
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))  # no activation in this last Dense layer. It's normal in regression problem. Refer to 3.6py

model.compile(optimizer=RMSprop(), loss='mae')  # not crossentropy but mae as loss. see 3.6-regression.py
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)
# loss: 0.2004 - val_loss: 0.3380


# plot
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


print('build a simple Dense network model done')
print('---------------------------------------------------------------------------------------------------------------')
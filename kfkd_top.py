# -*- encoding: utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, EarlyStopping

start = 0.01
stop = 0.001
nb_epoch = 300

train_data = np.load(open('bottleneck_features_train.npy'))
train_labels = np.load(open('label_train.npy'))

validation_data = np.load(open('bottleneck_features_validation.npy'))
validation_labels = np.load(open('label_validation.npy'))

model_top = Sequential()
model_top.add(Flatten(input_shape=train_data.shape[1:]))
model_top.add(Dense(1000))
model_top.add(Activation('relu'))
model_top.add(Dropout(0.5))
model_top.add(Dense(1000))
model_top.add(Activation('relu'))
model_top.add(Dense(30))

sgd = SGD(lr=start, momentum=0.9, nesterov=True)
model_top.compile(loss='mean_squared_error', optimizer=sgd)

early_stop = EarlyStopping(patience=100)
learning_rates = np.linspace(start, stop, nb_epoch)
change_lr = LearningRateScheduler(lambda epoch: float(learning_rates[epoch]))

hist = model_top.fit(train_data, train_labels,
         nb_epoch=nb_epoch,
         validation_data=(validation_data, validation_labels),
         callbacks=[change_lr, early_stop])

model_top.save_weights("model_top.h5")
np.savetxt("model_top_loss.csv", hist.history['loss'])
np.savetxt("model_top_val_loss.csv", hist.history['val_loss'])
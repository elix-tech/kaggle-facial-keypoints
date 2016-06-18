# -*- encoding: utf-8 -*-
import os
import h5py
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from sklearn.utils import shuffle

# Download from https://www.kaggle.com/c/facial-keypoints-detection/data
FTRAIN = 'data/training.csv'
FTEST = 'data/test.csv'

# Download from https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
weight_path = '../examples/vgg16_weights.h5'
img_width = 96
img_height = 96

def load(test=False, cols=None):

    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))

    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:
        df = df[list(cols) + ['Image']]

    print(df.count())
    df = df.dropna()

    X = np.vstack(df['Image'].values) / 255.
    X = X.astype(np.float32)

    if not test:
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48
        X, y = shuffle(X, y, random_state=42)
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

def load2d(test=False, cols=None):
    X, y = load(test, cols)
    X = X.reshape(-1, 1, 96, 96)
    return X, y

def flip_image(X, y):
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]

    X_flipped = np.array(X[:, :, :, ::-1])
    y_flipped = np.array(y)
    y_flipped[:, ::2] = y_flipped[:, ::2] * -1

    for i in range(len(y)):
        for a, b in flip_indices:
            y_flipped[i, a], y_flipped[i, b] = (y_flipped[i, b], y_flipped[i, a])
    return X_flipped, y_flipped

def gray_to_rgb(X):
    X_transpose = np.array(X.transpose(0, 2, 3, 1))
    ret = np.empty((X.shape[0], img_width, img_height, 3), dtype=np.float32)
    ret[:, :, :, 0] = X_transpose[:, :, :, 0]
    ret[:, :, :, 1] = X_transpose[:, :, :, 0]
    ret[:, :, :, 2] = X_transpose[:, :, :, 0]
    return ret.transpose(0, 3, 1, 2)

def save_bottleneck_features():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))


    assert os.path.exists(weight_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weight_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            break

        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    X, y = load2d()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_flipped, y_flipped = flip_image(X_train, y_train)

    X_train = np.vstack((X_train, X_flipped))
    y_train = np.vstack((y_train, y_flipped))
    X_train = gray_to_rgb(X_train)
    X_val = gray_to_rgb(X_val)

    bottleneck_features_train = model.predict(X_train)
    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
    np.save(open('label_train.npy', 'w'), y_train)

    bottleneck_features_validation = model.predict(X_val)
    np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
    np.save(open('label_validation.npy', 'w'), y_val)


def train_top_model():
    start = 0.03
    stop = 0.001
    nb_epoch = 300

    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.load(open('label_train.npy'))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.load(open('label_validation.npy'))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(30))

    sgd = SGD(lr=start, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    learning_rates = np.linspace(start, stop, nb_epoch)
    change_lr = LearningRateScheduler(lambda epoch: float(learning_rates[epoch]))
    hist = model.fit(train_data, train_labels,
                     nb_epoch=nb_epoch,
                     validation_data=(validation_data, validation_labels),
                     callbacks=[change_lr])

    model.save_weights('model_top_vgg.h5')
    np.savetxt('model_top_vgg_flip_loss.csv', hist.history['loss'])
    np.savetxt('model_top_vgg_flip_val_loss.csv', hist.history['val_loss'])


save_bottleneck_features()
train_top_model()
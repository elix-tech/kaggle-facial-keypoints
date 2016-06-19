# -*- encoding: utf-8 -*-
import os
import h5py
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.cross_validation import train_test_split
from collections import OrderedDict
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras import backend as K
from sklearn.utils import shuffle

# Download from https://www.kaggle.com/c/facial-keypoints-detection/data
FTRAIN = 'data/training.csv'
FTEST = 'data/test.csv'

# Download from https://github.com/elix-tech/kaggle-facial-keypoints
weights_path = '../examples/model6_weights_5000.h5'
img_width = 96
img_height = 96

SPECIALIST_SETTINGS = [
    dict(
        columns=(
            'left_eye_center_x', 'left_eye_center_y',
            'right_eye_center_x', 'right_eye_center_y',
            ),
        flip_indices=((0, 2), (1, 3)),
        ),

    dict(
        columns=(
            'nose_tip_x', 'nose_tip_y',
            ),
        flip_indices=(),
        ),

    dict(
        columns=(
            'mouth_left_corner_x', 'mouth_left_corner_y',
            'mouth_right_corner_x', 'mouth_right_corner_y',
            'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
            ),
        flip_indices=((0, 2), (1, 3)),
        ),

    dict(
        columns=(
            'mouth_center_bottom_lip_x',
            'mouth_center_bottom_lip_y',
            ),
        flip_indices=(),
        ),

    dict(
        columns=(
            'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
            'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
            'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
            'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
            ),
        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
        ),

    dict(
        columns=(
            'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
            'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
            'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
            'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
            ),
        flip_indices=((0, 2), (1, 3), (4, 6), (5, 7)),
        ),
    ]

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

def save_bottleneck_features():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(1, img_width, img_height)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(64, 2, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, 2, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        if k >= len(model.layers):
            break
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        if len(weight_names):
            weight_values = [g[weight_name] for weight_name in weight_names]
            layer = model.layers[k]
            symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
            if len(weight_values) != len(symbolic_weights):
                raise Exception('Layer #' + str(k) +
                                ' (named "' + layer.name +
                                '" in the current model) was found to '
                                'correspond to layer ' + name +
                                ' in the save file. '
                                'However the new layer ' + layer.name +
                                ' expects ' + str(len(symbolic_weights)) +
                                ' weights, but the saved weights have ' +
                                str(len(weight_values)) +
                                ' elements.')
            weight_value_tuples += zip(symbolic_weights, weight_values)
    K.batch_set_value(weight_value_tuples)
    f.close()
    print('Model loaded.')

    X, y = load2d()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_flipped, y_flipped = flip_image(X_train, y_train)

    X_train = np.vstack((X_train, X_flipped))
    y_train = np.vstack((y_train, y_flipped))

    bottleneck_features_train = model.predict(X_train)
    np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
    np.save(open('label_train.npy', 'w'), y_train)

    bottleneck_features_validation = model.predict(X_val)
    np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
    np.save(open('label_validation.npy', 'w'), y_val)

def fit_specialists():
    specialists = OrderedDict()
    start = 0.01
    stop = 0.001
    nb_epoch = 300

    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.load(open('label_train.npy'))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.load(open('label_validation.npy'))

    df = read_csv(os.path.expanduser(FTRAIN))

    for setting in SPECIALIST_SETTINGS:

        cols = setting['columns']
        indices = [index for index, column in enumerate(df.columns) if column in cols]
        train_labels_specialist = train_labels[:, indices]
        validation_labels_specialist = validation_labels[:, indices]

        model_specialist = Sequential()
        model_specialist.add(Flatten(input_shape=train_data.shape[1:]))
        model_specialist.add(Dense(1000))
        model_specialist.add(Activation('relu'))
        model_specialist.add(Dropout(0.5))
        model_specialist.add(Dense(1000))
        model_specialist.add(Activation('relu'))
        model_specialist.add(Dense(len(cols)))

        sgd = SGD(lr=start, momentum=0.9, nesterov=True)
        model_specialist.compile(loss='mean_squared_error', optimizer=sgd)

        early_stop = EarlyStopping(patience=100)
        learning_rates = np.linspace(start, stop, nb_epoch)
        change_lr = LearningRateScheduler(lambda epoch: float(learning_rates[epoch]))

        print("Training model for columns {} for {} epochs".format(cols, nb_epoch))

        hist = model_specialist.fit(train_data, train_labels_specialist,
                 nb_epoch=nb_epoch,
                 validation_data=(validation_data, validation_labels_specialist),
                 callbacks=[change_lr, early_stop])

        model_specialist.save_weights("model_{}.h5".format(cols[0]))
        np.savetxt("model_{}_loss.csv".format(cols[0]), hist.history['loss'])
        np.savetxt("model_{}_val_loss.csv".format(cols[0]), hist.history['val_loss'])

        specialists[cols] = model_specialist

save_bottleneck_features()
fit_specialists()
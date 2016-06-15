# -*- encoding: utf-8 -*-
from __future__ import print_function
from scipy.misc import imsave
import numpy as np
import time
import os
import h5py

from keras.models import Sequential
from keras.layers import Convolution2D, Activation, MaxPooling2D, Dropout
from keras import backend as K

img_width = 96
img_height = 96
weights_path = '../examples/model6_weights_5000.h5'
layer_name = 'conv3'


model = Sequential()

model.add(Convolution2D(32, 3, 3, input_shape=(1, 96, 96), name='conv1'))
first_layer = model.layers[-1]
input_img = first_layer.input

model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Convolution2D(64, 2, 2, name='conv2'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Convolution2D(128, 2, 2, name='conv3'))
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

layer_dict = dict([(layer.name, layer) for layer in model.layers])

def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

kept_filters = []
for filter_index in range(0, 128):
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, filter_index, :, :])
    grads = K.gradients(loss, input_img)[0]
    grads = normalize(grads)
    iterate = K.function([input_img, K.learning_phase()], [loss, grads])

    step = 5.

    input_img_data = np.random.random((1, 1, img_width, img_height)) * 20 + 128.

    for i in range(200):
        loss_value, grads_value = iterate([input_img_data, 0])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            break

    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))


nb_img_x = 6
nb_img_y = 2

kept_filters.sort(key=lambda x: x[1], reverse=True)
kept_filters = kept_filters[:nb_img_x * nb_img_y]

margin = 5
width = nb_img_x * img_width + (nb_img_x - 1) * margin
height = nb_img_y * img_height + (nb_img_y - 1) * margin
stitched_filters = np.zeros((height, width, 3))

for i in range(nb_img_x):
    for j in range(nb_img_y):
        img, loss = kept_filters[j * nb_img_y + i]
        stitched_filters[(img_height + margin) * j: (img_height + margin) * j + img_height,
            (img_width + margin) * i: (img_width + margin) * i + img_width, :] = img

imsave('stitched_filters_%s_%dx%d.png' % (layer_name, nb_img_x, nb_img_y), stitched_filters)



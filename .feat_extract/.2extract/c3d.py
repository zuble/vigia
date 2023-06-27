import globo

from tensorflow import keras
import keras.backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D
from keras.utils.data_utils import get_file

import numpy as np
#from scipy.misc import imresize
import cv2


C3D_MEAN_PATH = 'https://github.com/adamcasson/c3d/releases/download/v0.1/c3d_mean.npy'


def preprocess_input(video):
    """Preprocess video input to make it suitable for feature extraction.

    The video is resized, cropped, resampled and training mean is substracted
    to make it suitable for the network

    :param video: Video to be processed
    :returns: Preprocessed video
    :rtype: np.ndarray

    """

    intervals = np.ceil(np.linspace(0, video.shape[0] - 1, 16)).astype(int)
    frames = video[intervals]

    # Reshape to 128x171
    reshape_frames = np.zeros((frames.shape[0], 128, 171, frames.shape[3]))
    for i, img in enumerate(frames):
        img = cv2.resize(img, (171 , 128) , interpolation=cv2.INTER_CUBIC)
        #img = imresize(img, (128, 171), 'bicubic')
        reshape_frames[i, :, :, :] = img

    mean_path = get_file('c3d_mean.npy',
                         C3D_MEAN_PATH,
                         cache_subdir='models',
                         md5_hash='08a07d9761e76097985124d9e8b2fe34')

    mean = np.load(mean_path)
    reshape_frames -= mean
    # Crop to 112x112
    reshape_frames = reshape_frames[:, 8:120, 30:142, :]
    # Add extra dimension for samples
    reshape_frames = np.expand_dims(reshape_frames, axis=0)

    return reshape_frames



def C3D(weights='sports1M'):
    """Creation of the full C3D architecture

    :param weights: Weights to be loaded into the network. If None,
    the network is randomly initialized.
    :returns: Network model
    :rtype: keras.model

    """

    if weights not in {'sports1M', None}:
        raise ValueError('weights should be either be sports1M or None')

    if K.image_data_format() == 'channels_last':
        shape = (16, 112, 112, 3)
    else:
        shape = (3, 16, 112, 112)

    model = Sequential()
    model.add(
        Conv3D(64,
               3,
               activation='relu',
               padding='same',
               name='conv1',
               input_shape=shape))
    model.add(
        MaxPooling3D(pool_size=(1, 2, 2),
                     strides=(1, 2, 2),
                     padding='same',
                     name='pool1'))

    model.add(Conv3D(128, 3, activation='relu', padding='same', name='conv2'))
    model.add(
        MaxPooling3D(pool_size=(2, 2, 2),
                     strides=(2, 2, 2),
                     padding='valid',
                     name='pool2'))

    model.add(Conv3D(256, 3, activation='relu', padding='same', name='conv3a'))
    model.add(Conv3D(256, 3, activation='relu', padding='same', name='conv3b'))
    model.add(
        MaxPooling3D(pool_size=(2, 2, 2),
                     strides=(2, 2, 2),
                     padding='valid',
                     name='pool3'))

    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv4a'))
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv4b'))
    model.add(
        MaxPooling3D(pool_size=(2, 2, 2),
                     strides=(2, 2, 2),
                     padding='valid',
                     name='pool4'))

    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv5a'))
    model.add(Conv3D(512, 3, activation='relu', padding='same', name='conv5b'))
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(
        MaxPooling3D(pool_size=(2, 2, 2),
                     strides=(2, 2, 2),
                     padding='valid',
                     name='pool5'))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(0.5))
    model.add(Dense(487, activation='softmax', name='fc8'))

    if weights == 'sports1M':
        model.load_weights(globo.C3DSPORTS1M)

    return model


def c3d_feature_extractor():
    """Creation of the feature extraction architecture. This network is
    formed by a subset of the original C3D architecture (from the
    beginning to fc6 layer)

    :returns: Feature extraction model
    :rtype: keras.model

    """
    model = C3D()
    layer_name = 'fc6'
    feature_extractor_model = Model(inputs=model.input,
                                    outputs=model.get_layer(layer_name).output)
    return feature_extractor_model

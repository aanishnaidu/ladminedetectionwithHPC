import os
# change TensorFlow logging verbosity
# 0: all logs
# 1: filter out INFO logs
# 2: filter out WARNINGS logs
# 3: filter out ERROR logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# use the GPU with the lowest memory usage
import GPUtil
DEVICE_ID = str(GPUtil.getFirstAvailable(order='memory')[0])
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID
print('GPU selected:', DEVICE_ID)


import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = tf.compat.v1.Session(config=config)
##############################################ADDITION####################################
#import tensorflow as tf
tf.device('/gpu:1')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
##################################################################################################



import keras.backend as K
#K.set_session(session)

from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Flatten, Dense, MaxPooling2D, Lambda
from keras.models import Model
from keras.optimizers import Adam, SGD


class Settings:
    def __init__(self):
        self.patience = 10
        self.epochs = 100
        self.lr_factor = 0.1
        self.batch_size = 128


def Auto1(patch_size, opt=Adam()):
    """
    Autoencoder with a hidden representation of 32 elements
    """
    img = Input(shape=patch_size, name='g_in_0')

    x = Conv2D(16, (6, 6), strides=1, padding='same', name='g_conv_0')(img)
    x = Conv2D(16, (5, 5), strides=2, padding='same', name='g_conv_1')(x)
    x = Conv2D(16, (4, 4), strides=2, padding='same', name='g_conv_2')(x)
    x = Conv2D(16, (3, 3), strides=2, padding='same', name='g_conv_3')(x)

    enc = Conv2D(8, (1, 1), strides=2, padding='same', name='encoder')(x)

    x = Conv2DTranspose(16, (2, 2), strides=2, padding='same')(enc)
    x = Conv2DTranspose(16, (3, 3), strides=2, padding='same', name='g_deconv_1')(x)
    x = Conv2DTranspose(16, (4, 4), strides=2, padding='same', name='g_deconv_2')(x)
    x = Conv2DTranspose(16, (5, 5), strides=2, padding='same', name='g_deconv_3')(x)

    dec = Conv2DTranspose(1, (6, 6), strides=1, padding='same',
                          activation='tanh', name='g_deconv_4')(x)

    autoencoder = Model(inputs=img, outputs=dec)
    encoder = Model(inputs=img, outputs=enc)

    autoencoder.compile(loss='mean_squared_error', optimizer=opt)

    return autoencoder, encoder



def Auto2(patch_size, opt=Adam()):
    """
    Autoencoder with a hidden representation of 16 elements
    """
    img = Input(shape=patch_size, name='g_in_0')

    x = Conv2D(16, (6, 6), strides=1, padding='same', name='g_conv_0')(img)
    x = Conv2D(16, (5, 5), strides=2, padding='same', name='g_conv_1')(x)
    x = Conv2D(16, (4, 4), strides=2, padding='same', name='g_conv_2')(x)
    x = Conv2D(16, (3, 3), strides=2, padding='same', name='g_conv_3')(x)
    x = Conv2D(16, (2, 2), strides=2, padding='same')(x)

    enc = Conv2D(16, (1, 1), strides=2, padding='same', name='encoder')(x)

    x = Conv2DTranspose(16, (2, 2), strides=2, padding='same')(enc)
    x = Conv2DTranspose(16, (2, 2), strides=2, padding='same', name='g_deconv_0')(x)
    x = Conv2DTranspose(16, (3, 3), strides=2, padding='same', name='g_deconv_1')(x)
    x = Conv2DTranspose(16, (4, 4), strides=2, padding='same', name='g_deconv_2')(x)
    x = Conv2DTranspose(16, (5, 5), strides=2, padding='same', name='g_deconv_3')(x)

    dec = Conv2DTranspose(1, (6, 6), strides=1, padding='same',
                          activation='tanh', name='g_deconv_4')(x)

    autoencoder = Model(inputs=img, outputs=dec)
    encoder = Model(inputs=img, outputs=enc)

    autoencoder.compile(loss='mean_squared_error', optimizer=opt)

    return autoencoder, encoder



def Auto3(patch_size, opt=Adam()):
    """
    Autoencoder with a hidden representation of 64 elements
    """
    img = Input(shape=patch_size, name='g_in_0')

    x = Conv2D(16, (6, 6), strides=1, padding='same', name='g_conv_0')(img)
    x = Conv2D(16, (5, 5), strides=2, padding='same', name='g_conv_1')(x)
    x = Conv2D(16, (4, 4), strides=2, padding='same', name='g_conv_2')(x)
    x = Conv2D(16, (3, 3), strides=2, padding='same', name='g_conv_3')(x)

    enc = Conv2D(16, (2, 2), strides=2, padding='same', name='encoder')(x)

    x = Conv2DTranspose(16, (2, 2), strides=2, padding='same', name='g_deconv_0')(enc)
    x = Conv2DTranspose(16, (3, 3), strides=2, padding='same', name='g_deconv_1')(x)
    x = Conv2DTranspose(16, (4, 4), strides=2, padding='same', name='g_deconv_2')(x)
    x = Conv2DTranspose(16, (5, 5), strides=2, padding='same', name='g_deconv_3')(x)

    dec = Conv2DTranspose(1, (6, 6), strides=1, padding='same',
                          activation='tanh', name='g_deconv_4')(x)

    autoencoder = Model(inputs=img, outputs=dec)
    encoder = Model(inputs=img, outputs=enc)

    autoencoder.compile(loss='mean_squared_error', optimizer=opt)

    return autoencoder, encoder







def Auto3D1(patch_size, opt=Adam()):
    """
    Autoencoder with a hidden representation of 32 elements
    """
    img = Input(shape=patch_size, name='g_in_0')

    x = Conv2D(16, (6, 6), strides=1, padding='same', name='g_conv_0')(img)
    x = Conv2D(16, (5, 5), strides=2, padding='same', name='g_conv_1')(x)
    x = Conv2D(16, (4, 4), strides=2, padding='same', name='g_conv_2')(x)
    x = Conv2D(16, (3, 3), strides=2, padding='same', name='g_conv_3')(x)

    enc = Conv2D(8, (1, 1), strides=2, padding='same', name='encoder')(x)

    x = Conv2DTranspose(16, (2, 2), strides=2, padding='same')(enc)
    x = Conv2DTranspose(16, (3, 3), strides=2, padding='same', name='g_deconv_1')(x)
    x = Conv2DTranspose(16, (4, 4), strides=2, padding='same', name='g_deconv_2')(x)
    x = Conv2DTranspose(16, (5, 5), strides=2, padding='same', name='g_deconv_3')(x)

    dec = Conv2DTranspose(3, (6, 6), strides=1, padding='same',
                          activation='tanh', name='g_deconv_4')(x)

    autoencoder = Model(inputs=img, outputs=dec)
    encoder = Model(inputs=img, outputs=enc)

    autoencoder.compile(loss='mean_squared_error', optimizer=opt)

    return autoencoder, encoder


def Auto3D2(patch_size, opt=Adam()):
    """
    Autoencoder with a hidden representation of 16 elements
    """
    img = Input(shape=patch_size, name='g_in_0')

    x = Conv2D(16, (6, 6), strides=1, padding='same', name='g_conv_0')(img)
    x = Conv2D(16, (5, 5), strides=2, padding='same', name='g_conv_1')(x)
    x = Conv2D(16, (4, 4), strides=2, padding='same', name='g_conv_2')(x)
    x = Conv2D(16, (3, 3), strides=2, padding='same', name='g_conv_3')(x)
    x = Conv2D(16, (2, 2), strides=2, padding='same')(x)

    enc = Conv2D(16, (1, 1), strides=2, padding='same', name='encoder')(x)

    x = Conv2DTranspose(16, (2, 2), strides=2, padding='same')(enc)
    x = Conv2DTranspose(16, (2, 2), strides=2, padding='same', name='g_deconv_0')(x)
    x = Conv2DTranspose(16, (3, 3), strides=2, padding='same', name='g_deconv_1')(x)
    x = Conv2DTranspose(16, (4, 4), strides=2, padding='same', name='g_deconv_2')(x)
    x = Conv2DTranspose(16, (5, 5), strides=2, padding='same', name='g_deconv_3')(x)

    dec = Conv2DTranspose(3, (6, 6), strides=1, padding='same',
                          activation='tanh', name='g_deconv_4')(x)

    autoencoder = Model(inputs=img, outputs=dec)
    encoder = Model(inputs=img, outputs=enc)

    autoencoder.compile(loss='mean_squared_error', optimizer=opt)

    return autoencoder, encoder



def Auto3D3(patch_size, opt=Adam()):
    """
    Autoencoder with a hidden representation of 64 elements
    """
    img = Input(shape=patch_size, name='g_in_0')

    x = Conv2D(16, (6, 6), strides=1, padding='same', name='g_conv_0')(img)
    x = Conv2D(16, (5, 5), strides=2, padding='same', name='g_conv_1')(x)
    x = Conv2D(16, (4, 4), strides=2, padding='same', name='g_conv_2')(x)
    x = Conv2D(16, (3, 3), strides=2, padding='same', name='g_conv_3')(x)

    enc = Conv2D(16, (2, 2), strides=2, padding='same', name='encoder')(x)

    x = Conv2DTranspose(16, (2, 2), strides=2, padding='same', name='g_deconv_0')(enc)
    x = Conv2DTranspose(16, (3, 3), strides=2, padding='same', name='g_deconv_1')(x)
    x = Conv2DTranspose(16, (4, 4), strides=2, padding='same', name='g_deconv_2')(x)
    x = Conv2DTranspose(16, (5, 5), strides=2, padding='same', name='g_deconv_3')(x)

    dec = Conv2DTranspose(3, (6, 6), strides=1, padding='same',
                          activation='tanh', name='g_deconv_4')(x)

    autoencoder = Model(inputs=img, outputs=dec)
    encoder = Model(inputs=img, outputs=enc)

    autoencoder.compile(loss='mean_squared_error', optimizer=opt)

    return autoencoder, encoder

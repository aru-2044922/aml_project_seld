#
# The SELDnet architecture
#

import numpy as np
from keras.layers import Bidirectional, Conv2D, MaxPooling2D, Input, Concatenate
from keras.layers.core import Dense, Activation, Dropout, Reshape
from keras.layers.recurrent import LSTM
from keras.layers import BatchNormalization
from keras.layers import Lambda, RepeatVector, Permute, multiply
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam
import keras
# keras.backend.set_image_data_format('channels_first')
keras.backend.set_image_data_format('channels_last')
from IPython import embed

# Disable eager mode
# tf.compat.v1.disable_eager_execution()


def get_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, f_pool_size, t_pool_size,
                                rnn_size, fnn_size, weights):
    # model definition
    # Passed Input shape -> 16, 7, 128, 64

    spec_start = Input(shape=(data_in[-2], data_in[-1], data_in[-3]))
    # Network Input shape -> 128, 64, 7 (sequence_length x bins x features)

    # CNN
    spec_cnn = spec_start
    for i, convCnt in enumerate(f_pool_size):
        spec_cnn = Conv2D(filters=nb_cnn2d_filt[i], kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn) # add batch normalization layer
        spec_cnn = Activation('relu')(spec_cnn) # use relu activation function
        spec_cnn = MaxPooling2D(pool_size=(t_pool_size[i], f_pool_size[i]))(spec_cnn) # downsample along the frequency dimension only
        spec_cnn = Dropout(dropout_rate)(spec_cnn)
    # output shape of CNNs is (None, 128, 2, 256)

    spec_cnn = Reshape((data_in[-2], -1))(spec_cnn) # reshape input
    # After reshape (None, 128, 512)
    spec_rnn = attention_block(spec_cnn) # pass extracted features into attention block
    # After attention (None, 128, 512)
    
    #--- RNN --
    for nb_rnn_filt in rnn_size:
        spec_rnn = Bidirectional(
            LSTM(nb_rnn_filt, activation='tanh', dropout=dropout_rate, return_sequences=True),
            merge_mode='mul'
        )(spec_rnn)
    # After RNN (None, 128, 256)

    
    # FC - DOA
    doa = spec_rnn
    for nb_fnn_filt in fnn_size:
        doa = TimeDistributed(Dense(nb_fnn_filt))(doa)  # share weights across time
        doa = Dropout(dropout_rate)(doa)

    doa = TimeDistributed(Dense(data_out[1][-1]))(doa)
    doa = Activation('linear', name='doa_out')(doa) # no non-linear activation is needed for a regression task

    # FC - SED
    sed = spec_rnn
    for nb_fnn_filt in fnn_size:
        sed = TimeDistributed(Dense(nb_fnn_filt))(sed)
        sed = Dropout(dropout_rate)(sed)
    sed = TimeDistributed(Dense(data_out[0][-1]))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)


    model = Model(inputs=spec_start, outputs=[sed, doa])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse'], loss_weights=weights)

    model.summary()
    return model

def attention_block(inputs):
  # inputs.shape = (batch_size, time_steps, input_dim)
  input_dim = int(inputs.shape[2])
  timestep = int(inputs.shape[1])
  a = Permute((2, 1))(inputs) #Permutes the dimensions of the input according to a given pattern.
  a = Dense(timestep, activation='softmax')(a) #// Alignment Model + Softmax
  a = Lambda(lambda x: keras.backend.mean(x, axis=1), name='dim_reduction')(a)
  a = RepeatVector(input_dim)(a)
  a_probs = Permute((2, 1), name='attention_vec')(a)
  output_attention_mul = multiply([inputs, a_probs], name='attention_mul') #// Weighted Average 
  return output_attention_mul

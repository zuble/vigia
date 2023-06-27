# -- coding: UTF-8 --

from collections import namedtuple
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Activation, BatchNormalization, Concatenate, Conv3D, Dense, GlobalAveragePooling3D, Input,Lambda, MaxPooling3D, Dropout, Reshape, Add
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2

resnet_config = {
            'resnet50': [3, 4, 6, 3],
            'resnet101': [3, 4, 23, 3],
            'resnet152': [3, 8, 36, 3],
            'resnet200': [3, 24, 36, 3]
        }

class SLOWFAST(object):
    """SLOWFAST model."""

    def __init__(self, num_classes, mode):

        self.num_classes = num_classes
        self.mode = mode

        if K.image_data_format() == 'channels_last': self.bn_axis = 4 # or =-1
        else: self.bn_axis = 1
        
    
    def conv1pool1(self,input,sf):
        if sf == 'slow':
            x = Conv3D(64, (1, 7, 7), strides=(1, 2, 2), padding='same', kernel_regularizer=l2(1e-4), name='slow_conv1')(input)
            x = BatchNormalization(axis=self.bn_axis, name='slow_bn_conv1')(x)
            x = Activation('relu')(x)
            x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2),name='poo1_slow')(x)
        elif sf == 'fast':
            x = Conv3D(8, (5, 7, 7), strides=(1, 2, 2), padding='same', kernel_regularizer=l2(1e-4), name='fast_conv1')(input)
            x = BatchNormalization(axis=self.bn_axis, name='fast_bn_conv1')(x)
            x = Activation('relu')(x)
            x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2),name='poo1_fast')(x)
        return x

    def _bottleneck_residual(self, x, out_filter, stride=[1, 1, 1], inflate=False, need_short = False):
        orig_x = x
        ## a
        if inflate: length = 3
        else: length = 1
        x = Conv3D(filters=int(out_filter/4), kernel_size=[length, 1, 1], strides=stride, padding='SAME')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')()(x)
        ## b
        x = Conv3D(filters=int(out_filter/4), kernel_size=[1, 3, 3],  padding='SAME')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')()(x)
        ## c
        x = Conv3D(filters=out_filter, kernel_size=[1, 1, 1], padding='SAME')(x)
        x = BatchNormalization()(x)
        ## shorcut
        if need_short :
            orig_x = Conv3D(filters=out_filter, kernel_size=[1, 1, 1],strides=stride, padding='SAME')(orig_x)
            orig_x = BatchNormalization()(orig_x)
        x += orig_x
        x = Activation('relu')()(x)
        return x
    
    
    def build_model(self,input_shape):

        x_input = Input(shape=input_shape,dtype=tf.float32, name='input_node')

        ## conv1 + pool1
        fast_x = x_input
        fast_x = Conv3D(filters=8, kernel_size=[5, 7, 7], strides=(1, 2, 2), padding='SAME',
                                        name='fast_conv1')(fast_x)
        fast_x = BatchNormalization()(fast_x)
        fast_x = Activation('relu')(fast_x)
        fast_x = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2))(fast_x)

        ## conv1 + pool1
        #slow_x = x_input[:, ::8, ...]
        slow_x = tf.gather(x_input, tf.range(0, input_shape[0], 8), axis=1)
        slow_x = Conv3D(filters=64,kernel_size=[1, 7, 7],strides=(1, 2, 2),padding = 'SAME',
                                                        name='slow_conv1')(slow_x)
        slow_x = BatchNormalization()(slow_x)
        slow_x = Activation('relu')()(slow_x)
        slow_x = MaxPooling3D(pool_size=(1, 3, 3), strides=(1,2,2))(slow_x)

        concat1 =  Conv3D(filters=16, kernel_size=[5, 1, 1], strides=(8, 1, 1), padding='SAME', name='concat1')(fast_x)
        slow_x = Concatenate(axis=-1)([slow_x, concat1])

        block_num = [3, 4, 6, 3]
        filters_slow_out = [256, 512, 1024, 2048]
        filters_fast_out = [32, 64, 128, 256]
        inflate_list_slow = [False, False, True, True]
        inflate_list_fast = [True, True, True, True]

        res_func = self._bottleneck_residual
        for index in range(0, 4):
            ## res2 , res3 , res4 , res5
            for i in range(0, block_num[index]):
                ## slow
                ## frist block
                if i == 0:
                    ## res2
                    if index == 0:
                        slow_x = res_func(slow_x,  filters_slow_out[index], 
                                          stride=[1, 1, 1],
                                          inflate=inflate_list_slow[index],
                                          need_short=True)
                    ## res3 , res4 , res5
                    else:
                        slow_x = res_func(slow_x, filters_slow_out[index], 
                                          stride=[1, 2, 2],
                                          inflate=inflate_list_slow[index],
                                          need_short=True)
                
                ## others block
                else:
                    slow_x = res_func(slow_x, filters_slow_out[index], 
                                      stride=[1, 1, 1],
                                      inflate=inflate_list_slow[index])
                
                ## fast
                if i == 0:
                    if index != 0 :
                        fast_x = res_func(  fast_x,  filters_fast_out[index], 
                                            stride=[1, 2, 2],
                                            inflate=inflate_list_fast[index],
                                            need_short=True)
                    else:
                        fast_x = res_func(  fast_x, filters_fast_out[index],
						                    stride=[1, 1, 1], 
                                            inflate=inflate_list_fast[index],
                                            need_short=True)

                else:
                    fast_x = res_func(  fast_x, filters_fast_out[index], 
                                        stride=[1, 1, 1], 
                                        inflate=inflate_list_fast[index])

            if index != 3:
                concat = Conv3D(filters=filters_fast_out[index]*2, kernel_size=[5, 1, 1], strides=(8, 1, 1), padding='SAME')(fast_x)
                slow_x = Concatenate(axis=-1)([slow_x, concat])

        slow_x = GlobalAveragePooling3D()(slow_x)
        fast_x = GlobalAveragePooling3D()(fast_x)
        global_pool= Concatenate(axis=-1)([slow_x, fast_x])

        if self.mode == 'train':
            global_pool = Dropout(0.5)(global_pool)
        logits = Dense(units=self.num_classes)(global_pool)
        predictions = Dense(self.num_classes,activation='softmax')(global_pool)
        model = tf.keras.Model(inputs=x_input, outputs=[predictions, logits])

        return model



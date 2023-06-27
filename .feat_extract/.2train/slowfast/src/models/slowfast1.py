'''
    https://github.com/xuzheyuan624/slowfast-keras/tree/master/model
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, ReLU, Add, MaxPool3D, GlobalAveragePooling3D, Concatenate, Dropout, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
#from tensorflow.keras.utils import plot_model #ImportError: Failed to import pydot. You must install pydot and graphviz for `pydotprint` to work

class SlowFast(Model):
    def __init__(self, model_type, input_shape, num_classes, dropout=0.5):
        super(SlowFast, self).__init__()
        self.num_classes = num_classes
        self.inputs = Input(shape=input_shape)
        self.dropout = dropout
        
        resnet_config = {
            'resnet50': [3, 4, 6, 3],
            'resnet101': [3, 4, 23, 3],
            'resnet152': [3, 8, 36, 3],
            'resnet200': [3, 24, 36, 3]
        }
        self._numblocks = resnet_config[model_type]
        
        self._block = self.bottleneck
    
    def Conv_BN_ReLU(self, planes, kernel_size, strides=(1, 1, 1), padding='same', use_bias=False):
        return Sequential([
            Conv3D(planes, kernel_size, strides=strides, padding=padding, use_bias=use_bias),
            BatchNormalization(),
            ReLU()
        ])

    def bottleneck(self, x, planes, stride=1, downsample=None, head_conv=1, use_bias=False):
        residual = x
        if head_conv == 1:      x = self.Conv_BN_ReLU(planes, kernel_size=(1,1,1))(x)
        elif head_conv == 3:    x = self.Conv_BN_ReLU(planes, kernel_size=(3,1,1))(x)
        else: raise ValueError('Unsupported head_conv!!!')
        
        x = self.Conv_BN_ReLU(planes, kernel_size=(1, 3, 3), strides=(1, stride, stride))(x)
        x = Conv3D(planes*4, kernel_size=1, use_bias=use_bias)(x)
        x = BatchNormalization()(x)
        if downsample is not None:
            residual = downsample(residual)
        x = Add()([x, residual])
        x = ReLU()(x)
        return x

    def datalayer(self, x, stride):
        return x[:, ::stride, :, :, :]


    def Fast_body(self, x):
        ''' return the laterals to be used in the slow paths, and the GAP3D of res5'''
        fast_inplanes = 8
        lateral = []

        ## conv1pool1
        x = self.Conv_BN_ReLU(8, kernel_size=(5, 7, 7), strides=(1, 2, 2))(x)
        x = MaxPool3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(x)
        lateral_p1 = Conv3D(8*2, kernel_size=(5, 1, 1), strides=(8, 1, 1), padding='same', use_bias=False)(x)
        lateral.append(lateral_p1) ## fast lateral conv1pool1
        ## res2
        x, fast_inplanes = self.make_layer_fast(x, 8, self._numblocks[0], head_conv=3, fast_inplanes=fast_inplanes)
        lateral_res2 = Conv3D(32*2, kernel_size=(5, 1, 1), strides=(8, 1, 1), padding='same', use_bias=False)(x)
        lateral.append(lateral_res2) ## fast lateral res2
        ## res3
        x, fast_inplanes = self.make_layer_fast(x, 16, self._numblocks[1], stride=2, head_conv=3, fast_inplanes=fast_inplanes)
        lateral_res3 = Conv3D(64*2, kernel_size=(5, 1, 1), strides=(8, 1, 1), padding='same', use_bias=False)(x)
        lateral.append(lateral_res3) ## fast lateral res3
        ## res4
        x, fast_inplanes = self.make_layer_fast(x, 32, self._numblocks[2], stride=2, head_conv=3, fast_inplanes=fast_inplanes)
        lateral_res4 = Conv3D(128*2, kernel_size=(5, 1, 1), strides=(8, 1, 1), padding='same', use_bias=False)(x)
        lateral.append(lateral_res4) ## fast lateral res4
        ## res5
        x, fast_inplanes = self.make_layer_fast(x, 64, self._numblocks[3], stride=2, head_conv=3, fast_inplanes=fast_inplanes)
        x = GlobalAveragePooling3D()(x)
        return x, lateral
    
    def make_layer_fast(self, x, planes, nblocks, stride=1, head_conv=1, fast_inplanes=8, block_expansion=4):
            downsample = None
            if stride != 1 or fast_inplanes != planes * block_expansion:
                downsample = Sequential([
                    Conv3D(planes*block_expansion, kernel_size=1, strides=(1, stride, stride), use_bias=False),
                    BatchNormalization()
                ])
            fast_inplanes = planes * block_expansion
            x = self.bottleneck(x, planes, stride, downsample=downsample, head_conv=head_conv)
            for _ in range(1, nblocks):
                x = self.bottleneck(x, planes, head_conv=head_conv)
            return x, fast_inplanes
   
   
    def Slow_body(self, x, lateral):
        slow_inplanes = 64 + 64//8*2 #=80
        ## conv1pool1
        x = self.Conv_BN_ReLU(64, kernel_size=(1, 7, 7), strides=(1, 2, 2))(x)
        x = MaxPool3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(x)
        x = Concatenate()([x, lateral[0]]) ## lateral conn conv1pool1
        ## res2
        x, slow_inplanes = self.make_layer_slow(x, 64, self._numblocks[0], head_conv=1, slow_inplanes=slow_inplanes)
        x = Concatenate()([x, lateral[1]]) ## lateral conn res2
        ## res3
        x, slow_inplanes = self.make_layer_slow(x, 128, self._numblocks[1], stride=2, head_conv=1, slow_inplanes=slow_inplanes)
        x = Concatenate()([x, lateral[2]]) ## lateral conn res3
        ## res4
        x, slow_inplanes = self.make_layer_slow(x, 256, self._numblocks[2], stride=2, head_conv=1, slow_inplanes=slow_inplanes)
        x = Concatenate()([x, lateral[3]]) ## lateral conn res4
        ## res5
        x, slow_inplanes = self.make_layer_slow(x, 512, self._numblocks[3], stride=2, head_conv=1, slow_inplanes=slow_inplanes)
        x = GlobalAveragePooling3D()(x) 
        return x

    def make_layer_slow(self, x, planes, blocks, stride=1, head_conv=1, slow_inplanes=80, block_expansion=4):
        ''' residual stage'''
        downsample = None
        if stride != 1 or slow_inplanes != planes * block_expansion:
            downsample = Sequential([
                Conv3D(planes*block_expansion, kernel_size=1, strides = (1, stride, stride), use_bias=False),
                BatchNormalization()
            ])
        ## w/shortcut | conv_block
        x = self.bottleneck(x, planes, stride, downsample, head_conv=head_conv)
        for _ in range(1, blocks): ## w/o sc | identity_block 
            x = self.bottleneck(x, planes, head_conv=head_conv)
        slow_inplanes = planes * block_expansion + planes * block_expansion//8*2
        return x, slow_inplanes
    
    
    def construct(self):
        inputs_fast = Lambda(self.datalayer, name='data_fast', arguments={'stride':2})(self.inputs)
        inputs_slow = Lambda(self.datalayer, name='data_slow', arguments={'stride':16})(self.inputs)
        fast, lateral = self.Fast_body(inputs_fast)
        slow = self.Slow_body(inputs_slow, lateral)
        x = Concatenate()([slow, fast])
        x = Dropout(self.dropout)(x)
        out = Dense(self.num_classes, activation='softmax')(x)
        return Model(self.inputs, out)
    
    '''
    def call(self, inputs, training=True):
        inputs_fast = Lambda(self.datalayer, name='data_fast', arguments={'stride':2})(inputs)
        inputs_slow = Lambda(self.datalayer, name='data_slow', arguments={'stride':16})(inputs)
        
        fast, lateral = self.Fast_body(inputs_fast, self._layers, self.bottleneck)
        slow = self.Slow_body(inputs_slow, lateral, self._layers, self.bottleneck)
        
        x = Concatenate()([slow, fast])
        x = Dropout(self.dropout)(x)
        out = Dense(num_classes, activation='softmax')(x)
        return out
    '''

    #def summary(self, input_shape):
    #    x = Input(shape=input_shape)
    #    model = Model(inputs=x, outputs=self.call(x), name='X3D')
    #    return model.summary()


if __name__=="__main__":
    import numpy as np
    model_type = "resnet50"
    input_shape = (64, 224, 224, 3)
    num_classes = 10
    model = SlowFast(model_type, input_shape, num_classes).construct()
    
    #model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    
    #x=np.ones((1 ,64, 224, 224, 3))
    #y1=model(x)
    #print(np.shape(y1))
    
    
    #from zulowfat import SlowFast_Network
    #model = SlowFast_Network(clip_shape=[64,224,224,3],num_class=10,alpha=8,beta=1/8,tau=16,method='T_conv')
    #y2=model(x)
    #print(np.shape(y2))

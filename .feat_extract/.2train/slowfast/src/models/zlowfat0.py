#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Activation, BatchNormalization, Concatenate, Conv3D, Dense, GlobalAveragePooling3D, Input, Lambda, MaxPooling3D, Dropout, Reshape, Add
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2

from yacs.config import CfgNode
from src.models.builder import register_model

resnet_config = {
            'resnet50': [3, 4, 6, 3],
            'resnet101': [3, 4, 23, 3],
            'resnet152': [3, 8, 36, 3],
            'resnet200': [3, 24, 36, 3]
        }

@register_model("SlowFastRes50")
class SlowFastRes50(object):
    '''
    Constructs the X3D model, given the model configurations
    See: https://arxiv.org/abs/2004.04730v1
    '''
    def __init__(self, cfg: CfgNode):
        '''
        Initialize an instance of the model given the model configurations

        Args:
            cfg (CfgNode): the model configurations
        '''
        #super(SlowFastRes50, self).__init__()
        self.data_layer = cfg.SLOWFAST.DATALAYER
        if not self.data_layer:
            ## input is ready for fast pathway -> get slow from it
            self.clip_shape=[32,224,224,3]
        elif self.data_layer == 1:
            ## input is not temporal steped
            self.clip_shape=[64,224,224,3]
        else: raise Exception(f"data_layer {self.data_layer} not accepted")
        
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.alpha = cfg.SLOWFAST.ALPHA
        self.beta = cfg.SLOWFAST.BETA
        self.tau = cfg.SLOWFAST.TAU
        self.method= cfg.SLOWFAST.FUSION_METHOD
        
        #pass the l2 wight decay as variable
        
        self.bn_momentum = cfg.SLOWFAST.BN.MOMENTUM
        self.bn_eps = cfg.SLOWFAST.BN.EPS
        
        if K.image_data_format() == 'channels_last': self.bn_axis = 4 # or =-1
        else: self.bn_axis = 1
        
        
        self.model = self.construct()
        
        
    def lateral_conn(self, fast_res_block, slow_res_block, stage):
        
        lateral_name = 'lateral'+'_stage_'+str(stage)
        connection_name = 'connection'+'_stage_'+str(stage)
        if self.method not in ['T_conv','T_sample','TtoC_sum','TtoC_concat']:
            raise ValueError("method should be one of ['T_conv','T_sample','TtoC_sum','TtoC_concat']")
        
        ## log info ablout the shpae input
        
        if self.method == 'T_conv':
            ''' We perform a 3D convolution of a 5,1,1 kernel with 2βC out channels and stride=alpha 
                FuseFast2Slow class use bn+ReLU'''
            ## 8 filters if original setup / strides = (8,1,1)
            x = Conv3D(   filters=int(2*self.beta*int(fast_res_block.shape[4])),
                                padding='same', 
                                kernel_size=(5, 1, 1),
                                strides=(int(self.alpha), 1, 1), 
                                kernel_regularizer=l2(1e-4),
                                use_bias=False,
                                name=lateral_name+'conv3d')(fast_res_block)
            x = BatchNormalization( axis=self.bn_axis, 
                                    momentum=self.bn_momentum,
                                    epsilon=self.bn_eps,
                                    name=lateral_name+'bn')(x)
            lateral = Activation('relu')(x) 
            #print(stage,slow_res_block.shape,fast_res_block.shape,lateral.shape)
            connection = Concatenate(axis=-1,name=connection_name)([slow_res_block,lateral])

        if self.method == 'T_sample':
            def sample(input, stride):
                return tf.gather(input, tf.range(0, input.shape[1], stride), axis=1)
            lateral = Lambda(sample,arguments={'stride':self.alpha},name=lateral_name)(fast_res_block)
            connection = Concatenate(axis=-1,name=connection_name)([slow_res_block,lateral])
        
        if self.method =='TtoC_concat':
            lateral = Reshape((int(int(fast_res_block.shape[1])/self.alpha),int(fast_res_block.shape[2]),int(fast_res_block.shape[3]),int(self.alpha*fast_res_block.shape[4]))
                            ,name=lateral_name)(fast_res_block)
            connection = Concatenate(axis=-1,name=connection_name)([slow_res_block,lateral])
        
        if self.method =='TtoC_sum':
            if self.alpha*self.beta!=1:
                raise ValueError("The product of alpha and beta must equal 1 in TtoC_sum method")
            lateral = Reshape((int(int(fast_res_block.shape[1])/self.alpha),int(fast_res_block.shape[2]),int(fast_res_block.shape[3]),int(self.alpha*fast_res_block.shape[4]))
                            ,name=lateral_name)(fast_res_block)
            connection = Add(name=connection_name)([slow_res_block,lateral])

        return connection
    
    
    def identity_block(self, input, filters, k_s, id, non_degenerate_tconv=False):
        """The identity block is the block that has no conv layer at shortcut.

        Arguments:
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names

        Returns:
            Output tensor for the block.
        """
        filters1, filters2, filters3 = filters
        stage , block , path = id
        conv_id = str(path) + 'res' + str(stage) + block + '_branch'
        bn_id = str(path) + 'bn' + str(stage) + block + '_branch'

        if non_degenerate_tconv == True:
            aux_k_s=3 ; aux_pad='same'
        else: aux_k_s=1 ; aux_pad='valid'   
        
        ## Tx1x1 , BN , ReLU
        x = Conv3D( filters1, 
                    (aux_k_s,1,1), 
                    padding=aux_pad, 
                    kernel_regularizer=l2(1e-4),
                    use_bias=False, 
                    name=conv_id+'2A')(input)
        x = BatchNormalization( axis=self.bn_axis, 
                                momentum=self.bn_momentum,
                                epsilon=self.bn_eps,
                                name=bn_id+'2A')(x)
        x = Activation('relu')(x)
        
        ## 1x3x3, BN , ReLU
        x = Conv3D( filters2, 
                    kernel_size=k_s, 
                    padding='same', 
                    kernel_regularizer=l2(1e-4), 
                    use_bias=False, 
                    name=conv_id+'2B')(x)
        x = BatchNormalization( axis=self.bn_axis,
                                momentum=self.bn_momentum,
                                epsilon=self.bn_eps,
                                name=bn_id+'2B')(x)
        x = Activation('relu')(x)
        
        ## 1x1x1 , BN
        x = Conv3D( filters3, 
                    kernel_size=(1,1,1), 
                    kernel_regularizer=l2(1e-4), 
                    use_bias=False, 
                    name=conv_id+'2C')(x)
        x = BatchNormalization( axis=self.bn_axis, 
                                momentum=self.bn_momentum,
                                epsilon=self.bn_eps,
                                name=bn_id+'2C')(x)

        x = layers.add([x, input])
        x = Activation('relu')(x)
        return x
    
    def conv_block(self, input, filters, k_s, id , strides=(1,2,2), non_degenerate_tconv=False):
        """A block that has a conv layer at shortcut.

        Arguments:
            input: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            strides: Strides for the first conv layer in the block.

        Returns:
            Output tensor for the block.

        Note that from stage 3,
        the first conv layer at main path is with strides=(2, 2)
        And the shortcut should have strides=(2, 2) as well
        """
        filters1, filters2, filters3 = filters
        stage , block , path = id
        conv_id = str(path) + 'res' + str(stage) + block + '_branch'
        bn_id = str(path) + 'bn' + str(stage) + block + '_branch'

        ## Tx1x1(ks) , BN , ReLU
        if non_degenerate_tconv == True: 
            aux_k_s=3
            aux_pad='same' #[3//2,0,0]
        else: 
            aux_k_s=1
            aux_pad='valid' #[1//2,0,0]
            
        x = Conv3D( filters1,
                    kernel_size=(aux_k_s,1,1), 
                    strides=strides, 
                    padding=aux_pad, 
                    use_bias=False, 
                    kernel_regularizer=l2(1e-4), 
                    name=conv_id+'2A')(input)
        x = BatchNormalization( axis=self.bn_axis, 
                                momentum=self.bn_momentum,
                                epsilon=self.bn_eps,
                                name=bn_id+'2A')(x)
        x = Activation('relu')(x)
        
        ## 1x3x3, BN , ReLU
        x = Conv3D( filters2, 
                    kernel_size=k_s, 
                    padding='same', 
                    kernel_regularizer=l2(1e-4),
                    use_bias=False, 
                    name=conv_id + '2B')(x)
        x = BatchNormalization( axis=self.bn_axis, 
                                momentum=self.bn_momentum,
                                epsilon=self.bn_eps,
                                name=bn_id + '2B')(x)
        x = Activation('relu')(x)

        ## 1x1x1 , BN
        x = Conv3D( filters3, 
                    kernel_size=(1,1,1), 
                    kernel_regularizer=l2(1e-4),
                    use_bias=False, 
                    name=conv_id + '2c')(x)
        x = BatchNormalization( axis=self.bn_axis, 
                                momentum=self.bn_momentum,
                                epsilon=self.bn_eps,
                                name=bn_id + '2C')(x)

        shortcut = Conv3D(  filters3, 
                            kernel_size=(1,1,1), 
                            strides=strides, 
                            kernel_regularizer=l2(1e-4), 
                            use_bias=False,
                            name=conv_id+'SC')(input)
        shortcut = BatchNormalization(  axis=self.bn_axis,
                                        momentum=self.bn_momentum,
                                        epsilon=self.bn_eps,
                                        name=bn_id+'SC')(shortcut)

        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        return x

    def conv1pool1(self,input,typer):
        if typer == 'slow':
            aux_filters = 64
            aux_k_s = (1,7,7)
        elif typer == 'fast':
            aux_filters = 8
            aux_k_s = (5,7,7)
            
        x = Conv3D( filters=aux_filters, 
                    kernel_size=aux_k_s, 
                    strides=(1,2,2), 
                    padding='same',
                    use_bias=False, 
                    kernel_regularizer=l2(1e-4), 
                    name=typer+'_conv1')(input)
        x = BatchNormalization( axis=self.bn_axis,
                                momentum=self.bn_momentum,
                                epsilon=self.bn_eps, 
                                name=typer+'_bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling3D(   pool_size=(1,3,3), 
                            strides=(1,2,2),##video_model_builder.py/resnet/padding=[0,0,0]='valid'
                            name=typer+'_poo11')(x)
        return x
    
    '''
    resize_and_rescale = Sequential([
        layers.Resizing(224, 224),
        layers.Rescaling(1./255)])
    '''

    def slow_data(self,input,stride):
        return tf.gather(input,tf.range(0,self.clip_shape[0],stride),axis=1)
        
    def construct(self):
        clip_input = Input(shape=self.clip_shape)
        
        if not self.data_layer:
            slow_input = Lambda(self.slow_data,arguments={'stride':self.alpha},name='slow_input')(clip_input)
            fast_input = clip_input
        else:
            slow_input = Lambda(self.slow_data,arguments={'stride':self.tau},name='slow_input')(clip_input)
            fast_input = Lambda(self.slow_data,arguments={'stride':int(self.tau/self.alpha)},name='fast_input')(clip_input)
        print('slow:',slow_input.shape)
        print('fast:',fast_input.shape)

        ## --- fast pathway ---
        pool1_fast = self.conv1pool1(fast_input,'fast')
        ## res2 
        x_fast = self.conv_block(pool1_fast, [8,8,32] , [1,3,3] , id=[2,'a','fast'] , strides=(1, 1, 1), non_degenerate_tconv=True)
        x_fast = self.identity_block(x_fast, [8,8,32] , [1,3,3] , id=[2,'b','fast'] , non_degenerate_tconv=True)
        res2_fast = self.identity_block(x_fast, [8,8,32] , [1,3,3] , id=[2,'c','fast'], non_degenerate_tconv=True)
        ## res3
        x_fast = self.conv_block(res2_fast , [16,16,64] , [1,3,3] , id=[3,'a','fast'] , non_degenerate_tconv=True)
        x_fast = self.identity_block(x_fast , [16,16,64] , [1,3,3] , id=[3,'b','fast'] , non_degenerate_tconv=True)
        x_fast = self.identity_block(x_fast , [16,16,64] , [1,3,3] , id=[3,'c','fast'] , non_degenerate_tconv=True)
        res3_fast = self.identity_block(x_fast , [16,16,64] , [1,3,3] , id=[3,'d','fast'] , non_degenerate_tconv=True)
        ## res4
        x_fast = self.conv_block(res3_fast, [32,32,128] , [1,3,3],  id=[4,'a','fast'], non_degenerate_tconv=True)
        x_fast = self.identity_block(x_fast, [32,32,128] , [1,3,3] , id=[4,'b','fast'], non_degenerate_tconv=True)
        x_fast = self.identity_block(x_fast, [32,32,128] , [1,3,3] , id=[4,'c','fast'], non_degenerate_tconv=True)
        x_fast = self.identity_block(x_fast, [32,32,128] , [1,3,3] , id=[4,'d','fast'], non_degenerate_tconv=True)
        x_fast = self.identity_block(x_fast, [32,32,128] , [1,3,3] , id=[4,'e','fast'], non_degenerate_tconv=True)
        res4_fast = self.identity_block(x_fast, [32,32,128] , [1,3,3] , id=[4,'f','fast'], non_degenerate_tconv=True)
        ## res5
        x_fast = self.conv_block(res4_fast , [64,64,256] , [1,3,3] ,  id=[5,'a','fast'], non_degenerate_tconv=True)
        x_fast = self.identity_block(x_fast, [64,64,256] , [1,3,3] , id=[5,'b','fast'], non_degenerate_tconv=True)
        res5_fast = self.identity_block(x_fast, [64,64,256] , [1,3,3] , id=[5,'c','fast'], non_degenerate_tconv=True)


        ## --- slow pathway ---
        pool1 = self.conv1pool1(slow_input,'slow')
        pool1_conection = self.lateral_conn(pool1_fast,pool1,stage=1)
        ## res2
        x = self.conv_block(pool1_conection , [64,64,256] , [1,3,3] , id=[2,'a','slow'] , strides=(1,1,1) )
        x = self.identity_block(x, [64,64,256] , [1,3,3] , id=[2,'b','slow'])
        res2 = self.identity_block(x, [64,64,256] , [1,3,3] , id=[2,'c','slow'])
        res2_conn = self.lateral_conn(res2_fast,res2,stage=2)
        ## res3
        x = self.conv_block(res2_conn , [128,128,512] , [1,3,3] , id=[3,'a','slow'])
        x = self.identity_block(x, [128,128,512] , [1,3,3] , id=[3,'b','slow'])
        x = self.identity_block(x, [128,128,512] , [1,3,3] , id=[3,'c','slow'])
        res3 = self.identity_block(x, [128,128,512] , [1,3,3] , id=[3,'d','slow'])
        res3_conn = self.lateral_conn(res3_fast , res3 , stage=3)
        ## res4
        x = self.conv_block(res3_conn , [256,256,1024] , [1,3,3] , id=[4,'a','slow'] , non_degenerate_tconv=True)
        x = self.identity_block(x , [256,256,1024] , [1,3,3] , id=[4,'b','slow'] , non_degenerate_tconv=True)
        x = self.identity_block(x , [256,256,1024] , [1,3,3] , id=[4,'c','slow'] , non_degenerate_tconv=True)
        x = self.identity_block(x , [256,256,1024] , [1,3,3] , id=[4,'d','slow'] , non_degenerate_tconv=True)
        x = self.identity_block(x , [256,256,1024] , [1,3,3] , id=[4,'e','slow'] , non_degenerate_tconv=True)
        res4 = self.identity_block(x, [256,256,1024] , [1,3,3] ,  id=[4,'f','slow'] , non_degenerate_tconv=True)
        res4_conn = self.lateral_conn(res4_fast,res4,stage=4)
        ## res5
        x = self.conv_block(res4_conn , [512,512,2048] , [1,3,3] , id=[5,'a','slow'] , non_degenerate_tconv=True)
        x = self.identity_block(x , [512,512,2048] , [1,3,3] , id=[5,'b','slow'] , non_degenerate_tconv=True)
        res5 = self.identity_block(x , [512,512,2048], [1,3,3] , id=[5,'c','slow'] , non_degenerate_tconv=True)

        fast_output = GlobalAveragePooling3D(name='avg_pool_fast')(res5_fast)
        slow_output = GlobalAveragePooling3D(name='avg_pool_slow')(res5)
        concat_output = Concatenate(axis=-1)([slow_output,fast_output])
        concat_output = Dropout(0.5)(concat_output)
        output = Dense(self.num_classes,activation='softmax', kernel_regularizer=l2(1e-4), name = 'fc')(concat_output)

        return Model(clip_input, output, name='slowfast_resnet50')


if __name__=="__main__":
    import numpy as np
    model_type = "resnet50"
    input_shape = (64, 224, 224, 3)
    num_classes = 10
    model = SlowFastRes50(model_type, input_shape, num_classes).construct()
    
    #model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    
    #x=np.ones((1 ,64, 224, 224, 3))
    #y1=model(x)
    #print(np.shape(y1))
    
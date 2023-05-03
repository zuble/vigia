import os , logging , time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint

from utils import globo

#--------------------------------------------------------#
# GPU TF CONFIGURATION
# https://www.tensorflow.org/guide/gpu 
# https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth

def set_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        #print("\nAvaiable GPU's",gpus)
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                print(gpu,"set memory growth True\n")
                tf.config.experimental.set_memory_growth(gpu, True)
            
            #logical_gpus = tf.config.list_logical_devices('GPU')
            #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)

    gpus = tf.config.list_physical_devices('GPU')
    print("\nNum GPUs Available: ", len(gpus))
    #for i in range(len(gpus)) :print(str(gpus[i]))

def limit_gpu_gb(i):
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_logical_device_configuration(
        gpus[i],
        [tf.config.LogicalDeviceConfiguration(memory_limit=31744)]
    )


#--------------------------------------------------------#
## MODEL
def all_operations(args):
    x = args[0]
    x = tf.reshape(x, [1, -1,x.shape[1]*x.shape[2]*x.shape[3]])
    return x
@tf.function
def loss_category(y_true, y_pred):    
    #tf.print(y_pred, y_true, 'Prediction')
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return cce
def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))
#get_custom_objects().update({'gelu': Activation(gelu)})


def form_model_wav(params):
    print("\nFORM_MODEL @",print(params['full_or_max']),"\n")
   
    ''' waves coming '''
   
    #https://www.tensorflow.org/api_docs/python/tf/keras/activations
    #https://www.tensorflow.org/api_docs/python/tf/nn/leaky_relu
    if params["ativa"]=='leakyrelu': ativa = keras.layers.LeakyReLU()
    elif params["ativa"]=='gelu': ativa = gelu
    elif params["ativa"]=='relu': ativa = 'relu'
    else: raise Exception("no ativa named assim")


    ## (TIMESTEPS,AAS)
    if params["full_or_max"] == 'full':
        aas_input = tf.keras.layers.Input(shape=(None, params["sinet_aas_len"]), name='input_layer')
        
        gloabl_aas = keras.layers.GlobalMaxPooling1D()(aas_input) 
        
        hidden_dense_1 = keras.layers.Dense(128, activation=ativa)(gloabl_aas)
        sigmoid = keras.layers.Dense(1, activation='sigmoid')(hidden_dense_1)    
        
        model = keras.Model(inputs=[aas_input], 
                            outputs=[sigmoid])
    
    ## or np.max before
    elif params['full_or_max'] == 'max':
        model = tf.keras.Sequential([
            layers.Input(shape=(None,params["sinet_aas_len"]), name='input_layer'),
            layers.Dense(128, activation=ativa, name='hidden_layer'),
            layers.Dense(1, activation='sigmoid', name='output_layer')
        ])
    
    
    #https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
    if params["optima"]=='sgd':optima = keras.optimizers.SGD(learning_rate = 0.0002)
    elif params["optima"]=='adam':optima = keras.optimizers.Adam(learning_rate = 0.0002)
    elif params["optima"]=='adamamsgrad':optima = keras.optimizers.Adam(learning_rate = 0.0002,amsgrad=True)
    else: raise Exception("no optima named assim")

    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'), 
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]

    model.compile(optimizer=optima, 
                    loss = 'binary_crossentropy', 
                    metrics = METRICS
                )
    model.summary()    
    
    print("\n\t",params,"\n\n\tOPTIMA",optima,"\n\tATIVA",ativa)

    time_str = str(time.time()); 
    model_name = time_str + '_'+params["ativa"]+'_'+params["optima"]+'_'+str(params["full_or_max"])+'_'+str(params["frame_max"])
    print("\n\t",model_name)
    return model , model_name

#--------------------------------------------------------#
## CALLBACKS

def ckpt_clbk(model_name):
    #https://keras.io/api/callbacks/model_checkpoint/
    p = os.path.join(globo.CKPT_PATH,model_name)
    if not os.path.exists(p):
        os.makedirs(p)
    else:raise Exception(f"{p} eristes")
    
    return ModelCheckpoint( filepath=p+'/'+model_name+'_ckpt-{epoch:02d}-{loss:.2f}.h5' , \
                            monitor='loss',\
                            save_weights_only=True,\
                            #save_best_only=True,\
                            mode='auto',\
                            save_freq='epoch',\
                            verbose = 1)
        

#--------------------------------------------------------#
## MIL MODEL IDEA FORMULATION
def build_mil_model():
    image_input = keras.Input(shape=(None, 120, 160, 3))

    c3d_mp = keras.Sequential([
        keras.layers.Conv3D(4,(2,3,3), activation='relu'),  
        keras.layers.MaxPooling3D((1,2,2)),
        keras.layers.Conv3D(8,(4,3,3), activation='relu'),
        keras.layers.MaxPooling3D((2,2,2)),
        keras.layers.Conv3D(16,(8,3,3), activation='relu'),
        keras.layers.MaxPooling3D((4,2,2))     
    ])
    c3d_mp_out = c3d_mp(image_input)

    c3d_mp_flatten = keras.layers.Flatten()(c3d_mp_out)  
    lstm1 = keras.layers.LSTM(1024, return_sequences=True)(c3d_mp_flatten)
    global_rgb_feature = keras.layers.GlobalMaxPooling1D()(lstm1)

    dense_1 = keras.layers.Dense(128, activation='relu')(global_rgb_feature)
    soft_max = keras.layers.Dense(1, activation='sigmoid')(dense_1)
    
    return keras.Model(inputs=[image_input], outputs=[soft_max])

def mil_bag_model():
    bag_input = keras.Input(shape=(None, None, 120, 160, 3))
    mil_model = build_mil_model()
    bag_outputs = keras.layers.TimeDistributed(mil_model)(bag_input)
    bag_level_output = keras.layers.GlobalMaxPooling1D()(bag_outputs)
    return keras.Model(inputs=[bag_input], outputs=[bag_level_output])

#model_rgb = mil_bag_model()
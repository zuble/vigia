import os , logging , time

import numpy as np
import tensorflow as tf
from tensorflow import keras
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

def check_gpu_conn():
    device_name = tf.test.gpu_device_name()
    if not device_name:
        raise SystemError('GPU device not found')
    print('\n\nFound GPU at: {}'.format(device_name))


#--------------------------------------------------------#
## MODEL
def name_it(cfg,string_it=False):
    aa = cfg["ativa"]+'_'+cfg["optima"]+'_'+str(cfg["batch_type"])+'_'+str(cfg["frame_step"])+'_'+str(cfg["frame_max"])
    if string_it: return aa.split("_")
    return aa 


def all_operations(args):
    x = args[0]
    #tf.print(x.shape)
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


def form_model(params):
    print("\nFORM_MODEL\n")

    in_height = 120; in_width = 160
    image_input = keras.Input(shape=(None, in_height, in_width, 3), name='input_layer')
    # ( 1 , frame_max , h , w , ch)    
    # ( 1 , 1000 , 120 , 160 , 3)

    
    #https://www.tensorflow.org/api_docs/python/tf/keras/activations
    #https://www.tensorflow.org/api_docs/python/tf/nn/leaky_relu
    if params["ativa"]=='leakyrelu': ativa = keras.layers.LeakyReLU()
    elif params["ativa"]=='gelu': ativa = gelu
    elif params["ativa"]=='relu': ativa = 'relu'
    else: raise Exception("no ativa named assim")

    
    c3d_mp = keras.Sequential([
        keras.layers.Conv3D(4,(2,3,3), activation=ativa),   #c3d_layer1
        keras.layers.MaxPooling3D((1,2,2)),                 #c3d_pooling1

        keras.layers.Conv3D(8,(4,3,3), activation=ativa),   #c3d_layer2
        keras.layers.MaxPooling3D((2,2,2)),                 #c3d_pooling2

        keras.layers.Conv3D(16,(8,3,3), activation=ativa),  #c3d_layer3
        keras.layers.MaxPooling3D((4,2,2))                  #c3d_pooling3
    ])
    c3d_mp_out = c3d_mp(image_input)
    # ( 1 , time_steps , spatl_featr1 , spatl_featr2 , spatl_featr3 ) 
    # ( 1 , 122 , 13 , 18 , 16 )  
    
    c3d_mp_flatten = keras.layers.Lambda(all_operations)(c3d_mp_out)  # flatten spatial features to time series
    # ( 1 , time_steps , spatl_featr_flattned ) 
    # ( 1 , 122        , 3744 )
    
    lstm1 = keras.layers.LSTM(1024, return_sequences=True)(c3d_mp_flatten) #input_shape=(120,c3d_mp_flatten.shape[2]),
    # ( 1 , time_steps , units) 
    # ( 1 , 122        , 1024 ) 
    
    global_rgb_feature = keras.layers.GlobalMaxPooling1D()(lstm1)
    # ( 1 , 1024 ) 
    
    '''    
    c3d_layer1 = keras.layers.Conv3D(4,(2,3,3), activation=ativa)(image_input)
    c3d_pooling1 = keras.layers.MaxPooling3D((1,2,2))(c3d_layer1)
    
    c3d_layer2 = keras.layers.Conv3D(8,(4,3,3), activation=ativa)(c3d_pooling1)
    c3d_pooling2 = keras.layers.MaxPooling3D((2,2,2))(c3d_layer2)
    
    c3d_layer3 = keras.layers.Conv3D(16,(8,3,3), activation=ativa)(c3d_pooling2)
    c3d_pooling3 = keras.layers.MaxPooling3D((4,2,2))(c3d_layer3)

    feature_conv_4 = keras.layers.Lambda(all_operations)(c3d_pooling3) #flatten spatial features to time series
    
    lstm1 = keras.layers.LSTM(1024,input_shape=(1200,feature_conv_4.shape[2]), return_sequences=True)(feature_conv_4)
    #lstm2 = keras.layers.LSTM(512, return_sequences=True)(lstm1)
    
    global_feature = keras.layers.GlobalMaxPooling1D()(lstm1)
    '''
    
    #waves 
    
    hidden_dense_1 = keras.layers.Dense(128, activation=ativa)(global_rgb_feature)
    sigmoid = keras.layers.Dense(1, activation='sigmoid', name='output_layer')(hidden_dense_1)
    
    model = keras.Model(inputs=[image_input], outputs=[sigmoid])

   
    #class_weights = [10,10,10,10,10,10,10,10,10,10,10,10,0.1,10]
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
                    #loss= {'output_layer':'binary_crossentropy'}, 
                    #loss_weights = class_weights,
                    metrics = METRICS
                    #metrics={'output_layer':METRICS}
                )
    
    print("\n\t",params,"\n\n\tOPTIMA",optima,"\n\tATIVA",ativa)
    
    model_name = str(time.time()) + '_' + name_it(params)
    print("\n\t",model_name)
    return model , model_name


def multi_gpu_model(cfg):
    ## MULTI GPU STRATEGY
    strategy = tf.distribute.MirroredStrategy()
    print('\nSTATEGY\nNumber of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model,model_name = form_model(cfg)
    return model,model_name


#---------#
## h5 files

def find_h5(path,find_string):
    '''
        if find_string=('') it returns all .h5 files in path descarding subdiretories
        if find_string=('str1','str2',..) it returns .h5 files with all str in name 
    '''
    h5_fn = [];h5_pth = []

    for root, dirs, files in os.walk(path):
        if len(find_string) == 0:
            for fil in files:
                fname, fext = os.path.splitext(fil)
                if fext == ".h5":
                    h5_pth.append(os.path.join(root, fil))
                    h5_fn.append(fname)
            break
        else:
            for fil in files:
                fname, fext = os.path.splitext(fil) 
                aux = 0
                for i in range(len(find_string)):    
                    if str(find_string[i]) in fname:aux = aux + 1
                
                if fext == ".h5" and aux == len(find_string):
                    h5_pth.append(os.path.join(root, fil))
                    h5_fn.append(fname)

    if not h5_fn:
        raise Exception("no h5 file with ",find_string,"in ",path)          
        
    return h5_fn, h5_pth

def load_h5(model,path,config):
    # eg. tfh5.get_weight_from_ckpt("1679349568.3157873_leakyrelu_adamamsgrad_0_4000_ckpt-00000009/1679349568.3157873_leakyrelu_adamamsgrad_0_4000_ckpt-00000009-00000017.h5",train_config)
    print('load_h5')

    h5fn , h5pth = find_h5(path, name_it(config,True) )
    print(h5fn,h5pth)
    
    #full_ckpt_path = os.path.join(aux.CKPT_PATH,h5_folder_nd_file)
    #model.load_weights(full_ckpt_path)
    #print("\t\nweights loaded from",full_ckpt_path)

    #model_name = os.path.splitext(os.path.basename(h5_folder_nd_file))[0]
    #model_weights_path = aux.WEIGHTS_PATH + model_name + '_weights.h5'
    #model.save_weights(model_weights_path)
    #print("\tweights saved to",model_weights_path)

    return model

#--------------------------------------------------------#
## CALLBACKS

def ckpt_clbk(model_name):
    #https://keras.io/api/callbacks/model_checkpoint/
    p = os.path.join(globo.CKPT_PATH,model_name)
    if not os.path.exists(p): os.makedirs(p)
    else: raise Exception(f"{p} eristes")
    
    return ModelCheckpoint( filepath=p+'/'+model_name+'_ckpt-{epoch:02d}.h5' , \
                            monitor='loss',\
                            save_weights_only=True,\
                            #save_best_only=True,\
                            mode='auto',\
                            save_freq='epoch',\
                            verbose = 1)
    
def tnsrboard_clbk(model_name,batch_start,batch_end):        
    logs = globo.LOGS_PATH + model_name
    return tf.keras.callbacks.TensorBoard(  log_dir = logs,
                                            histogram_freq = 1,
                                            profile_batch = str(batch_start)+","+str(batch_end))


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


#--------------------------------------------------------#
## TEST 
def init_test_model(params,from_path=globo.WEIGHTS_PATH,pb=False):
    ''' returns model with params config present in name over aux.WEIGHTS_PATH
        or if from_path is aux.MODEL_PATH it will recreate model from folders
    '''
    
    if from_path == str(globo.WEIGHTS_PATH):
        model = form_model(params)
    
        find_string=[params["ativa"]+'_'+params["optima"]+'_'+str(params["batch_type"])+'_'+params["frame_max"]]
        para_file_name, para_file_path = find_h5(from_path,find_string,ruii=False)
        model_path = para_file_path[0]
        model.load_weights(model_path)
    
        print("\n\tWEIGHTS from ", '/'+os.path.split(os.path.split(para_file_path[0])[0])[1]+'/'+os.path.split(para_file_path[0])[1])
        
        
    elif from_path==str(globo.MODEL_PATH):
        
        find_string=[params["ativa"]+'_'+params["optima"]+'_'+str(params["batch_type"])+'_'+params["frame_max"]]
        para_file_name, para_file_path = find_h5(from_path,find_string,ruii=False)
        
        if pb:
            model_path = os.path.join(para_file_path[0].replace(".h5","") , "/saved_model.pb")
            print(para_file_path[0].replace(".h5","") + "/saved_model.pb")
            model = tf.keras.models.load_model(model_path)
        else:
            model_path = para_file_path[0].replace(".h5","")
            model = tf.keras.models.load_model(model_path)
        
        print("\n\tMODEL from ", model_path)
        model.summary()
        
    else:
        raise Exception("give valid from_path")
    
    return model , para_file_name[0]
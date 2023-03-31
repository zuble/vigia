import os , logging
import numpy as np

import tensorflow as tf
from tensorflow import keras
os.system("conda list | grep -E 'tensorflow|cudnn|cudatoolkit|numpy'")

import utils.auxua as aux


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
                tf.config.experimental.set_memory_growth(gpu, True)
            
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
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


#---------------------------#

def get_weight_from_ckpt(h5_folder_nd_file,config):
    # eg. tf_formh5.get_weight_from_ckpt("1679349568.3157873_leakyrelu_adamamsgrad_0_4000_ckpt-00000009/1679349568.3157873_leakyrelu_adamamsgrad_0_4000_ckpt-00000009-00000017.h5",train_config)
    print('get_weight_from_ckpt')
    model = form_model(config)

    full_ckpt_path = os.path.join(aux.CKPT_PATH,h5_folder_nd_file)
    model.load_weights(full_ckpt_path)
    print("\t\nweights loaded from",full_ckpt_path)

    model_name = os.path.splitext(os.path.basename(h5_folder_nd_file))[0]
    model_weights_path = aux.WEIGHTS_PATH + model_name + '_weights.h5'
    model.save_weights(model_weights_path)
    print("\tweights saved to",model_weights_path)


def find_h5(path,find_string,ruii):
    '''
        if find_string=('') it returns all .h5 files in path descarding subdiretories
        if find_string=('str1','str2',..) it returns .h5 files with all str in name 
    '''
    if ruii:
        import PySimpleGUI as sg
        layout = [  [sg.Input(key="ckpt_h5" ,change_submits=True), sg.FileBrowse(key="browse",initial_folder=aux.MODEL_PATH)],
                    [sg.Button("check")]  # identify the multiline via key option]
                ]
        window = sg.Window("h5ckpt", layout)
        h5_pth=''
        h5_fn=''
        while True:
            event, values = window.read()
            if event in (sg.WIN_CLOSED, 'Exit'):
                break
            elif event == "check":
                h5_pth = values["ckpt_h5"]
                
        window.close()
        h5_fn = os.path.basename(h5_pth)
        return h5_fn,h5_pth
    if not ruii:
        h5_fn = []
        h5_pth = []

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

#---------------------------#

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


in_height = 120; in_width = 160
def form_model(params):
    print("\nFORM_MODEL\n")
    image_input = keras.Input(shape=(None, in_height, in_width, 3))
    #Freeze the batch normalization
    
    #https://www.tensorflow.org/api_docs/python/tf/keras/activations
    #https://www.tensorflow.org/api_docs/python/tf/nn/leaky_relu
    if params["ativa"]=='leakyrelu': ativa = keras.layers.LeakyReLU()
    elif params["ativa"]=='gelu': ativa = gelu
    elif params["ativa"]=='relu': ativa = 'relu'
    else: raise Exception("no ativa named assim")

    c3d_layer1 = keras.layers.Conv3D(4,(2,3,3), activation=ativa)(image_input)
    #c3d_layer1 = keras.layers.Activation(activation=ativa)(c3d_layer1) #another way
    c3d_pooling1 = keras.layers.MaxPooling3D((1,2,2))(c3d_layer1)
    c3d_layer2 = keras.layers.Conv3D(8,(4,3,3), activation=ativa)(c3d_pooling1)
    c3d_pooling2 = keras.layers.MaxPooling3D((2,2,2))(c3d_layer2)
    c3d_layer3 = keras.layers.Conv3D(16,(8,3,3), activation=ativa)(c3d_pooling2)
    c3d_pooling3 = keras.layers.MaxPooling3D((4,2,2))(c3d_layer3)
    #c3d_layer4 = keras.layers.Conv3D(32,(2,3,3), activation=activa)(c3d_pooling3)
    #c3d_pooling4 = keras.layers.MaxPooling3D((2,2,2))(c3d_layer4)
    
    feature_conv_4 = keras.layers.Lambda(all_operations)(c3d_pooling3) #flatten spatial features to time series
    
    lstm1 = keras.layers.LSTM(1024,input_shape=(1200,feature_conv_4.shape[2]), return_sequences=True)(feature_conv_4)
    #lstm2 = keras.layers.LSTM(512, return_sequences=True)(lstm1)
    global_feature = keras.layers.GlobalMaxPooling1D()(lstm1)
    
    #ADD THE AUDIO FEATURE HERE 
    
    dense_1 = keras.layers.Dense(128, activation=ativa)(global_feature)
    #dense_2 = keras.layers.Dense(13, activation='sigmoid')(dense_1)
    soft_max = keras.layers.Dense(1, activation='sigmoid')(dense_1)
    
    model = keras.Model(inputs=[image_input], outputs=[soft_max])
    #model.summary()
   
   
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
                    loss= 'binary_crossentropy', 
                    #loss_weights = class_weights,
                    #metrics=['accuracy']
                    metrics=METRICS)
    
    print("\n\t",params,"\n\n\tOPTIMA",optima,"\n\tATIVA",ativa)
    
    return model


def init_test_model(params,run=None):
    ''' returns model with params in config present in name over aux.WEIGHTS_PATH'''
    
    model = form_model(params)
    
    find_string=[params["ativa"]+'_'+params["optima"]+'_'+str(params["batch_type"])+'_'+params["frame_max"]]
    para_file_name, para_file_path = find_h5(aux.WEIGHTS_PATH,find_string,ruii=False)
    
    model.load_weights(para_file_path[0])
    
    print("\n\tWEIGHTS from ", '/'+os.path.split(os.path.split(para_file_path[0])[0])[1]+'/'+os.path.split(para_file_path[0])[1])
    if run:run["test/model_name"] = para_file_name[0]
    
    return model , para_file_name[0]



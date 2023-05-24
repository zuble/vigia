import os , logging , time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers , regularizers
from tensorflow.keras.callbacks import ModelCheckpoint

from concurrent.futures import ProcessPoolExecutor

from utils import globo , sinet


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

    #gpus = tf.config.list_physical_devices('GPU')
    #print("\nNum GPUs Available: ", len(gpus))
    ##for i in range(len(gpus)) :print(str(gpus[i]))

def limit_gpu_gb(i):
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_logical_device_configuration(
        gpus[i],
        [tf.config.LogicalDeviceConfiguration(memory_limit=31744)]
    )


#--------------------------------------------------------#
## DATA GENERATION FOR TRAINING AND VALIDATION
''' APPROACH 1 w/o batch on xdv test_BG (0) and (1) '''
class DataGenFL1(tf.keras.utils.Sequence):
    def __init__(self, cfg_sinet , cfg_wav , mode = 'train' , dummy = 0 , debug = False):
 
        self.mode = mode
        if mode == 'valdt' : self.valdt = True ;  self.train = False
        elif mode == 'train': self.train = True ; self.valdt = False
        print("\n\nDataGen",mode,self.train,self.valdt)
        
        self.data = np.load(os.path.join(globo.AAS_PATH+"/full_interval",f"{cfg_sinet['sinet_version']}--fl_{self.mode}.npz"), allow_pickle=True)["data"]
        self.len_data = len(self.data)
        
        if dummy:
            self.data = self.data[:dummy]
            self.len_data = dummy
        
        #self.sinet = sinet.Sinet(CFG_SINET)
        
        self.wav_arch = cfg_wav["arch"]
        self.batch_size = cfg_wav["batch_size"]
        self.sigm_norm = cfg_wav["sigm_norm"]
        self.mm_norm = cfg_wav["mm_norm"]
        
        self.shuffle = cfg_wav["shuffle"]
        
        self.debug = debug


    def sigmoid_rescale(self,data):
    
        def sigmoid(x):return 1 / (1 + np.exp(-x))
        
        num_features = data.shape[1]
        scaled_data = np.zeros_like(data)

        for feature_idx in range(num_features):
            feature_data = data[:, feature_idx]

            # Center the data around the mean
            centered_data = feature_data - np.mean(feature_data)

            # Apply sigmoid function
            scaled_feature_data = sigmoid(centered_data)
            scaled_data[:, feature_idx] = scaled_feature_data

        return scaled_data
    
    def min_max_rescale(self,data):
        ## data is (time_steps, feature_dim)
        scaled_data = []
        
        for sample in data:
            min_val = np.min(sample, axis=0)
            max_val = np.max(sample, axis=0)
            
            # Avoid division by zero
            feature_range = max_val - min_val
            if np.isscalar(feature_range):
                if feature_range == 0:
                    feature_range = 1
            else:
                feature_range[feature_range == 0] = 1

            scaled_sample = (sample - min_val) / feature_range
            scaled_data.append(scaled_sample)

        return np.array(scaled_data)
 
 
    def __len__(self):
        if self.debug: print("\n\n__len__",self.mode,"= n batchs = ",int(np.ceil(self.len_data / float(self.batch_size))), " w/ ",self.batch_size," vid_frames each")
        return int(np.ceil(self.len_data / float(self.batch_size)))
        
        
    def __getitem__(self, idx):
        
        #p_es_arr_total = [data[k]['p_es_array'] for k in range(len(data))]
        #label_total = [data[k]['label'] for k in range(len(data))]
        
        vpath = self.data[idx]['vpath']
        fi = self.data[idx]['frame_interval']
        p_es_arr = self.data[idx]['p_es_array']
        label = self.data[idx]['label']
        label_str = 'NORMAL' if not label else 'ANOMALY'
        #print("\n",idx,os.path.basename(vpath),label,"\n",fi,"\np_es_arr @ ",np.shape(p_es_arr))
        
        
        if self.mm_norm:
            p_es_arr = self.min_max_rescale(p_es_arr)
            
        if self.sigm_norm:
            p_es_arr = self.sigmoid_rescale(p_es_arr)
            
        if self.wav_arch == 'topgurlmax':
            p_es_arr = np.max(p_es_arr , axis = 0)
        
        if np.isnan(p_es_arr).any() : print("Input data contains NaN values:")
        
        X = np.expand_dims(np.array(p_es_arr).astype(np.float32),0)
        y = np.expand_dims(np.array(label).astype(np.float32),0)
         
         
        ## prints
        if self.debug:
            print(f"\n********** {self.mode}_{idx} **** {label_str} ***************\n")
            print(  f"\n\n\n£££ {self.mode}_{self.wav_arch}_{idx} @ {os.path.basename(vpath)}\n"
                    f"    X {X.shape} @{X.dtype} , y {y} , {y.shape} @{y.dtype}\n\n")
        
        return X , y



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
    print("\nFORM_MODEL @",print(params['arch']),"\n")
   
    ''' waves coming '''
   
    #https://www.tensorflow.org/api_docs/python/tf/keras/activations
    #https://www.tensorflow.org/api_docs/python/tf/nn/leaky_relu
    if params["ativa"]=='leakyrelu': ativa = keras.layers.LeakyReLU()
    elif params["ativa"]=='gelu': ativa = gelu
    elif params["ativa"]=='relu': ativa = 'relu'
    else: raise Exception("no ativa named assim")

    ## raw proccesing of sinet output with patterning
    '''
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
    '''
    
    ## c1d or lstm
    if params['arch'] == 'c1d':
        model = tf.keras.Sequential([
            layers.Input(shape=(None, params["sinet_aas_len"]), name='input_layer'),
            layers.Conv1D(64, kernel_size=3, activation=ativa, name='conv1d_layer1'),
            layers.MaxPooling1D(pool_size=2, name='maxpool1d_layer1'),
            layers.Conv1D(128, kernel_size=3, activation=ativa, name="conv1d_layer2"),
            #layers.BatchNormalization(),
            #layers.Activation(ativa),
            layers.GlobalMaxPooling1D(name='globalmaxpooling1d_layer'),
            layers.Dropout(0.5),
            layers.Dense(64, activation=ativa, name='hidden_layer'),
            layers.Dense(1, activation='sigmoid', name='output_layer')
        ])
                
    elif params['arch'] == 'lstm' :
        model = tf.keras.Sequential([
            layers.Input(shape=(None,params["sinet_aas_len"]), name='input_layer'),
            layers.LSTM(units = params["lstm_units"], activation=ativa, return_sequences=True, name='lstm_layer'),
            layers.GlobalMaxPooling1D(),
            #layers.Dense(128, activation=ativa, kernel_regularizer=regularizers.l1(0.01) , name='hidden_layer'),
            #layers.Dropout(0.5),
            layers.Dense(64, activation=ativa, name='hidden_layer2'),
            layers.Dense(1, activation='sigmoid', name='output_layer')
        ])
        #model = tf.keras.Sequential([
        #    layers.Input(shape=(None,params["sinet_aas_len"]), name='input_layer'),
        #    # LSTM layer
        #    layers.LSTM(units = params["lstm_units"], return_sequences=False , name='lstm_layer'),
        #    layers.Dropout(0.5),  # Dropout after LSTM layer
        #    # First Dense layer with L1 regularization
        #    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
        #    layers.Dropout(0.5),  # Dropout after first Dense layer
        #    # Last Dense layer with sigmoid activation and L1 regularization
        #    layers.Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(0.01))
        #])

    elif params['arch'] == 'topgurlmax':
        model = tf.keras.Sequential([
            layers.Input(shape=(None,params["sinet_aas_len"]), name='input_layer'),
            #layers.Lambda(lambda x: tf.reduce_max(x, axis=1), name='max_pooling'), # = np.max(input , axis = 0)
            #layers.Dense(128, activation=ativa, name='hidden_layer1'),
            #layers.Dropout(0.5),
            layers.GlobalMaxPooling1D(),
            #layers.Dense(64, activation=ativa, name='hidden_layer2'),
            #layers.Dropout(0.5),
            layers.Dense(32, activation=ativa, name='hidden_layer3'),
            layers.Dense(1, activation='sigmoid', name='output_layer')
        ])
    
    #https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
    if params["optima"]=='sgd':
        if params["lr_agenda"]:
            lr = tf.keras.optimizers.schedules.ExponentialDecay(
                params["lr"],
                decay_steps=1000,
                decay_rate=0.9,
                staircase=True
            )
        else: lr = params["lr"]
        optima = keras.optimizers.SGD(learning_rate = lr)
    elif params["optima"]=='adam':optima = keras.optimizers.Adam(learning_rate = params["lr"])
    elif params["optima"]=='adamamsgrad':optima = keras.optimizers.Adam(learning_rate = params["lr"],amsgrad=True)
    elif params["optima"]=='nadam':optima = keras.optimizers.Nadam(learning_rate = params["lr"])
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
    
    ## MODEL NAME
    time_str = str(time.time()); 
    if params["arch"] == 'lstm' : params["arch"] = 'lstm' + str(params["lstm_units"])
    if params["optima"]=='sgd' and params["lr_agenda"]: params["lr"] = str(params["lr"])+'A'
    model_name = time_str + '_'+params["ativa"]+'_'+params["optima"]+'-'+str(params["lr"])+'_'+str(params["arch"]+'_'+str(params["sinet_fi"])+'-'+str(params["sinet_fi_iter"])+'_'+str(params["batch_size"])+'bs')
   
    print(  "\n\tCFG_WAV\n\t   ","\n\t  ".join(f"{key}: {value}" for key, value in params.items()),\
            "\n\n\tOPTIMA",optima,"with lr=",params["lr"],\
            "\n\tATIVA",ativa,\
            "\n\n\tMODEL NAME\n\t",model_name)

    return model , model_name


#--------------------------------------------------------#
## CALLBACKS

def ckpt_clbk(model_name , monitor , save_best = False):
    #https://keras.io/api/callbacks/model_checkpoint/
    p = os.path.join(globo.CKPT_PATH,model_name)
    if not os.path.exists(p):
        os.makedirs(p)
    else:raise Exception(f"{p} eristes")
    
    print("\n\n\tCKPT_CLBK saving at",p)
    ckpts=[]
    for i,metric in enumerate(monitor):
        print('\t  ckpt callback monotoring',metric)
        ckpts.append ( ModelCheckpoint( filepath=p+'/'+model_name+'_'+metric+'_ckpt-{epoch:02d}-{loss:.2f}-{val_loss:.2f}.h5' , \
                            monitor=metric,\
                            save_weights_only=True,\
                            save_best_only=save_best,\
                            mode='auto',\
                            save_freq='epoch',\
                            verbose = 1) )
    return  ckpts
        

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
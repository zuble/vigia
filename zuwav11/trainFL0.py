import os, time, random, logging , datetime , cv2 , csv , subprocess , json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
print("tf",tf.version.VERSION)

#from tensorflow import keras
from tqdm.keras import TqdmCallback

from utils import globo , xdv , tfh5 , sinet


''' GPU CONFIGURATION '''
os.environ["CUDA_VISIBLE_DEVICES"]="2"
tfh5.set_tf_loglevel(logging.ERROR)
tf.debugging.set_log_device_placement(False) #Enabling device placement logging causes any Tensor allocations or operations to be printed.
#tfh5.set_memory_growth()
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
#tfh5.limit_gpu_gb(2)


''' TRAIN & VALDT '''
## its done inside DataGen
#TV1_DICT=xdv.train_valdt_test_from_xdvtest_bg_from_npy()


''' CONFIGS '''
CFG_SINET = {
    'sinet_version': 'fsd-sinet-vgg42-tlpf_aps-1',
    
    'graph_filename' : os.path.join(globo.FSDSINET_PATH,"fsd-sinet-vgg42-tlpf_aps-1.pb"),
    'metadata_file'  : os.path.join(globo.FSDSINET_PATH,"fsd-sinet-vgg42-tlpf_aps-1.json"),
    
    'audio_fs_input':22050,
    'batchSize' : 64,
    'lastPatchMode': 'repeat',
    'patchHopSize' : 50,
    
    
    'anom_labels' : ["Alarm","Boom","Crowd","Dog","Drill","Explosion","Fire","Gunshot and gunfire","Hammer","Screaming","Screech",\
                    "Shatter","Shout","Siren","Slam","Squeak","Yell"],
    'anom_labels_i' : [4,18,51,59,65,72,78,92,94,145,146,147,148,152,154,161,198],
    
    'anom_labels2' : ["Boom","Explosion","Fire","Gunshot and gunfire","Screaming",\
                    "Shout","Siren","Yell"],
    'anom_labels_i2' : [18,72,78,92,147,148,152,198],
    
    #'full_or_max' : 'max', #chose output to (timesteps,labels_total) ot (1,labels_total)
    'labels_total' : 200   
}

CFG_WAV= {
    "arch" : 'topgurlmax', #c1d , lstm , topgurlmax
    "sinet_aas_len" : CFG_SINET["labels_total"], 
    
    "sigm_norm" : False,
    "mm_norm" : False, #MinMaxNorm
    
    "shuffle" : False,
    
    "ativa" : 'relu',
    "optima" : 'adamamsgrad',
    "lr": 1e-4,
    "batch_type" : 0,   # =0 all batch have frame_max or video length // =1 last batch has frame_max frames // =2 last batch has no repetead frames
    "frame_max" : 8000,
    "ckpt_start" : f"{0:0>8}",  #used in train_model: if 00000000 start from scratch, else start from ckpt with CFG_WAV stated
    
    "epochs" : 100,
    "batch_size" : 4
}


''' Data Gerador w/o batch on xdv_test set'''
class DataGenFL0(tf.keras.utils.Sequence):
    def __init__(self, mode = 'train' , dummy = 0 , debug = False):
 
        self.mode = mode
        if mode == 'valdt' : self.valdt = True ;  self.train = False
        elif mode == 'train': self.train = True ; self.valdt = False
        print("\n\nDataGen",mode,self.train,self.valdt)
        
        self.data = np.load(os.path.join(globo.AAS_PATH,f"{CFG_SINET['sinet_version']}--fl_{self.mode}.npz"), allow_pickle=True)["data"]
        self.len_data = len(self.data)
        
        if dummy:
            self.data = self.data[:dummy]
            self.len_data = dummy
        
        #self.sinet = sinet.Sinet(CFG_SINET)
        
        self.wav_arch = CFG_WAV["arch"]
        self.batch_size = CFG_WAV["batch_size"]
        self.sigm_norm = CFG_WAV["sigm_norm"]
        self.mm_norm = CFG_WAV["mm_norm"]
        
        self.shuffle = CFG_WAV["shuffle"]
        
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

   
          
if __name__ == "__main__":
    
    ''' DATA GERADOR '''
    
    train_generator = DataGenFL0( 'train' )
    valdt_generator = DataGenFL0( 'valdt' )
    
    
    ''' MODEL WAV'''
    
    ## SINGLE
    model,model_name = tfh5.form_model_wav(CFG_WAV)
    
    ## CLBK's
    #ckpt_clbk = tfh5.ckpt_clbk(model_name)
    #early_stop_clbk = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
    tqdm_clbk = TqdmCallback(CFG_WAV["epochs"], len(train_generator), CFG_WAV["batch_size"])
    clbks = [ tqdm_clbk ]
    
    ''' FIT '''
    history = model.fit(train_generator, 
                        epochs = CFG_WAV["epochs"] ,
                        #steps_per_epoch = len(train_generator),
                        
                        verbose=2,
                        
                        validation_data = valdt_generator,
                        #validation_steps = len(valdt_fp),
                        
                        #use_multiprocessing = True , 
                        #workers = 8 #,
                        callbacks= clbks
                    )


    ''' SAVINGS '''
    #model.save(globo.MODEL_PATH + model_name + '.h5')
    #model.save(globo.MODEL_PATH + model_name )
    #
    #model.save_weights(globo.WEIGHTS_PATH + model_name + '_weights.h5')
    #
    #hist_csv_file = globo.HIST_PATH + model_name + '_history.csv'
    #with open(hist_csv_file, 'w', newline='') as file:writer = csv.writer(file);writer.writerow(history.history.keys());writer.writerows(zip(*history.history.values()))
      
      
      
      
    ## TF.DATA FROM GENERATOR
    '''
    def data_gen_wrapper(data_gen):
        for i in range(len(data_gen)):
            yield data_gen[i]
        
    output_types = (tf.float32)
    output_shapes = (
        tf.TensorShape((None , CFG_WAV["sinet_aas_len"])),
        tf.TensorShape((None,))
    )
    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_gen_wrapper(train_generator),
        output_types=output_types,
        output_shapes=output_shapes
    )
    valdt_dataset = tf.data.Dataset.from_generator(
        lambda: data_gen_wrapper(valdt_generator),
        output_types=output_types,
        output_shapes=output_shapes
    )
    '''       
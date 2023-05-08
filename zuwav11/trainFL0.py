import os, time, random, logging , datetime , cv2 , csv , subprocess , json

import matplotlib.pyplot as plt
import numpy as np
from tqdm.keras import TqdmCallback
from concurrent.futures import ProcessPoolExecutor

''' GPU CONFIGURATION '''
os.environ["CUDA_VISIBLE_DEVICES"]="2"
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'


import tensorflow as tf
print("tf",tf.version.VERSION)
#from tensorflow import keras
from tensorflow.keras import backend as K

from utils import globo , xdv , tfh5 , sinet , sinet2

tf.debugging.set_log_device_placement(False) #Enabling device placement logging causes any Tensor allocations or operations to be printed.
tfh5.set_tf_loglevel(logging.ERROR)
#tfh5.set_memory_growth()
#tfh5.limit_gpu_gb(2)

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
    
    'full_or_max' : 'full', #chose output to (timesteps,labels_total) ot (1,labels_total)
    'labels_total' : 200   
}

CFG_WAV= {
    "arch" : 'lstm', #c1d , lstm , topgurlmax
    "sinet_aas_len" : CFG_SINET["labels_total"], 
    "timesteps" : 8,
    
    "sigm_norm" : False,
    "mm_norm" : False, #MinMaxNorm
    
    "shuffle" : False,
    
    "ativa" : 'relu',
    "optima" : 'adam',
    "lr": 1e-4,
    "batch_type" : 0,   # =0 all batch have frame_max or video length // =1 last batch has frame_max frames // =2 last batch has no repetead frames
    "frame_max" : 8000,
    "ckpt_start" : f"{0:0>8}",  #used in train_model: if 00000000 start from scratch, else start from ckpt with CFG_WAV stated
    
    "epochs" : 50,
    "batch_size" : 8
}



def simple_test_function(i):
    return i, i * 2


def call_get_sigmoid(model, model_config, metadata, i, vpath, start_frame, end_frame, label, debug):
    try:
        print("\t call for ", i, os.path.basename(vpath), start_frame, end_frame)
        result = i, sinet2.get_sigmoid(model, model_config, metadata, vpath, start_frame, end_frame, debug=debug) , label
    except Exception as e:
        print(f"Exception in call_get_sigmoid for index {i}: {e}")
        result = i, None, label
    return result
    
    
''' APPROACH 2 w/ batch on xdv test_BG (1) + train_A (0) '''
class DataGenFL2(tf.keras.utils.Sequence):   
    def __init__(self, cfg_sinet , cfg_wav , mode = 'train' , dummy = 0 , debug = False):
 
        self.mode = mode
        if mode == 'valdt' : self.valdt = True ;  self.train = False
        elif mode == 'train': self.train = True ; self.valdt = False
        print()
        
        data = np.load(os.path.join(globo.SERVER_TEST_COPY_PATH,'npy/dataset_from_xdvtest_bg_train_a_data.npy'), allow_pickle=True).item()
        if dummy:
            self.data = data[self.mode][:dummy]
            self.len_data = dummy
        else:
            self.data = data[self.mode]
            self.len_data = len(self.data)
        print("\n\nDataGen",mode,self.train,self.valdt,"\n",np.shape(self.data)[0],\
            "\n\tNORMAL intervals", sum(1 for _, (_, _, label) in self.data if label == 0),\
            "\n\tABNORMAL intervals", sum(1 for _, (_, _, label) in self.data if label == 1),"\n\n")
        
        
        self.wav_arch = cfg_wav["arch"]
        self.timesteps = cfg_wav["timesteps"]
        self.batch_size = cfg_wav["batch_size"]
        self.sigm_norm = cfg_wav["sigm_norm"]
        self.mm_norm = cfg_wav["mm_norm"]
        self.shuffle = cfg_wav["shuffle"]
        
        self.debug = debug

        self.cfg_sinet = cfg_sinet
        ## SingleP
        self.sinet = sinet.Sinet(cfg_sinet)
        ## MultiP
        #self.sinet_model, self.sinet_metadata = sinet2.create_sinet(cfg_sinet)
        
        #test_p_es_array = self.sinet.get_sigmoid(self.data[0][0], self.data[0][1][:2][0], self.data[0][1][:2][1])
        #self.model_wav_shape = np.shape(test_p_es_array)
        #print("test sinet call shape",self.model_wav_shape)


    def __len__(self):
        if self.debug: print("\n\n__len__",self.mode,"= n batchs = ",int(np.ceil(self.len_data / float(self.batch_size))), " w/ ",self.batch_size," vid_frames each")
        return int(np.ceil(self.len_data / float(self.batch_size)))
        

    def __getitem__(self, batch_idx):
        batch_size = self.batch_size
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        
        # Clamp end_idx to the length of self.data
        end_idx = min(end_idx, len(self.data))
        
        '''
        current_batch_size = end_idx - start_idx
        
        vpath_list , sf_list , ef_list , label_list = [] , [] , [] , []
        for idx in range(start_idx, end_idx):
            vpath = self.data[idx][0]
            sf, ef = self.data[idx][1][:2]
            label = self.data[idx][1][2]
            
            vpath_list.append(vpath)
            sf_list.append(sf)
            ef_list.append(ef)
            label_list.append(label)
            
            if self.debug : print("\n__get_item__\n",idx,"label",label,os.path.basename(vpath),sf,ef)
            
                
        #with ProcessPoolExecutor() as executor:
        #    results = list(executor.map(lambda i: call_get_sigmoid(i, vpath_list[i], sf_list[i], ef_list[i], label_list[i], True), range(current_batch_size)))
        
        #results = list(map(lambda i: call_get_sigmoid(self.sinet, i, vpath_list[i], sf_list[i], ef_list[i], label_list[i], True), range(current_batch_size)))
        
        
        #with ProcessPoolExecutor(max_workers = 2) as executor:
        #    results = list(executor.map(lambda i: call_get_sigmoid(self.sinet_model, self.cfg_sinet , self.sinet_metadata, i, vpath_list[i], sf_list[i], ef_list[i], label_list[i], True), range(current_batch_size), timeout=20))
        results = [call_get_sigmoid(self.sinet_model, self.cfg_sinet, self.sinet_metadata, i, vpath_list[i], sf_list[i], ef_list[i], label_list[i], True) for i in range(current_batch_size)]
        
        
        #with ProcessPoolExecutor() as executor:
        #    results = list(executor.map(simple_test_function, range(current_batch_size)))
        #print(results)
        
    
        X_aux, y_aux = [] , []
        for i, p_es_i , label_i in results:
            
            if self.wav_arch == 'topgurlmax':
                p_es_i = np.max(p_es_i , axis = 0)
                
            X_aux.append(p_es_i)
            y_aux.append(label_i)
            
        X_batch = np.stack(X_aux, axis=0)  # Shape: (batch_size, timesteps, params["sinet_aas_len"])
        y_batch = np.array(y_aux).astype(np.float32).reshape(-1, 1)  # Shape: (batch_size, 1)    
        if self.debug: print(  f"\n\tBATCH @ data_gen after\n\tX {X_batch.shape} @{X_batch.dtype} , y {y_batch} , {y_batch.shape} @{y_batch.dtype}\n\n")
        
        return X_batch , y_batch
        '''
        
        
        X_aux, y_aux = [] , []
        for idx in range(start_idx, end_idx):
            vpath = self.data[idx][0]
            sf, ef = self.data[idx][1][:2]
            label = self.data[idx][1][2]
            
            p_es_arr = self.sinet.get_sigmoid(vpath, sf, ef, debug=True)

            if self.wav_arch == 'topgurlmax':
                p_es_arr = np.max(p_es_arr , axis = 0)
            
            X_aux.append(p_es_arr)
            y_aux.append(label)
            
            if self.debug : print("\n__get_item__\n",idx,"label",label,os.path.basename(vpath),sf,ef,\
                                    "\np_es_arr",np.shape(p_es_arr))
            
        X_batch = np.stack(X_aux, axis=0)  # Shape: (batch_size, timesteps, params["sinet_aas_len"])
        y_batch = np.array(y_aux).astype(np.float32).reshape(-1, 1)  # Shape: (batch_size, 1)
    
        if self.debug: 
            print(f"\nEND BATCH\n    X {X_batch.shape} @{X_batch.dtype} , y {y_batch} , {y_batch.shape} @{y_batch.dtype}\n\n")
        
        return X_batch, y_batch
        


if __name__ == "__main__":
    
    ''' DATA GERADOR '''
    train_generator = DataGenFL2( CFG_SINET , CFG_WAV , 'train' , debug = True)
    valdt_generator = DataGenFL2( CFG_SINET , CFG_WAV , 'valdt' , debug = True)
    
    K.clear_session()
    
    ''' MODEL WAV'''
    model,model_name = tfh5.form_model_wav(CFG_WAV)
    
    ## CLBK's
    #ckpt_clbk = tfh5.ckpt_clbk(model_name)
    #early_stop_clbk = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
    tqdm_clbk = TqdmCallback(CFG_WAV["epochs"], len(train_generator), CFG_WAV["batch_size"])
    clbks = [ tqdm_clbk ]
    
    ''' FIT '''
    history = model.fit(train_generator, 
                        epochs = CFG_WAV["epochs"] ,
                        steps_per_epoch = len(train_generator),
                        
                        verbose=2,
                        
                        validation_data = valdt_generator,
                        validation_steps = len(valdt_generator),
                        
                        #use_multiprocessing = True , 
                        #workers = 16 ,
                        callbacks= clbks
                    )


    ''' SAVINGS '''
    #model.save(globo.MODEL_PATH + model_name + '.h5')
    #model.save(globo.MODEL_PATH + model_name )
    
    model.save_weights(globo.WEIGHTS_PATH + model_name + '_weights.h5')
    
    hist_csv_file = globo.HIST_PATH + model_name + '_history.csv'
    with open(hist_csv_file, 'w', newline='') as file:writer = csv.writer(file);writer.writerow(history.history.keys());writer.writerows(zip(*history.history.values()))
      
      
      
      
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

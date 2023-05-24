from utils import globo , tfh5 , xdv
#from utils import sinet , sinet2

import os, time, random, logging , csv

import numpy as np
from tqdm.keras import TqdmCallback
#from concurrent.futures import ProcessPoolExecutor


import tensorflow as tf
print("tf",tf.version.VERSION)
#from tensorflow import keras
from tensorflow.keras import backend as K

tf.debugging.set_log_device_placement(False) #Enabling device placement logging causes any Tensor allocations or operations to be printed.
tfh5.set_tf_loglevel(logging.ERROR)
#tfh5.set_memory_growth()
#tfh5.limit_gpu_gb(2)



''' APPROACH 2 w/ batch on xdv test_BG (1) + train_A (0) '''
class DataGenFL2(tf.keras.utils.Sequence):   
    def __init__(self, mode = 'train' , dummy = 0 , debug = False):
        
        self.debug = debug
        self.mode = mode
        if mode == 'valdt' : self.valdt = True ;  self.train = False
        elif mode == 'train': self.train = True ; self.valdt = False
        
        
        data = xdv.load_train_valdt_test_from_xdvtest_bg_train_a(self.mode , globo.CFG_WAV_TRAIN["sinet_fi_iter"])
        if dummy:
            self.data = data[:dummy]
            self.len_data = dummy
        else:
            self.data = data
            self.len_data = len(self.data)
        print("\n\nDataGen",mode,self.train,self.valdt,"\n",np.shape(self.data)[0],\
            "\n\tNORMAL intervals", sum(1 for i in range(len(data)) if data[i]["label"] == 0 ),\
            "\n\tABNORMAL intervals", sum(1 for i in range(len(data)) if data[i]["label"] == 1 ),"\n\n")
        
        
        self.wav_arch = globo.CFG_WAV_TRAIN["arch"]
        self.batch_size = globo.CFG_WAV_TRAIN["batch_size"]

        self.sigm_norm = globo.CFG_WAV_TRAIN["sigm_norm"]
        self.mm_norm = globo.CFG_WAV_TRAIN["mm_norm"]
        self.anom_filter = globo.CFG_WAV_TRAIN["anom_filter"]
        
        self.shuffle = globo.CFG_WAV_TRAIN["shuffle"]
        
        
        #self.globo.CFG_SINET = globo.CFG_SINET
        
        ## SingleP
        #self.sinet = sinet.Sinet(globo.CFG_SINET)
        ## MultiP
        #self.sinet_model, self.sinet_metadata = sinet2.create_sinet(globo.CFG_SINET)
        
        #test_p_es_array = self.sinet.get_sigmoid(self.data[0][0], self.data[0][1][:2][0], self.data[0][1][:2][1])
        #self.model_wav_shape = np.shape(test_p_es_array)
        #print("test sinet call shape",self.model_wav_shape)



    
    def enhance_probabilities(self, probs, power=2):
        return np.where(probs < 0.5, (probs ** power), 1 - (1 - probs) ** power)

    
    def __len__(self):
        if self.debug: print("\n\n__len__",self.mode,"= n batchs = ",int(np.ceil(self.len_data / float(self.batch_size))), " w/ ",self.batch_size," vid_frames each")
        return int(np.ceil(self.len_data / float(self.batch_size)))
    
    def on_epoch_end(self):
        if self.shuffle : 
            print("\nSHUFFLING\n")
            np.random.shuffle(self.data)    

    def __getitem__(self, batch_idx):
        batch_size = self.batch_size
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        
        # Clamp end_idx to the length of self.data
        end_idx = min(end_idx, len(self.data))
        
        X_aux, y_aux = [] , []
        for idx in range(start_idx, end_idx):
            
            ## pre processed arrays
            vpath = self.data[idx]['vpath']
            sf = self.data[idx]['sf']
            ef = self.data[idx]['ef']
            p_es_arr = self.data[idx]['p_es_array']
            label = self.data[idx]['label']
            
            ## live usage 
            #p_es_arr = self.sinet.get_sigmoid(vpath, sf, ef, debug=True)
            
            
            #p_es_arr = self.enhance_probabilities(p_es_arr)
            
            if self.anom_filter:
                p_es_arr = p_es_arr[:, globo.CFG_SINET["anom_labels_i2"]]
            
            #if self.wav_arch == 'topgurlmax':
            #    p_es_arr = np.max(p_es_arr , axis = 0)
            
            X_aux.append(p_es_arr)
            y_aux.append(label)
            
            if self.debug : 
                print("\n__get_item__",idx,"\nlabel",label,'\n',os.path.basename(vpath),'\n',sf,ef,\
                                    "\np_es_arr",np.shape(p_es_arr))
                print("explosion",np.max(p_es_arr , axis = 0)[72],"gun",np.max(p_es_arr , axis = 0)[92])
                
        X_batch = np.stack(X_aux, axis=0)  # Shape: (batch_size, timesteps, params["sinet_aas_len"])
        y_batch = np.array(y_aux).astype(np.float32).reshape(-1, 1)  # Shape: (batch_size, 1)
    
        if self.debug: 
            print(f"\n£££ END BATCH {batch_idx} £££\n    X {X_batch.shape} @{X_batch.dtype}\n    y {[y_batch[iii][0] for iii in range(len(y_batch))]} , {y_batch.shape} @{y_batch.dtype}\n\n")
        
        return X_batch, y_batch
        


if __name__ == "__main__":
    
    ''' DATA GERADOR '''
    train_generator = DataGenFL2( 'train' )
    valdt_generator = DataGenFL2( 'valdt' )
    
    #K.clear_session()
    
    ''' MODEL WAV '''
    model,model_name = tfh5.form_model_wav(globo.CFG_WAV_TRAIN)
    
    ''' CLBK's '''
    #ckpt_clbk1 , ckpt_clbk2 = tfh5.ckpt_clbk(model_name,['loss','val_loss'],True)
    early_stop_clbk = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tqdm_clbk = TqdmCallback(globo.CFG_WAV_TRAIN["epochs"], len(train_generator), globo.CFG_WAV_TRAIN["batch_size"])
    clbks = [ tqdm_clbk , early_stop_clbk ] # ckpt_clbk1 , ckpt_clbk2 ,
    
    ''' FIT '''
    history = model.fit(train_generator, 
                        epochs = globo.CFG_WAV_TRAIN["epochs"] ,
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
        tf.TensorShape((None , globo.CFG_WAV_TRAIN["sinet_aas_len"])),
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

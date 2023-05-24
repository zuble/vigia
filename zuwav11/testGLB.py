import os, time, random, logging , datetime , cv2 , csv , subprocess , json

import matplotlib.pyplot as plt
import numpy as np
from tqdm.keras import TqdmCallback
from concurrent.futures import ProcessPoolExecutor


import tensorflow as tf
print("tf",tf.version.VERSION)
#from tensorflow import keras
from tensorflow.keras import backend as K

from utils import globo , xdv , tfh5 
from utils import sinet , rslts

tf.debugging.set_log_device_placement(False) #Enabling device placement logging causes any Tensor allocations or operations to be printed.
tfh5.set_tf_loglevel(logging.ERROR)
#tfh5.set_memory_growth()
#tfh5.limit_gpu_gb(2)



''' APPROACH 2 w/ batch on xdv test_BG (1) + train_A (0) '''
class DataTest():   
    def __init__(self, model , dummy = 0 , debug = False , watch = False):
        
        self.model = model
        self.mode = 'test'
        self.watch = watch
        self.debug = debug
        
        npz_path = os.path.join(globo.AAS_PATH+"/2_sub_interval",f"{globo.CFG_SINET['sinet_version']}--fl2_{self.mode}-{globo.CFG_SINET['chunck_fsize']}fi.npz")
        if not os.path.exists(npz_path):
            print("npz no existe:", npz_path)
            return 
        data = np.load(npz_path, allow_pickle=True)["data"]
        if dummy:
            self.data = data[:dummy]
            self.len_data = dummy
        else:
            self.data = data
            self.len_data = len(self.data)
        print("\n\nDataGen",self.mode,np.shape(self.data)[0],\
            "\n\tNORMAL intervals", sum(1 for i in range(len(data)) if data[i]["label"] == 0 ),\
            "\n\tABNORMAL intervals", sum(1 for i in range(len(data)) if data[i]["label"] == 1 ),"\n\n")
        
        
        self.wav_arch = globo.CFG_WAV_TEST["arch"]
       
        self.CFG_SINET = globo.CFG_SINET
        self.sinet = sinet.Sinet(globo.CFG_SINET)

        ## watch stuff
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.5;self.thickness = 1;self.lineType = cv2.LINE_AA
        self.green = (0, 255, 0);self.red = (0, 0, 255)
    
    
    def watch_this_interval(self,idx,aas):
        vpath = self.data[idx]['vpath']
        sf = self.data[idx]['sf']
        ef = self.data[idx]['ef']
        label = self.data[idx]['label']
        
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            print("Error opening video file")
            return

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:break

            frame_count += 1
            if sf <= frame_count <= ef:
                cv2.putText(frame, str(label), (10, 30), self.font, self.fontScale, self.green, self.thickness, self.lineType)
                cv2.putText(frame, str(aas), (10, 60), self.font, self.fontScale, self.red, self.thickness, self.lineType)
                cv2.imshow('Video', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):break

        cap.release()
        cv2.destroyAllWindows()
        

    @tf.function
    def aas_predict(self,x):
        return self.model(x, training=False)    

    def run(self):
        
        aas_list = []
        for idx in range(self.len_data):
            print("\n################ START test.run idx",idx,"##############")
            
            vpath = self.data[idx]['vpath']
            sf = self.data[idx]['sf']
            ef = self.data[idx]['ef']
            p_es_arr = self.data[idx]['p_es_array']
            label = self.data[idx]['label']
            
            
            t0 = time.time()
            
            ## LIVE
            #p_es_arr = self.sinet.get_sigmoid(vpath, sf, ef, debug=True)
            
            ## FROM .npz
            tf_out = self.aas_predict(np.expand_dims(p_es_arr, 0))
            tf_out = tf.constant(tf_out)
            aas = tf_out.numpy().item()

            t1 = time.time()
            aas_list.append(aas) 
            
            
            #if self.wav_arch == 'topgurlmax':
            #    p_es_arr = np.max(p_es_arr , axis = 0)

            if self.debug : print(  "\n",idx,os.path.basename(vpath),sf,ef,\
                                    "\np_es_arr",np.shape(p_es_arr),\
                                    "\nlabel",label,\
                                    "\naas",aas,\
                                    '\nin',format(t1-t0,f".{3}f"),"secs",\
                                    "\n################ END test.run idx",idx,"##############")
        
        
        label_list = [ self.data[i]["label"] for i in range(self.len_data)]
        assert len(aas_list) == len(label_list)

        if self.watch:
            for idz in range(self.len_data):
                label = label_list[idz] 
                aas = aas_list[idz]
                if (label == 0 and aas > 0.5) or (label == 1 and aas < 0.5):
                    self.watch_this_interval(idz,aas)
            
        rslts.get_cm_accuracy_precision_recall_f1(model_name,label_list,aas_list,plot=True)

if __name__ == "__main__":
      
    ''' MODEL WAV'''
    model,model_name = tfh5.form_model_wav(globo.CFG_WAV_TEST)
    model.load_weights('/raid/DATASETS/.zuble/vigia/zuwav11/model/weights/1683578246.295987_relu_adam-0.0001_lstm_weights.h5')

    test = DataTest(model,debug=True)
    test.run()

    
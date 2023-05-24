from utils import globo , rslts
import os , glob , cv2 
import numpy as np
import multiprocessing as mp

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input


VALDT_1 = '/raid/DATASETS/anomaly/RWF-2000/val/Fight'
VALDT_0 = '/raid/DATASETS/anomaly/RWF-2000/val/NonFight'

IN_SHAPES = {
    "original":     (120, 160, 3),
    "resnet50":     (224, 224, 3),
    "mobilenetv2":  (224, 224, 3),  #(224, 224, 3)
    "xception":     (150, 150, 3)   #(299, 299, 3)
}
BACKBONE = 'original'  


def process_video(args):
    label, vpath, injector = args
    frames, fn, label = injector.vinject(label, vpath)
    return label, fn, frames

class TestRWD():
    def __init__(self):
        self.vpaths1 = sorted(glob.glob(os.path.join(VALDT_1, "*.avi")))
        self.vpaths0 = sorted(glob.glob(os.path.join(VALDT_0, "*.avi")))
        self.len1=len(self.vpaths1)
        self.len0=len(self.vpaths0)
        print(f'\n{self.len1} Abnormal videos\n{self.len0} Normal videos\n')
        
        
        self.in_height = IN_SHAPES[BACKBONE][0]
        self.in_width = IN_SHAPES[BACKBONE][1]
        print("\nRESOLUTION", self.in_height , self.in_width)
        
        self.data , self.fn , self.labels , self.predictions = [] , [] , [] , []
        self.allocate_videos()
        
        
    def allocate_videos(self):
        """ Inject frames in memory """
        with mp.Pool(mp.cpu_count()) as pool:
            results1 = pool.map(process_video, [(1, vpath1, self) for vpath1 in self.vpaths1])
            results0 = pool.map(process_video, [(0, vpath0, self) for vpath0 in self.vpaths0])
        
        for i, (label, filename, frames) in enumerate(results1):
            self.data.append(frames)
            self.fn.append(filename)
            self.labels.append(label)
            print(f'Alloced ABNORMAL {i} {np.shape(frames)}')

        for j, (label, filename, frames) in enumerate(results0):
            self.data.append(frames)
            self.fn.append(filename)
            self.labels.append(label)
            print(f'Alloced NORMAL {j} {np.shape(frames)}')

        print(f'Data length: {len(self.data)}, FN length: {len(self.fn)}, Labels length: {len(self.labels)}')
    
    def vinject(self,label,vpath):
        
        frames = []
        vid = cv2.VideoCapture(vpath)
        while True:
            sucess, frame = vid.read()
            if not sucess:break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.in_width, self.in_height))
            frame = self.prep_fx(np.array(frame))
            #minn , maxx = np.min(frame, axis=(0, 1)) , np.max(frame, axis=(0, 1))
            #print("Min Max pixel values (R, G, B):", minn, maxx)
            frames.append(frame)
        vid.release()
        
        frames = np.expand_dims(np.array(frames) , 0)

        return frames, os.path.basename(vpath), label
        
        
    def prep_fx(self,x): 
        if BACKBONE == "original":      return (x/255.0).astype(np.float32)     ## uint8 [0,255] > float32 [0,1]
        elif BACKBONE == "mobilenetv2": return mobilenet_v2_preprocess_input(x) ## uint8 [0,255] > float32 [-1,1]
        elif BACKBONE == "xception":    return xception_preprocess_input(x)     ## uint8 [0,255] > float32 [-1,1]
        else: raise Exception("no backbone named assim")


    @tf.function
    def predict(self,x):
        return MODEL(x, training=False) 
    
    def runner(self):
        
        for k in range(len(self.data)):
            out = self.predict(self.data[k])
            out = tf.constant(out)
            out = out.numpy().item()
            self.predictions.append(out)
            print(f'\n*** {k} - {self.fn[k]} ***\n{self.labels[k]} .... {out}')

        rslts.get_cm_accuracy_precision_recall_f1(MODEL_NAME,self.labels,self.predictions,plot=True)


rwd = TestRWD()

pf = '/raid/DATASETS/.zuble/vigia/zurgb00/model/model'
models_saved_fodlers = [folder for folder in os.listdir(pf) if os.path.isdir(os.path.join(pf, folder))]
for model_folder in models_saved_fodlers:
    
    MODEL_NAME = model_folder     
    MODEL = tf.saved_model.load(os.path.join(pf,MODEL_NAME))
    print("\n",MODEL_NAME,"\n")
    rwd.runner()
    

from utils import globo , xdv

import keras
import cv2 as cv 
import numpy as np , os , random

fn, labels, tframes = xdv.load_test_npy()

## how data is feed to train feature extractor on UCF-101
## from tfm-anomaly-detection/proposal/video_data_generator.py
def openframe(fp):
    """Append ORIGNALS frames in memory, transformations are made on the fly"""
    
    nbframe = 16
    
    frames = []
    fn = os.path.basename(fp)
    vid = cv.VideoCapture(fp)

    while True:
        grabbed, frame = vid.read()
        if not grabbed: break
        #frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.resize(frame, (globo.CFG_RGB_TRAIN["in_width"] , globo.CFG_RGB_TRAIN["in_height"]))
        frames.append(frame)
        
    step = len(frames)//nbframe
    print("\n\n####",fn,"\n",len(frames) , "with step" , step)
    frames = frames[::step]
    if len(frames) >= nbframe:
        frames = frames[:nbframe]
        
    print(len(frames))
    
    for frame in frames:
        cv.imshow("nn", frame)
        key = cv.waitKey(500)  
        if key == ord('q'): break  # quit
        if key == ord(' '):  # pause
            while True:
                key = cv.waitKey(1)
                if key == ord(' '):break
                
    # add frames in memory
    frames = np.array(frames, dtype=np.float32)
    frames = keras.applications.xception.preprocess_input(frames)
    if len(frames) == nbframe:
        print("prep_frmaes_shape",np.shape(frames))
    else:
        print(f'\n{fn} has not enought frames {len(frames)}' )
      
for i in range(len(fn)):
    #if "label_A" not in f:
    openframe(fn[i+100])    
    
    

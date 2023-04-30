import json , time, os

import numpy as np
import essentia.standard as es
import moviepy.editor as mp
#import tensorflow as tf

from utils import util, globo, xdv

test_config = {
    
    'anom_labels' : ["Alarm","Boom","Crowd","Dog","Drill","Explosion","Fire","Gunshot and gunfire","Hammer","Screaming","Screech",\
                    "Shatter","Shout","Siren","Slam","Squeak","Yell"],
    'anom_labels_i' : [4,18,51,59,65,72,78,92,94,145,146,147,148,152,154,161,198],
    
    'anom_labels2' : ["Boom","Explosion","Fire","Gunshot and gunfire","Screaming",\
                    "Shout","Siren","Yell"],
    'anom_labels_i2' : [18,72,78,92,147,148,152,198],
    
    "audio_fs_input":22050
}


model_config = {
    
    'graph_filename' : os.path.join(globo.MODEL_PATH,"fsd-sinet-vgg42-tlpf_aps-1.pb"),
    'metadata_file'  : os.path.join(globo.MODEL_PATH,"fsd-sinet-vgg42-tlpf_aps-1.json"),
    
    'batchSize' : 64,
    'lastPatchMode': 'repeat',
    'patchHopSize' : 50
    
}


''' MODEL & METADATA '''
model = es.TensorflowPredictFSDSINet(
            graphFilename = model_config['graph_filename'],
            batchSize = model_config["batchSize"],
            lastPatchMode = model_config["lastPatchMode"],
            patchHopSize = model_config["patchHopSize"]
                                    )
metadata = json.load(open(model_config['metadata_file'], "r"))



aas_aar_dict = {}
train_mp4_paths = xdv.load_train_copy();print("\n  train_mp4_paths",np.shape(train_mp4_paths))
test_mp4_paths = xdv.load_test_copy() ; print('\n  test_mp4_paths',np.shape(test_mp4_paths))


paths = train_mp4_paths
tt = 0.0;count=0
for fp in paths:
    print('\n#-------------------#',count,'%--------------------#\n',fp)
    
    ## FS converter 
    mp4_fs_aac = util.print_acodec_from_mp4([fp],only_sr=True);print("mp4_fs_aac",mp4_fs_aac)
    resampler = es.Resample(inputSampleRate=mp4_fs_aac, outputSampleRate=test_config['audio_fs_input'])
    
    ## process aud data
    tt1=time.time()
    
    audio_es = mp.AudioFileClip(filename=fp,fps=mp4_fs_aac)
    aud_arr = audio_es.to_soundarray()
    aud_arr_mono_single = np.mean(aud_arr, axis=1).astype(np.float32)
    aud_arr_essentia = resampler(aud_arr_mono_single) 
    
    ## predict
    p_es = model(aud_arr_essentia)
    
    tt2=time.time();tt+=tt2-tt1
    print(np.shape(p_es),'@',str(tt2-tt1),"expls",np.amax(p_es[:,72])) 
    
    aas_aar_dict[fp] = np.asarray(p_es)
    
    
    print("\nMAX aas for anom labels 2")
    for i in range(len(test_config['anom_labels_i2'])):
        label_i = test_config['anom_labels_i2'][i]
        print(label_i,test_config['anom_labels2'][i],np.amax(np.asarray(p_es)[:,label_i]))
        
    count+=1
           

# Save all the arrays to a single file
#np.save("/raid/DATASETS/.zuble/vigia/zuwav/xdv_aas/xdvtrain_aases_tlpf_aps_full.npy", aas_aar_dict)

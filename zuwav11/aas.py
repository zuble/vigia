import os , numpy as np

from utils import globo , xdv , sinet
    
CFG_SINET = {
    'sinet_version': 'fsd-sinet-vgg42-tlpf_aps-1',
    'sinet_v': 'sinet42tlpf_aps',
    
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
    
    'full_or_max' : 'max', #chose output to (timesteps,labels_total) ot (1,labels_total)
    'labels_total' : 200
    
}

data=xdv.train_valdt_test_from_xdvtest_bg_from_npy(True)

sinet = sinet.Sinet(CFG_SINET)

data_dicts= {'train': [], 'valdt': [], 'test': []}

for i in range(len(data)): 
            
    if not i:typee = "train"
    elif i == 1:typee = "valdt"
    else: typee = "test"
    print("\n*****",typee,"******************\n")
    
    total_intervals = 0
    for video_j in range(len(data[typee])):
        line = data[typee][video_j]
        vpath = data[typee][video_j][0]
        frame_intervals = data[typee][video_j][1] #(sf,ef,label)
        
        total_intervals += len(frame_intervals)
        
        p_es_array = sinet.get_sigmoid_fl(vpath,frame_intervals)
        
        # Store the vpath, frame_interval, p_es_array, and label for each interval
        for k in range(len(p_es_array)):
            data_dicts[typee].append({
                'vpath': vpath,
                'frame_interval': frame_intervals[k],
                'p_es_array': p_es_array[k],
                'label': frame_intervals[k][2]
            })
            print(vpath,frame_intervals[k],np.shape(p_es_array[k]),frame_intervals[k][2])
        
        #print("\n\n\n")
        #for k in range(len(p_es_array)):
        #    print("interval",k,"@",np.shape(p_es_array[k]),frame_intervals[k][2])
        
    print("\n\n\n***********************\n",total_intervals,len(data_dicts[typee]))
        
# Save the data for each typee to a single .npz file
for typee in data_dicts:
    ofn = f"{CFG_SINET['sinet_v']}-fl-{typee}.npz"
    ofp = os.path.join('/raid/DATASETS/.zuble/vigia/zuwav11/aas',ofn)
    print(ofn,'\n',ofp)
    # Save the vpath, frame_interval, p_es_array, and label as .npz files
    np.savez_compressed(ofp, data=data_dicts[typee])
import os , numpy as np

from utils import globo , xdv , sinet

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
    
    'full_or_max' : 'max', #chose output to (timesteps,labels_total) ot (1,labels_total)
    'labels_total' : 200
    
}

frame_intervals , time_intervals = xdv.get_frame_intervals_chuncked_from_test_bg()
data = xdv.divide_into_train_valdt_test(frame_intervals)


sinett = sinet.Sinet(CFG_SINET)

data_pes_arr = {'train': [], 'valdt': [], 'test': []}

for i in range(len(frame_intervals)):
    vpath = frame_intervals[i][0]
    fi = frame_intervals[i][1]

    p_es_arr = sinett.get_sigmoid_fl(vpath,fi,debug=True)
    print(len(p_es_arr),len(fi))
    
    # Store the vpath, frame_interval, p_es_array, and label for each interval
    for k in range(len(p_es_arr)):
        data_pes_arr['train'].append({
            'vpath': vpath,
            'frame_interval': fi[k],
            'p_es_array': p_es_arr[k],
            'label': fi[k][2]
        })
        print(vpath,fi[k],np.shape(p_es_arr[k]),fi[k][2])
    
    break
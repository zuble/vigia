import os , numpy as np

from utils import globo , sinet

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

data = np.load(os.path.join(globo.AAS_PATH+"/2_sub_interval/",'dataset_from_xdvtest_bg_train_a_data.npy'), allow_pickle=True).item()

sinett = sinet.Sinet(CFG_SINET)

data_dict = {'train': [], 'valdt': [], 'test': []}
for j in range(len(data_dict)):

    if not j:typee = "train"
    elif j == 1:typee = "valdt"
    else: typee = "test"
    
    print("\n\n*****",typee,"******************\n")
    
    for i in range(len(data[typee])):
        vpath = data[typee][i][0]
        sf, ef = data[typee][i][1][:2]
        label = data[typee][i][1][2]

        p_es_arr = sinett.get_sigmoid(vpath, sf, ef, debug=True)
        
        data_dict[typee].append({
            'vpath': vpath,
            'sf': sf,
            'ef': ef,
            'p_es_array': p_es_arr,
            'label': label
        })
        print("\ndata_dict\n",i,"\n",vpath,"\n",sf,ef,"\n",np.shape(p_es_arr),"\nlabel",label)
       

    print("\n\n***********************\n",len(data[typee]), len(data_dict[typee]))
    assert len(data[typee]) == len(data_dict[typee]), "original data == to data_dict"


#print(data_dict["train"][0])
#print(data_dict["valdt"][0])
#print(data_dict["test"][0])

# Save the data for each typee to a single .npz file
for typee in data_dict:
    ofn = f"{CFG_SINET['sinet_version']}--fl2_{typee}.npz"
    ofp = os.path.join('/raid/DATASETS/.zuble/vigia/zuwav11/aas/2_sub_interval',ofn)
    print(ofn,'\n',ofp)
    np.savez_compressed(ofp, data=data_dict[typee])        
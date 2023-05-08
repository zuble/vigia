import os , numpy as np

from utils import globo , sinet , sinet2

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

sinet_model, sinet_metadata = sinet2.create_sinet(CFG_SINET)


def call_get_sigmoid(i,vpath, start_frame, end_frame, label, debug):
    try:
        print("\t call for ", i, os.path.basename(vpath), start_frame, end_frame)
        result = i, sinet2.get_sigmoid(sinet_model, CFG_SINET, sinet_metadata, vpath, start_frame, end_frame, debug=debug) , label
    except Exception as e:
        print(f"Exception in call_get_sigmoid for index {i}: {e}")
        result = i, None, label
    return result
    

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

        results = list(map(lambda i: call_get_sigmoid(0,vpath, sf, ef , label , True), range(1)))
        #print(results)
        data_dict[typee].append({
            'vpath': vpath,
            'sf': sf,
            'ef': ef,
            'p_es_array': results[1],
            'label': label
        })
        print("\ndata_dict\n",i,"\n",vpath,"\n",sf,ef,"\n",np.shape(results[1]),"\nlabel",label)
       

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
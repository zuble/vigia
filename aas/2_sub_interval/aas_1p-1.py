from utils import globo , sinet

import os , sys , numpy as np


if __name__ == "__main__":
    
    npz_path = os.path.join(globo.AAS_PATH+"/2_sub_interval/temp",f"dataset_from_xdvtest_bg_train_a_data-{globo.CFG_SINET['chunck_fsize']}fi.npy")
    if not os.path.exists(npz_path):
        print("npz no existe:", npz_path)
        sys.exit()
    data = np.load(npz_path, allow_pickle=True).item()    

    sinett = sinet.Sinet(globo.CFG_SINET)

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
        ofn = f"{globo.CFG_SINET['sinet_version']}--fl2_{typee}-{globo.CFG_SINET['chunck_fsize']}fi.npz"
        ofp = os.path.join('/raid/DATASETS/.zuble/vigia/zuwav11/aas/2_sub_interval/temp',ofn)
        print(ofn,'\n',ofp)
        np.savez_compressed(ofp, data=data_dict[typee])        
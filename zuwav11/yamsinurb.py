import os, time, random , cv2
import numpy as np

from utils import globo , xdv , sinet , yammet , urbnet

sinnet = sinet.Sinet(globo.CFG_SINET)
yammet = yammet.Yammet(globo.CFG_YAMMET)
urbnet = urbnet.Urbnet(globo.CFG_URBNET)

def xdvtrain_wavtry(dummy = 20):
    
    def showfr(fr1):
        for frame in fr1:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(0)
            if key == ord('q'): break  # quit
        cv2.destroyAllWindows()
    
    frame_step = 1
    frame_max = 8000
    maxpool3_min_tframes = 21 * frame_step
    
    data = xdv.train_valdt_from_npy()
    vpath_list = data["train_fn"]
    label_list = data["train_labels"]
    tframe_list = data["train_tot_frames"]
    class_list = xdv.get_index_per_label_from_filelist(vpath_list)
    
    if dummy:
        vpath_list = vpath_list[:dummy]
        label_list = label_list[:dummy]

    glob_tp , glob_fn , glob_tn , glob_fp = 0 , 0 , 0 , 0 
    yam_tp , yam_fn , yam_tn , yam_fp = 0 , 0 , 0 , 0
    sin_tp , sin_fn , sin_tn , sin_fp = 0 , 0 , 0 , 0
    urb_tp , urb_fn , urb_tn , urb_fp = 0 , 0 , 0 , 0

    for idx in range(len(vpath_list)):    
        vpath = vpath_list[idx]
        label = label_list[idx]
        tframe = tframe_list[idx]
        
        label_str = ''
        if idx in class_list['A']:label_str=str('NORMAL')
        else:
            if idx in class_list['B1']:label_str+=str('FIGHT ')
            if idx in class_list['B2']:label_str+=str('SHOOT ')
            if idx in class_list['B4']:label_str+=str('RIOT')
            if idx in class_list['B5']:label_str+=str('ABUSE')
            if idx in class_list['B6']:label_str+=str('CARACC')
            if idx in class_list['G']:label_str+=str('EXPLOS')


        if tframe >= maxpool3_min_tframes:
            
            if label == 0 and tframe > frame_max:
                sf = random.randint(0, tframe - frame_max)
                ef = sf + frame_max
            else: 
                sf = 0
                ef = tframe             
        else: 
            sf = 0 
            ef = tframe
        
        print("\n",idx,label_str,os.path.basename(vpath),"\n\t",\
            sf,ef,'{:.2f} secs'.format((ef-sf)/24))
        
        
        ###################
        print("\n\tYAMMET")
        ty = time.time()
        p_yam_arr = yammet.get_sigmoid(vpath,sf,ef)
        p_yam_arr_max = np.max(p_yam_arr,axis=0)
        tty = time.time(); ttty=tty-ty 
        yam_tp_flag , yam_fn_flag , yam_tn_flag , yam_fp_flag  = False , False , False, False
        
        for kkk,anom_yam_k in enumerate(globo.CFG_YAMMET["anom_labels_i"]):
            aux_class_max = p_yam_arr_max[anom_yam_k]
            if label: #1
                if aux_class_max >= 0.4:
                    print('\t\t-TP- {}: {:.4f}'.format(np.array(globo.CFG_YAMMET['labels'])[anom_yam_k], aux_class_max))
                    yam_tp_flag = True
                    yam_fn_flag = False
                    break
                else:
                    print('\t\t-FN- {}: {:.4f}'.format(np.array(globo.CFG_YAMMET['labels'])[anom_yam_k], aux_class_max))
                    yam_fn_flag = True
            else: #0
                if aux_class_max >= 0.4:
                    print('\t\t-FP- {}: {:.4f}'.format(np.array(globo.CFG_YAMMET['labels'])[anom_yam_k], aux_class_max))
                    yam_fp_flag = True
                    yam_tn_flag = False
                    break
                else:
                    #print('\t-TN- {}: {:.4f}'.format(np.array(globo.CFG_YAMMET['labels'])[anom_yam_k], aux_class_max))
                    yam_tn_flag = True
                    
        if yam_tp_flag: 
            print("\t\tOK")
            yam_tp += 1
        if yam_fn_flag: yam_fn += 1
        if yam_tn_flag: 
            print("\t\tOK")
            yam_tn += 1
        if yam_fp_flag: yam_fp += 1
    
        print('\t\tin {:.2f} secs'.format(ttty))
        
        ####################
        print("\n\tSINNET")
        ts = time.time()
        p_sin_arr = sinnet.get_sigmoid(vpath,sf,ef)
        p_sin_arr_max = np.max(p_sin_arr,axis=0)
        tts = time.time(); ttts = tts-ts
        
        sin_tp_flag , sin_fn_flag , sin_tn_flag , sin_fp_flag  = False , False , False, False
        
        for kkk,anom_sin_k in enumerate(globo.CFG_SINET["anom_labels_i2"]):
            aux_class_max = p_sin_arr_max[anom_sin_k]
            aux_class_label = np.array(globo.CFG_SINET['labels'])[anom_sin_k]
            
            if label: #1
                if aux_class_max >= 0.4:
                    print('\t\t-TP- {}: {:.4f}'.format(aux_class_label , aux_class_max))
                    sin_tp_flag = True
                    sin_fn_flag = False
                    break
                else:
                    print('\t\t-FN- {}: {:.4f}'.format(aux_class_label , aux_class_max))
                    sin_fn_flag = True
            
            else: #0
                if aux_class_max >= 0.4:
                    print('\t\t-FP- {}: {:.4f}'.format(aux_class_label , aux_class_max))
                    sin_fp_flag = True
                    sin_tn_flag = False
                    break
                else:
                    #print('\t-TN- {}: {:.4f}'.format(aux_class_label , aux_class_max))
                    sin_tn_flag = True
                    
        if sin_tp_flag: 
            print("\t\tOK")
            sin_tp += 1
        if sin_fn_flag: sin_fn += 1
        if sin_tn_flag: 
            print("\t\tOK")
            sin_tn += 1
        if sin_fp_flag: sin_fp += 1

        print('\t\tin {:.2f} secs'.format(ttts))


        ####################
        print("\n\tURBNET")
        tu = time.time()
        p_urb_arr = urbnet.get_sigmoid(vpath,sf,ef)
        p_urb_arr_max = np.max(p_urb_arr,axis=0)
        ttu = time.time(); tttu = ttu-tu
        
        urb_tp_flag , urb_fn_flag , urb_tn_flag , urb_fp_flag  = False , False , False, False
        
        for kkk,anom_urb_k in enumerate(globo.CFG_URBNET["anom_labels_i"]):
            aux_class_max = p_urb_arr_max[anom_urb_k]
            aux_class_label = np.array(globo.CFG_URBNET['labels'])[anom_urb_k]
            
            if label: #1
                if aux_class_max >= 0.4:
                    print('\t\t-TP- {}: {:.4f}'.format(aux_class_label , aux_class_max))
                    urb_tp_flag = True
                    urb_fn_flag = False
                    break
                else:
                    print('\t\t-FN- {}: {:.4f}'.format(aux_class_label , aux_class_max))
                    urb_fn_flag = True
            
            else: #0
                if aux_class_max >= 0.4:
                    print('\t\t-FP- {}: {:.4f}'.format(aux_class_label , aux_class_max))
                    urb_fp_flag = True
                    urb_tn_flag = False
                    break
                else:
                    #print('\t-TN- {}: {:.4f}'.format(aux_class_label , aux_class_max))
                    urb_tn_flag = True
                    
        if urb_tp_flag: 
            print("\t\tOK")
            urb_tp += 1
        if urb_fn_flag: urb_fn += 1
        if urb_tn_flag: 
            print("\t\tOK")
            urb_tn += 1
        if urb_fp_flag: urb_fp += 1

        print('\t\tin {:.2f} secs'.format(tttu))


        ## GLOBAL
        if yam_tp_flag or sin_tp_flag: 
            print("\n\tOK")
            glob_tp += 1
        if yam_fn_flag and sin_fn_flag: glob_fn += 1
        if yam_tn_flag or sin_tn_flag: 
            print("\n\tOK")
            glob_tn += 1
        if yam_fp_flag and sin_fp_flag: glob_fp += 1
    
    
    print("\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("SINNET\n","tp",sin_tp,"fn",sin_fn,"tn",sin_tn,"fp",sin_fp)
    print("YAMMET\n","tp",yam_tp,"fn",yam_fn,"tn",yam_tn,"fp",yam_fp)
    print("URBNET\n","tp",urb_tp,"fn",urb_fn,"tn",urb_tn,"fp",urb_fp)
    print("GLOBAL\n","tp",glob_tp,"fn",glob_fn,"tn",glob_tn,"fp",glob_fp)
    

xdvtrain_wavtry()
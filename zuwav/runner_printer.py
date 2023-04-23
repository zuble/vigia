import cv2 , os , time , json , sys

import essentia
from essentia import standard as es
#print('\nessentia_version',essentia.__version__,'\n')
#print('essentia_file',essentia.__file__,'\n')
#print('sys.path',sys.path,'\n')

import matplotlib.pyplot as plt
import numpy as np
import moviepy.editor as mp


import utils.util as util



'''
https://essentia.upf.edu/reference/streaming_TensorflowPredictFSDSINet.html

batchSize:
    integer ∈ [-1,inf) (default = 64)
    the batch size for prediction. This allows parallelization when GPUs are
    available. Set it to -1 or 0 to accumulate all the patches and run a single
    TensorFlow session at the end of the stream

graphFilename:
    string (default = "")
    the name of the file from which to load the TensorFlow graph
    
        tlpf : Trainable Low-Pass Filters
        aps : Adaptive Polyphase Sampling

        fsd-sinet-vgg42-tlpf_aps-1 - best
        fsd-sinet-vgg41-tlpf-1 - lighter
    
input:
    string (default = "x")
    the name of the input node in the TensorFlow graph

lastPatchMode:
    string ∈ {discard,repeat} (default = "discard")
    what to do with the last frames: `repeat` them to fill the last patch or
    `discard` them

normalize:
    bool ∈ {false,true} (default = true)
    whether to normalize the input audio signal. Note that this parameter is
    only available in standard mode

output:
    string (default = "model/predictions/Sigmoid")
    the name of the node from which to retrieve the output tensors

patchHopSize:
    integer ∈ [0,inf) (default = 50)
    number of frames between the beginnings of adjacent patches. 0 to avoid
    overlap

savedModel:
    string (default = "")
    the name of the TensorFlow SavedModel. Overrides parameter `graphFilename`
    '''


model_config = {
    
    'graph_filename' : "/raid/DATASETS/.zuble/vigia/zuwav/fsd-sinet-essentia/models/fsd-sinet-vgg41-tlpf-1.pb",
    'metadata_file' : "/raid/DATASETS/.zuble/vigia/zuwav/fsd-sinet-essentia/models/fsd-sinet-vgg41-tlpf-1.json",
    
    'batchSize' : 64,
    'lastPatchMode': 'repeat',
    'patchHopSize' : 50
    
}


test_config = {
    
    'anom_labels' : ["Alarm","Boom","Crowd","Dog","Drill","Explosion","Fire","Gunshot and gunfire","Hammer","Screaming","Screech",\
                    "Shatter","Shout","Siren","Slam","Squeak","Yell"],
    'anom_labels_i' : [4,18,51,59,65,72,78,92,94,145,146,147,148,152,154,161,198],
    
    'anom_labels2' : ["Boom","Explosion","Fire","Gunshot and gunfire","Screaming",\
                    "Shout","Siren","Yell"],
    'anom_labels_i2' : [18,72,78,92,147,148,152,198],
    
    
    "batch_len_secs":1,
    "batch_step_secs": 1,
    
    "audio_fs_input":22050,
    "nlabels2predict" : 15,
    
    
    #updated after knowing the video fps
    "video_fps":0,
    "video_total_frames":0,
    "video_total_time":0,
    
    "batch_len":0,
    "batch_step_len":0
}



''' MODEL & METADATA '''
model = es.TensorflowPredictFSDSINet(
            graphFilename = model_config['graph_filename'],
            batchSize = model_config["batchSize"],
            lastPatchMode = model_config["lastPatchMode"],
            patchHopSize = model_config["patchHopSize"]
                                    )
metadata = json.load(open(model_config['metadata_file'], "r"))
    
labels = metadata["classes"]

simple_output = True

def get_aas_total(mp4path):

    def plot_predictions(top_preds, top_labels_w_av,top_labels_with_av, save_plot):
            # Generate plots and improve formatting
            matfig = plt.figure(figsize=(8, 3))
            plt.matshow(top_preds, fignum=matfig.number, aspect="auto")

            plt.yticks(np.arange(len(top_labels_w_av)), top_labels_with_av)
            locs, _ = plt.xticks()
            ticks = np.array(locs // 2).astype("int")
            plt.xticks(locs[1: -1], ticks[1: -1])
            plt.tick_params(bottom=True, top=False, labelbottom=True, labeltop=False)
            plt.xlabel("(s)")
            plt.show()
            if save_plot : plt.savefig("activations.png", bbox_inches='tight')
    
    # ALL   
    def prediction_all(audio2predict,plott=False,save_plot=False):
        
        def process_rslt_all(predictions):
            def top_from_average(data):
                av = np.mean(data, axis=0)
                sorting = np.argsort(av)[::-1]
                return sorting[:test_config["nlabels2predict"]], [av[i] for i in sorting] ,av

            top_labels_i, averages_sorted , averages = top_from_average(predictions)
            top_labels = [labels[i] for i in top_labels_i]
            top_labels_with_av = [f"{label} ({av:.3f})" for label, av in zip(top_labels, averages_sorted)]
            
            if plott:
                top_predictions = np.array([predictions[:,i] for i in top_labels_i])
                plot_predictions(top_predictions,top_labels_with_av,top_labels_with_av,save_plot)
            
            return top_labels_with_av 
        
        predictions = model(audio2predict)
        print("predictions_shape_all",np.shape(predictions))
        return process_rslt_all(predictions) , predictions   

    # ONLY ANOMALIES
    def prediction_anom(audio2predict, plott=False,save_plot=False):
        
        def process_rslt_anom(predictions):
            def top_from_anomaly(data):
                av = np.mean(data, axis=0)
                sorting = np.argsort(av)[::-1]
                sorting_anom = [x for x in sorting if x in test_config['anom_labels_i'] ]
                return sorting_anom[:test_config["nlabels2predict"]],[av[i] for i in sorting_anom]

            top_labels_anom_i, averages_anom_sorted = top_from_anomaly(predictions)
            top_labels_anom = [labels[i] for i in top_labels_anom_i]
            top_labels_anom_with_av = [f"{label} ({av:.3f})" for label, av in zip(top_labels_anom, averages_anom_sorted)]
            
            if plott:
                top_predictions_anom = np.array([predictions[:,i] for i in top_labels_anom_i])
                plot_predictions(top_predictions_anom,top_labels_anom_with_av,top_labels_anom_with_av,save_plot)
            
            return top_labels_anom_with_av
        
        predictions = model(audio2predict) 
        print("predictions_shape_anom",np.shape(predictions))
        return process_rslt_anom(predictions) , predictions
  
  
    # FS converter 
    mp4_fs_aac = util.print_acodec_from_mp4([mp4path],only_sr=True) # get audio stream fs from mp4 
    print("mp4_fs_aac",mp4_fs_aac)
    resampler = es.Resample(inputSampleRate=mp4_fs_aac, outputSampleRate=test_config["audio_fs_input"])
    
  
    # PREDICTION OVER TOTAL VIDEO
    ttt=time.time()
    audio_tot = mp.AudioFileClip(filename=mp4path,fps=mp4_fs_aac)#.fx(mp.afx.audio_normalize)
    audio_tot_array2 = audio_tot.to_soundarray()
    audio_tot_array_mono_single2 = np.mean(audio_tot_array2, axis=1).astype(np.float32)
    audio_tot_array_essentia = resampler(audio_tot_array_mono_single2) 
    top_tot1,*_ = prediction_all(audio_tot_array_essentia,plott=True)
    top_tot2,p = prediction_anom(audio_tot_array_essentia,plott=True)
    tttt=time.time()
    print("\nTOTAL",tttt-ttt,'secs\naud_arr',np.shape(audio_tot_array_essentia),'\n\ntop_avg_all\n',top_tot1,'\n\ntop_avg_anom\n',top_tot2)
    audio_tot.close()
    
    print("\nMAX aas for anom labels 2")
    for i in range(len(test_config['anom_labels_i2'])):
        label_i = test_config['anom_labels_i2'][i]
        print(label_i,test_config['anom_labels2'][i],np.amax(np.asarray(p)[:,label_i]))

    

def get_aas_step(mp4path,printt=False):
    
    def plot_predictions(top_preds, top_labels_w_av,top_labels_with_av, save_plot):
        # Generate plots and improve formatting
        matfig = plt.figure(figsize=(8, 3))
        plt.matshow(top_preds, fignum=matfig.number, aspect="auto")

        plt.yticks(np.arange(len(top_labels_w_av)), top_labels_with_av)
        locs, _ = plt.xticks()
        ticks = np.array(locs // 2).astype("int")
        plt.xticks(locs[1: -1], ticks[1: -1])
        plt.tick_params(bottom=True, top=False, labelbottom=True, labeltop=False)
        plt.xlabel("(s)")
        plt.show()
        if save_plot : plt.savefig("activations.png", bbox_inches='tight')
    
    # ALL   
    def prediction_all(audio2predict,plott=False,save_plot=False):
        
        def process_rslt_all(predictions):
            def top_from_average(data):
                av = np.mean(data, axis=0)
                sorting = np.argsort(av)[::-1]
                return sorting[:test_config["nlabels2predict"]], [av[i] for i in sorting] ,av

            top_labels_i, averages_sorted , averages = top_from_average(predictions)
            top_labels = [labels[i] for i in top_labels_i]
            top_labels_with_av = [f"{label} ({av:.3f})" for label, av in zip(top_labels, averages_sorted)]
            
            if plott:
                top_predictions = np.array([predictions[:,i] for i in top_labels_i])
                plot_predictions(top_predictions,top_labels_with_av,top_labels_with_av,save_plot)
            
            return top_labels_with_av
        
        predictions = model(audio2predict)
        #print("predictions_shape",np.shape(predictions))
        return process_rslt_all(predictions)     

    # ONLY ANOMALIES
    def get_anom_top_sort(predictions, plott=False,save_plot=False):
        
        def top_from_anomaly(data):
            av = np.mean(data, axis=0)
            sorting = np.argsort(av)[::-1]
            sorting_anom = [x for x in sorting if x in test_config['anom_labels_i'] ]
            return sorting_anom[:test_config["nlabels2predict"]],[av[i] for i in sorting_anom]

        top_labels_anom_i, averages_anom_sorted = top_from_anomaly(predictions)
        top_labels_anom = [labels[i] for i in top_labels_anom_i]
        top_labels_anom_with_av = [f"{label} ({av:.3f})" for label, av in zip(top_labels_anom, averages_anom_sorted)]
        
        if plott:
            top_predictions_anom = np.array([predictions[:,i] for i in top_labels_anom_i])
            plot_predictions(top_predictions_anom,top_labels_anom_with_av,top_labels_anom_with_av,save_plot)
  
        return top_labels_anom_with_av
  
    def get_anom2(p):
        av = np.mean(p, axis=0)
        av_anom = [av[i] for i in test_config['anom_labels_i2']]
        labl_anom_w_av = [f"{label} ({av:.3f})" for label, av in zip(test_config['anom_labels2'], av_anom)]
        return labl_anom_w_av,av_anom
    

    ## FS converter 
    mp4_fs_aac = util.print_acodec_from_mp4([mp4path],only_sr=True) # get audio stream fs from mp4 
    #print("mp4_fs_aac",mp4_fs_aac)
    resampler = es.Resample(inputSampleRate=mp4_fs_aac, outputSampleRate=test_config["audio_fs_input"])
    
    ## Get full audio from atual mp4
    #audio_mp = mp.AudioFileClip(filename=mp4path,fps=mp4_fs_aac)#.fx(mp.afx.audio_normalize)
    audio_es = mp.AudioFileClip(filename=mp4path,fps=mp4_fs_aac)
    
    
    #print("while true")
    tt=t=0.0
    vaf=batch_step_atual=0
    
    #top_mp=[];top_mp_all=[];top_mp_total=[];aasmp_wlabel_total=[];aasmp_total=[]
    top_es=[];top_es_all=[];top_es_total=[];aases_wlabel_total=[];aases_total=[]
    
    while True:
        
        vaf += 1

        if vaf == test_config["batch_len"] + test_config["batch_step_len"] * batch_step_atual:
            
            ## set start and end of batch in secs to use in mp.subclip
            start = (test_config["batch_step_len"] * batch_step_atual)/test_config["video_fps"]
            end = vaf / test_config["video_fps"]
            #print('\n******************************************\n',vaf,'@',batch_step_atual,'[',start,end,']')
            
            '''
            #########
            ## MOVIEPY 
            t1=time.time()
            
            ## prep array
            aud_bat_mp = audio_mp.subclip(t_start=start,t_end=end)
            aud_bat_arr = aud_bat_mp.to_soundarray(fps = test_config["audio_fs_input"])
            aud_bat_arr_mp = np.mean(aud_bat_arr, axis=1).astype(np.float32)
            
            ## predict
            p_mp = model(aud_bat_arr_mp)#;print("predictions_shape",np.shape(p_mp))
            
            ## process rslt sorted/type2
            top_mp = get_anom_top_sort(p_mp) ;top_mp_total.append(top_mp)
            aasmp_wlabel,aasmp = get_anom2(p_mp) ;aasmp_wlabel_total.append(aasmp_wlabel);aasmp_total.append(aasmp)
            
            ## timer & print
            t2=time.time();t+=t2-t1#;print("\nmp",t2-t1,'\n',np.shape(aud_bat_arr_mp),'\n',top_mp,'\n',aasmp_wlabel,'\n',aasmp)  
            
            ## all
            #top_mp_all = prediction_all(aud_bat_arr_mp);print('all',top_mp_all)
            '''
            
            ##########
            # ESSENTIA
            tt1=time.time()
            
            ## prep array
            aud_bat_es = audio_es.subclip(t_start=start,t_end=end)
            aud_bat_arr2 = aud_bat_es.to_soundarray()
            aud_bat_arr_mono_single2 = np.mean(aud_bat_arr2, axis=1).astype(np.float32)
            aud_bat_arr_essentia = resampler(aud_bat_arr_mono_single2) 
            
            ## predict
            p_es = model(aud_bat_arr_essentia)#;print("predictions_shape",np.shape(p_es))  
            
            ## precoess rslt sorted/type2
            top_es = get_anom_top_sort(p_es);top_es_total.append(top_es)
            aases_wlabel,aases = get_anom2(p_es);aases_wlabel_total.append(aases_wlabel);aases_total.append(aases)
            
            ## timer & print
            tt2=time.time();tt+=tt2-tt1#print("\nes",tt2-tt1,'\n',np.shape(aud_bat_arr_essentia),'\n',top_es,'\n',aases_wlabel,'\n',aases)  
            
            
            ## all
            #top_es_all = prediction_all(aud_bat_arr_essentia) ;print('all',top_es_all)
            
            
            batch_step_atual += 1
            
        
        if vaf == test_config["video_total_frames"]: 
            #audio_mp.close()
            audio_es.close() 
            print("\n\nTOTAL TIME MP",t,"\nTOTAL TIME ESSENTIA",tt)
            
            if printt:
                for i in range(len(aases_wlabel_total)):
                    #print(aasmp_wlabel_total[i])
                    print(i,aases_wlabel_total[i])
            
                #print("\n")
                #for x in top_mp_total: print(x)
                #print("\n")
                #for y in top_es_total: print(y)

            #return aasmp_total,aases_total
            return aases_total


def FSDSINET_runner(mp4path , total_or_step):
  
    ''' CV video info '''
    video = cv2.VideoCapture(mp4path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_time = total_frames/fps
    print("total_time",total_time,"fps",fps,"total_frames",total_frames)
    video.release()
    
    test_config["video_total_frames"] = total_frames
    test_config["video_total_time"] = total_time
    
    ''' after getting video fps, modify test_config '''
    test_config["video_fps"] = fps
    test_config["batch_len"] = test_config["batch_len_secs"]*fps
    test_config["batch_step_len"] = test_config["batch_step_secs"]*fps
    
    
    
    if total_or_step == 'total':
        get_aas_total(mp4path)
    
    elif total_or_step == 'step':
        #aasmp,aases=get_aas_step(mp4path)
        #print(np.shape(aasmp),np.shape(aases))
        
        aases=get_aas_step(mp4path,printt=True)
        print(np.shape(aases))
        print("MAX")
        for i in range(np.shape(aases)[1]): print(np.amax(np.asarray(aases)[:,i]),'  |  ', end=' ')
        print("\nAVG")
        for i in range(np.shape(aases)[1]): print(np.average(np.asarray(aases)[:,i]),'  |  ', end=' ')
        
        return aases
    else: Exception(" 'total' or 'step' ")



def init(train_or_test,watch_this,total_or_step,fast=False):
    print("\n\nINIT WATCH LIVE")
    
    train_mp4_paths = util.load_xdv_train() ; print('\n  train_mp4_paths',np.shape(train_mp4_paths))
    train_labels_indexs = util.get_index_per_label_from_filelist(train_mp4_paths)
    
    test_mp4_paths = util.load_xdv_test() ; print('\n  ******\n  test_mp4_paths',np.shape(test_mp4_paths))
    test_labels_indexs = util.get_index_per_label_from_filelist(test_mp4_paths)

    if train_or_test == 'train': paths = train_mp4_paths;labels_indexs = train_labels_indexs
    elif train_or_test == 'test': paths = test_mp4_paths;labels_indexs = test_labels_indexs
    else : Exception(" 'train' or 'test' ")     
        
    if not fast :
        print('\n  watching',watch_this)
        for labels_2_watch in watch_this:
            print('  ',labels_2_watch,' : ',test_labels_indexs[labels_2_watch])
            
            all_or_specific = input("\n\nall indxs : enter  |  specific indxs : ex 3,4,77,7  |  dry_run no as window : dr\n\n")
            
            if all_or_specific == "": # all
                for i in range(len(labels_indexs[labels_2_watch])):
                    index = labels_indexs[labels_2_watch][i]
                    path = paths[index]
                    print('\n#-------------------#$%--------------------#\n',labels_2_watch,index,path)
                    FSDSINET_runner(path,total_or_step)
                    
            elif all_or_specific == "dr": 
                for i in range(len(labels_indexs[labels_2_watch])):
                    index = labels_indexs[labels_2_watch][i]
                    path = paths[index]
                    print('\n#-------------------#$%--------------------#\n',labels_2_watch,index,path) 
            
            else: # specific
                all_or_specific = all_or_specific.split(",")
                all_or_specific = [int(num) for num in all_or_specific]
                for index in all_or_specific:
                    path = paths[index]
                    print('\n#-------------------#$%--------------------#\n',labels_2_watch,index,path)
                    FSDSINET_runner(path,total_or_step)

    else:
        #all_or_specific = [52]
        #for index in all_or_specific:
        #    path = train_mp4_paths[index]
        #    print('\n#-------------------#$%--------------------#\n',index,path)
        #    FSDSINET_runner(path,total_or_step)
        
        #paths = ["/raid/DATASETS/anomaly/XD_Violence/training_copy/v=WpxJJFcWM8s__#1_label_A.mp4"]
        train_alter_paths = util.load_xdv_train_alter()
        test_labels_indexs = util.get_index_per_label_from_filelist(train_alter_paths)
        for path in train_alter_paths:
            print('\n#-------------------#$%--------------------#\n',path.replace("/raid/DATASETS/anomaly/XD_Violence/training_copy_alter/",""))
            FSDSINET_runner(path,total_or_step)



if __name__ == "__main__":        
    
    '''
        A  NORMAL  
        B1 FIGHT | B2 SHOOTING | B4 RIOT | B5 ABUSE | B6 CAR ACCIDENT | G  EXPLOSION 
        BG ALL ANOMALIES
    ''' 
    
    init('train',['G'],total_or_step='total',fast=True)

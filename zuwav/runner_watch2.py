#import essentia
#print(essentia.__version__)
#print(essentia.__file__)
import essentia.standard as es

# let's have a look at what is in there
#print(dir(essentia.standard))


import threading , cv2 , os , time , json
import matplotlib.pyplot as plt
from queue import Queue
import numpy as np

import utils.util as util
import moviepy.editor as mp


'''
FSDSINET_player injects the vaf to queue
FSDSINET_atuator predicts according to the test_config batch len/steps
'''

test_config = {
        "batch_len":0, #updated after knowing the video fps
        "batch_step_len":0,
        "video_fps":0,
        
        "batch_len_secs":2,
        "batch_step_secs": 1,
        
        "audio_fs_input":22050,
        "nlabels2predict" : 3   
    }


def FSDSINET_atuator(mp4path, vaf_queque , aasmp_queque , aases_queque , test_conf):

    # Loading the model
    graph_filename = "/raid/DATASETS/.zuble/vigia/zuwav/fsd-sinet-essentia/models/fsd-sinet-vgg41-tlpf-1.pb"
    model = es.TensorflowPredictFSDSINet(graphFilename=graph_filename)

    # Read the metadata
    metadata_file = "/raid/DATASETS/.zuble/vigia/zuwav/fsd-sinet-essentia/models/fsd-sinet-vgg41-tlpf-1.json"
    metadata = json.load(open(metadata_file, "r"))
    labels = metadata["classes"]
    anom_labels = ["Alarm","Boom","Crowd","Dog","Drill","Explosion","Fire","Gunshot and gunfire","Hammer","Screaming","Screech",\
                        "Shatter","Shout","Siren","Slam","Squeak","Yell"]
    anom_labels_i = [4,18,51,59,65,72,78,92,94,145,146,147,148,152,154,161,198]

    
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
    def prediction_anom(audio2predict, plott=False,save_plot=False):
        
        def process_rslt_anom(predictions):
            def top_from_anomaly(data):
                av = np.mean(data, axis=0)
                sorting = np.argsort(av)[::-1]
                sorting_anom = [x for x in sorting if x in anom_labels_i]
                return sorting_anom[:test_config["nlabels2predict"]],[av[i] for i in sorting_anom]

            top_labels_anom_i, averages_anom_sorted = top_from_anomaly(predictions)
            top_labels_anom = [labels[i] for i in top_labels_anom_i]
            top_labels_anom_with_av = [f"{label} ({av:.3f})" for label, av in zip(top_labels_anom, averages_anom_sorted)]
            
            if plott:
                top_predictions_anom = np.array([predictions[:,i] for i in top_labels_anom_i])
                plot_predictions(top_predictions_anom,top_labels_anom_with_av,top_labels_anom_with_av,save_plot)
            
            return top_labels_anom_with_av
        
        predictions = model(audio2predict) 
        print("predictions_shape",np.shape(predictions))
        return process_rslt_anom(predictions)     
  
  
    # FS converter 
    mp4_fs_aac = util.print_acodec_from_mp4([mp4path],only_sr=True) # get audio stream fs from mp4 
    print("mp4_fs_aac",mp4_fs_aac)
    resampler = es.Resample(inputSampleRate=mp4_fs_aac, outputSampleRate=test_config["audio_fs_input"])
    
    # Get full audio from atual mp4
    audio_mp = mp.AudioFileClip(filename=mp4path,fps=mp4_fs_aac)#.fx(mp.afx.audio_normalize)
    audio_es = mp.AudioFileClip(filename=mp4path,fps=mp4_fs_aac)
  
  
    # PREDICTION OVER TOTAL VIDEO
    
    '''ttt=time.time()
    audio_tot = mp.AudioFileClip(filename=mp4path)#.fx(mp.afx.audio_normalize)
    audio_tot_array2 = audio_tot.to_soundarray()
    audio_tot_array_mono_single2 = np.mean(audio_tot_array2, axis=1).astype(np.float32)
    audio_tot_array_essentia = resampler(audio_tot_array_mono_single2) 
    top_tot1 = prediction_all(audio_tot_array_essentia,plott=True)
    top_tot2 = prediction_anom(audio_tot_array_essentia,plott=True)
    tttt=time.time()
    print("\nTOTAL",tttt-ttt,'\n',np.shape(audio_tot_array_essentia),'\n',top_tot1,'\n',top_tot2)
    audio_tot.close()'''
    
    

    print("while true")
    tt=t=0.0
    batch_step_atual=0
    top_mp = top_es = ['','','']
    aases_total = []
    while True:
        
        vaf = vaf_queque.get()

        # checks for close signal
        if vaf == -1: 
            audio_mp.close() ; audio_es.close() 
            print("\n\nTOTAL TIME MP",t,"\nTOTAL TIME ESSENTIA",tt)
            for aas in aases_total:print(aas)
            break
        
        if vaf == test_conf["batch_len"] + test_conf["batch_step_len"] * batch_step_atual:
            
            # set start and end of batch in secs to use in mp.subclip
            start = (test_conf["batch_step_len"] * batch_step_atual)/test_conf["video_fps"]
            end = vaf / test_conf["video_fps"]
            print('\n******************************************\n',batch_step_atual,vaf,start,end)
            
            
            # MOVIEPY 
            t1=time.time()
            audio_batch_mp = audio_mp.subclip(t_start=start,t_end=end)
            audio_batch_array = audio_batch_mp.to_soundarray(fps = test_config["audio_fs_input"])
            audio_batch_array_mp = np.mean(audio_batch_array, axis=1).astype(np.float32)  
            #audio_batch_array_mp = np.random.randint(1, size=(66150)).astype(np.float32)
            top_mp = prediction_anom(audio_batch_array_mp)
            t2=time.time()
            t+=t2-t1
            print("\nmp",t2-t1,'\n',np.shape(audio_batch_array_mp),'\n',top_mp)  
            
            aasmp_queque.put((top_mp,batch_step_atual))
            
            
            # ESSENTIA
            tt1=time.time()
            audio_batch_es = audio_es.subclip(t_start=start,t_end=end)
            audio_batch_array2 = audio_batch_es.to_soundarray()
            audio_batch_array_mono_single2 = np.mean(audio_batch_array2, axis=1).astype(np.float32)
            audio_bacth_array_essentia = resampler(audio_batch_array_mono_single2) 
            top_es = prediction_anom(audio_bacth_array_essentia)
            tt2=time.time()
            tt+=tt2-tt1
            print("\nessentia",tt2-tt1,'\n',np.shape(audio_bacth_array_essentia),'\n',top_es)

            aases_queque.put((top_es,batch_step_atual))    
            aases_total.append(top_es)   
               
            batch_step_atual += 1
        

def FSDSINET_player(index, mp4path):
  
    ''' CV video info '''
    video = cv2.VideoCapture(mp4path)
    width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_time = total_frames/fps
    print("total_time",total_time,"fps",fps)
    
    
    ''' after getting video fps, modify test_config '''
    test_config["video_fps"] = fps
    test_config["batch_len"] = test_config["batch_len_secs"]*fps
    test_config["batch_step_len"] = test_config["batch_step_secs"]*fps
    

    ''' CV window info '''
    frame_time_ms = int(1000/fps)
    font = cv2.FONT_HERSHEY_SIMPLEX;fontScale = 0.5;thickness = 1;lineType = cv2.LINE_AA
    strap_video_name = os.path.splitext(os.path.basename(mp4path))[0]
    wn='asVwR @ '+str(index)+strap_video_name
    cv2.namedWindow(wn) 
    
    
    ''' THREAD prediction creation '''
    ## Create the input and as queues
    vaf_queque = Queue()
    aasmp_queque = Queue()
    aases_queque = Queue()
    
    # Create the prediction thread
    FSDSINet_atuactor_thread = threading.Thread(target=FSDSINET_atuator, args=(mp4path , vaf_queque , aasmp_queque , aases_queque , test_config,) )
    FSDSINet_atuactor_thread.start()
    

    batch_step_atual = 0 #var2print
    top_mp3 = top_es3 = ['','','']
    
    while True:
        
        ret, frame = video.read()
        if not ret:break
        else:
            video_atual_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
            
            # inject vaf to queque
            try: vaf_queque.put_nowait((video_atual_frame))
            except: pass
            
            # Get aas if ready
            try: top_mp3,batch_step_atual = aasmp_queque.get_nowait()
            except: pass
            try: top_es3,batch_step_atual = aases_queque.get_nowait()
            except: pass    


            # frames / secs   
            cv2.putText(frame, '%d' % (video_atual_frame)+'/'+str(int(total_frames)), (5, int(height)-7),font,fontScale,[60,250,250],thickness,lineType)    
            cv2.putText(frame, '%.2f' % (video_atual_frame/fps)+'s',(5,int(height)-25),font,fontScale, [80,100,250],thickness,lineType)
        
            # aas
            cv2.putText(frame,str(top_mp3[0])+'\n'+str(top_mp3[1])+'\n'+str(top_mp3[2])+'\n'+str(batch_step_atual),(10,15),font,fontScale,[0,0,255],thickness,lineType)  
            cv2.putText(frame,str(top_es3[0])+'\n'+str(top_es3[1])+'\n'+str(top_es3[2])+str(batch_step_atual),(10,30),font,fontScale,[0,0,255],thickness,lineType)  
            
            cv2.imshow(wn, frame)
            
            key = cv2.waitKey(frame_time_ms)  
            if key == ord('q'): break  # quit
            if key == ord(' '):  # pause
                while True:
                    key = cv2.waitKey(1)
                    if key == ord(' '):break
        
        
            #if batch_step_atual == 2: break
       
    video.release()
    cv2.destroyAllWindows()

    print("signal frame queue to close")
    vaf_queque.put_nowait((-1))
    
    print("closing predict thread")
    FSDSINet_atuactor_thread.join()



def init_watch_live(watch_this,from_train_or_test,fast=False):
    print("\n\nINIT WATCH LIVE")
    

    test_mp4_paths = util.load_xdv_test()
    print('\n  test_mp4_paths',np.shape(test_mp4_paths))
    test_labels_indexs = util.get_index_per_label_from_filelist(test_mp4_paths)

    train_mp4_paths = util.load_xdv_train()
    print('\n  train_mp4_paths',np.shape(train_mp4_paths))
    train_labels_indexs = util.get_index_per_label_from_filelist(train_mp4_paths)

    if from_train_or_test == 'train':
        paths = train_mp4_paths; labels_indexs = train_labels_indexs   
    if from_train_or_test == 'test':    
        paths = test_mp4_paths; labels_indexs = test_labels_indexs     

    if not fast:
        print('\n  watching',watch_this)
        for labels_2_watch in watch_this:
            print('  ',labels_2_watch,' : ',labels_indexs[labels_2_watch])
            
            all_or_specific = input("\n\nall indxs : enter  |  specific indxs : ex 3,4,77,7  |  dry_run no as window : dr\n\n")
            
            if all_or_specific == "": # all
                for i in range(len(labels_indexs[labels_2_watch])):
                    index = labels_indexs[labels_2_watch][i]
                    path = paths[index]
                    print('\n#-------------------#$%--------------------#\n',labels_2_watch,index,path)
                    FSDSINET_player(index,path)
                    
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
                    FSDSINET_player(index,path)
                   
    else: ## FOR FAST INPUT
        
        #all_or_specific = [50]
        #for index in all_or_specific:
        #    path = test_mp4_paths[index]
        #    print('\n#-------------------#$%--------------------#\n',index,path)
        #    FSDSINET_player(index,path)
        
        train_alter_paths = util.load_xdv_train_alter()
        for path in train_alter_paths:
            print('\n#-------------------#$%--------------------#\n',path.replace("/raid/DATASETS/anomaly/XD_Violence/training_copy_alter/",""))
            FSDSINET_player(0,path)

'''
    A  NORMAL  
    B1 FIGHT | B2 SHOOTING | B4 RIOT | B5 ABUSE | B6 CAR ACCIDENT | G  EXPLOSION 
    BG ALL ANOMALIES
'''

init_watch_live(watch_this=['G'],from_train_or_test='train')#,fast=True


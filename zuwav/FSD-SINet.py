import essentia
print(essentia.__version__)
print(essentia.__file__)
import essentia.standard as es

# let's have a look at what is in there
#print(dir(essentia.standard))

import json
import matplotlib.pyplot as plt
import numpy as np
import cv2

import utils.util as util
import moviepy.editor as mp

''' HELPERS '''
#print(dir(essentia.standard))
#print(help(es.MonoLoader))

# Loading the model
graph_filename = "/raid/DATASETS/.zuble/vigia/zuwav/fsd-sinet-essentia/models/fsd-sinet-vgg41-tlpf-1.pb"
model = es.TensorflowPredictFSDSINet(graphFilename=graph_filename)

# Read the metadata
metadata_file = "/raid/DATASETS/.zuble/vigia/zuwav/fsd-sinet-essentia/models/fsd-sinet-vgg41-tlpf-1.json"
metadata = json.load(open(metadata_file, "r"))
labels = metadata["classes"]


def predicition_complete(audio,nlabels2predict,printt=False,plott=False,save_plot=False):
    
    def top_from_average(data, top_n):
        av = np.mean(data, axis=1)
        sorting = np.argsort(av)[::-1]
        return sorting[:top_n], [av[i] for i in sorting]

    
    def plot_predictions(top_preds, top_labels_w_av):
    
        # Generate plots and improve formatting
        matfig = plt.figure(figsize=(8, 3))
        plt.matshow(top_preds, fignum=matfig.number, aspect="auto")

        plt.yticks(np.arange(len(top_labels_w_av)), top_labels_with_av)
        locs, _ = plt.xticks()
        ticks = np.array(locs // 2).astype("int")
        plt.xticks(locs[1: -1], ticks[1: -1])
        plt.tick_params(
            bottom=True, top=False, labelbottom=True, labeltop=False
        )
        plt.xlabel("(s)")

        if save_plot : plt.savefig("activations.png", bbox_inches='tight')
    
    
    predictions = model(audio)
    print('predictions_shape',np.shape(predictions))

    if printt:
        for label, probability in zip(metadata['classes'], predictions.mean(axis=0)):
            print(f'{label}: {100 * probability:.1f}%') 

    # Compute the top-n labels and predictions
    top_n, averages = top_from_average(predictions,nlabels2predict)
    
    for i in top_n:
        print(i)

    top_labels = [labels[i] for i in top_n]
    if printt : print(top_labels)
    
    top_labels_with_av = [
        f"{label} ({av:.3f})" for label, av in zip(top_labels, averages)
    ]
    if printt: print(top_labels_with_av)
    
    top_predictions = np.array([predictions[i, :] for i in top_n])
    if plott: plot_predictions(top_predictions, top_labels_with_av)
    
    return top_labels


def fdspredict_from_video(index,path):
    
    # cv video info
    video = cv2.VideoCapture(path)
    width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_time_ms = int(1000/fps)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5;thickness = 1;lineType = cv2.LINE_AA
    
    
    # Extract the audio from the video
    audio = mp.AudioFileClip(path)#.fx(mp.afx.audio_normalize)
    audio_total_array = audio.to_soundarray()
    audio_total_array_mono = np.mean(audio_total_array, axis=1)
    audio_total_array_mono_single = audio_total_array_mono.astype(np.float32)
    
    
    # predict over total array
    top_n1 = predicition_complete(audio_total_array_mono_single,5)
    print('total_array0',top_n1,'\n\n')
    
    
    '''# predict over total saved wavfile
    fn2 = 'audio_mono_total'+str(index)+'.wav'
    es.MonoWriter(filename=fn2)(audio_total_array_mono_single)#sampleRate = int(22050)
    
    audio_monoloader = es.MonoLoader(filename=fn2)()#, sampleRate=22050
    top_n2 = predicition_complete(audio_monoloader,5)
    print('total_wav2',top_n2,'\n\n')'''
    
    
    '''#predict over framed explosion
    #t_s=47;t_e=49 #680
    t_s=3;t_e=4 #675
    audio_explosion = audio.subclip(t_start=t_s,t_end=t_e)
    audio_explosion_array = audio_explosion.to_soundarray()
    audio_explosion_array_mono = np.mean(audio_explosion_array, axis=1)
    audio_explosion_array_mono_single = audio_explosion_array_mono.astype(np.float32)
    top_n3 = predicition_complete(audio_explosion_array_mono_single,5)
    print('explosion_array000',top_n3,'\n\n')
    fn3 = 'audio_mono_explosion'+str(index)+'.wav'
    es.MonoWriter(filename=fn3)(audio_explosion_array_mono_single)'''
    
    
    batch_len = 2*int(fps)
    batch_step_len = batch_len
    batch_steap_atual = 0
    
    batch_frame_step = 1
    atual_label0 = atual_label00 = ''
    
    while True:
        
        ret, frame = video.read()
        if not ret:break
        video_atual_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        
        #video_atual_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
        #audio_frame = audio.subclip(video_atual_time , video_atual_time+(1/fps))
        #audio_frame_array = np.array(audio_frame.to_soundarray())
        
        if video_atual_frame == batch_len + batch_step_len * batch_steap_atual:
            print(video_atual_frame,batch_steap_atual)
            
            start = (batch_step_len * batch_steap_atual)/fps
            end = video_atual_frame / fps
            print(start,end)
            
            audio_batch = audio.subclip(t_start=start,t_end=end)
            audio_batch_array = audio_batch.to_soundarray()
            
            nsamples = np.shape(audio_batch_array)[0]
            secs = end-start
            sample_rate = nsamples / secs
            #print("soundarray",audio_batch_array.dtype,np.shape(audio_batch_array), sample_rate)
            
            audio_batch_array_mono = np.mean(audio_batch_array, axis=1)
            audio_batch_array_mono_single = audio_batch_array_mono.astype(np.float32)
            print("mono_single",audio_batch_array_mono_single.dtype,np.shape(audio_batch_array_mono_single))
            top_n0 = predicition_complete(audio_batch_array_mono_single,3)
            atual_label0 = str(top_n0[0],top_n0[1],top_n0[2])
            print('batch_array',top_n0)
            
            
            # 2 save as stereo
            #es.AudioWriter(filename='audio_stereo.wav',sampleRate = int(sample_rate))(audio_batch_array)
            
            '''# predict over batch saved wav
            fn00 = 'audio_mono_'+str(batch_steap_atual)+'.wav'
            es.MonoWriter(filename=fn00,sampleRate = int(sample_rate))(audio_batch_array_mono)
            
            audio_monoloader = es.MonoLoader(filename=fn00, sampleRate=22050)()
            top_n00 = predicition_complete(audio_monoloader,2)
            atual_label00 = str(top_n00[0])
            print('essentia',top_n00,'\n\n')'''
            
            
            batch_steap_atual += 1
            
        cv2.putText(frame, '%d' % (video_atual_frame)+'/'+str(int(total_frames)), (5, int(height)-7),font,fontScale,[60,250,250],thickness,lineType)    
        cv2.putText(frame,str(atual_label0)+' '+str(batch_steap_atual),(10,15),font,fontScale,[0,0,255],thickness,lineType)  
        #cv2.putText(frame,str(atual_label00)+' '+str(batch_steap_atual),(10,30),font,fontScale,[0,0,255],thickness,lineType)  
        cv2.imshow('frame', frame)
        
        key = cv2.waitKey(frame_time_ms)  
        if key == ord('q'): break  # quit
        if key == ord(' '):  # pause
            while True:
                key = cv2.waitKey(1)
                if key == ord(' '):break

    video.release()
    cv2.destroyAllWindows()
    

def init_watch_live(watch_this):
    print("\n\nINIT WATCH LIVE")
    
    test_mp4_paths,test_mp4_labels,test_aac_paths,test_aac_labels = util.load_xdv_test(util.SERVER_TEST_AUD_ORIG_PATH)
    print('\n  test_mp4_paths',np.shape(test_mp4_paths))

    test_labels_indexs = util.get_index_per_label_from_filelist(test_mp4_paths)

    print('\n  watching',watch_this)
    for labels_2_watch in watch_this:
        print('  ',labels_2_watch,' : ',test_labels_indexs[labels_2_watch])
        
        all_or_specific = input("\n\nall indxs : enter  |  specific indxs : ex 3,4,77,7  |  dry_run no as window : dr\n\n")
        
        if all_or_specific == "": # all
            for i in range(len(test_labels_indexs[labels_2_watch])):
                index = test_labels_indexs[labels_2_watch][i]
                path = test_mp4_paths[index]
                print('\n#-------------------#$%--------------------#\n',labels_2_watch,index,path)
                fdspredict_from_video(index,path)
        if all_or_specific == "dr": 
            for i in range(len(test_labels_indexs[labels_2_watch])):
                index = test_labels_indexs[labels_2_watch][i]
                path = test_mp4_paths[index]
                print('\n#-------------------#$%--------------------#\n',labels_2_watch,index,path) 
        else: # specific
            all_or_specific = all_or_specific.split(",")
            all_or_specific = [int(num) for num in all_or_specific]
            for index in all_or_specific:
                path = test_mp4_paths[index]
                print('\n#-------------------#$%--------------------#\n',labels_2_watch,index,path)
                fdspredict_from_video(index,path)


'''
    A  NORMAL
    B1 FIGHT
    B2 SHOOTING
    B4 RIOT
    B5 ABUSE
    B6 CAR ACCIDENT
    G  EXPLOSION 
    BG ALL ANOMALIES
'''

init_watch_live(watch_this=['G'])

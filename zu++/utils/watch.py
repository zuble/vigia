import cv2, time 

import numpy as np


def get_as_total_from_res_list(res_list,txt_i):
    '''
    retrieves the as score per frame for each video 
    taking as input txt file index from res_list['txt_path']
    '''
    
    as_total = []
    printt=False
    print("as_total from",res_list['txt_path'][txt_i])
    for video_j in range(len(res_list['videopath'])):
        file_path = "".join(res_list['videopath'][video_j][txt_i])
        if printt:print(video_j,file_path+"\n")

        nbatch_in_video = int(res_list['full'][video_j][txt_i][0])
        if printt:print("nbatch_in_video",nbatch_in_video)

        nframes_in_video = int(res_list['full'][video_j][txt_i][(nbatch_in_video*3)-2])
        if printt:print("nframes_in_video",nframes_in_video)

        as_per_frame = []
        for i in range(nframes_in_video): as_per_frame.append(0)
        if printt:print("as_per_frame",np.shape(as_per_frame))

        for batch in range(0,nbatch_in_video):
            # frist batch
            if batch == 0:
                if printt:print("\nbatch",batch)

                if nbatch_in_video==1: end_frame_batch = int(res_list['full'][video_j][txt_i][1])
                else: end_frame_batch = int(res_list['full'][video_j][txt_i][1])-1
                if printt:print("end_frame_batch",end_frame_batch)

                res_batch = '{:.4f}'.format(float(res_list['full'][video_j][txt_i][2]))
                if printt:print("res_batch",res_batch)

                for i in range(end_frame_batch):as_per_frame[i] = res_batch
                #as_per_frame = [res_batch for _ in range(end_frame_batch)]

            else:
                # last batch
                if batch == nbatch_in_video-1:
                    if printt:print("\nbatch",batch)

                    start_frame_batch = int(res_list['full'][video_j][txt_i][batch*3])
                    end_frame_batch = int(res_list['full'][video_j][txt_i][(batch*3)+1])   
                    last_batch_end_frame = int(res_list['full'][video_j][txt_i][(batch*3)-2])
                    res_last_batch = '{:.4f}'.format(float(res_list['full'][video_j][txt_i][(batch*3)-1]))

                    # batch type 2 (last batch has no repetead frames)
                    if start_frame_batch == last_batch_end_frame - 1:

                        if printt:print("(bt2)start_frame_batch",start_frame_batch,"\nend_frame_batch",end_frame_batch)

                        res_batch = '{:.4f}'.format(float(res_list['full'][video_j][txt_i][(batch*3)+2]))
                        if printt:print("res_batch",res_batch)

                        for i in range(start_frame_batch,end_frame_batch):as_per_frame[i] = res_batch

                    # batch type 1 (last batch has frame_max frames)
                    else:
                        if printt:print("(bt1)start_frame_batch",start_frame_batch,"\nend_frame_batch",end_frame_batch,"\nlast_batch_end_frame",last_batch_end_frame-1)

                        res_batch = '{:.4f}'.format(float(res_list['full'][video_j][txt_i][(batch*3)+2]))
                        if printt:print("res_batch",res_batch)

                        for i in range(start_frame_batch,last_batch_end_frame):as_per_frame[i]=(res_last_batch,res_batch)
                        for i in range(last_batch_end_frame,end_frame_batch):as_per_frame[i] = res_batch

                # intermidiate batchs        
                else:
                    if printt:print("\nbatch",batch)

                    start_frame_batch = int(res_list['full'][video_j][txt_i][batch*3])
                    end_frame_batch = int(res_list['full'][video_j][txt_i][(batch*3)+1])-1

                    if printt:print("start_frame_batch",start_frame_batch,"\nend_frame_batch",end_frame_batch)

                    res_batch = '{:.4f}'.format(float(res_list['full'][video_j][txt_i][(batch*3)+2]))
                    if printt:print("res_batch",res_batch)

                    for i in range(start_frame_batch,end_frame_batch):as_per_frame[i] = res_batch

        if printt:print(np.shape(as_per_frame))         
        as_total.append(as_per_frame)        
        if printt:print("\n")
    
    return as_total


def cv2_test_from_aslist(res_list,vas_list,video_j,txt_i):
    global is_quit, is_paused, frame_counter
    
    #as_total = get_as_total_from_rslt_txti(txt_i)
    print("\nres_list['full'] for video ",video_j,"| txt",txt_i,res_list['full'][video_j][txt_i])
    
    file_path = "".join(res_list['videopath'][video_j][txt_i])
    
    video = cv2.VideoCapture(str(file_path))
    window_name = "anoml vwr"+str(video_j)+":"+"/"+".".join(res_list['videoname'][video_j])
    cv2.namedWindow(window_name)
    
    # Video information
    fps = video.get(cv2.CAP_PROP_FPS)
    width  = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    total_frame = video.get(cv2.CAP_PROP_FRAME_COUNT)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # We can set up keys to pause, go back and forth.
    # **params can be used to pass parameters to key actions.
    def quit_key_action(**params):
        global is_quit
        is_quit = True
    def rewind_key_action(**params):
        global frame_counter
        frame_counter = max(0, int(frame_counter - (fps * 5)))
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
    def forward_key_action(**params):
        global frame_counter
        frame_counter = min(int(frame_counter + (fps * 5)), total_frame - 1)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
    def pause_key_action(**params):
        global is_paused
        is_paused = not is_paused
    # Map keys to buttons
    key_action_dict = {
        ord('q'): quit_key_action,
        ord('a'): rewind_key_action,
        ord('d'): forward_key_action,
        ord('s'): pause_key_action,
        ord(' '): pause_key_action
    }
    def key_action(_key):
        if _key in key_action_dict:key_action_dict[_key]()
            
    prev_time = time.time() # Used to track real fps
    is_quit = False         # Used to signal that quit is called
    is_paused = False       # Used to signal that pause is called
    
    frame_counter = 0       # Used to track which frame are we.
    try:
        while video.isOpened():
            # If the video is paused, we don't continue reading frames.
            if is_quit:# Do something when quiting
                break
            elif is_paused:# Do something when paused
                pass
            else:
                sucess, frame = video.read() # Read the frames
                if not sucess:break
                frame_counter = int(video.get(cv2.CAP_PROP_POS_FRAMES))
                #print(frame_counter)

                #predict_atual = as_total[video_j][frame_counter-1]
                predict_atual = vas_list[frame_counter-1]
                #print(predict_atual)
                
                # Display frame numba/AS/time/fps
                cv2.putText(frame, 'Frame: %d' % (frame_counter), (10, 10), font, 0.5, [60,250,250], 2)
                cv2.putText(frame, 'AS:'+str(predict_atual), (10, 30), font, 0.5, [80,100,250], 2)
                
                cv2.putText(frame, 'Time: %.4f' % (frame_counter/fps), (int(width*2/8), 10), font, 0.5, [100,250,10], 2)
                new_time = time.time()
                cv2.putText(frame, 'fps: %.2f' % (1/(new_time-prev_time)), (int(width*4/8), 10), font, 0.5, [0,50,200], 2)
                prev_time = new_time
                
            # Display the image
            cv2.imshow(window_name,frame)

            # Wait for any key press and pass it to the key action
            frame_time_ms = int(1000/fps)#print(frame_time_ms,fps)
            key = cv2.waitKey(frame_time_ms)
            key_action(key)
            
    finally:
        video.release()
        cv2.destroyAllWindows()

import math, os , glob, random, time, cv2, tensorflow as tf, numpy as np

def get_num_loops(size, end_index):
    num_loops = tf.math.ceil(end_index / size)
    return tf.cast(num_loops, tf.int32)


def get_frames_idx(video_length, clip_length , frame_step):
    '''
        gets clip_length indices equaly frame_step spaced
        from [0,video_length] 
    '''
    num_frames = clip_length * frame_step
    
    delta = max(video_length - num_frames, 0)
    #print(delta)
    start_idx = np.random.randint(0, delta)
    end_idx = start_idx + num_frames
    #print(start_idx,end_idx)

    indices,steps = np.linspace(start_idx, end_idx, clip_length, retstep=True, endpoint=False, dtype=int)
    #indices = list(range(start_idx,end_idx,frame_step))
    assert int(steps) == int(frame_step)

    return indices.tolist() , start_idx , end_idx

def get_len_ds(dataset):
    count = 0
    for _ in dataset:count += 1
    #print("DATASET WITH",count,"ELEMENTS")
    return count


def get_ds_info(path):
    '''
        path to the video folder
        get_ds_info(str(cfg.DS.kinetics400.vpaths[0]))
    '''
    def get_video_length(video_path):
        _ext = ['.avi', '.mp4']
        _, ext = os.path.splitext(video_path)
        if not ext in _ext:raise ValueError('Extension "%s" not supported' % ext)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():raise ValueError("Could not open the file.\n{}".format(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        tframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return tframes , fps
    
    paths = glob.glob(path+'/*/*.mp4')
    nvideos = len(paths)
    print(f'ds with {nvideos} videos')

    atf , afps = 0 , 0.0
    for p in paths:
        tf , fps = get_video_length(p)
        atf += tf
        afps += fps
        print(f'{p}\n   {str(tf)} frames | {str(fps)} fps')
    print(f'mean_frames {str(atf/nvideos)}  mean_fps {str(tfps/nvideos)}')

def bench_ds(ds,steps,view=False):
    t=time.time()
    for step, batch in enumerate(ds.take(steps)):
        
        videos, labels = batch
        print(f'{step + 1}  V {videos.shape}{videos.dtype} | L {labels.numpy()}')
        #print("Min/Max pixel values:", videos.numpy().min(), videos.numpy().max())
        
        if view:
            video = tf.squeeze(videos)
            for i in range(len(video)):
                image_bgr = cv2.cvtColor(video[i].numpy(), cv2.COLOR_RGB2BGR)
                cv2.imshow("Image",image_bgr )
                cv2.waitKey(int(1000/30))  # Wait for a key press to close the window
            cv2.destroyAllWindows()
            
    print("\n\nTOTAL TIME :",str(time.time()-t))
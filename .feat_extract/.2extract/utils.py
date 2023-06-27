import cv2 , numpy as np , os


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


## ORIGINAL
## had problems with loading full video slides into memory
## Normal_Videos307_x264.mp4
## numpy.core._exceptions.MemoryError: Unable to allocate 135. GiB for an array with shape (628016, 240, 320, 3) and data type uint8
def get_video_clips_full(video_path , sliding = True , clip_length = 16 , resolution = 0 ):
    '''
        gets frames from vid segmented in non-overlaped clips with clip_length frames
        return is dtype uint8
    '''
    
    def sliding_window(arr, size = clip_length, stride = clip_length):
        ##  stride = clip_length -> no overlapping
        num_chunks = int((len(arr) - size) / stride) + 2
        result = []
        for i in range(0,  num_chunks * stride, stride):
            if len(arr[i:i + size]) == size:
                #print(i,len(arr[i:i + size]))
                result.append(arr[i:i + size])
                
        print("\tSLIDED IN2",np.shape(result)[0],np.shape(result)[1:])
        #return np.array(result)
        return result
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    #video_max_length = 1800
    #frame_max = int(fps * video_max_length) ## 30 mins
    frames = []
    while (cap.isOpened()):
        sucess, frame = cap.read()
        if not sucess: break
        ## in case of usinng the c3d , resize is done inside c3d.py
        if resolution: frame = cv2.resize(frame, (resolution,resolution)) ## default-cv::INTER_LINEAR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        #if len(frames) >= frame_max: break
    cap.release()
    
    ## no clips without full clip_lenght
    cut = len(frames) % clip_length
    if cut != 0 : frames = frames[:-cut]

    print("\tFRAMED",np.shape(frames)[0],np.shape(frames)[1:],frames[0].dtype)
    
    vinfo = (len(frames) , fps) 
    
    if not sliding: return frames , vinfo
    else: return  sliding_window(frames) , vinfo
    

def gerador_clips(video_path , sliding=True, clip_length=16):

    '''
        gets batch_size of frames from video
        yields chucnks of 16 frames
        repeat until end
    '''
    batch_size = clip_length * 8
        
    def sliding_window(arr, size=16, stride=16):
        num_chunks = int((len(arr) - size) / stride) + 2
        for i in range(0, num_chunks * stride, stride):
            current_chunk = np.array(arr[i:i + size]) 
            if len(current_chunk) == size: ## only clip_length chuncks
                yield current_chunk
    
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_batch = []
    while True:
        success, frame = cap.read()
        if not success: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_batch.append(frame)

        if len(frame_batch) >= batch_size:
            if len(frame_batch) < clip_length: break
            
            if sliding: yield from sliding_window(frame_batch, size=clip_length)
            else: yield frame_batch
            frame_batch = []

    if frame_batch:
        if sliding: yield from sliding_window(frame_batch, size=clip_length)
        else: yield frame_batch

    cap.release()



###########################################

## https://github.com/sven2101/Preprocess-I3D-Deepmind
def compute_rgb(video_path):
    """Compute RGB"""
    rgb = []
    vidcap = cv2.VideoCapture(video_path)
    success,frame = vidcap.read()
    while success:
        frame = cv2.resize(frame, (342,256)) 
        frame = (frame/255.)*2 - 1
        frame = frame[16:240, 59:283]    
        rgb.append(frame)        
        success,frame = vidcap.read()
    vidcap.release()
    rgb = rgb[:-1]
    rgb = np.asarray([np.array(rgb)])
    print('save rgb with shape ',rgb.shape)
    #np.save(SAVE_DIR+'/rgb.npy', rgb)
    return rgb
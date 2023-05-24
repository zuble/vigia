import cv2 , numpy as np


def get_video_clips(video_path, clip_length = 16 ):
    
    ## stride = clip_length > no overlapping
    def sliding_window(arr, size = clip_length, stride = clip_length):
        num_chunks = int((len(arr) - size) / stride) + 2
        result = []
        for i in range(0,  num_chunks * stride, stride):
            if len(arr[i:i + size]) > 0:
                #print(i,len(arr[i:i + size]))
                result.append(arr[i:i + size])
                
        # Remove last clip if number of frames is not equal to clip_length
        if len(result[-1]) % clip_length != 0: result = result[:-1]
        print("\tSLIDED IN2",np.shape(result)[0],np.shape(result)[1:])
        return np.array(result)
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    while (cap.isOpened()):
        sucess, frame = cap.read()
        if not sucess: break
        ## resize done by c3d.py
        #frame = cv2.resize(frame, (resolution,resolution))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    print("\tFRAMED",np.shape(frames)[0],np.shape(frames)[1:])
    
    return sliding_window(frames) , len(frames)
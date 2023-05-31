import globo
import cv2 , sys , time , os , random
import numpy as np


## https://github.com/sven2101/Preprocess-I3D-Deepmind

def get_video_length(video_path):
    _ext = ['.avi', '.mp4']
    _, ext = os.path.splitext(video_path)
    if not ext in _ext:
        raise ValueError('Extension "%s" not supported' % ext)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        raise ValueError("Could not open the file.\n{}".format(video_path))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length

def compute_TVL1(video_path):
    """Compute the TV-L1 optical flow."""
    flow = []
    TVL1 = cv2.optflow.createOptFlow_DualTVL1()
    vidcap = cv2.VideoCapture(video_path)
    success,frame1 = vidcap.read()
    bins = np.linspace(-20, 20, num=256)
    prev = cv2.cvtColor(frame1,cv2.COLOR_RGB2GRAY)
    vid_len = get_video_length(video_path)
    for _ in range(0,vid_len-1):
        success, frame2 = vidcap.read()
        curr = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY) 
        curr_flow = TVL1.calc(prev, curr, None)
        assert(curr_flow.dtype == np.float32)

        #Truncate large motions
        curr_flow[curr_flow >= 20] = 20
        curr_flow[curr_flow <= -20] = -20

        #digitize and scale to [-1;1]
        curr_flow = np.digitize(curr_flow, bins)
        curr_flow = (curr_flow/255.)*2 - 1

        #cropping the center
        curr_flow = curr_flow[8:232, 48:272]  
        flow.append(curr_flow)
        prev = curr
    vidcap.release()
    
    #flow = np.asarray([np.array(flow)])
    flow = np.array(flow)
    #print('flow', flow.shape , flow.dtype)
    #np.save(SAVE_DIR+'/flow.npy', flow)
    return flow


#################


def get_vinfo(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    tframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return tframes , fps

def compute_optical_flow(algorithm):
    flow , frames = [] , []
    
    vidcap = cv2.VideoCapture(VPATH)
    if not vidcap.isOpened():
        print("Error: Cannot open video file.")
        sys.exit()

    success, frame1 = vidcap.read()
    prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    prev = cv2.resize(prev, (224, 224))

    while success:
        success, frame2 = vidcap.read()
        if not success:
            break
        
        frames.append(cv2.resize(frame2, (224, 224)))  

        curr = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        curr = cv2.resize(curr, (224, 224))

        if algorithm == 'TVL1':
            TVL1 = cv2.optflow.createOptFlow_DualTVL1()
            curr_flow = TVL1.calc(prev, curr, None)
        elif algorithm == 'Farneback':
            curr_flow = cv2.calcOpticalFlowFarneback(prev, curr, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        else:
            raise ValueError("Invalid algorithm. Choose from 'TVL1', 'Farneback', or 'LucasKanade'.")

        flow.append(curr_flow)
        prev = curr

    vidcap.release()
    return frames, flow  # Return both RGB frames and optical flow


##################


def visualize_flow(frames, flow, tvl1_flow):
    images = []
    for frame, flow_frame, tvl1_flow_frame in zip(frames, flow, tvl1_flow):
        
        hsv = np.zeros((flow_frame.shape[0], flow_frame.shape[1], 3), dtype=np.uint8)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow_frame[..., 0], flow_frame[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        tvl1_hsv = np.zeros((tvl1_flow_frame.shape[0], tvl1_flow_frame.shape[1], 3), dtype=np.uint8)
        tvl1_hsv[..., 1] = 255
        tvl1_mag, tvl1_ang = cv2.cartToPolar(tvl1_flow_frame[..., 0], tvl1_flow_frame[..., 1])
        tvl1_hsv[..., 0] = tvl1_ang * 180 / np.pi / 2
        tvl1_hsv[..., 2] = cv2.normalize(tvl1_mag, None, 0, 255, cv2.NORM_MINMAX)
        tvl1_bgr = cv2.cvtColor(tvl1_hsv, cv2.COLOR_HSV2BGR)

        # Combine RGB, optical flow, and the new TV-L1 flow side by side
        images.append(np.hstack((frame, bgr, tvl1_bgr)))

    images = np.asarray(images)
    for image in images:
        cv2.imshow("Optical Flow Visualization", image)
        key = cv2.waitKey(int(1000/FPS))  
        if key == ord('q'): break  # quit
        if key == ord(' '):  # pause
            while True:
                key = cv2.waitKey(1)
                if key == ord(' '):break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    
    vpaths = (list(open(globo.UCFCRIME_VPATHS_LISTS["train_abnormal"])))
    
    ## choses random video with less than n secs
    while True:
        i = random.randint(0, len(vpaths)-1 )
        vpath = vpaths[i].strip("\n")
        tframes , fps = get_vinfo(vpath)
        if tframes < fps * 10:
            VPATH = vpath
            FPS = fps
            TFRAMES = tframes
            break
    print(os.path.basename(VPATH) , TFRAMES , FPS)


    algorithm = "TVL1"  ## TVL1  Farneback
    t = time.time()
    frames , flow = compute_optical_flow(algorithm)
    tt = time.time()
    print("frames",np.shape(frames))
    print("\nflow1 in",int(tt-t),"secs")
    print("flow1",np.shape(flow))
    
    t = time.time()
    flow2 = compute_TVL1(VPATH)
    tt = time.time()
    print("\nflow2 in",int(tt-t),"secs")
    print("flow2",np.shape(flow))
    print(np.allclose(flow , flow2))


    visualize_flow(frames , flow , flow2)
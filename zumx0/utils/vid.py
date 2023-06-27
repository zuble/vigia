import cv2 , numpy as np , os
from gluoncv.utils.filesystem import try_import_decord
from  utils.log import get_logger
logger = get_logger(__name__)

decord = try_import_decord()
def gerador_clips(vpath , cfg , transform , use_decord = True):

    '''
        yields cfg.DATA.NEW_LENGTH frames spaced cfg.DATA.NEW_STEP
        from vpath until end
    '''
    
    clip_length = cfg.DATA.NEW_LENGTH
    if not use_decord:
        vid = cv2.VideoCapture(vpath)
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        tframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f'{vpath}\n   {str(tframes)} frames | {str(fps)} fps')
        
        frames = []
        while True:
            success, frame = vid.read()
            if not success: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

            if len(frames) >= clip_length:
                
                #if cfg.DATA.SLOWFAST:     
                
                #else:
                frames = transform(frames) 
                frames = np.stack(frames, axis=0)
                frames = frames.reshape((-1,) + (cfg.DATA.NEW_LENGTH, 3, cfg.TRANSFORM.INPUT_SIZE, cfg.TRANSFORM.INPUT_SIZE))
                frames = np.transpose(frames, (0, 2, 1, 3, 4))    
                yield frames
                frames = []
        vid.release()
    
    else:
        vid = decord.VideoReader(vpath, width=cfg.DATA.NEW_WIDTH, height=cfg.DATA.NEW_HEIGHT)
        duration = len(vid)
        
        idx_vid = list(range(0,duration,cfg.DATA.NEW_STEP))
        segments = int(len(idx_vid) / cfg.DATA.NEW_LENGTH)
        logger.info(f'yielding {segments} segments of {cfg.DATA.NEW_LENGTH} from {len(idx_vid)} frames ({str(duration)}/{cfg.DATA.NEW_STEP})')
        yield segments
        
        for segment in range(segments):
            idx_batch = idx_vid[segment*32:(segment+1)*32]
            #logger.info(len(idx_batch),idx_batch)
            frames = vid.get_batch(idx_batch).asnumpy()
            frames = transform(frames) 
            frames = np.stack(frames, axis=0)
            frames = frames.reshape((-1,) + (cfg.DATA.NEW_LENGTH, 3, cfg.TRANSFORM.INPUT_SIZE, cfg.TRANSFORM.INPUT_SIZE))
            frames = np.transpose(frames, (0, 2, 1, 3, 4))    
            yield frames



def viewer(video):
    for i in range(len(video)):
        image_bgr = cv2.cvtColor(video[i], cv2.COLOR_RGB2BGR)
        cv2.imshow("Image",image_bgr )
        cv2.waitKey(int(1000/30))  # Wait for a key press to close the window
    cv2.destroyAllWindows()


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
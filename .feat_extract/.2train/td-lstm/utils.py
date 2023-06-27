import cv2 , os , glob


def get_video_length(video_path):
    _ext = ['.avi', '.mp4']
    _, ext = os.path.splitext(video_path)
    if not ext in _ext:raise Exception('Extension "%s" not supported' % ext)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open the file.\n{}".format(video_path))
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    tframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return tframes , fps , width , height


def get_ds_info(ds):

    invalids=[]
    for type, dir in ds.items():
        
        tff , fpss = 0 , 0
        print("\n",type,"\n")
        
        
        files = glob.glob(dir+'/*/*.mp4')
        files.sort()
        
        for i,file in enumerate(files):
            print("\n",i,file)
            try:
                tf,fps,w,h=get_video_length(file)
                print("\n\ttframes:",tf,"\n\tfps:",fps,"\n\t(w,h)",w,h)

                tff += tf
                fpss += fps
            except:
                invalids.append(file)
                print("\n\tINVALID INVALID INVALID") 
                pass
            
        print("\nAVG TFRAMES",str(int(tff/len(files))))
        print("\nAVG FPS",str(int(fpss/len(files))))
    
        print("\nINVALIDS")
        for inv in invalids:
            print(inv)
    


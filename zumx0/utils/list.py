import glob

def get(dir,save=False):
    paths = glob.glob(dir+"/*.mp4")
    paths.sort()
    print(f'from {dir} got {len(paths)} vids')
    
    if save:
        with open("train_copy.list", "w") as f:
            for p in pathvs:
                line = f'{p} 0 0\n'
                print(line)
                f.write(line)
        f = open('train_copy.list', 'r')
        return f.readlines()    
    
    else: return paths
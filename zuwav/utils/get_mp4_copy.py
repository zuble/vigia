import util , numpy as np

mp4_paths_train = util.load_xdv_train()
print(np.shape(mp4_paths_train))

for path in mp4_paths_train:
    util.recreate_mp4_with_right_duration(path,False)
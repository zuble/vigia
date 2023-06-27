import globo
import numpy as np , csv

def segment_feat(feat, length = globo.NSEGMENTS):
    """
        segments (ts,features) into (length,features)
    """
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
    r = np.linspace(0, len(feat), length+1, dtype=np.int)
    for i in range(length):
        if r[i]!=r[i+1]:
            new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
        else:
            new_feat[i,:] = feat[r[i],:]
    return new_feat


def segment_feat_crop(feat, length = globo.NSEGMENTS):
    """
        segments (ncrops,ts,features) into (ncrops,length,features)
    """
    #print(feat.shape) 
    divided_features = []        
    for f in feat: 
        new_f = np.zeros((length, f.shape[1])).astype(np.float32) #32x1024 
        r = np.linspace(0, len(f), length+1, dtype=int)
        #print(r) 
        for i in range(length):
            if r[i]!=r[i+1]:
                new_f[i,:] = np.mean(f[r[i]:r[i+1],:], 0)
            else:
                new_f[i,:] = f[r[i],:]
        divided_features.append(new_f) 

    divided_features = np.array(divided_features, dtype=np.float32)
    return divided_features


####################################################


## DeepMIL
def transform_into_segments_deepmil(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
    r = np.linspace(0, len(feat), length+1, dtype=np.int)
    for i in range(length):
        if r[i]!=r[i+1]:
            new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
        else:
            new_feat[i,:] = feat[r[i],:]
    return new_feat


## Espana
def transform_into_segments(features, n_segments=32):
    if features.shape[0] < n_segments:
        raise RuntimeError(
            "Number of prev segments lesser than expected output size"
        )

    cuts = np.linspace(0, features.shape[0], n_segments,
                       dtype=int, endpoint=False)

    new_feats = []
    for i, j in zip(cuts[:-1], cuts[1:]):
        new_feats.append(np.mean(features[i:j,:], axis=0))

    new_feats.append(np.mean(features[cuts[-1]:,:], axis=0))

    new_feats = np.array(new_feats)
    new_feats = sklearn.preprocessing.normalize(new_feats, axis=1)
    return new_feats


## AnomalyDetection_CVPR18 ~ (SULTANI)
def transform_into_segments_sultani(features, features_per_bag = 32):
    """
    Transform a bag with an arbitrary number of features into a bag
    with a fixed amount, using interpolation of consecutive features

    :param features: Bag of features to pad
    :param features_per_bag: Number of features to obtain
    :returns: Interpolated features
    :rtype: np.ndarray

    """
    
    feature_size = np.array(features).shape[1]
    interpolated_features = np.zeros((features_per_bag, feature_size))
    interpolation_indicies = np.round(np.linspace(0, len(features) - 1, num=features_per_bag + 1))
    count = 0
    for index in range(0, len(interpolation_indicies)-1):
        
        start = int(interpolation_indicies[index])
        end = int(interpolation_indicies[index + 1])

        assert end >= start

        if start == end: temp_vect = features[start, :]
        else: temp_vect = np.mean(features[start:end+1, :], axis=0)

        temp_vect = temp_vect / np.linalg.norm(temp_vect)

        if np.linalg.norm(temp_vect) == 0: print("Error")

        interpolated_features[count,:]=temp_vect
        count = count + 1
        
    return np.array(interpolated_features)


def extrapolate(outputs, num_frames):
    """
    Expand output to match the video length

    :param outputs: Array of predicted outputs
    :param num_frames: Expected size of the output array
    :returns: Array of output size
    :rtype: np.ndarray

    """
    
    print("exterpolate")
    extrapolated_outputs = []
    extrapolation_indicies = np.round(np.linspace(0, len(outputs) - 1, num=num_frames))
    for index in extrapolation_indicies:
        extrapolated_outputs.append(outputs[int(index)])
        
    return np.array(extrapolated_outputs)


def test_interpolate():
    test_case1 = np.random.randn(24, 2048)
    output_case1 = transform_into_segments_sultani(test_case1, 32)
    assert output_case1.shape == (32, 2048)

    test_case2 = np.random.randn(32, 2048)
    output_case2 = transform_into_segments_sultani(test_case2, 32)
    assert output_case2.shape == (32, 2048)

    test_case3 = np.random.randn(42, 2048)
    output_case3 = transform_into_segments_sultani(test_case3, 32)
    assert output_case3.shape == (32, 2048)
    

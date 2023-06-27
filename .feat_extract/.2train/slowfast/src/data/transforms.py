import tensorflow as tf
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input
#from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input

import src.utils.log as log
logger = log.get_logger(__name__)


@tf.function
def normalize(clips, mean, std, norm_value=255):
    """
    Standardizes an n-dimensional tensor of images by first
    normalizing with the given norm_value, then subtracting
    the mean and dividing by the standard deviation channelwise.

    Args:
    clips (tf.Tensor): video clips to normalize
    mean (list): mean value of the video raw pixels across the R G B channels.
    std (list): standard deviation of the video raw pixels across the R G B channels.
    norm_value (int, optional): value to normalize raw pixels by.
    Defaults to 255.

    Returns:
    tf.Tensor float32: tensor of the same shape as clips
    """
    clips = tf.cast(clips, tf.float32)
    mean = tf.cast(mean, tf.float32)
    std = tf.cast(std, tf.float32)
    
    shapes = tf.shape(clips) ## bs * nclips * crop * crop * 3
    all_frames = tf.reshape(clips, [-1, shapes[-3], shapes[-2], shapes[-1]]) ## nclips * crop * crop * 3
    all_frames /= norm_value
    
    def _normalize(frame, mean, std):
        frame = frame - mean
        return frame / std
    
    all_frames = tf.vectorized_map(
        lambda x: _normalize(x, mean, std),
        all_frames
    )
    ## bs * nclips * crop * crop * 3
    return tf.reshape(all_frames, tf.shape(clips))


class TemporalTransforms:
  def __init__(self,
              frame_step: int,
              num_frames: int,
              num_views: int=1):
    """
      Args:
        frame_step (int): 
        num_frames (int):
        num_views (int): default is 1
      
      when num_views is 1 , clips simply get a batch dimension
      as they are already decoded according to cfg data frame_max/frame_step
      
      when num_vies > 1 , for each num_view from a random window of 
      frame_step * num_frames frames, num_frames frames equally divided frame_step
      are gather from video 
    """
    self._frame_step = frame_step
    self._num_frames = num_frames
    self._num_views = num_views
  
  
  @tf.function
  def get_temporal_views(self, video):
    """
      Temporally samples num_views clips from the given video by
      randomly select windows of num_frames*frame_step frames
      and gather num_frames equally frame_step spaced 
      
    Args:
      video (tf.Tensor): Full video

    Returns:
      tuple (tf.Tensor, tf.Tensor): clips from video, clip label
    """
    
    size = tf.shape(video)[0]
    needed = self._frame_step * self._num_frames
    indices = tf.range(size)

    num_views_idx = []
    for _ in range(self._num_views):
        delta = tf.maximum(0, size - needed)
        start_idx = tf.random.uniform(shape=[], minval=0, maxval=delta + 1, dtype=tf.int32)
        end_idx = start_idx + needed
        idx = tf.range(start_idx, end_idx, self._frame_step)
        logger.info(delta,start_idx,end_idx,idx)
        num_views_idx.append(idx)
    indices = tf.stack(num_views_idx)
    logger.info(indices)
        
    clip = tf.gather(video, indices, axis=0)
    return tf.reshape(clip, [self._num_views, self._num_frames,tf.shape(video)[1],tf.shape(video)[2],tf.shape(video)[3]])
    
    
  def __call__(self, video, label):
    tf.assert_rank(video, 4, 'video must be 4-dimensional tensor')
    #logger.info(f'TEMPORAL_IN {tf.shape(video)},{tf.shape(label)}')

    if self._num_views > 1 :
      clips = self.get_temporal_views(video)
      #num_clips = clips.shape[0]
      #labels = tf.tile(tf.expand_dims(label, axis=0), [num_clips])
    
    else: ## train val 
      clips = tf.expand_dims(video, axis=0)
      #labels = label
    
    #logger.info(f'TEMPORAL_OUT {tf.shape(clips)},{tf.shape(label)}')
    return clips, label


class SpatialTransforms:
  def __init__(self, num_crops , jitter_min, jitter_max, crop_size, mode,
                random_hflip):
    """
    Args:
      jitter_min (int): minimum size to scale frames to
      jitter_max (int): maximum size to scale frames to
      crop_size (int): final size of frames after cropping
      is_training (bool): whether transformation is being applied
        on training data
      num_crops (int, optional): number of crops to take. Only for
        non-training data. Defaults to 1.
      random_hflip (bool, optional): whether to perform horizontal flip
        on frames (with probability of 0.5). Defaults to False.
      
    1:frames.float/255.0 ([0,1]) -> (frames-mean)/std (color norm)
    2:transform.random_short_side_scale_jitter
    3:transform.random_crop
    4:transform.horizontal_flip(0.5, frames)
    """
    self.mode = mode
    self._num_crops = num_crops
    self._crop_size = crop_size
    self._min_size = float(jitter_min)
    self._max_size = float(jitter_max)
    self._random_hflip = random_hflip
  
  #@tf.function (iterate over tf.tensor)
  def random_short_side_resize(self, clips, min_size, max_size):
    """
    Randomly scale the short side of frames in `clips`.
    Reference: https://github.com/facebookresearch/SlowFast/blob/a521bc407fb4d58e05c51bde1126cddec3081841/slowfast/datasets/transform.py#L9

    Args:
      clips (tf.Tensor): a tensor of rank 5 with dimensions
        `num_clips` x `num frames` x `height` x `width` x `channel`.
      min_size (int): minimum scale size
      max_size (int): maximum scale size

    Returns:
      tf.Tensor: transformed clips scaled to new height and width
    """
    size = tf.random.uniform([], min_size, max_size, tf.float32)
    num_views = tf.shape(clips)[0]

    height = tf.cast(tf.shape(clips)[2], tf.float32)
    width = tf.cast(tf.shape(clips)[3], tf.float32)

    if (width <= height and width == size) or (
        height <= width and height == size):
        return clips
    new_width = size
    new_height = size
    if width < height:
      new_height = tf.math.floor((height / width) * size)
    else:
      new_width = tf.math.floor((width / height) * size)

    new_height = tf.cast(new_height, tf.int32)
    new_width = tf.cast(new_width, tf.int32)

    frames = [tf.image.resize(clips[i], [new_height, new_width])for i in range(num_views)]
    frames = tf.stack(frames, 0)
    frames.set_shape([None, *frames.shape[1:]])
    
    return tf.cast(frames, clips.dtype)

  
  def uniform_crop(self, clips, size, spatial_idx):
    """
    Perform uniform spatial sampling on the images.
    Reference: https://github.com/facebookresearch/SlowFast/blob/a521bc407fb4d58e05c51bde1126cddec3081841/slowfast/datasets/transform.py#L151
    
    Args:
      clips (tf.Tensor): images to perform uniform crop. The dimension is
          `num_clips` x `num frames` x `height` x `width` x `channel`.
      size (int): size of height and weight to crop the images.
      spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
          is larger than height. Or 0, 1, or 2 for top, center, and bottom
          crop if height is larger than width.
    Returns:
      cropped (tensor): images with dimension of
          `num_clips` x `num frames` x `size` x `size` x `channel`.
    """
    assert spatial_idx in [0, 1, 2]

    height = tf.shape(clips)[2]
    width = tf.shape(clips)[3]

    y_offset = tf.math.ceil((height - size) / 2)
    x_offset = tf.math.ceil((width - size) / 2)

    y_offset = tf.cast(y_offset, tf.int32)
    x_offset = tf.cast(x_offset, tf.int32)

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = clips[
        :, :, y_offset : y_offset + size, x_offset : x_offset + size, :
    ]

    return cropped


  def __call__(self, clips, label, per_channel_mean, per_channel_std):
    tf.assert_rank(clips, 5, 'clips must be 5-dimensional tensor')
    
    clips = normalize(clips, per_channel_mean, per_channel_std)
    
    #logger.info(f'SPATIAL_IN {tf.shape(clips)} , {tf.shape(label)}')
    
    if self.mode in ['train','val']:
      
      frames = tf.py_function(
        func=self.random_short_side_resize,
        inp=[clips, self._min_size, self._max_size],
        Tout=clips.dtype
      )
      ## random crop , frames changes to 4-D 
      frames = tf.image.random_crop(
                frames[0],
                size=[tf.shape(frames)[1], self._crop_size, self._crop_size, tf.shape(frames)[-1]]
              )
      if self._random_hflip: 
        if tf.random.uniform([],maxval=1) > 0.5:
          frames = tf.image.flip_left_right(frames)

      frames = tf.expand_dims(frames, axis=0) ## batch dimension 5-D
      
    elif self.mode == 'test':
      ## testing is deterministic so jitter should be performed
      frames = tf.numpy_function(
                func=self.random_short_side_resize,
                inp=[clips, self._crop_size, self._crop_size],
                Tout=clips.dtype
              )
      frames = [
          self.uniform_crop(
              frames,
              self._crop_size,
              i%3 if self._num_crops > 1 else 1) # LeftCenterRight vs Center crop
          for i in range(self._num_crops)]
      
      frames = tf.convert_to_tensor(frames)

    #frames = normalize(frames, per_channel_mean, per_channel_std)
    #frames = mobilenet_v2_preprocess_input(frames)
    
    #logger.info(f'SPATIAL_OUT {tf.shape(frames)} , {tf.shape(label)}')
    return frames, label
  
  
if __name__ == "__main__":
  spatial_transform = SpatialTransforms(
      num_crops=1,
      jitter_min=256,
      jitter_max=320,
      crop_size=224,
      mode='val',
      random_hflip=True
  )

  x = tf.random.uniform((1, 32, 300, 300, 3))
  label = tf.constant([1])
  spatial_transform(x, label , [0.45, 0.45, 0.45] , [0.225, 0.225, 0.225])
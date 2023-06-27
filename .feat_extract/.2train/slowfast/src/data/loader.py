import numpy as np , os.path as osp, time
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

from decord import VideoLoader, VideoReader , cpu , gpu
from decord.bridge import set_bridge

from src.data.utils import get_frames_idx , get_len_ds
from src.data.transforms import TemporalTransforms, SpatialTransforms
import src.utils.log as log
logger = log.get_logger(__name__)


## change to kinetics and register the ds
class DSBuilder:
  def __init__(self, cfg , mode, mixed_precision=False):
    """
    Args:
      cfg (CfgNode): the model configurations
      is_training (bool): boolean flag to indicate if
        reading training dataset
      use_tfrecord (bool): whether data is in tfrecord
        format.
      mixed_precision (bool): whether to use mixed precision.
    """
    assert mode in ["train","val","test",], "mode '{}' not correct".format(mode)
    
    self.mode = mode
    self._mixed_prec = cfg.TRAIN.MIXED_PRECISION
    self.ds = cfg.TRAIN.DATASET
    self.arch = cfg.MODEL.ARCH
         
    # For training or validation mode, one single clip is sampled from every
    # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
    # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
    # the frames.
    
    ## !! to modifi the way it decidies if train,val,test
    if self.mode in ["train" , 'val']:
      self.num_clips = 1
      self.num_spatial_crops = 1
      self.num_temporal_views = 1
      self.crop_size = cfg.DATA.TRAIN_CROP_SIZE
      self.channels = cfg.DATA.NINPUT_CHANNELS
      self.random_hflip = True if self.mode=='train' else False
      self.txt = cfg.DS[self.ds].lpaths[ 1 if self.mode=='val' else 0]
      self.batch_size = None if cfg.TRAIN.BATCH_SIZE == 1 else cfg.TRAIN.BATCH_SIZE

    ## dont have test ds videos
    elif self.mode in ["test"]:
      self.num_clips = ( cfg.TEST.NUM_TEMPORAL_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS) ## DEFAULT IS 10 * 3
      self.num_spatial_crops = cfg.TEST.NUM_SPATIAL_CROPS
      self.num_temporal_views = cfg.TEST.NUM_TEMPORAL_VIEWS
      self.crop_size = cfg.DATA.TEST_CROP_SIZE
      self.channels = cfg.DATA.NINPUT_CHANNELS
      self.random_hflip = False
      self.txt = cfg.DS[self.ds].lpaths[1]
      self.batch_size = None if cfg.TEST.BATCH_SIZE == 1 else cfg.TEST.BATCH_SIZE

    logger.info(f'{self.mode} {self.arch} {self.ds} , bs {self.batch_size}')


    self._init_dstxt()
    # do i need the len of ds for after ? 
    cfg.TRAIN.DATASET_SIZE = self.len_dstxt
    self._cfg = cfg
    
    self.assert_input()
    self.frame_max = cfg.DATA.FRAME_MAX
    self.frame_step = cfg.DATA.FRAME_STEP 
    logger.info(f'fmax: {self.frame_max} fstep: {self.frame_step}')
    
    self.data_mean = cfg.DATA.MEAN
    self.data_std = cfg.DATA.STD
    
    
  def _init_dstxt(self):
    ''' 
      creates the tf.TextLineDataset 
      updates the len of dataset in question
    '''
    dstxt = tf.data.TextLineDataset(self.txt)
    #len_dstxt = 0
    len_dstxt = get_len_ds(dstxt)
    logger.info(f'TextLineDataset @ {self.txt} w/ {len_dstxt} elements')
    self.dstxt , self.len_dstxt = dstxt , len_dstxt


  def assert_input(self):
    if 'slowfast' in self.arch:
      ## fmax/fstep = 2 = tau/alpha
      assert self._cfg.DATA.FRAME_STEP == self._cfg.SLOWFAST.TAU / self._cfg.SLOWFAST.ALPHA
      logger.info(f'fstep slow : {self._cfg.SLOWFAST.TAU} (tau)')
      logger.info(f'fstep fast : {str(self._cfg.SLOWFAST.TAU / self._cfg.SLOWFAST.ALPHA)} (tau/alpha) ')


  def decode_video(self,line):
    """
    https://github.com/dmlc/decord
    Given a line from a text file containing the link
    to a video and the numerical label, process the line
    and decode the video:
      train mode, 
        from a randomly select start_index,
        gets frame_max frames equally divided frame_step
        directly without decoding any unncessary frames
        using decord
        they represent the fast path frames 
        
      test mode, 
        gets the full length video decoded
        ( a temporal transform will be mapped to the dataset
        to further get cfg.TEST.NUM_TEMPORAL_VIEWS clips from
        video )

    Args:
      line (tf.Tensor): a string tensor containing the
        path to a video file and the label of the video.

    Returns:
      tf.uint8, tf.int32: the decoded video (with all its
        frames intact), the label of the video
    """
    
    line = tf.strings.strip(line)
    split = tf.strings.split(line, " ")
    path = tf.compat.as_str_any(split[0].numpy()) # convert byte tensor to python string object
    label = tf.strings.to_number(split[1], out_type=tf.int32) # convert label to integer
    #line_str = line.numpy().decode("utf-8")
    #path , label = line_str.split(" ")
    #label = tf.strings.to_number(label, out_type=tf.int32)
    #tf.print(path,label)
    
    attempts = 5
    for attempt in range(attempts):
      try:
          set_bridge('tensorflow') ## returns tf.tensor
          vr = VideoReader(str(path), ctx=cpu(0))
          len_vr = len(vr)
          
          if self.mode in ['train' , 'val']:
            fast_frames_idx , start , end = get_frames_idx(len_vr,self.frame_max,self.frame_step)
            video = vr.get_batch(fast_frames_idx)
            
          elif self.mode == 'test':
            video = vr.get_batch(range(len_vr))
            
          logger.info(f'{self.mode} {path},{label} {len_vr}->{tf.shape(video)}')
          
      except Exception as e:
        if attempt == attempts - 1:
          logger.warning(f"decode {attempt+1}/{attempts} failed @ {path} , sending 0000\nException{str(e)}")
          video = tf.zeros([32, 224, 224, 3], tf.uint8)
          #continue
        else:
          time.sleep(0.3)
        #  logger.warning(f"fail decode {attempt+1}/{attempts} @ {path}\nException{str(e)}")
    
    return video, label
    
  ## raascunho
  def set_shapes(self, video_tensor, label_tensor):
    video_tensor.set_shape([self.num_clips, self.frame_max, self.crop_size, self.crop_size, self.channels])
    label_tensor.set_shape([1]) # Assuming label is a scalar value
    return video_tensor, label_tensor


  @tf.function
  def process_batch(self, videos, label,  bs):
    """
      Reshapes the video tensor to be of the format
      bs x H x W x C`
    """
    
    if self.mode in ['train' , 'val']:
      videos = tf.squeeze(videos)

    elif self.mode in ["test"]:
      shapes = tf.shape(videos)
      videos = tf.reshape(videos, shape=[-1, shapes[-4], shapes[-3], shapes[-2], shapes[-1]]) 
    
    videos.set_shape((
        bs * self.num_clips,
        self.frame_max,
        self.crop_size,
        self.crop_size,
        self.channels
    ))
  
    if self._mixed_prec:
      dtype = tf.keras.mixed_precision.experimental.global_policy().compute_dtype
      videos = tf.cast(videos, dtype)
      logger.info(f'casted to {dtype}')
    return videos, label


  @property
  def dataset_options(self):
    """Returns set options for td.data.Dataset API"""
    options = tf.data.Options()
    options.experimental_optimization.map_vectorization.enabled = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_threading.max_intra_op_parallelism = 1
    options.experimental_deterministic = False if self.mode in ['train'] else True
    options.experimental_optimization.parallel_batch = True
    return options


  def __call__(self, epochs=None):
    """Loads, transforms and batches data"""
    #logger.info("autotune",AUTOTUNE)
    
    temporal_transform = TemporalTransforms(
      frame_step=self.frame_step,
      num_frames=self.frame_max,
      num_views=self.num_temporal_views
    )
    
    spatial_transform = SpatialTransforms(
      num_crops=self.num_spatial_crops,
      jitter_min=self._cfg.DATA.TRAIN_JITTER_SCALES[0],
      jitter_max=self._cfg.DATA.TRAIN_JITTER_SCALES[1],
      crop_size=self.crop_size,
      mode=self.mode,
      random_hflip=self.random_hflip
    )
    
    
    
    ds = self.dstxt#.cache('/media/jtstudents/T77/kinetics400/.cache')
    '''
    def load_dataset(file_pattern):
      return tf.data.TextLineDataset(file_pattern)
    dataset = tf.data.Dataset.list_files(self.txt)
    dataset = dataset.interleave(load_dataset, cycle_length=8, num_parallel_calls=AUTOTUNE)
    '''
    

    if self.mode in ['train']: ## shuffle while in txt format
      #cache will produce exactly the same elements during each iteration through the dataset. 
      #to randomize the iteration order, make sure to call shuffle after calling cache.
      ds = ds.shuffle(self.len_dstxt, reshuffle_each_iteration=True)

    ds = ds.with_options(self.dataset_options)


    ## 1000 elemetns: w/autotune = 79s , w/o = 116s
    ds = ds.map(lambda x: tf.py_function(self.decode_video, [x], [tf.uint8, tf.int32]),
                num_parallel_calls=AUTOTUNE)
    logger.info(f'out decode_video {ds.element_spec}')
    #dataset = self.dstxt.map(set_shapes) #py_function loss shape info
    
    #if self.mode == 'train':
    #  ds = ds.repeat() #.repeat(num_epochs)


    #, num_parallel_calls=AUTOTUNE
    ds = ds.map( lambda *args: temporal_transform(*args), num_parallel_calls=AUTOTUNE)
    logger.info(f'out temporal_transform {ds.element_spec}')
    
    ds = ds.map(lambda *args: spatial_transform( *args, self.data_mean , self.data_std), num_parallel_calls=AUTOTUNE)
    logger.info(f'out spatial_transform {ds.element_spec}')
    
    
    if self.batch_size is not None:
      ds = ds.batch(self.batch_size, drop_remainder=True)
      ds = ds.map(lambda *args: self.process_batch(*args, self.batch_size),
                  num_parallel_calls=AUTOTUNE)
      logger.info(f'out process_batch {ds.element_spec}')
    
    #ds = ds.prefetch(AUTOTUNE)

    return ds


if __name__ == "__main__":
  print()
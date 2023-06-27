import psutil

import tensorflow as tf
from tensorflow.python.client import device_lib

import src.utils.log as log

logger = log.get_logger(__name__)


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if tf.config.list_physical_devices('GPU'):
        device = tf.test.gpu_device_name()
        device_info = [x for x in device_lib.list_local_devices() if x.name == device]
        if device_info: mem_usage_bytes = device_info[0].memory_limit
        else: mem_usage_bytes = 0
    else: mem_usage_bytes = 0
    return mem_usage_bytes / 1024**3


def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024**3
    total = vram.total / 1024**3

    return usage, total


def get_precision(mixed_precision):
  """
  set up the precision to be used for training.

  Args:
    mixed_precision (bool): whether to use use mixed
      precision.
  
  Returns:
    (str): 'mixed_float16' if `mixed_precision`, otherwise
      `float32`
  """
  precision = 'float32'
  if mixed_precision and tf.config.list_physical_devices('GPU'): 
    precision = 'mixed_float16'

  return precision


def get_strategy(num_gpus):
  """sets up the training strategy - single or multi-gpu.

  Args:
    num_gpus (int): number of gpus to use for training

  Returns:
    (tf.distribute.Strategy): the strategy to use for the current
      training session.

  """
  # prevent runtime initialization from allocating all the memory in each gpu
  avail_gpus = tf.config.list_physical_devices('GPU')
  for gpu in avail_gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  if num_gpus > 1 and len(avail_gpus) > 1: # use multiple gpus
    devices = []
    for num in range(num_gpus):
      if num < len(avail_gpus):
        id = int(avail_gpus[num].name.split(':')[-1])
        devices.append(f'/gpu:{id}')
    assert len(devices) > 1 # confirm that more than 1 gpu is available to use
    strategy = tf.distribute.MirroredStrategy(devices)
  elif len(avail_gpus) == 1 and num_gpus == 1:
    log.info("using 1 gpu")
    strategy = tf.distribute.OneDeviceStrategy('device:GPU:0')
  else:
    log.warn('Using CPU')
    strategy = tf.distribute.OneDeviceStrategy('device:CPU:0')

  return strategy
import tensorflow as tf
import math as m

import src.utils.log as log
logger = log.get_logger(__name__)


class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, cfg , steps_per_epoch):
        super().__init__()
        self.base_lr = cfg.SOLVER.BASE_LR
        self.steps = cfg.SOLVER.STEPS
        #self.lrs = cfg.SOLVER.LRS
        self.warmup_start_lr = cfg.SOLVER.WARMUP_START_LR
        self.warmup_epochs = cfg.SOLVER.WARMUP_EPOCHS
        self.max_epoch = cfg.SOLVER.MAX_EPOCH
        
        self.cosine_after_warmup = cfg.SOLVER.COSINE_AFTER_WARMUP   
        self.cosine_end_lr = cfg.SOLVER.COSINE_END_LR
        
        self.steps_per_epoch = steps_per_epoch

    def __call__(self, step):
        epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
        #epoch2 = step // self.steps_per_epoch
        #print(epoch,epoch2)
        
        def true_fn():  return self._warmup_lr(epoch)
        def false_fn(): return self._calculate_lr(epoch)
        
        lr_epoch = tf.cond(tf.less(epoch, self.warmup_epochs), true_fn, false_fn)
        logger.info(f'lr{step}@epoc{epoch} = {lr_epoch}  | warmup{self.warmup_epochs} | max_epoch{self.max_epoch    }')
        return lr_epoch
        
        '''
        if tf.less(epoch, self.warmup_epochs): #epoch < self.warmup_epochs:
            lr_epoch = self._warmup_lr(epoch)
            logger.info(f'lr_warmup{step}@epoc{epoch} = {lr_epoch}')
            return lr_epoch
        else:
            lr_epoch = self._calculate_lr(epoch)
            logger.info(f'lr{step}@epoc{epoch} = {lr_epoch}')
            return lr_epoch
        '''
        

    def _warmup_lr(self, epoch):
        lr_end = self._calculate_lr(self.warmup_epochs)
        alpha = (lr_end - self.warmup_start_lr) / self.warmup_epochs
        return epoch * alpha + self.warmup_start_lr
    

    def _calculate_lr(self, epoch):
        #if self.lr_policy == 'cosine':
        return self.cosine_lr(epoch)
        #elif self.lr_policy == 'steps_with_relative_lrs':
        #    return self.steps_lr(epoch)
        #else:
        #    raise ValueError(f"Unsupported learning rate policy: {self.lr_policy}")


    def cosine_lr(self, epoch):
        offset = self.warmup_epochs if self.cosine_after_warmup else 0.0
        assert self.cosine_end_lr < self.base_lr
        return (
            self.cosine_end_lr
            + (self.base_lr - self.cosine_end_lr)
            * (
                tf.math.cos(
                    tf.constant(m.pi) * (epoch - offset) / (self.max_epoch - offset)
                )
                + 1.0
            )
            * 0.5
        )
    
    #def steps_lr(self, epoch):
    #    for i, step in enumerate(self.steps):
    #        if epoch < step:
    #            return self.base_lr * self.lrs[i]
    #    return self.base_lr * self.lrs[-1]


def get_optima(cfg,steps_per_epoch):
    
    #steps_per_epoch = cfg.TRAIN.DATASET_SIZE // cfg.TRAIN.BATCH_SIZE
    
    
    lr_schedule = LRSchedule(cfg,steps_per_epoch)
    #print(lr_schedule(100))
    
    opt_meth = cfg.SOLVER.OPTIMIZING_METHOD
    if opt_meth == 'sgd':
        optima = tf.optimizers.SGD(
            learning_rate=lr_schedule,
            momentum=cfg.SOLVER.MOMENTUM,
            nesterov=True)
    elif opt_meth == 'adam':
        optima = tf.optimizers.Adam(
                learning_rate=lr_schedule)
    else:
      raise NotImplementedError(f'{opt_meth} not supported')
  
    logger.info(f'optima @ {optima}')
    return optima 
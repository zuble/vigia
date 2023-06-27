import os

from src.cfg.default import get_cfg
import src.utils.ckpt as ckpt
import src.utils.log as log
logger = log.get_logger(__name__)


def init(args):
    """
    Given the arguemnts, load cfg and initialize experiment dir struture
    Args:
        args (argument):
    """

    cfg = get_cfg()

    ## Merge cfg with experiment values
    mainpy_path = os.getcwd()
    expryml_path = os.path.join(mainpy_path,args.experiment)
    cfg.merge_from_file(expryml_path)
    
    ## Creates experiment main dir
    cfg.EXPERIMENTID = os.path.basename(expryml_path).split(".")[0]
    exp_path = os.path.join(cfg.EXPERIMENTDIR,cfg.EXPERIMENTID)
    if not os.path.exists(exp_path):os.mkdir(exp_path)    
    cfg.EXPERIMENTPATH = exp_path
    
    ## sets defined gpu visible b4 import tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{str(cfg.GPUID)}'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = f'{str(cfg.GPUSLIM).lower()}'
    
    ## expr folder creation
    log_path = log.setup(exp_path)
    ckpt_path = ckpt.setup(exp_path)
    cfg.EXPERIMENTCKPTPATH = ckpt_path
    
    #cfg.freeze() ## freeze or not ? multigrid ? 2get ds len ?
    
    if cfg.LOG_CFG_INFO: logger.info('cfg:\n{}'.format(cfg.dump()))
    logger.info(f'log file @ {log_path}')
    logger.info(f'ckpt dir @ {ckpt_path}')
    
    return cfg
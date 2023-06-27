import os

from cfg.default import get_cfg
import utils.log as log
logger = log.get_logger(__name__)


def init(args):
    """
    Given the arguemnts, load cfg and initialize experiment dir struture
    Args:
        args (argument):
    """

    cfg = get_cfg(args)

    ## Merge cfg with experiment values
    mainpy_path = os.getcwd()
    expryml_path = os.path.join(mainpy_path,args.experiment)

    ## Creates experiment main dir
    cfg.EXPERIMENTID = os.path.basename(expryml_path).split(".")[0]
    exp_path = os.path.join(cfg.EXPERIMENTDIR,cfg.EXPERIMENTID)
    if not os.path.exists(exp_path):os.mkdir(exp_path)    
    cfg.EXPERIMENTPATH = exp_path
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'#f'{str(cfg.GPUID)}' #.replace("[","").replace("]","")
    
    log_path = log.setup(exp_path)
    
    #cfg.freeze()
    if cfg.LOG_CFG_INFO: logger.info('cfg:\n{}'.format(cfg.dump()))
    logger.info(f'log file @ {log_path}')
    
    return cfg
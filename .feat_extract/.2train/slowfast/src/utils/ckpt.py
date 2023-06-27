import os
import src.utils.log as log

logger = log.get_logger(__name__)

def setup(path_to_expr):
    """
    Creates the checkpoint directory (if not present already).
    Args:
        path_to_expr (string): the path to the folder of the current experiment.
    """
    ckpts_dir = os.path.join(path_to_expr, "ckpts")
    if not os.path.exists(ckpts_dir):os.mkdir(ckpts_dir)
    return ckpts_dir
import os , argparse

import src.utils.setup as setup
from train import train


parser = argparse.ArgumentParser(description='zulowfat')
parser.add_argument('--experiment', 
                    type=str, 
                    default='cfgs/kinetics400/slowfast.yml', 
                    help='relative path to the experiment .yml')
ARGS = parser.parse_args(args=[])


if __name__ == "__main__":
    
    cfg = setup.init(ARGS)

    ## init main train
    if cfg.TRAIN.ENABLE: train(cfg)

    ## init main test





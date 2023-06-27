import logging, os, sys


def setup(output_dir=None):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    # Set up logging format.
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"

    #if du.is_master_proc():
    #    # Enable logging for the master process.
    #    logging.root.handlers = []
    #else:
    #    # Suppress logging for non-master processes.
    #    _suppress_print()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(filename)s: %(lineno)4d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )

    #if du.is_master_proc():
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(plain_formatter)
    logger.addHandler(ch)
    
    if output_dir is not None:# and du.is_master_proc(du.get_world_size()):
        filename = os.path.join(output_dir, "stdout.log")
        #fh = logging.StreamHandler(_cached_log_stream(filename))
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return filename


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)
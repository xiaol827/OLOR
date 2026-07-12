import os
import sys
import logging
import functools


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = '[%(asctime)s %(name)s] %(levelname)s: %(message)s'
    formatter = logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')

    # only master rank prints to console
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # each rank writes its own log file
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

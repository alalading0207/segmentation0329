import logging
def get_logger(filename):
    logger = logging.getLogger('root')
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
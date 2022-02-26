import logging


def get_logger(name):
    """
    Set-up logger function
    """
    logger = logging.getLogger(name)
    formatter = logging.Formatter(
        '%(asctime)s: %(module)s:%(funcName)s %(levelname)s: %(message)s'
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    return logger

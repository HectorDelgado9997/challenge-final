import logging


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a reusable logger.

    Parameters
    ----------
    name : str
        Logger name, usually __name__.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
import datetime as dt
import logging

from src.constant import LOG_DIR

logger = logging.getLogger("experiment")
logger.setLevel(logging.DEBUG)

if len(logger.handlers) < 1:
    formatter = logging.Formatter(
        fmt="[{asctime}] {levelname: <9} {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # log_file = LOG_DIR / f"{dt.datetime.now():%Y_%m_%d_%H_%M_%S}.log"
    # file_handler = logging.FileHandler(log_file)
    # file_handler.setLevel(logging.DEBUG)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

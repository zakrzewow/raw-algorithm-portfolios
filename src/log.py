import logging

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

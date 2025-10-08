from __future__ import annotations
import logging
import sys

class LevelColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO:  "\033[0m",
        logging.WARNING:"\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL:"\033[41m",
    }
    RESET = "\033[0m"

    def format(self, record):
        base = super().format(record)
        color = self.COLORS.get(record.levelno, self.RESET)
        return f"{color}{base}{self.RESET}"

def get_logger(name: str = "isow", level: int = logging.INFO, red: bool = True) -> logging.Logger:
    """
    Logger simple : horodaté, envoie sur stdout, option 'rouge'.
    Usage:
        log = get_logger(__name__)
        log.info("Message")
        log.warning("Attention")
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # évite double configuration si appelé plusieurs fois
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(stream=sys.stdout)
    fmt = "%(asctime)s | %(levelname)s | %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handler.setFormatter(LevelColorFormatter(fmt, datefmt) if red else logging.Formatter(fmt, datefmt))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
import os
import sys
import logging
import logging.config
from pathlib import Path
from rich.console import Console
from rich.theme import Theme


def get_logger() -> logging.Logger:
    custom_theme = Theme(
        {
            "debug": "cyan",
            "info": "green",
            "warning": "yellow",
            "error": "red",
            "critical": "red on white",
        }
    )

    console = Console(theme=custom_theme)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging_config: dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "rich": {
                "format": "%(message)s",
                "datefmt": "[%X]",
            },
            "detailed": {
                "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n",
                "datefmt": "%Y-%m-%d %H:%M:%S,%f",
            },
        },
        "handlers": {
            "console": {
                "class": "rich.logging.RichHandler",
                "formatter": "rich",
                "console": console,
                "rich_tracebacks": True,
                "show_time": True,
                "show_path": True,
                "enable_link_path": True,
                "level": logging.DEBUG,
            },
            "info": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "logs/info.log",
                "maxBytes": 10485760,  # 10 MB
                "backupCount": 10,
                "formatter": "detailed",
                "level": logging.INFO,
            },
            "error": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "logs/error.log",
                "maxBytes": 10485760,  # 10 MB
                "backupCount": 10,
                "formatter": "detailed",
                "level": logging.ERROR,
            },
        },
        "root": {
            "handlers": ["console", "info", "error"],
            "level": logging.INFO,
            "propagate": True,
        },
    }

    logging.config.dictConfig(logging_config)
    logger = logging.getLogger()
    return logger

import logging
from pathlib import Path

from rich.logging import RichHandler

from modelgate.config import settings

_LOGGER_CONFIGURED = False


def get_logger(name: str = "modelgate") -> logging.Logger:
    global _LOGGER_CONFIGURED
    logger = logging.getLogger(name)
    if not _LOGGER_CONFIGURED:
        logging.basicConfig(
            level=settings.log_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        )
        _LOGGER_CONFIGURED = True
    return logger


def ensure_dirs() -> None:
    for p in [
        settings.data_dir,
        settings.raw_dir,
        settings.windows_dir,
        settings.logs_dir,
    ]:
        Path(p).mkdir(parents=True, exist_ok=True)

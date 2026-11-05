# ruff: noqa  # verbatim port from TAMP tmp3 (kept diffable against upstream)
LOG_LEVEL_NAMES = {
    "NONE": 0,
    "ERROR": 1,
    "WARN": 2,
    "INFO": 3,
    "DEBUG": 4,
}

class Logger:
    log_level = 2
    """The logging level"""
    def __init__(self):
        pass

    def error(self, message):
        if self.log_level >= 1:
            print(f"[ERROR]: {message}")

    def warn(self, message):
        if self.log_level >= 2:
            print(f"[WARN ]: {message}")

    def info(self, message):
        if self.log_level >= 3:
            print(f"[INFO ]: {message}")

    def debug(self, message):
        if self.log_level >= 4:
            print(f"[DEBUG]: {message}")
       

    def set_log_level(self, log_level: int | str):
        if isinstance(log_level, str):
            log_level = LOG_LEVEL_NAMES[log_level]
        elif isinstance(log_level, int):
            if log_level not in LOG_LEVEL_NAMES.values():
                raise ValueError(f"Invalid log level: {log_level}")
        else:
            raise ValueError(f"Invalid log level: {log_level}")
        self.log_level = log_level

logger = Logger()
"""Global logger instance"""
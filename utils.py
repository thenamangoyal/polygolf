import time
import signal
import logging

class TimeoutException(Exception):   # Custom exception class
    pass


def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException


class MainLoggingFilter(logging.Filter):
    def __init__(self, name: str) -> None:
        super().__init__(name=name)

    def filter(self, record):
        if record.name == self.name:
            return True
        else:
            return False


class PlayerLoggingFilter(logging.Filter):
    def __init__(self, name: str) -> None:
        super().__init__(name=name)

    def filter(self, record):
        if self.name in record.name or record.name == __name__:
            return True
        else:
            return False

def isiterable(obj):
    try:
        iterator = iter(obj)
    except TypeError as te:
        return False
    return True

def count_iterable(i):
    return sum(1 for e in i)
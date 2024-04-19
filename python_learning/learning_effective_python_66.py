"""

第66条 考虑用contextlib和with语句来改写可复用的try/finally代码

"""

import logging
from contextlib import contextmanager


def my_function():
    logging.debug("some debug")
    logging.error('some error messages')
    logging.debug('More debug messages')


@contextmanager
def debug_logging(level):
    logger = logging.getLogger()
    old_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield  # 纯粹转移控制权给外部代码，配合with使用确实不错
    finally:
        logger.setLevel(old_level)


with debug_logging(logging.DEBUG):
    print('* Inside:')
    my_function()

print('* After:')
my_function()


print('log level , modify log name .................................................')

@contextmanager
def log_level(level, name):
    logger = logging.getLogger(name)
    old_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield logger
    finally:
        logger.setLevel(old_level)


with log_level(logging.DEBUG, 'my-log') as logger:
    logger.debug(f'this is a message for {logger.name}')
    logging.debug(f'This will not print')

with log_level(logging.DEBUG, 'other-log') as logger:
    logger.debug(f'This is a message for {logger.name}')
    logging.debug('This will not print')


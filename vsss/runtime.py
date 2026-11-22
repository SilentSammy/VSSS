"""Cleanup for plain synchronous student programs."""
from functools import wraps
import logging


def run(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        from .car import Car
        from .gps import gps
        failed = False
        try:
            return function(*args, **kwargs)
        except BaseException:
            failed = True
            raise
        finally:
            try:
                try: Car.close_all()
                finally: gps.close()
            except BaseException:
                if not failed: raise
                logging.exception('Cleanup failed while preserving the original program error')
    return wrapped

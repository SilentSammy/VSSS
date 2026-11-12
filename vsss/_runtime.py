"""Cleanup wrapper for student script entry points."""
from functools import wraps
import inspect
import sys

from .car import Car
from .gps import gps


def run(function):
    """Decorate a synchronous main function to release cars and GPS on exit.

    Call the decorated function normally. Original exceptions (including Ctrl+C)
    are preserved if cleanup also fails. Cleanup errors otherwise propagate.
    """
    if (inspect.iscoroutinefunction(function) or inspect.isgeneratorfunction(function)
            or inspect.isasyncgenfunction(function)):
        raise TypeError('@run requires a synchronous, non-generator function')

    @wraps(function)
    def wrapped(*args, **kwargs):
        original_error = None
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            original_error = error
            raise
        finally:
            errors = []
            for cleanup in (Car.close_all, gps.close):
                try:
                    cleanup()
                except BaseException as error:
                    errors.append(error)
            if errors:
                if original_error is None:
                    raise errors[0]
                for error in errors:
                    print(f'[VSSS] Cleanup failed: {error}', file=sys.stderr)

    return wrapped

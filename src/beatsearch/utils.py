# coding=utf-8
import os
import sys
import typing as tp
from time import time
from inspect import isclass
from functools import wraps
from beatsearch.config import __USE_NUMPY__

if __USE_NUMPY__:
    import numpy as np


def merge_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def format_timespan(seconds):
    """
    Returns the given timespan as a string in hours:minutes:seconds.

    :param seconds: timespan in seconds
    :return: formatted string
    """

    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def print_progress_bar(iteration, total, prefix="", suffix="", decimals=1, length=50, fill='█', starting_time=None):
    """
    Call in a loop to create terminal progress bar

    :param iteration: current iteration
    :param total: total iterations
    :param prefix: prefix string
    :param suffix: suffix string
    :param decimals: number of decimals in percent progress
    :param length: character length of bar
    :param fill: bar fill character
    :param starting_time: starting time, for computing the time remaining for completion
    :return: terminal progress bar
    """

    progress = iteration / float(total)
    percent = ("{0:." + str(decimals) + "f}").format(100 * progress)
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)

    if starting_time is not None:
        elapsed_time = time() - starting_time
        if iteration < total:
            try:
                remaining = format_timespan((elapsed_time / progress) - elapsed_time)
            except ZeroDivisionError:
                remaining = "unknown"
            time_info = "Remaining: %s " % remaining
        else:
            time_info = "Elapsed: %s" % format_timespan(elapsed_time)
    else:
        time_info = ""

    sys.stdout.write('\r%s |%s| %s%% %s %s' % (prefix, bar, percent, suffix, time_info))
    sys.stdout.flush()

    if iteration == total:
        print("")


def friendly_named_class(name):
    # noinspection PyPep8Naming
    def _friendly_named_class_decorator(Cls):
        setattr(Cls, '__friendly_name__', name)
        return Cls

    return _friendly_named_class_decorator


def err_print(*args, **kwargs):
    """
    Print function wrapper for printing to stderr.
    """

    print(*args, file=sys.stderr, **kwargs)


def make_dirs_if_not_exist(directory):
    """
    Creates a directory if it doesn't exist yet.

    :param directory: path
    :return: path
    """

    if not os.path.isdir(directory):
        os.makedirs(directory)
    return directory


def head_trail_iter(iterable):
    """
    Returns an iterator which yields a tuple (is_first, is_last, item) on each iteration.

    :param iterable: iterable with length
    :return: iterator returning (is_first, is_last, item)
    """

    last_i = len(iterable) - 1
    for i, item in enumerate(iterable):
        first = i == 0
        last = i == last_i
        yield first, last, item


def get_beatsearch_dir(mkdir=True):
    """
    Returns the beatsearch directory. This directory is used to store *.ini files and output various beatsearch output
    data.

    :param mkdir: when True, this function will create the beatsearch directory in case it does not exist
    :return: beatsearch directory path
    """

    home = os.path.expanduser("~")
    beatsearch_dir = os.path.join(home, "beatsearch")
    if not os.path.isdir(beatsearch_dir) and mkdir:
        os.mkdir(beatsearch_dir)
    return beatsearch_dir


def no_callback(*_, **__):
    return None


def type_check_and_instantiate_if_necessary(thing, thing_base_type, allow_none=True, *args, **kwargs):
    """
    Instantiates the given 'thing' with the given args and kwargs and returns it. If the given 'thing' is already an
    instance of the given 'thing_base_type' it will return 'thing'.

    :param thing: either an instance or a (sub)-class of 'thing_base_type'
    :param thing_base_type: a class
    :param allow_none: when True, this function will allow None things (it will return None if 'thing' is None)
    :param args: if an initialization is needed, this positional arguments will be passed to the constructor
    :param kwargs: if an initialization is needed, this named arguments will be passed to the constructor
    :return: an instance of 'thing_base_type' (might return None if 'allow_none' is True)
    """
    # type: (tp.Any, tp.Type, bool) -> tp.Any

    if not isclass(thing_base_type):
        raise TypeError("Expected a class but got \"%s\"" % str(thing_base_type))

    if (allow_none and thing is None) or isinstance(thing, thing_base_type):
        return thing

    if not isclass(thing) or not issubclass(thing, thing_base_type):
        raise TypeError("Expected either a \"%s\" (sub)-class or "
                        "an instance but got \"%s\"" % (thing_base_type, str(thing)))

    return thing(*args, **kwargs)


def inject_numpy(func):
    """
    Adds a named "numpy" argument to the decorated function. If the __USE_NUMPY__ option is False, calling the
    decorated function will raise an exception.

    :param func: function to decorate
    :return: decorated function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not __USE_NUMPY__:
            raise Exception("%s needs numpy" % func.__name__)
        kwargs['numpy'] = np
        return func(*args, **kwargs)
    return wrapper


def eat_args(func):
    """
    Adds ignored positional and named arguments to the given function's signature.

    :param func: function to add the fake arguments to
    :return: wrapped function
    """

    @wraps(func)
    def wrapper(*_, **__):
        return func()
    return wrapper
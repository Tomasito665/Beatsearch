# coding=utf-8
import sys
from time import time


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
        print ""


def friendly_named_class(name):
    # noinspection PyPep8Naming
    def _friendly_named_class_decorator(Cls):
        setattr(Cls, '__friendly_name__', name)
        return Cls
    return _friendly_named_class_decorator

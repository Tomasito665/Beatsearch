# coding=utf-8
import os
import sys
import logging
import itertools
import collections
import numpy as np
import typing as tp
from time import time
from types import MappingProxyType
from inspect import isclass, isabstract
from functools import wraps, reduce
from matplotlib.colors import to_rgb, rgb_to_hsv, hsv_to_rgb, to_hex


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


def make_dir_if_not_exist(directory: str) -> str:
    """
    Creates a directory if it doesn't exist yet.

    :param directory: path
    :return: path
    """

    if not os.path.isdir(directory):
        os.mkdir(directory)
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


def current_next_pair_iter(seq: tp.Sequence, **kwargs) -> tp.Iterable[tp.Tuple[tp.Any, tp.Any]]:
    """
    Returns an iterator which yields a (curr_element, next_element) tuple on each iteration.

    :param seq: sequence over which to iterate
    :param kwargs:
        post_final - if set, the iterator will include the last iteration and use this parameter as next element

    :return: iterable yielding (curr_el, next_el) on each iteration (iteration count is N if given last or N-1 if not)
    """

    try:
        post_final = kwargs["post_final"]
        include_last = True
    except KeyError:
        post_final = None
        include_last = False

    for i, curr_el in enumerate(seq):
        if (i + 1) == len(seq):  # if last
            if not include_last:
                break
            next_el = post_final
        else:
            next_el = seq[i + 1]
        yield curr_el, next_el


def get_beatsearch_dir(mkdir=True):
    """Returns the beatsearch directory
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


def get_default_beatsearch_rhythms_fpath(mkdir=True):
    """Returns the default path to the rhythms pickle file

    Returns the default path to the rhythms pickle file. This path will be ${BSHome}/data/rhythms.pkl, where BSHome
    is the beatsearch directory returned by get_beatsearch_dir.

    :return: default path to the rhythms pickle file as a string
    """

    beatsearch_dir = get_beatsearch_dir(mkdir)
    return os.path.join(beatsearch_dir, "rhythms.pkl")


def no_callback(*_, **__) -> tp.Any:
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


def color_variant(color, brightness_offset=0):
    """
    Takes a color and produces a lighter or darker variant.

    :param color: color to convert (either as a rgb array or as a hexadecimal string)
    :param brightness_offset: brightness offset between -1 and 1
    :return: new color (either hexadecimal or rgb depending on whether the input is a hexadecimal color)
    """

    try:
        is_hexadecimal = color.startswith("#")
    except TypeError:
        is_hexadecimal = False

    rgb = to_rgb(color)
    hsv = rgb_to_hsv(rgb)

    curr_brightness = hsv[2]
    new_brightness = max(-1.0, min(curr_brightness + brightness_offset, 1.0))  # clamp to [-1.0, 1.0]

    new_hsv = (hsv[0], hsv[1], new_brightness)
    new_rgb = hsv_to_rgb(new_hsv)

    if is_hexadecimal:
        return to_hex(new_rgb)

    return new_rgb


def get_midi_files_in_directory(directory):
    """Iterates over the midi files in the given directory

    Recursively iterates oer the midi files in the given directory yielding the full MIDI-file paths.

    :param directory: directory to iterate over
    :return: generator yielding paths to the MIDI files
    """

    for directory, subdirectories, files in os.walk(directory):
        for f_name in files:
            if os.path.splitext(f_name)[1] != ".mid":
                continue
            yield os.path.join(directory, f_name)


class TupleView(collections.Sequence):
    @staticmethod
    def range_view(the_tuple, from_index, to_index):
        """Creates a range tuple view

        :param the_tuple: the tuple to create the view of
        :param from_index: the starting index
        :param to_index: the ending index (excluding)
        :return: TupleView
        """

        assert to_index >= from_index
        indices = range(from_index, to_index) if to_index > from_index else tuple()
        return TupleView(the_tuple, indices)

    def __init__(self, the_tuple, indices):
        """Creates a view of the given tuple

        :param the_tuple: the tuple to create the view of
        :param indices: the indices of the tuple
        """

        indices = tuple(indices)

        for ix in indices:
            try:
                the_tuple[ix]
            except IndexError:
                raise ValueError("invalid index: %i" % ix)

        self._the_tuple = the_tuple
        self._indices = indices

    def __str__(self):
        return str(list(self))

    def __getitem__(self, index):
        actual_index = self._indices[index]
        return self._the_tuple[actual_index]

    def __len__(self):
        return len(self._indices)


def most_common_element(sequence: tp.Sequence[tp.Any]) -> tp.Any:
    """Returns the most common element in the given sequence

    :param sequence: sequence
    :return: None
    """

    return max(set(sequence), key=sequence.count)


def sequence_product(iterable: tp.Iterable[tp.Union[int, float]]) -> tp.Union[int, float]:
    """Returns the product of the given iterable

    Returns 0 if reduce() raise a TypeError.

    :param iterable: iterable to return the product of
    :return: product of given iterable
    """

    try:
        return reduce((lambda x, y: x * y), iterable)
    except TypeError:
        return 0


def minimize_term_count(value: int, terms: tp.Sequence[int], assume_sorted: bool = False):
    """This function minimizes the number of terms necessary to sum up a value, given a set of terms

    This function returns terms that eventually sum up to the given value. It is only allowed to use terms given in the
    "terms" parameter or 1. For example, if given value 17 and terms [3, 5, 7], this method will yield [7, 7, 5, 1].

    :param value: the value that the terms should sum up to as an integer
    :param terms: sequence containing the terms to choose from
    :param assume_sorted: set this to true if the given terms are sorted (in ascending order) for better performance
    :return: a generator yielding maximized terms that sum up to the given value
    """

    if not assume_sorted:
        terms = sorted(terms)
    i = 0
    while i < value:
        remaining = value - i
        u = next((u for u in reversed(terms) if u <= remaining), 1)
        n = remaining // u
        for j in range(n):
            yield u
        i += n * u


FileInfo = collections.namedtuple("FileInfo", ["path", "modified_time"])


def normalize_directory(directory):
    """Normalizes a directory path

    Normalizes the directory path, replaces the backslashes with forward slashes and removes the ending slash.

    :param directory: directory path to normalize
    :return: normalized directory path
    """

    directory = os.path.normpath(directory)
    directory = directory.replace("\\", "/")
    if directory.endswith("/"):
        directory = directory[:-1]
    return directory


def generate_unique_abbreviation(
        label: str,
        max_len: int = 3,
        taken_abbreviations: tp.Optional[tp.Iterable[str]] = None,
        dictionary: tp.Union[tp.Tuple[str], str] = "cdfghjklmnpqrstvxz"
):
    """
    Returns a unique abbreviation of the given label.

    :param label: label to abbreviate
    :param max_len: maximum length of the abbreviation
    :param taken_abbreviations: abbreviations that are already taken
    :param dictionary: dictionary of characteristic characters (defaults to consonants)
    :return: abbreviation of given text
    """

    if not label:
        raise ValueError

    label = label.lower()
    taken_abbreviations = taken_abbreviations or set()

    if len(label) <= max_len and label not in taken_abbreviations:
        return label

    # filter out the characters which are not in the given dictionary (or keep the name if
    # it doesn't contain any characters of the given dictionary)
    essence = "".join(filter(lambda c: c in dictionary, label)) or label

    if len(essence) < max_len and essence not in taken_abbreviations:
        return essence

    key_chars = list(essence[i] for i in sorted(set(np.linspace(
        0, len(essence), max_len, endpoint=False, dtype=int))))

    abbreviation = "".join(key_chars)

    while abbreviation in taken_abbreviations:
        # append a character if the max length allows it
        if len(key_chars) < max_len:
            key_chars.append("0")
        else:
            last_char = key_chars[-1]
            key_chars[-1] = chr(ord(last_char) + 1)
        abbreviation = "".join(key_chars)

    return abbreviation


def generate_abbreviations(
        labels: tp.Iterable[str],
        max_abbreviation_len: int = 3,
        dictionary: tp.Union[tp.Tuple[str], str] = "cdfghjklmnpqrstvxz"):
    """
    Returns unique abbreviations for the given labels. Generates the abbreviations with
    :func:`beatsearch.utils.generate_unique_abbreviation`.

    :param labels: labels to abbreviate
    :param max_abbreviation_len: maximum length of the abbreviations
    :param dictionary: characteristic characters (defaults to consonants)
    :return: abbreviations of the given labels
    """

    abbreviations = list()

    for label in labels:
        abbreviations.append(generate_unique_abbreviation(
            label,
            max_len=max_abbreviation_len,
            taken_abbreviations=abbreviations,
            dictionary=dictionary
        ))

    return abbreviations


Point2D = collections.namedtuple("Point2D", ["x", "y"])

Dimensions2D = collections.namedtuple("Dimensions2D", ["width", "height"])


class Rectangle2D(collections.namedtuple("Rectangle2D", ["x", "y", "width", "height"])):
    @property
    def x_bounds(self):
        return self.x, self.x + self.width

    @property
    def y_bounds(self):
        return self.y, self.y + self.height

    @property
    def position(self) -> Point2D:
        return Point2D(x=self.x, y=self.y)

    @property
    def dimensions(self) -> Dimensions2D:
        return Dimensions2D(width=self.width, height=self.height)


class QuantizableMixin(object):
    """This mixin adds a 'quantize' property to classes that extend it. It also provides a hook that gets notified when
    the quantize property has been set."""

    # Whether the quantization of this object is enabled
    __quantize: bool

    @property
    def quantize(self) -> bool:
        """Whether the quantization of this object is enabled. Defaults to False."""
        try:
            return self.__quantize
        except AttributeError:
            return False

    @quantize.setter
    def quantize(self, quantize_enabled: bool):
        quantize_enabled = bool(quantize_enabled)
        self.__quantize = quantize_enabled
        self.__on_quantize_set__(quantize_enabled)

    def __on_quantize_set__(self, quantize: bool):
        """This method will be called when the quantize property is set

        Override this method to get notified whenever the quantize attribute is set.

        :param quantize: new value of quantize (same as self.quantize)
        :return: None (not used)
        """

        pass


def zip_equal(*iterables: tp.Iterable) -> tp.Generator[tp.Tuple[tp.Any, ...], None, None]:
    """
    Zip utility function that raises a value error if the given iterables don't yield the same number of elements.

    :param iterables: iterables yielding the same number of elements
    :return: generator yielding tuples containing each n-th element of the given iterables
    :raises: ValueError if the iterables don't yield the same number of iterables
    """

    sentinel = object()
    for combo in itertools.zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


class InstrumentWeightedMixin(object):
    """This mixin adds methods to set and get instrument weights. It also provides a hook that gets notified when new
    instrument weights have been set."""

    __instrument_weights_dict: tp.Dict[str, float]
    __instrument_weights_proxy: tp.Mapping[str, float]

    def set_instrument_weights(self, instrument_weights: tp.Optional[tp.Mapping[str, float]]) -> None:
        """
        Sets new instrument weights from the given mapping, which must map instrument names to their corresponding
        weights. The weights should be given as (interpretable) floats. Note that the object will not take ownership
        over the given mapping but make a copy of it.

        :param instrument_weights: mapping containing weights by instrument name or None to remove weights
        :return: None
        """

        instrument_weights = instrument_weights or {}

        try:
            internal_instrument_weights = self.__instrument_weights_dict
        except AttributeError:
            internal_instrument_weights = self.__instrument_weights_dict = {}
            self.__instrument_weights_proxy = MappingProxyType(internal_instrument_weights)

        # Clear old weights
        internal_instrument_weights.clear()

        for instrument, weight in instrument_weights.items():
            instrument = str(instrument)
            weight = float(weight)
            # If this happens, instrument is probably an object with a very generic __str__ method
            assert instrument not in internal_instrument_weights
            internal_instrument_weights[instrument] = weight

        # Notify about new weights
        self.__on_instrument_weights_set(self.__instrument_weights_proxy)

    def get_instrument_weights(self) -> tp.Mapping[str, float]:
        """
        Returns the instrument weights as a read-only dictionary, mapping instrument names to their corresponding
        weights.

        :return: a read-only dictionary mapping instrument names to their corresponding weights
        """

        try:
            return self.__instrument_weights_proxy
        except AttributeError:
            return MappingProxyType({})

    def get_instrument_weights_as_tuple(self):
        """
        Returns the instrument weights as a tuple containing the instrument/weight information.

        :return: tuple containing (instrument, weight) tuples
        """

        instrument_weights = self.get_instrument_weights()
        return tuple(instrument_weights.items())

    def get_normalized_weights(self, instruments: tp.Sequence[str]) -> tp.Tuple[float]:
        """
        Returns the weights of the given instruments as a list, normalized, so that the combined weights add up to one.
        This method will assign default weights to instruments for which no weight is known.

        :param instruments: instrument names as a sequence
        :return: iterator over
        """

        n_instruments = len(instruments)
        weight_list = []  # type: tp.List[float]

        if n_instruments == 0:
            return tuple(weight_list)

        known_weights = self.get_instrument_weights()
        summed_known_weight = sum(known_weights.values())

        if summed_known_weight > 0:
            default_weight = summed_known_weight / float(n_instruments)
        else:
            default_weight = 1.0

        for instrument in instruments:
            instr_weight = known_weights.get(instrument, default_weight)
            weight_list.append(instr_weight)

        summed_weight_out = sum(weight_list)

        try:
            return tuple(w / summed_weight_out for w in weight_list)
        except ZeroDivisionError:
            return tuple(itertools.repeat(0, n_instruments))

    def __on_instrument_weights_set(self, instrument_weights: tp.Mapping[str, float]) -> None:
        """This method is called when setting new instrument weights

        Override this method to get notified when new instrument weights are set.

        :param instrument_weights: new instrument weights (same as result of calling get_instrument_weights) as a
                                   read-only dictionary
        :return: None (not used)
        """

        pass


def get_logging_level_by_name(level_name: str) -> int:
    """
    Returns the numerical level of the given logging level, which can then be passed to :func:`logging.basicConfig`. The
    given level name must be one of:

        * DEBUG
        * INFO
        * WARNING
        * ERROR
        * CRITICAL

    :param level_name: one of ['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL']
    :return: numeric value of given logging level
    """

    numeric_level = getattr(logging, level_name.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s, choose between: %s" % (level_name, [
            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        ]))
    return numeric_level


def set_logging_level_by_name(level_name: str) -> None:
    """
    Sets the logging level of the :module:`logging` module by its logging name.

    :param level_name: interpreted as for :func:`beatsearch.utils.get_logging_level_by_name`
    :return: None
    """

    level = get_logging_level_by_name(level_name)
    logging.basicConfig(level=level)


def find_all_subclasses(cls: tp.Type) -> tp.List[tp.Type]:
    """
    Recursively finds all subclasses of the given class and returns them as a list.

    :param cls: class to find the subclasses of
    :return: list of subclasses of the given class
    """

    return cls.__subclasses__() + [g for s in cls.__subclasses__() for g in find_all_subclasses(s)]


def find_all_concrete_subclasses(cls: tp.Type) -> tp.List[tp.Type]:
    """
    Recursively finds all non-abstract subclasses of the given class and returns them as a list.

    :param cls: class to find the concrete subclasses of
    :return: list of non-abstract subclasses of the given class
    """

    return list(filter(lambda _cls: not isabstract(_cls), find_all_subclasses(cls)))


def iterable_to_str(
        iterable: tp.Sequence[tp.Any],
        nested_iterables: bool = True,
        iterable_types: tp.Iterable[tp.Type] = (tuple, list),
        separator: str = ",",
        boundary_chars: tp.Union[tp.Tuple[str, str], str] = "[]"
):
    """
    Converts the given tuple/list into a string, converting all elements to strings with str(element).

    :param iterable:                 iterable to convert into a string
    :param nested_iterables: when true, this method will also represent nested iterables as strings
    :param iterable_types:           iterable types to convert to strings, defaults to (tuple, list) (ignored if
                                     convert_nested_iterables is false)
    :param separator:                separator string between elements
    :param boundary_chars:           characters to use at the beginning and ending of the sequence, defaults to "[]"

    :return: string representation of iterable
    """

    str_elements = []
    for e in iterable:
        if not nested_iterables:
            str_elements.append(str(e))
            continue
        if any(isinstance(e, t) for t in iterable_types):
            e_str = iterable_to_str(e, nested_iterables, iterable_types, separator)
        else:
            e_str = str(e)
        str_elements.append(e_str)
    return "%s%s%s" % (boundary_chars[0], ("%s " % separator).join(str_elements), boundary_chars[1])


def iterable_nth(iterable, n, default=None):
    """
    Returns the nth item or a default value.

    From: https://docs.python.org/3/library/itertools.html#recipes

    :param iterable: iterable
    :param n: item index
    :param default: default value
    :return: nth item of given iterable
    """

    return next(itertools.islice(iterable, n, None), default)


__all__ = [
    'merge_dicts', 'format_timespan', 'print_progress_bar', 'friendly_named_class',
    'err_print', 'make_dir_if_not_exist', 'head_trail_iter', 'current_next_pair_iter', 'get_beatsearch_dir',
    'get_default_beatsearch_rhythms_fpath', 'no_callback', 'type_check_and_instantiate_if_necessary',
    'eat_args', 'color_variant', 'get_midi_files_in_directory', 'TupleView', 'most_common_element',
    'sequence_product', 'minimize_term_count', 'FileInfo', 'normalize_directory', 'QuantizableMixin', 'zip_equal',
    'InstrumentWeightedMixin', 'generate_unique_abbreviation', 'generate_abbreviations', 'Point2D', 'Dimensions2D',
    'Rectangle2D', 'get_logging_level_by_name', 'set_logging_level_by_name', 'find_all_subclasses',
    'find_all_concrete_subclasses', 'iterable_to_str', 'iterable_nth'
]

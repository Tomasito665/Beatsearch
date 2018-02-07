import os
import inspect
import itertools
from abc import abstractmethod, ABCMeta
from io import IOBase
from functools import wraps
import typing as tp
from collections import OrderedDict, namedtuple
import math
import numpy as np
import midi


class Unit(object):
    _units = OrderedDict()  # units by unit values
    _unit_names = []  # unit names

    def __init__(self, name, value, scale_factor_from_quarter_note):
        self._name = str(name)
        self._value = value
        self._scale_from_quarter = float(scale_factor_from_quarter_note)
        Unit._units[value] = self

    class UnknownTimeUnit(Exception):
        pass

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value

    @property
    def scale_factor_from_quarter_note(self):
        return self._scale_from_quarter

    @staticmethod
    def exists(unit_value):
        return unit_value in Unit._units

    @staticmethod
    def get(unit):  # type: (tp.Union[str, Unit]) -> Unit
        """
        Returns the Unit object for the given unit value, e.g. "quarters" or "eighths". If the given unit is already
        a Unit object, this method will return that.

        :return: Unit object for given unit value or Unit if given Unit as argument
        """

        try:
            unit_value = unit.value
        except AttributeError:
            unit_value = unit

        try:
            return Unit._units[unit_value]
        except KeyError:
            raise Unit.UnknownTimeUnit(unit_value)

    @staticmethod
    def get_unit_names():
        """
        Returns a tuple containing the names of the units.

        :return: tuple with unit names
        """

        if len(Unit._unit_names) != len(Unit._units):
            Unit._unit_names = tuple(unit.name for unit in Unit._units.values())
        return Unit._unit_names

    @classmethod
    def get_unit_by_name(cls, unit_name):
        """
        Returns a unit given its unit name.

        :param unit_name: unit name
        :return: Unit object
        """

        try:
            unit_ix = cls._unit_names.index(unit_name)
        except IndexError:
            raise Unit.UnknownTimeUnit("No time unit named \"%s\"" % unit_name)
        return tuple(cls._units.values())[unit_ix]


Unit.FULL = Unit("Full", "fulls", 0.25)
Unit.HALF = Unit("Half", "halves", 0.5)
Unit.QUARTER = Unit("Quarter", "quarters", 1.0)
Unit.EIGHTH = Unit("Eighth", "eighths", 2.0)
Unit.SIXTEENTH = Unit("Sixteenth", "sixteenths", 4.0)
Unit.THIRTY_SECOND = Unit("Thirty-second", "thirty-seconds", 8.0)


def convert_time(time, unit_from, unit_to, quantize=False):
    """
    Converts the given time to a given unit. The given units can be either a resolution (in Pulses
    Per Quarter Note) or one of these musical time units: 'fulls', 'halves', 'quarters', 'eighths or,
    'sixteenths'.

    :param time: time value
    :param unit_from: the original unit of the given time value.
    :param unit_to: the unit to convert the time value to.
    :param quantize: whether or not to round to the nearest target unit. If quantize is true, this
                     function will return an integer. If false, this function will return a float.
    :return: converted time
    """

    if unit_from == unit_to:
        return int(round(time)) if quantize else time

    if time == 0:
        return int(time) if quantize else float(time)

    try:
        # if unit_from is a time resolution
        time_in_quarters = time / float(unit_from)  # if unit_from is a time resolution
    except (ValueError, TypeError):
        # if unit_from is musical unit
        unit_from = Unit.get(unit_from)
        time_in_quarters = time / unit_from.scale_factor_from_quarter_note

    try:
        # if unit_to is a time resolution
        converted_time = time_in_quarters * float(unit_to)
    except (ValueError, TypeError):
        # if unit_to is a musical unit
        unit_to = Unit.get(unit_to)
        converted_time = time_in_quarters * unit_to.scale_factor_from_quarter_note

    return int(round(converted_time)) if quantize else converted_time


def concretize_unit(f_get_res=lambda *args, **kwargs: args[0].get_resolution()):
    """
    This function returns a decorator which converts the "unit" argument of a function to a resolution in PPQ.

    :param f_get_res: callable that should return the resolution. The result of this function will be replaced for
                      "unit" arguments with a value of 'ticks'. This callable is passed the *args and **kwargs of the
                      function being decorated.
    :return: a decorator function
    """

    def concretize_unit_decorator(func):
        arg_spec = inspect.getfullargspec(func)

        try:
            unit_arg_index = arg_spec.args.index("unit")
            n_args_before_default_args = len(arg_spec.args) - len(arg_spec.defaults)
            default_unit = arg_spec.defaults[unit_arg_index - n_args_before_default_args]
        except ValueError:
            unit_arg_index = -1
            if not arg_spec.keywords:
                raise ValueError("Function %s doesn't have a 'unit' argument, "
                                 "neither does it accept **kwargs" % func.__name__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Retrieve the unit or get the default unit if not given
            try:
                unit = kwargs["unit"]
                unit_in_kwargs = True
            except KeyError:
                try:
                    unit = args[unit_arg_index]
                except IndexError:
                    try:
                        unit = default_unit
                    except NameError:
                        raise ValueError("Required named 'unit' argument not found")
                unit_in_kwargs = False

            # Concretize the unit to the rhythm's resolution for tick unit
            if unit == 'ticks':
                unit = f_get_res(*args, **kwargs)
                if unit <= 0:
                    raise ValueError("Requested unit is \"ticks\", but resolution %i <= 0" % unit)

            # Replace unit by concretized unit
            if unit_in_kwargs:
                kwargs["unit"] = unit
            else:
                try:
                    args = list(args)
                    args[unit_arg_index] = unit
                except IndexError:
                    kwargs["unit"] = unit

            return func(*args, **kwargs)

        return wrapper

    return concretize_unit_decorator


class TimeSignature(object):
    """
    This class represents a musical time signature, consisting of a numerator and a denominator.
    """

    def __init__(self, numerator, denominator):
        self._numerator = numerator
        self._denominator = denominator

        if denominator == 4:
            self._beat_unit = Unit.QUARTER
        elif denominator == 8:
            self._beat_unit = Unit.EIGHTH
        else:
            raise ValueError("Unknown denominator: %s" % str(denominator))

    @property
    def numerator(self):
        return self._numerator

    @property
    def denominator(self):
        return self._denominator

    def get_beat_unit(self):
        return self._beat_unit

    def to_midi_event(self, metronome=24, thirty_seconds=8):
        return midi.TimeSignatureEvent(
            numerator=self.numerator,
            denominator=self.denominator,
            metronome=metronome,
            thirtyseconds=thirty_seconds
        )

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.numerator == other.numerator and self.denominator == other.denominator

    def __str__(self):
        return "%i/%i" % (self.numerator, self.denominator)

    @staticmethod
    def from_midi_event(midi_event: midi.TimeSignatureEvent):
        if not isinstance(midi_event, midi.TimeSignatureEvent):
            raise ValueError("Expected midi.TimeSignatureEvent")
        numerator = midi_event.get_numerator()
        denominator = midi_event.get_denominator()
        return TimeSignature(numerator, denominator)


class Rhythm(object, metaclass=ABCMeta):
    """Rhythm interface

    This class consists of abstract rhythm functionality. All rhythm classes must implement this interface. This class
    also provides generic functionality which makes use of the abstract methods.
    """

    class Precondition(object):
        """Preconditions for Rhythm methods"""

        class ResolutionNotSet(Exception):
            pass

        class TimeSignatureNotSet(Exception):
            pass

        @classmethod
        def needs_resolution(cls, f):
            @wraps(f)
            def wrapper(rhythm, *args, **kwargs):
                if rhythm.get_resolution() == 0:
                    raise cls.ResolutionNotSet
                return f(rhythm, *args, **kwargs)

            return wrapper

        @classmethod
        def needs_time_signature(cls, f):
            @wraps(f)
            def wrapper(rhythm, *args, **kwargs):
                if not rhythm.get_time_signature():
                    raise cls.TimeSignatureNotSet
                return f(rhythm, *args, **kwargs)

            return wrapper

    @abstractmethod
    def get_resolution(self) -> int:
        """
        Returns the tick resolution of this rhythm in pulses-per-quarter-note.

        :return: tick resolution of this rhythm in PPQN
        """

        raise NotImplementedError

    @abstractmethod
    def set_resolution(self, resolution: int):
        """
        Sets this rhythm's tick resolution and rescales the onsets to the new resolution.

        :param resolution: new tick resolution in PPQN
        :return: None
        """

        raise NotImplementedError

    @abstractmethod
    def __rescale_onset_ticks__(self, old_resolution: int, new_resolution: int) -> None:
        """
        Rescales the onset positions from one resolution to another. The given resolutions must be greater than zero.

        :param old_resolution: current resolution of the onsets
        :param new_resolution: resolution to scale the onsets to
        :return: None
        """

        raise NotImplementedError

    @abstractmethod
    def get_bpm(self) -> float:
        """
        Returns the tempo of this rhythm in beats per minute.

        :return: tempo of this rhythm in beats per minute
        """

        raise NotImplementedError

    @abstractmethod
    def set_bpm(self, bpm: tp.Union[float, int]) -> None:
        """
        Sets this rhythm's tempo in beats per minute.

        :param bpm: new tempo in beats per minute
        :return: None
        """

        raise NotImplementedError

    @abstractmethod
    def get_time_signature(self) -> tp.Union[TimeSignature, None]:
        """
        Returns the time signature of this rhythm.

        :return: the time signature of this rhythm as a TimeSignature object
        """

        raise NotImplementedError

    @abstractmethod
    def set_time_signature(self, time_signature: tp.Union[TimeSignature,
                                                          tp.Tuple[int, int],
                                                          tp.Sequence[int], None]) -> None:
        """
        Sets the time signature of this rhythm. None to remove the time signature.

        :param time_signature: new time signature either as an iterable (numerator, denominator) or as a TimeSignature
                               or None to remove the time signature
        :return: None
        """

        raise NotImplementedError

    @abstractmethod
    def get_duration_in_ticks(self) -> int:
        """
        Returns the duration of this rhythm in ticks.

        :return: duration of this rhythm in ticks
        """

        raise NotImplementedError

    @abstractmethod
    def set_duration_in_ticks(self, requested_duration: int) -> int:
        """
        Sets the duration of the rhythm to the closest duration possible to the requested duration and returns the
        actual new duration.

        :param requested_duration: requested new duration
        :return: actual new duration
        """

        raise NotImplementedError

    @abstractmethod
    def get_last_onset_tick(self) -> int:
        """
        Returns the position of the last onset of this rhythm in ticks or -1 if this rhythm is empty.

        :return: position of last onset in ticks or -1 if this rhythm is empty
        """

        raise NotImplementedError

    @abstractmethod
    def get_onset_count(self) -> int:
        """
        Returns the number of onsets in this rhythm.

        :return: number of onsets in this rhythm
        """

        raise NotImplementedError

    ########################
    # Non-abstract methods #
    ########################

    @concretize_unit()
    def get_duration(self, unit="ticks") -> float:
        """
        Returns the duration of this rhythm in the given time unit.

        :param unit: time unit
        :return: duration of this rhythm in the given time unit
        """

        duration_in_ticks = self.get_duration_in_ticks()
        resolution = self.get_resolution()
        return convert_time(duration_in_ticks, resolution, unit, quantize=False)

    @concretize_unit()
    def set_duration(self, duration: tp.Union[int, float], unit="ticks") -> None:
        """
        Sets the duration of this rhythm in the given time unit.

        :param duration: new duration
        :param unit: time unit of the given duration
        :return: None
        """

        resolution = self.get_resolution()
        duration_in_ticks = convert_time(duration, unit, resolution)
        self.set_duration_in_ticks(round(duration_in_ticks))

    @Precondition.needs_time_signature
    @concretize_unit()
    def get_beat_duration(self, unit="ticks") -> float:
        """
        Returns the duration of one musical beat, based on the time signature.

        :param unit: unit of the returned duration
        :return: the duration of one beat in the given unit
        :raises TimeSignatureNotSet: if no time signature has been set
        """

        time_signature = self.get_time_signature()
        beat_unit = time_signature.get_beat_unit()
        return convert_time(1.0, beat_unit, unit, quantize=False)

    @Precondition.needs_time_signature
    @concretize_unit()
    def get_measure_duration(self, unit="ticks") -> float:
        """
        Returns the duration of one musical measure, based on the time signature.

        :param unit: unit of the returned duration
        :return: the duration of one measure in the given unit
        :raises TimeSignatureNotSet: if no time signature has been set
        """

        time_signature = self.get_time_signature()
        numerator = time_signature.numerator
        beat_unit = time_signature.get_beat_unit()
        return convert_time(numerator, beat_unit, unit, quantize=False)

    def get_duration_in_measures(self):
        """
        Returns the duration of this rhythm in musical measures as a floating point number.

        :return: the duration of this rhythm in measures as a floating point number
        :raises TimeSignatureNotSet: if no time signature has been set
        """

        measure_duration = self.get_measure_duration("ticks")
        duration = self.get_duration_in_ticks()
        return duration / measure_duration

    ##############
    # Properties #
    ##############

    # Resolution

    @property
    def resolution(self) -> int:
        """See Rhythm.set_resolution and Rhythm.get_resolution"""
        return self.get_resolution()

    @resolution.setter
    def resolution(self, resolution: tp.Union[float, int]):  # setter
        self.set_resolution(resolution)

    @resolution.deleter
    def resolution(self):
        self.set_resolution(0)

    # BPM

    @property
    def bpm(self) -> float:
        """See Rhythm.set_bpm and Rhythm.get_bpm"""
        return self.get_bpm()

    @bpm.setter
    def bpm(self, bpm: tp.Union[float, int]):  # setter
        self.set_bpm(bpm)

    @bpm.deleter
    def bpm(self):
        self.set_bpm(0)

    # Time signature

    @property
    def time_signature(self) -> tp.Union[TimeSignature, None]:
        """See Rhythm.set_time_signature and Rhythm.get_time_signature"""
        return self.get_time_signature()

    @time_signature.setter
    def time_signature(self, time_signature: tp.Union[TimeSignature,
                                                      tp.Tuple[int, int],
                                                      tp.Sequence[int], None]) -> None:
        self.set_time_signature(time_signature)

    @time_signature.deleter
    def time_signature(self):
        self.set_time_signature(None)

    # Duration in ticks

    @property
    def duration_in_ticks(self) -> int:
        """See Rhythm.set_duration_in_ticks and Rhythm.get_duration_in_ticks"""
        return self.get_duration_in_ticks()

    @duration_in_ticks.setter
    def duration_in_ticks(self, duration: int) -> None:  # setter
        self.set_duration_in_ticks(duration)


class RhythmBase(Rhythm, metaclass=ABCMeta):
    """Rhythm abstract base class

    This class extends the Rhythm interface and adds state for all of its properties (resolution, bpm, time_signature
    and duration_in_ticks). It also implements the getters and setters of these properties.

    Note that this class does not add onset state and does not implement onset-related functionality. Rhythm.\
    get_last_onset_tick, Rhythm.get_onset_count and Rhythm.__rescale_onset_ticks__ remain abstract and should be
    implemented in subclasses.
    """

    def __init__(self, **kwargs):
        """Sets up state for generic rhythm properties

        :param kwargs: unused
        """

        self._resolution = 0              # type: int
        self._bpm = 0                     # type: int
        self._time_signature = None       # type: tp.Union[TimeSignature, None]
        self._duration_in_ticks = 0       # type: int

        # Note: we don't call any setters because subclasses might not be finished initializing

    def post_init(self, **kwargs) -> None:
        """
        Multi-setter that can be used in RhythmBase subclass constructors to make sure that these are initialised after
        specific subclass initialization. Only the properties will be set that are given.

        :param kwargs
            resolution: sets Rhythm.resolution
            bpm: sets Rhythm.bpm
            time_signature: sets Rhythm.time_signature
            duration_in_ticks: sets Rhythm.duration_in_ticks
            duration: also sets Rhythm.duration_in_ticks

        :return: None
        """

        # TODO c'mon, I can do better than this...

        if "resolution" in kwargs:
            self.set_resolution(kwargs['resolution'])
        if "bpm" in kwargs:
            self.set_bpm(kwargs['bpm'])
        if "time_signature" in kwargs:
            self.set_time_signature(kwargs['time_signature'])
        if "duration_in_ticks" in kwargs:
            self.set_duration_in_ticks(kwargs['duration_in_ticks'])
        elif "duration" in kwargs:
            self.set_duration_in_ticks(kwargs['duration'])

    class TimeSignatureNotSet(Exception):
        pass

    def get_resolution(self) -> int:
        """
        Returns the tick resolution of this rhythm in PPQN.

        :return: tick resolution of this rhythm in PPQN
        """

        return self._resolution

    def set_resolution(self, new_res: int):
        """
        Sets the tick resolution of this rhythm. If this rhythm already has a resolution, this method will automatically
        scale the onsets within this rhythm from the old resolution to the new resolution. This method will always call
        set_duration_in_ticks, even if the duration didn't rescale.

        :param new_res: new tick resolution in PPQN
        :return: None
        """

        new_res = int(new_res)
        old_res = self.resolution

        if new_res < 0:
            raise ValueError("expected positive resolution but got %i" % new_res)

        old_dur = self.get_duration_in_ticks()

        if old_res > 0 and new_res > 0:
            self.__rescale_onset_ticks__(old_res, new_res)
            new_dur = convert_time(old_dur, old_res, new_res, quantize=True)
        else:
            new_dur = old_dur

        self._resolution = new_res
        self.set_duration_in_ticks(new_dur)

    def get_bpm(self) -> float:
        """
        Returns the tempo of this rhythm in beats per minute.

        :return: tempo of this rhythm in beats per minute
        """

        return self._bpm

    def set_bpm(self, bpm: tp.Union[float, int]) -> None:
        """
        Sets this rhythm's tempo in beats per minute.

        :param bpm: new tempo in beats per minute
        :return: None
        """

        self._bpm = float(bpm)

    def get_time_signature(self) -> tp.Union[TimeSignature, None]:
        """
        Returns the time signature of this rhythm.

        :return: the time signature of this rhythm as a TimeSignature object
        """

        return self._time_signature

    def set_time_signature(self, time_signature: tp.Union[TimeSignature,
                                                          tp.Tuple[int, int],
                                                          tp.Sequence[int], None]) -> None:
        """
        Sets the time signature of this rhythm. None to remove the time signature.

        :param time_signature: new time signature either as an iterable (numerator, denominator) or as a TimeSignature
                               or None to remove the time signature
        :return: None
        """

        if not time_signature:
            self._time_signature = None
            return

        try:
            # if given an iterable
            numerator, denominator = time_signature
        except TypeError:
            # if given a TimeSignature object
            numerator = time_signature.numerator
            denominator = time_signature.denominator

        self._time_signature = TimeSignature(numerator, denominator)

    def get_duration_in_ticks(self) -> int:
        """
        Returns the duration of this rhythm in ticks.

        :return: duration of this rhythm in ticks
        """

        return self._duration_in_ticks

    def set_duration_in_ticks(self, requested_duration: int) -> int:
        """
        Tries to set the duration of this rhythm to the requested duration and returns the actual new duration. The
        duration of the rhythm can't be less than the position of the last note in this rhythm. If a duration is
        requested that is less than the last note's position, the duration will be set to that last note's position.

        :param requested_duration: new duration in ticks
        :return: the new duration
        """

        last_onset_position = self.get_last_onset_tick()
        self._duration_in_ticks = max(last_onset_position, int(requested_duration))
        return self._duration_in_ticks


# TODO Add test to check that all fields of Onset are equal
Onset = np.dtype([("tick", np.uint32), ("velocity", np.uint32)])
"""Numpy structured array type for onsets

This is a numpy structured array type for numpy onset arrays. Each onset has two properties:
    tick      - the absolute tick position of this onset within the rhythm as an integer (np.int32)
    velocity  - the MIDI velocity of this note with a range of [0, 127] as an integer (np.int32)
"""


class OnsetsNotInChronologicalOrder(Exception):
    """Exception thrown when two adjacent onsets are not in chronological order"""

    def __init__(self, tick_a: int, tick_b: int):
        msg = "<..., %i, !%i!, ...>" % (tick_a, tick_b)
        super().__init__(msg)


class OnsetFactory(object, metaclass=ABCMeta):
    """Utility class providing helper methods to create onsets

    This is a utility class that provides functionality to create onsets.
    """

    def __init__(self):
        raise TypeError("this is a utility class and shouldn't be initialized")

    @classmethod
    def create(cls, onset_descriptions, default_velocity=100):
        """Creates a series of onsets, given an iterable with onset descriptions

        Creates and returns a series of onsets, given an iterable with onset descriptions. The onset descriptions must
        be in chronological order. If not, this method will raise an OnsetsNotInChronologicalOrder exception. Onset
        descriptions consist of two properties:
            - tick (required):        absolute tick position of the onset
            - velocity (optional):    the velocity of the onset

        :param onset_descriptions: an iterable of onset descriptions, where each onset description can be either:
                                        - list/tuple [tick] or [tick, onset]
                                        - dictionary {'tick'} or {'tick', 'onset'}
                                        - object {tick} or {tick, onset}
                                        - scalar representing the tick

        :param default_velocity: this velocity will be used when an onset description has no velocity specified
        :return: a numpy ndarray of structured onset arrays (see Onset)
        :raises: OnsetsNotInChronologicalOrder
        """

        n_onsets = len(onset_descriptions)

        def parse_onsets():
            prev_tick = -1
            for desc in onset_descriptions:
                tick, velocity = cls.__parse_onset(desc, default_velocity)

                if tick < 0:
                    raise ValueError("expected tick equal or greater than zero but got %i" % tick)
                if velocity < 0:
                    raise ValueError("expected velocity equal or greater than zero but got %i" % velocity)

                if tick < prev_tick:
                    raise OnsetsNotInChronologicalOrder(prev_tick, tick)

                yield tick, velocity
                prev_tick = tick

        return np.fromiter(parse_onsets(), dtype=Onset, count=n_onsets)

    @staticmethod
    def empty():
        """Creates an empty onset array

        :return: empty onset array
        """

        return np.empty(0, dtype=Onset)

    @classmethod
    def __parse_onset(cls, onset_description, default_velocity):
        tick, velocity = None, None

        for parse in cls.__ONSET_PARSERS:
            try:
                result = parse(onset_description)
                tick, velocity = (int(x) for x in result)
            except (TypeError, ValueError):
                tick, velocity = None, None
                continue
            break

        if tick is None:
            assert velocity is None
            raise ValueError("failed to parse onset description \"%s\"" % onset_description)

        if velocity < 0:
            velocity = default_velocity

        return tick, velocity

    @staticmethod
    def __from_iterator(onset):
        iterator = iter(onset)
        try:
            tick = next(iterator)
        except StopIteration:
            raise ValueError("iterator should yield at least one value (tick)")
        try:
            velocity = next(iterator)
        except StopIteration:
            velocity = -1
        return tick, velocity

    @staticmethod
    def __from_dictionary(onset):
        try:
            tick = onset['tick']
        except KeyError:
            raise TypeError
        try:
            velocity = onset['velocity']
        except KeyError:
            velocity = -1
        return tick, velocity

    @staticmethod
    def __from_object(onset):
        try:
            tick = onset.tick
        except AttributeError:
            raise TypeError
        try:
            velocity = onset.velocity
        except AttributeError:
            velocity = -1
        return tick, velocity

    @staticmethod
    def __from_scalar(onset):
        tick = round(onset)
        return tick, -1

    # noinspection PyUnresolvedReferences
    __ONSET_PARSERS = tuple(m.__func__ for m in (__from_iterator, __from_dictionary, __from_object, __from_scalar))


def rescale_onset_series(onsets: np.ndarray, resolution_from: tp.Union[int, float], resolution_to: tp.Union[int, float]):
    """Rescales a given onset series from one resolution to another

    Rescales the tick position of the onsets in the given onset series from one resolution to another. This function is
    destructive.

    :param onsets: a numpy array with dtype Onset
    :param resolution_from: original resolution of the onset tick position
    :param resolution_to: new resolution to rescale the onsets to
    :return: None
    """

    dtype_tick = Onset[0]  # np.int32
    flat_view = onsets.view(dtype=dtype_tick)  # [t, v, t, v, ...]
    ticks_view = flat_view[::2]  # [t, t, t, ...]
    scale_factor = float(resolution_to) / float(resolution_from)

    # noinspection PyArgumentList
    np.multiply(ticks_view, scale_factor, out=ticks_view, casting="unsafe")


class MonophonicRhythm(Rhythm, metaclass=ABCMeta):
    """Monophonic rhythm interface

    Interface for monophonic rhythms.
    """

    @abstractmethod
    def get_onsets(self):
        """
        Returns the onsets within this rhythm as a tuple of onsets, where each onset is an instance of Onset.

        :return: the onsets within this rhythm as a tuple of Onset objects
        """

        raise NotImplementedError

    @abstractmethod
    def set_onsets(self, onsets):  #: tp.Union[tp.Iterable[Onset], tp.Iterable[tp.Tuple[int, int]]]):
        """
        Sets the onsets of this rhythm.

        :param onsets: onsets as an iterable of (absolute tick, velocity) tuples or as Onset objects
        :return: None
        """

        raise NotImplementedError

    @property
    def onsets(self):
        """See MonophonicRhythm.get_onsets"""
        return self.get_onsets()

    @onsets.setter
    def onsets(self, onsets):
        """See MonophonicRhythm.set_onsets"""
        self.set_onsets(onsets)

    @onsets.deleter
    def onsets(self):
        self.set_onsets([])

    def get_last_onset_tick(self) -> int:  # implements Rhythm.get_last_onset_tick
        try:
            return self.onsets[-1]['tick']
        except IndexError:
            return -1

    def get_onset_count(self) -> int:  # implements Rhythm.get_onset_count
        return len(self.onsets)


class MonophonicRhythmRepresentationsMixin(MonophonicRhythm, metaclass=ABCMeta):

    #####################################
    # Monophonic rhythm representations #
    #####################################

    @Rhythm.Precondition.needs_resolution
    @concretize_unit()
    def get_binary(self, unit="eighths"):
        """
        Returns the binary representation of the note onsets of this rhythm where each step where a note onset
        happens is denoted with a 1; otherwise with a 0. The given resolution is the resolution in PPQ (pulses per
        quarter note) of the binary vector.

        :param unit:
        :return: the binary representation of the note onsets of the given pitch
        """

        resolution = self.get_resolution()
        duration = self.get_duration(unit)
        binary_string = [0] * int(math.ceil(duration))

        for onset in self.onsets:
            pulse = convert_time(onset.tick, resolution, unit, quantize=True)
            try:
                binary_string[pulse] = 1
            except IndexError:
                pass  # when quantization pushes note after end of rhythm

        return binary_string

    @Rhythm.Precondition.needs_resolution
    @concretize_unit()
    def get_pre_note_inter_onset_intervals(self, unit="ticks"):
        """
        Returns the time difference between the current note and the previous note for all notes of this rhythm in
        eighths. The first note will return the time difference with the start of the rhythm.

        For example, given the Rumba Clave rhythm:
          X--X---X--X-X---
          0  3   4  3 2

        :return: pre note inter-onset interval vector
        """

        resolution = self.get_resolution()
        current_tick = 0
        intervals = []

        for onset in self.onsets:
            delta_tick = onset.tick - current_tick
            intervals.append(convert_time(delta_tick, resolution, unit))
            current_tick = onset.tick

        return intervals

    @Rhythm.Precondition.needs_resolution
    @concretize_unit()  # TODO Add cyclic option to include the offset in the last onset's interval
    def get_post_note_inter_onset_intervals(self, unit="ticks", quantize=False):
        """
        Returns the time difference between the current note and the next note for all notes in this rhythm in eighths.
        The last note will return the time difference with the end (duration) of the rhythm.

        For example, given the Rumba Clave rhythm:
          X--X---X--X-X---
          3  4   3  2 4

        :return: post note inter-onset interval vector
        """

        intervals = []
        onset_positions = itertools.chain((onset.tick for onset in self.onsets), [self.duration_in_ticks])
        last_onset_tick = -1

        for onset_tick in onset_positions:
            if last_onset_tick < 0:
                last_onset_tick = onset_tick
                continue

            delta_in_ticks = onset_tick - last_onset_tick
            delta_in_units = convert_time(delta_in_ticks, self.get_resolution(), unit, quantize=quantize)

            intervals.append(delta_in_units)
            last_onset_tick = onset_tick

        return intervals

    def get_interval_histogram(self, unit="eighths"):
        """
        Returns the number of occurrences of all the inter-onset intervals from the smallest to the biggest
        interval. The inter onset intervals are retrieved with get_post_note_inter_onset_intervals().

        For example, given the Rumba Clave rhythm, with inter-onset vector [3, 4, 3, 2, 4]:
            (
                [1, 2, 2],  # occurrences
                [2, 3, 4]   # bins (interval durations)
            )


        :return: an (occurrences, bins) tuple
        """

        intervals = self.get_post_note_inter_onset_intervals(unit, quantize=True)
        histogram = np.histogram(intervals, tuple(range(min(intervals), max(intervals) + 2)))
        occurrences = histogram[0].tolist()
        bins = histogram[1].tolist()[:-1]
        return occurrences, bins

    def get_binary_schillinger_chain(self, unit='ticks', values=(1, 0)):
        """
        Returns the Schillinger notation of this rhythm where each onset is a change of a "binary note".

        For example, given the Rumba Clave rhythm and with values (0, 1):
          X--X---X--X-X---
          0001111000110000

        However, when given the values (1, 0), the schillinger chain will be the opposite:
          X--X---X--X-X---
          1110000111001111

        :param unit: the unit to quantize on ('ticks' is no quantization)
        :param values: binary vector to be used in the schillinger chain. E.g. when given ('a', 'b'), the returned
                       schillinger chain will consist of 'a' and 'b'.
        :return: Schillinger rhythm vector as a list
        """

        chain, i = self.get_binary(unit), 0
        value_i = 1
        while i < len(chain):
            if chain[i] == 1:
                value_i = 1 - value_i
            chain[i] = values[value_i]
            i += 1
        return chain

    def get_chronotonic_chain(self, unit="ticks"):
        """
        Returns the chronotonic chain representation of this rhythm.

        For example, given the Rumba Clave rhythm:
          X--X---X--X-X---
          3334444333224444

        :param unit: unit
        :return: the chronotonic chain as a list
        """

        chain, i, delta = self.get_binary(unit), 0, 0
        while i < len(chain):
            if chain[i] == 1:
                j = i + 1
                while j < len(chain) and chain[j] == 0:
                    j += 1
                delta = j - i
            chain[i] = delta
            i += 1
        return chain

    def get_interval_difference_vector(self, cyclic=True, unit="ticks", quantize=False):
        """
        Returns the interval difference vector (aka difference of rhythm vector). For each
        note, this is the difference between the current onset interval and the next onset
        interval. So, if N is the number of onsets, the returned vector will have a length
        of N - 1. This is different with cyclic rhythms, where the last onset's interval is
        compared with the first onset's interval. In this case, the length will be N.

        For example, given the post-note inter-onset interval vector for the Rumba clave:
          [3, 4, 3, 2, 4]

        The interval difference vector would be:
           With cyclic set to False: [4/3, 3/4, 2/3, 4/2]
           With cyclic set to True:  [4/3, 3/4, 2/3, 4/2, 3/4]

        :param cyclic: whether or not to treat this rhythm as a cyclic rhythm or not
        :param unit: time unit
        :param quantize: whether or not the inter onset interval vector should be quantized
        :return: interval difference vector
        """

        vector = self.get_post_note_inter_onset_intervals(unit, quantize)
        if cyclic:
            vector.append(vector[0])
        i = 0
        while i < len(vector) - 1:
            try:
                vector[i] = vector[i + 1] / float(vector[i])
            except ZeroDivisionError:
                vector[i] = float('inf')
            i += 1
        vector.pop()
        return vector

    @Rhythm.Precondition.needs_resolution
    @concretize_unit()
    def get_onset_times(self, unit="ticks", quantize=False):
        """
        Returns the absolute onset times of the notes in this rhythm.

        :param unit: the unit of the onset times
        :param quantize: whether or not the onset times must be quantized to the given unit
        :return: a list with the onset times of this rhythm
        """

        return [convert_time(onset[0], self.get_resolution(), unit, quantize) for onset in self.onsets]


class MonophonicRhythmBase(MonophonicRhythmRepresentationsMixin, MonophonicRhythm, metaclass=ABCMeta):
    """Monophonic rhythm base class implementing MonophonicRhythm

    Abstract base class for monophonic rhythms. This class implements MonophonicRhythm.get_onsets, adding onset state
    to subclasses. Note that this class does not extend RhythmBase and therefore does NOT add rhythm base state like
    bpm, resolution, time signature, etc.

    This class inherits all monophonic rhythm representations from the MonophonicRhythmRepresentationsMixin class.
    """

    def __init__(self, onsets):
        """
        Creates a new monophonic rhythm from the given onsets. The onsets will be stored as MonophonicOnset.Onset
        named tuples.

        :param onsets: An iterable returning an (absolute tick, velocity) tuple for each iteration. The onsets should
                       be given in chronological order.
        :raises OnsetsNotInChronologicalOrder
        """

        self._onsets = OnsetFactory.empty()
        self.set_onsets(onsets)

    def get_onsets(self):
        """
        Returns the onsets within this rhythm as a tuple of onsets, where each onset is an instance of Onset.

        :return: the onsets within this rhythm as a tuple of MonophonicRhythmImpl.Onset objects
        """

        return self._onsets

    def set_onsets(self, onset_descriptions):
        """Sets the onsets of this rhythm

        Sets the onsets of this rhythm. The given onsets must be in chronological order. If not, this method will raise
        an OnsetsNotInChronologicalOrder exception.

        :param onset_descriptions: an iterable of onset descriptions (see OnsetFactory.create) or None to remove the
                                   onsets
        :return: None
        :raises OnsetsNotInChronologicalOrder
        """

        if onset_descriptions is None:
            self._onsets = OnsetFactory.empty()
        else:
            self._onsets = OnsetFactory.create(onset_descriptions)

    # implements Rhythm.__rescale_onset_ticks__
    def __rescale_onset_ticks__(self, old_resolution: int, new_resolution: int):
        rescale_onset_series(self._onsets, old_resolution, new_resolution)


class MonophonicRhythmImpl(RhythmBase, MonophonicRhythmBase):
    """Implements both rhythm base and monophonic rhythm base"""

    def __init__(self, onsets=None, **kwargs):
        """
        Creates a new monophonic rhythm with the given onsets.

        :param onsets: An iterable returning an (absolute tick, velocity) tuple for each iteration. The tick
                       resolution should equal the parent's resolution. The onsets should be given in chronological
                       order.
        :param kwargs: Post-init keyword arguments. See RhythmBase.post_init.
        """

        RhythmBase.__init__(self)
        MonophonicRhythmBase.__init__(self, onsets)
        self.post_init(**kwargs)


class MonophonicRhythmFactory(object):
    def __init__(self):
        raise Exception("can't instantiate this class, it is a factory and only contains static methods")

    @classmethod
    def from_string(cls, onset_string, onset_character="x", velocity=100, resolution=4) -> MonophonicRhythmImpl:
        """
        Creates a new monophonic rhythm, given a given string. Each character in the string will represent one tick and
        each onset character will represent an onset in the rhythm, e.g. "x--x---x--x-x---", given onset character "x".

        :param onset_string: onset string where each onset character will result in an onset, the length of the string
                             will determine the duration of the rhythm
        :param onset_character: the onset character
        :param velocity: the velocity of the onsets (velocity is the same for all onsets)
        :param resolution: resolution in pulses per quarter note (if res=4, four characters in the onset
                           string will represent one quarter note)
        :return: monophonic rhythm object
        """

        binary_string = tuple((char == onset_character) for char in onset_string)
        return cls.from_binary_chain(binary_string, velocity, resolution)

    @classmethod
    def from_binary_chain(cls, binary_chain, velocity=100, resolution=4) -> MonophonicRhythmImpl:
        """
        Creates a new monophonic rhythm, given a binary chain (iterable). Each element in the iterable represents one
        tick. If the element is True, that will result in an onset, e.g. [1, 0, 1, 0, 1, 1, 1, 0].

        :param binary_chain: iterable containing true or false elements
        :param velocity: the velocity of the onsets (velocity is the same for all onsets)
        :param resolution: resolution in pulses per quarter note (if res=4, four elements in the binary chain will
                           represent one quarter note)
        :return: monophonic rhythm object
        """

        onsets = filter(None, ((ix, velocity) if atom else None for ix, atom in enumerate(binary_chain)))
        return MonophonicRhythmImpl(onsets=tuple(onsets), duration=len(binary_chain), resolution=resolution)


MonophonicRhythm.create = MonophonicRhythmFactory


class SlavedRhythmBase(Rhythm, metaclass=ABCMeta):
    """Rhythm abstract base class

    This class extends the Rhythm interface and implements its property getters and setters. Each SlavedRhythmBase
    instance is slaved to a parent rhythm. Calls to rhythm property getters are redirected to the parent. Calls to
    setters will result in an AttributeError. Thus, the slaved rhythm can read but not write the properties of the
    parent.
    """

    class ParentPropertyAccessError(Exception):
        def __init__(self, method_name):
            super().__init__("Slaved rhythms have read-only properties, use parent.%s" % method_name)

    class ParentNotSet(Exception):
        pass

    def __init__(self, parent: Rhythm = None):
        """
        Creates a new dependent rhythm parented to the given parent-rhythm or parented, if given.

        :param parent: parent rhythm
        """

        self._parent = None  # type: Rhythm
        self.set_parent(parent)

    def get_parent(self) -> tp.Union[Rhythm, None]:
        """Returns the parent

        Returns the parent pf this slave rhythm or None if it doesn't have a parent.
        """
        return self._parent

    def set_parent(self, parent: tp.Union[Rhythm, None]):
        """Sets the parent

        Sets the parent of this slave rhythm.

        :param parent: parent or None to remove parent
        :return: None
        """

        self._parent = parent

    @property
    def parent(self) -> Rhythm:
        """See SlavedRhythmBase.set_parent and SlavedRhythmBase.get_parent"""
        return self.get_parent()

    @parent.setter
    def parent(self, parent: tp.Union[Rhythm, None]):
        self.set_parent(parent)

    ###############################
    # Redirected property getters #
    ###############################

    def get_resolution(self) -> int:
        """Returns the resolution of the parent"""
        return self.__check_and_get_parent().get_resolution()

    def get_bpm(self) -> float:
        """Returns the bpm of the parent"""
        return self.__check_and_get_parent().get_bpm()

    def get_time_signature(self) -> tp.Union[TimeSignature, None]:
        """Returns the time signature of the parent"""
        return self.__check_and_get_parent().get_time_signature()

    def get_duration_in_ticks(self) -> int:
        """Returns the tick duration of the parent"""
        return self.__check_and_get_parent().get_duration_in_ticks()

    def set_resolution(self, resolution: int):
        """Raises a ParentPropertyAccessError exception"""
        raise self.ParentPropertyAccessError("set_resolution")

    def set_bpm(self, bpm: tp.Union[float, int]) -> None:
        """Raises a ParentPropertyAccessError exception"""
        raise self.ParentPropertyAccessError("set_bpm")

    def set_time_signature(
            self, time_signature: tp.Union[TimeSignature, tp.Tuple[int, int], tp.Sequence[int], None]) -> None:
        """Raises a ParentPropertyAccessError exception"""
        raise self.ParentPropertyAccessError("set_time_signature")

    def set_duration_in_ticks(self, requested_duration: int) -> None:
        """Raises a ParentPropertyAccessError exception"""
        raise self.ParentPropertyAccessError("set_duration_in_ticks")

    # used internally, raises a ParentNotSet exception if parent not set and returns the parent
    def __check_and_get_parent(self) -> Rhythm:
        parent = self._parent
        if not parent:
            raise self.ParentNotSet
        return parent


class Track(MonophonicRhythmBase, SlavedRhythmBase):
    """Represents one track of a polyphonic rhythm

    A polyphonic rhythm consists of multiple monophonic rhythms; tracks. Each of those tracks is represented by one
    instance of this class. A track is a slaved rhythm, parented to the polyphonic rhythm to which it belongs.
    """

    def __init__(self, onsets=None, track_name: str = "", parent: Rhythm = None):
        """
        Creates a new rhythm track.

        :param onsets: An iterable returning an (absolute tick, velocity) tuple for each iteration. The tick
                       resolution should equal the parent's resolution. The onsets should be given in chronological
                       order.
        :param track_name: The name of this track. This can't be changed after instantiation.
        :param parent: The polyphonic rhythm which this track belongs to
        """

        MonophonicRhythmBase.__init__(self, onsets=onsets)
        SlavedRhythmBase.__init__(self, parent=parent)

        self._track_name = str(track_name)  # type: str
        self._parent = parent               # type: PolyphonicRhythmImpl

    def get_name(self):  # type: () -> str
        """
        Returns the name of this track. Note that there is a getter but not a setter for a track's name. The name of
        a track can not be changed after initialization.
        :return: track name
        """

        return self._track_name

    @property
    def name(self):
        """See Track.get_name. This property is read-only"""
        return self.get_name()


class PolyphonicRhythm(Rhythm, metaclass=ABCMeta):

    class TrackNameError(Exception):
        """Thrown if there's something wrong with a track name"""
        pass

    class EquallyNamedTracksError(TrackNameError):
        """Thrown by set_tracks when given multiple tracks with same name"""
        pass

    class IllegalTrackName(TrackNameError):
        """Thrown by set_tracks if __validate_track_name__ returns False"""
        pass

    @abstractmethod
    def set_tracks(self, tracks: tp.Iterable[Track], resolution: int) -> None:
        """
        Sets the tracks and updates the resolution of this rhythm. The given tracks will automatically be parented to
        this polyphonic rhythm. This method also resets the duration of the rhythm to the position of the last onset
        in the given tracks.

        :param tracks: iterator yielding PolyphonicRhythmImpl.Track objects
        :param resolution: resolution of the onsets in the given tracks
        :return: None
        """

        raise NotImplementedError

    @abstractmethod
    def get_track_iterator(self) -> tp.Iterator[Track]:
        """
        Returns an iterator over the tracks of this polyphonic rhythm. Each iteration yields a PolyphonicRhythmImpl.Track
        object.

        :return: iterator over this rhythm's tracks yielding PolyphonicRhythmImpl.Track objects
        """

        raise NotImplementedError

    @abstractmethod
    def get_track_by_name(self, track_name: str):
        """
        Returns the track with the given name or None if this rhythm has no track with the given name.

        :param track_name: track name
        :return: Track object or None
        """

        raise NotImplementedError

    @abstractmethod
    def get_track_count(self):
        """
        Returns the number of tracks within this rhythm.

        :return: number of tracks
        """

        raise NotImplementedError

    @classmethod
    def create_tracks(cls, **onsets_by_track_name: tp.Iterable[tp.Tuple[int, int]]) -> tp.Generator[Track, None, None]:
        # TODO add docstring
        for name, onsets in onsets_by_track_name.items():
            yield Track(onsets=onsets, track_name=name)

    #####################################
    # Polyphonic rhythm representations #
    #####################################

    # TODO remove duplicate functionality (see MonophonicRhythm.get_interval_histogram)
    def get_interval_histogram(self, unit="ticks") -> (int, int):
        """
        Returns the interval histogram of all the tracks combined.

        :return: combined interval histogram of all the tracks in this rhythm
        """

        intervals = []

        for track in self.get_track_iterator():
            track_intervals = track.get_post_note_inter_onset_intervals(unit, quantize=True)
            intervals.extend(track_intervals)

        histogram = np.histogram(intervals, tuple(range(min(intervals), max(intervals) + 2)))
        occurrences = histogram[0].tolist()
        bins = histogram[1].tolist()[:-1]
        return occurrences, bins


class PolyphonicRhythmImpl(RhythmBase, PolyphonicRhythm):

    def __init__(
            self, tracks: tp.Iterable[Track] = tuple(),
            resolution: int = 0,
            **kwargs,
    ):
        """
        Creates a new polyphonic rhythm with the given tracks. The tick resolution of the onsets in the tracks is passed
        as second argument and is required when the tracks are given to the constructor.

        :param tracks: sequence of tracks
        :param resolution: the resolution of the given tracks, this parameter is unused if no tracks were given
        :param kwargs: Post-init keyword arguments. See RhythmBase.post_init.
        """

        super().__init__()
        self._tracks = OrderedDict()  # type: tp.Dict[str, Track]

        if tracks:
            if resolution is None:
                raise TypeError("given tracks without resolution")
            self.set_tracks(tracks, resolution)

        self.post_init(**kwargs)

    def set_tracks(self, tracks: tp.Iterable[Track], resolution: int) -> None:
        """
        Sets the tracks and updates the resolution of this rhythm. The given tracks will automatically be parented to
        this polyphonic rhythm. This method also resets the duration of the rhythm to the position of the last onset
        in the given tracks.

        Note that the given tracks are not deep copied.

        :param tracks: iterator yielding Track objects
        :param resolution: resolution of the onsets in the given tracks
        :return: None
        :raises EquallyNamedTracksError: when given multiple tracks with the same name
        """

        tracks_by_name = {}

        for t in tracks:
            name = t.get_name()
            if name in tracks_by_name:
                raise self.EquallyNamedTracksError(name)
            naming_error = self.__get_track_naming_error__(name)
            if naming_error:
                raise self.IllegalTrackName(naming_error)
            tracks_by_name[name] = t
            t.parent = self

        # by clearing the old tracks we prevent .set_resolution of rescaling
        # the onsets of the previous tracks, that we don't need anymore
        self._tracks.clear()
        self.set_resolution(resolution)
        self._tracks = tracks_by_name

        # update duration to position of last note, this works because RhythmBase.set_duration_in_ticks ensures that
        # the duration is at least the position of the last note
        self.set_duration_in_ticks(0)

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def __get_track_naming_error__(self, track_name: str) -> str:
        """
        Override this method for custom track name validation. If this method returns a non-empty string, an
        IllegalTrackName exception will be thrown while trying to set tracks in set_tracks. The message of the exception
        will be set to the result of this method.

        :param track_name: track name
        :return: error message or empty string if the track name is ok
        """

        return ""

    def get_track_iterator(self) -> tp.Iterator[Track]:
        """
        Returns an iterator over the tracks of this polyphonic rhythm. Each iteration yields a Track object.

        :return: iterator over this rhythm's tracks yielding PolyphonicRhythmImpl.Track objects
        """

        return iter(self._tracks.values())

    def get_track_by_name(self, track_name: str) -> tp.Union[Track, None]:
        """
        Returns the track with the given name or None if this rhythm has no track with the given name.

        :param track_name: track name
        :return: Track object or None
        """

        return self._tracks.get(str(track_name), None)

    def get_track_count(self) -> int:
        """
        Returns the number of tracks within this rhythm.

        :return: number of tracks
        """

        return len(self._tracks)

    # implements Rhythm.__rescale_onset_ticks__
    def __rescale_onset_ticks__(self, old_resolution: int, new_resolution: int):
        for track in self.get_track_iterator():
            track.__rescale_onset_ticks__(old_resolution, new_resolution)

    def get_last_onset_tick(self):
        """
        Returns the absolute position of the last note in this rhythm or -1 if this rhythm has no tracks or if the
        tracks are empty.

        :return: tick position of last note or -1 if no tracks or tracks empty
        """

        try:
            return max(track.get_last_onset_tick() for track in self.get_track_iterator())
        except ValueError:
            return -1

    def get_onset_count(self) -> int:
        """
        Returns the combined onset count in all tracks.

        :return: sum of onset count in all tracks
        """

        return sum(track.get_onset_count() for track in self.get_track_iterator())

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_tracks']
        state['__track_onsets_by_name__'] = {}
        for track in self.get_track_iterator():
            state['__track_onsets_by_name__'][track.name] = track.onsets
        return state

    def __setstate__(self, state):
        state['_tracks'] = {}
        for track_name, onsets in state['__track_onsets_by_name__'].items():
            track = Track(onsets, track_name, self)
            state['_tracks'][track.name] = track
        del state['__track_onsets_by_name__']
        self.__dict__.update(state)


MidiKey = namedtuple("MidiKey", [
    "pitch",
    "name",
    "abbreviation"
])


class MidiMapping(object):
    """Midi mapping base class"""

    def __init__(self, name: str, keys: tp.Iterable[tp.Union[MidiKey, tp.Tuple[int, str, str]]]):
        keys = tuple(MidiKey(pitch, name, abbr.lower()) for [pitch, name, abbr] in keys)
        keys_by_pitch = dict((key.pitch, key) for key in keys)
        keys_by_name = dict((key.name, key) for key in keys)
        keys_by_abbreviation = dict((key.abbreviation, key) for key in keys)

        if len(keys_by_pitch) < len(keys):
            raise ValueError("Keys should have unique pitch values")
        if len(keys_by_name) < len(keys):
            raise ValueError("Keys should have unique names")
        if len(keys_by_abbreviation) < len(keys):
            raise ValueError("Keys should have unique abbreviations")

        self._name = name
        self._keys = keys
        self._keys_by_pitch = keys_by_pitch
        self._keys_by_name = keys_by_name
        self._keys_by_abbreviation = keys_by_abbreviation

    def find(self, query: tp.Union[str, int]) -> tp.Union[MidiKey, None]:
        """
        Finds a Midi key by either its midi pitch, name or abbreviation.

        :param query: MIDI pitch, name or abbreviation
        :return: MidiKey namedtuple or None if not found
        """

        return self.find_by_pitch(query) or self.find_by_name(query) or self.find_by_abbreviation(query)

    def find_by_pitch(self, pitch: int) -> tp.Union[MidiKey, None]:
        """
        Finds a Midi key by its pitch.

        :param pitch: midi pitch as an integer
        :return: MidiKey namedtuple or None if not found
        """

        return self._keys_by_pitch.get(pitch, None)

    def find_by_name(self, name: str) -> tp.Union[MidiKey, None]:
        """
        Finds a Midi key by its name.

        :param name: key name
        :return: MidiKey namedtuple or None if not found
        """

        return self._keys_by_name.get(name, None)

    def find_by_abbreviation(self, abbreviation: str) -> tp.Union[MidiKey, None]:
        """
        Finds a Midi key by its abbreviation.

        :param abbreviation: key abbreviation
        :return: MidiKey namedtuple or None if not found
        """

        return self._keys_by_abbreviation.get(abbreviation.lower(), None)

    @property
    def keys(self) -> tp.Tuple[MidiKey, ...]:
        """
        A tuple containing all the Midi keys of this mapping. This property is read-only.

        :return: tuple containing all the Midi keys of this mapping
        """

        return self._keys

    @property
    def name(self) -> str:
        """
        The name of this mapping. This property is read-only.

        :return: name of this mapping
        """

        return self._name


GMDrumMapping = MidiMapping("GMDrumMapping", [
    (35, "Acoustic bass drum", "abd"),
    (36, "Bass drum 1", "bd1"),
    (37, "Side stick", "sst"),
    (38, "Acoustic snare", "asn"),
    (39, "Hand clap", "hcl"),
    (40, "Electric snare", "esn"),
    (41, "Low floor tom", "lft"),
    (42, "Closed hi-hat", "chh"),
    (43, "High floor tom", "hft"),
    (44, "Pedal hi-hat", "phh"),
    (45, "Low tom", "ltm"),
    (46, "Open hi-hat", "ohh"),
    (47, "Low mid tom", "lmt"),
    (48, "High mid tom", "hmt"),
    (49, "Crash cymbal 1", "cr1"),
    (50, "High tom", "htm"),
    (51, "Ride cymbal 1", "rc1"),
    (52, "Chinese cymbal", "chc"),
    (53, "Ride bell", "rbl"),
    (54, "Tambourine", "tmb"),
    (55, "Splash cymbal", "spl"),
    (56, "Cowbell", "cwb"),
    (57, "Crash cymbal 2", "cr2"),
    (58, "Vibraslap", "vbs"),
    (59, "Ride cymbal 2", "rc2"),
    (60, "Hi bongo", "hbg"),
    (61, "Low bongo", "lbg"),
    (62, "Muted high conga", "mhc"),
    (63, "Open high conga", "ohc"),
    (64, "Low conga", "lcn"),
    (65, "High timbale", "htb"),
    (66, "Low timbale", "ltb"),
    (67, "High agogo", "hgo"),
    (68, "Low agogo", "lgo"),
    (69, "Cabasa", "cbs"),
    (70, "Maracas", "mcs"),
    (71, "Short whistle", "swh"),
    (72, "Long whistle", "lwh"),
    (73, "Short guiro", "sgr"),
    (74, "Long guiro", "lgr"),
    (75, "Claves", "clv"),
    (76, "Hi wood block", "hwb"),
    (77, "Low wood block", "lwb"),
    (78, "Muted cuica", "mcu"),
    (79, "Open cuica", "ocu"),
    (80, "Muted triangle", "mtr"),
    (81, "Open triangle", "otr")
])


class RhythmLoop(PolyphonicRhythmImpl):
    """Rhythm loop with a name and duration always snapped to a downbeat"""

    def __init__(self, name: str = "", **kwargs):
        super().__init__(**kwargs)
        self._name = ""
        self.set_name(name)

    def set_duration_in_ticks(self, requested_duration):
        """
        Sets the duration in ticks to the first downbeat after the given duration position. The actual duration will
        be always greater than the requested duration unless the requested duration lays exactly on a downbeat or if
        the requested duration is less than the position of the last note within this rhythm.

        Note: If no time signature has been set, this method has no clue where the downbeat is and the requested
        duration will be set without snapping to the downbeat. As soon as the time signature is set, the duration will
        be updated.

        :param requested_duration: requested duration
        :return: the actual new duration. Unless the requested duration lays exactly on a downbeat and it lays behind
                 the position of the last note, this will always be greater than the requested duration
        """

        if not self.get_time_signature() or not self.get_resolution():
            # In order to know the tick position of the downbeat, we need both a resolution and a time signature. If
            # one of those is not, set the duration freely to the requested duration, it will automatically snap to
            # a downbeat whenever a time signature or resolution is set (it is updated in those methods)
            return super().set_duration_in_ticks(requested_duration)

        measure_duration = int(self.get_measure_duration("ticks"))
        n_measures = int(math.ceil(requested_duration / measure_duration))
        t_next_downbeat = n_measures * measure_duration
        assert t_next_downbeat >= requested_duration

        # adjust the duration and check if it was a success
        actual_duration = super().set_duration_in_ticks(round(n_measures * measure_duration))

        # If the actual duration was not the requested one this means that the requested duration was less than the
        # position of the last onset in this rhythm. We know that the actual duration is legal, that all durations
        # greater than that will also be legal and that this method only rounds measures up (not down). So, try again,
        # effectively setting the duration to the first downbeat after the last note
        if actual_duration != t_next_downbeat:
            return self.set_duration_in_ticks(actual_duration)

        return actual_duration

    def set_time_signature(self, time_signature: tp.Union[TimeSignature,
                                                          tp.Tuple[int, int],
                                                          tp.Sequence[int], None]) -> None:
        """
        Sets the time signature of this rhythm loop and updates the duration if necessary to set the duration to a
        downbeat.

        :param time_signature: new time signature
        :return: None
        """

        old_time_signature = self.get_time_signature()
        super().set_time_signature(time_signature)

        # We don't use the time_signature argument because that
        # is not guaranteed to be a TimeSignature object
        new_time_signature = self.get_time_signature()

        # If time signature has changed, update the duration to snap the duration to a downbeat
        if new_time_signature and new_time_signature != old_time_signature:
            self.set_duration_in_ticks(self.get_duration_in_ticks())

    def set_name(self, name: str) -> None:
        """
        Sets the name of this loop.

        :param name: name of the loop
        :return: None
        """

        self._name = str(name)

    def get_name(self) -> str:
        """
        Returns the name of this loop.

        :return: name of the loop
        """

        return self._name

    @property
    def name(self) -> str:
        """See RhythmLoop.set_name and RhythmLoop.get_name"""
        return self.get_name()

    @name.setter
    def name(self, name: str) -> None:
        self.set_name(name)


class MidiRhythm(RhythmLoop):
    def __init__(self, midi_file: tp.Union[IOBase, str] = "",
                 midi_pattern: midi.Pattern = None,
                 midi_mapping: MidiMapping = GMDrumMapping,
                 name: str = "", preserve_midi_duration: bool = False, **kwargs):
        super().__init__(**kwargs)

        if all(a for a in [midi_file, midi_pattern]):
            raise Exception("Given both midi file and midi pattern. Please provide one max one.")

        if midi_file:
            try:
                if isinstance(midi_file, str):
                    midi_file = open(midi_file, "rb")
                midi_pattern = midi.read_midifile(midi_file)
            finally:
                midi_file.close()
            name = name or os.path.splitext(os.path.basename(midi_file.name))[0]

        self.set_name(name)
        self._midi_mapping = midi_mapping  # type: MidiMapping
        self._midi_metronome = -1  # type: int

        # loads the tracks and sets the bpm, time signature, midi metronome and resolution
        if midi_pattern:
            self.load_midi_pattern(midi_pattern, preserve_midi_duration)

    @property
    def midi_mapping(self):
        return self._midi_mapping

    def as_midi_pattern(self, note_length: int = 0, midi_channel: int = 9,
                        midi_format: int = 0) -> midi.Pattern:
        """
        Converts this rhythm to a MIDI pattern.

        :param note_length: note duration in ticks
        :param midi_channel: NoteOn/NoteOff events channel (defaults to 9, which is the default for drum sounds)
        :param midi_format: midi format
        :return: MIDI pattern
        """

        midi_track = midi.Track(tick_relative=False)  # create track and add metadata events

        midi_track.append(midi.TrackNameEvent(text=self.get_name()))  # track name
        midi_metronome = 24 if self._midi_metronome is None else self._midi_metronome
        midi_track.append(self.get_time_signature().to_midi_event(midi_metronome))  # time signature

        if self.bpm:
            midi_track.append(midi.SetTempoEvent(bpm=self.bpm))  # tempo

        # add note events
        for track in self.get_track_iterator():
            midi_note = self._midi_mapping.find_by_abbreviation(track.name)
            assert midi_note is not None  # if track has invalid midi note abbreviation set_track would have failed
            pitch = midi_note.pitch
            onsets = track.onsets
            for onset in onsets:
                note_abs_tick = onset[0]
                velocity = onset[1]
                # channel 9 for drums
                note_on = midi.NoteOnEvent(tick=note_abs_tick, pitch=pitch, velocity=velocity, channel=midi_channel)
                note_off = midi.NoteOffEvent(tick=note_abs_tick + note_length, pitch=pitch, channel=midi_channel)
                midi_track.extend([note_on, note_off])

        # sort the events in chronological order and convert to relative delta-time
        midi_track = midi.Track(sorted(midi_track, key=lambda event: event.tick), tick_relative=False)
        midi_track.make_ticks_rel()

        # add end of track event
        midi_track.append(midi.EndOfTrackEvent())

        # save the midi file
        return midi.Pattern(
            [midi_track],
            format=midi_format,
            resolution=self.get_resolution()
        )

    def write_midi_out(self, midi_file: tp.Union[str, IOBase], **kwargs):
        """
        Writes this rhythm loop as a MIDI file.

        :param midi_file: midi file or path
        :param kwargs: arguments passed to as_midi_pattern, see documentation of that method
        :return: None
        """

        midi_pattern = self.as_midi_pattern(**kwargs)
        midi.write_midifile(midi_file, midi_pattern)

    def __get_track_naming_error__(self, track_name: str) -> str:
        """
        Checks if the given track name is a valid MidiKey abbreviation according to this loop's midi mapping.

        :param track_name: track name to check
        :return: a message telling that the track name is not a valid abbreviation or an empty string if the track name
                 is ok
        """

        mapping = self._midi_mapping
        if mapping.find_by_abbreviation(track_name):
            return ""
        return "No midi key found with abbreviation \"%s\" in %s" % (track_name, mapping.name)

    def load_midi_pattern(self, pattern: midi.Pattern, preserve_midi_duration: bool = False) -> None:
        """
        Loads a midi pattern and sets this rhythm's tracks, time signature bpm and duration. The given midi pattern must
        have a resolution property and can't have more than one track containing note events. The midi events map to
        rhythm properties like this:

            midi.NoteOnEvent            ->  adds an onset to this rhythm
            midi.TimeSignatureEvent *   ->  set the time signature of this rhythm
            midi.SetTempoEvent          ->  sets the bpm of this rhythm
            midi.EndOfTrackEvent **     ->  sets the duration of this rhythm (only if preserve_midi_duration is true)

                * required
                ** required only if preserve_midi_duration is true

        If preserve_midi_duration is false, the duration of this rhythm will be set to the first downbeat after the last
        note position.

        :param pattern: the midi pattern to load
        :param preserve_midi_duration: when true, the duration will be set to the position of the midi EndOfTrackEvent,
                                       otherwise it will be set to the first downbeat after the last note position
        :return: None
        """

        mapping = self._midi_mapping
        n_tracks_containing_note_events = sum(any(isinstance(e, midi.NoteEvent) for e in track) for track in pattern)

        if n_tracks_containing_note_events > 1:
            raise ValueError("Given MIDI pattern has multiple tracks with note events (%i)",
                             n_tracks_containing_note_events)

        pattern.make_ticks_abs()
        track = list(itertools.chain(*pattern))  # merge all tracks into one
        track = midi.Track(sorted(track, key=lambda event: event.tick))  # sort in chronological order

        bpm = 0               # type: tp.Union[float]
        track_data = {}       # type: tp.Dict[MidiKey, tp.List[tp.Tuple[int, int], ...]]
        ts_midi_event = None  # type: tp.Union[midi.TimeSignatureEvent, None]
        eot_event = None      # type: tp.Union[midi.EndOfTrackEvent, None]

        for msg in track:
            if isinstance(msg, midi.NoteOnEvent):
                midi_pitch = msg.get_pitch()  # type: int
                midi_key = mapping.find_by_pitch(midi_pitch)
                if midi_key not in track_data:
                    if midi_key is None:
                        print("Unknown midi key: %i (Mapping = %s)" % (midi_pitch, mapping.name))
                        continue
                    track_data[midi_key] = []
                onset = (int(msg.tick), int(msg.get_velocity()))
                track_data[midi_key].append(onset)
            elif isinstance(msg, midi.TimeSignatureEvent):
                if ts_midi_event is None:
                    ts_midi_event = msg
                elif ts_midi_event.data != msg.data:
                    raise ValueError("Time signature changes are "
                                     "not supported (from %s to %s)" % (ts_midi_event, msg))
            elif isinstance(msg, midi.SetTempoEvent):
                bpm = float(msg.get_bpm())
            elif isinstance(msg, midi.EndOfTrackEvent):
                eot_event = msg
                break

        if ts_midi_event is None:
            raise ValueError("Given pattern contains no time signature")

        time_signature = TimeSignature.from_midi_event(ts_midi_event)
        midi_metronome = ts_midi_event.get_metronome()  # type: int

        # noinspection PyShadowingNames
        def track_generator():
            # sort the rhythm tracks by midi pitch
            sorted_midi_keys = sorted(track_data.keys(), key=lambda midi_key: midi_key.pitch)
            for midi_key in sorted_midi_keys:
                onsets = track_data[midi_key]
                yield Track(onsets, midi_key.abbreviation)

        self._midi_metronome = midi_metronome
        self.set_time_signature(time_signature)
        self.set_bpm(bpm)
        self.set_tracks(track_generator(), pattern.resolution)

        if preserve_midi_duration:
            self.set_duration_in_ticks(eot_event.tick)


def create_rumba_rhythm(resolution=240):
    """
    Utility function that creates a one-bar rumba rhythm.

    :param resolution: rhythm resolution
    :return: rumba rhythm
    """

    track = Track(((0, 127), (3, 127), (7, 127), (10, 127), (12, 127)), "main_track")
    rhythm = PolyphonicRhythmImpl([track], 4, time_signature=(4, 4))
    rhythm.set_resolution(resolution)
    return rhythm

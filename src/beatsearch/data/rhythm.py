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
from beatsearch.utils import friendly_named_class
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


def check_iterables_meet_length_policy(iterables, len_policy):
    """
    Checks whether the given iterables meet a certain duration policy. Duration policies are:
        'exact' - met when all iterables have the exact same length and are not empty
        'multiple' - met when the length of the largest iterable is a multiple of all the other iterable lengths
        'fill' - met when any of the chains is empty

    :param iterables: iterables to check the lengths of
    :param len_policy: one of {'exact', 'multiple' or 'fill'}
    :return length of the largest iterable
    """

    if not iterables:
        return

    l = [len(c) for c in iterables]  # lengths

    if len_policy == 'exact':
        if not all(x == l[0] for x in l):
            raise ValueError("When length policy is set to 'exact', iterables should have the same lengths")
    elif len_policy == 'multiple':
        if not all(x % l[0] == 0 or l[0] % x == 0 for x in l):
            raise ValueError("When length policy is set to 'multiple', the length of the largest "
                             "iterable should be a multiple of all the other iterable lengths")
    elif len_policy != 'fill':
        raise ValueError("Unknown length policy: '%s'" % len_policy)

    return max(l)


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


class IRhythm(object, metaclass=ABCMeta):
    """Rhythm interface"""

    class Precondition(object):
        """Preconditions for IRhythm methods"""

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
        Sets this rhythm's tick resolution.

        :param resolution: new tick resolution in PPQN
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
    def set_duration_in_ticks(self, requested_duration: int) -> None:
        """
        Sets the duration of the rhythm to the closest duration possible to the requested duration and returns the
        actual new duration.

        :param requested_duration: requested new duration
        :return: actual new duration
        """

        raise NotImplementedError

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

    @property
    def resolution(self) -> int:
        """See IRhythm.set_resolution and IRhythm.get_resolution"""
        return self.get_resolution()

    @resolution.setter
    def resolution(self, resolution: tp.Union[float, int]):  # setter
        self.set_resolution(resolution)

    @property
    def bpm(self) -> float:
        """See IRhythm.set_bpm and IRhythm.get_bpm"""
        return self.get_bpm()

    @bpm.setter
    def bpm(self, bpm: tp.Union[float, int]):  # setter
        self.set_bpm(bpm)

    @property
    def time_signature(self) -> tp.Union[TimeSignature, None]:
        """See IRhythm.set_time_signature and IRhythm.get_time_signature"""
        return self.get_time_signature()

    @time_signature.setter
    def time_signature(self, time_signature: tp.Union[TimeSignature,
                                                      tp.Tuple[int, int],
                                                      tp.Sequence[int], None]) -> None:  # setter
        self.set_time_signature(time_signature)

    @property
    def duration_in_ticks(self) -> int:
        """See IRhythm.set_duration_in_ticks and IRhythm.get_duration_in_ticks"""
        return self.get_duration_in_ticks()

    @duration_in_ticks.setter
    def duration_in_ticks(self, duration: int) -> None:  # setter
        self.set_duration_in_ticks(duration)


class RhythmBase(IRhythm, metaclass=ABCMeta):
    """Rhythm abstract base class implementing the IRhythm interface"""

    def __init__(self):
        """
        Creates a rhythm base object containing generic meta-data. As this is an abstract base class, this constructor
        should only be called by subclass constructors.
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
            resolution: sets IRhythm.resolution
            bpm: sets IRhythm.bpm
            time_signature: sets IRhythm.time_signature
            duration_in_ticks: sets IRhythm.duration_in_ticks
            duration: also sets IRhythm.duration_in_ticks

        :return: None
        """

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

    def set_resolution(self, resolution: int):
        """
        Sets this rhythm's tick resolution and updates the duration, which is in ticks and will change according to the
        resolution.

        :param resolution: new tick resolution in PPQN
        :return: None
        """

        if not isinstance(resolution, int) or resolution < 0:
            raise ValueError("Resolution should be a positive integer or "
                             "zero for no resolution but got %s" % str(resolution))

        prev_resolution = self.resolution
        prev_duration = self.duration_in_ticks

        if prev_duration != 0.0:
            rescaled_duration = convert_time(prev_duration, prev_resolution, resolution, quantize=True)
        else:
            rescaled_duration = 0

        self._resolution = resolution
        self.set_duration_in_ticks(rescaled_duration)

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


class Onset(namedtuple("Onset", ["tick", "velocity"])):
    """Onset in a rhythm

    Each onset represents a note within rhythm and has the following (read-only) properties:
       tick      - the absolute tick position of this onset within the rhythm as an integer
       velocity  - the MIDI velocity of this note with a range of [0, 127] as an integer
    """

    def scale(self, resolution_from: tp.Union[int, float], resolution_to: tp.Union[int, float]):
        """
        Returns a new Onset object with a scaled position.

        :param resolution_from: original resolution of the onset tick position in PPQ
        :param resolution_to: resolution of the new onset's tick position in PPQN
        :return: new Onset object with the given new resolution
        """

        scaled_tick = convert_time(self.tick, resolution_from, resolution_to, quantize=True)
        return Onset(scaled_tick, self.velocity)


class IMonophonicRhythm(IRhythm, metaclass=ABCMeta):
    """Monophonic rhythm interface

    Interface for monophonic rhythms.
    """

    @abstractmethod
    def get_onsets(self) -> tp.Tuple[Onset, ...]:
        """
        Returns the onsets within this rhythm as a tuple of onsets, where each onset is an instance of
        MonophonicRhythm.Onset.

        :return: the onsets within this rhythm as a tuple of MonophonicRhythm.Onset objects
        """

        raise NotImplementedError

    @property
    def onsets(self) -> tp.Tuple[Onset, ...]:
        """See IMonophonicRhythm.get_onsets. This property is read-only."""
        return self.get_onsets()

    def get_last_onset_tick(self) -> int:  # implements IRhythm.get_last_onset_tick
        try:
            return self.onsets[-1].tick
        except IndexError:
            return -1

    def get_onset_count(self) -> int:  # implements IRhythm.get_onset_count
        return len(self.onsets)

    #####################################
    # Monophonic rhythm representations #
    #####################################

    @IRhythm.Precondition.needs_resolution
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

    @IRhythm.Precondition.needs_resolution
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

    @IRhythm.Precondition.needs_resolution
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

    def get_binary_schillinger_chain(self, unit='ticks', values=(0, 1)):
        """
        Returns the Schillinger notation of this rhythm where each onset is a change of a "binary note".

        For example, given the Rumba Clave rhythm and with values (0, 1):
          X--X---X--X-X---
          0001111000110000

        :param unit: the unit to quantize on ('ticks' is no quantization)
        :param values: binary vector to be used in the schillinger chain. E.g. when given ('a', 'b'), the returned
                       schillinger chain will consist of 'a' and 'b'.
        :return: Schillinger rhythm vector as a list
        """

        chain, i = self.get_binary(unit), 0
        value_i = 0
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

    @IRhythm.Precondition.needs_resolution
    @concretize_unit()
    def get_onset_times(self, unit="ticks", quantize=False):
        """
        Returns the absolute onset times of the notes in this rhythm.

        :param unit: the unit of the onset times
        :param quantize: whether or not the onset times must be quantized to the given unit
        :return: a list with the onset times of this rhythm
        """

        return [convert_time(onset[0], self.get_resolution(), unit, quantize) for onset in self.onsets]


class MonophonicRhythmBase(IMonophonicRhythm, metaclass=ABCMeta):
    """Monophonic rhythm base class implementing IMonophonicRhythm

    Abstract base class for monophonic rhythms. This class implements IMonophonicRhythm.get_onsets, adding onset state
    to subclasses. Note that this class does not extend RhythmBase and therefore does NOT add rhythm base state like
    bpm, resolution, time signature, etc.
    """

    class OnsetsNotInChronologicalOrder(Exception):
        """Exception thrown when two adjacent onsets are not in chronological order"""
        def __init__(self, tick_a: int, tick_b: int):
            msg = "<..., %i, %i <--!, ...>" % (tick_a, tick_b)
            super().__init__(msg)

    def __init__(
            self,
            onsets: tp.Union[tp.Iterable[Onset],
                             tp.Iterable[tp.Tuple[int, int]],
                             tp.Iterable[tp.Sequence[int]]]
    ):
        """
        Creates a new monophonic rhythm from the given onsets. The onsets will be stored as MonophonicOnset.Onset
        named tuples.

        :param onsets: An iterable returning an (absolute tick, velocity) tuple for each iteration. The tick
                       resolution should equal the parent's resolution. The onsets should be given in chronological
                       order.
        :raises OnsetsNotInChronologicalOrder
        """

        def validate_onsets_gen():
            prev_tick = -1
            for [tick, velocity] in onsets:
                if tick < prev_tick:
                    raise self.OnsetsNotInChronologicalOrder(prev_tick, tick)
                yield Onset(tick, velocity)
                prev_tick = tick

        self._onsets = tuple(validate_onsets_gen())  # type: (tp.Tuple[Onset, ...])

    def get_onsets(self) -> tp.Tuple[Onset, ...]:
        """
        Returns the onsets within this rhythm as a tuple of onsets, where each onset is an instance of
        MonophonicRhythm.Onset.

        :return: the onsets within this rhythm as a tuple of MonophonicRhythm.Onset objects
        """

        return self._onsets


class MonophonicRhythm(RhythmBase, MonophonicRhythmBase):
    """Implements both rhythm base and monophonic rhythm base"""

    def __init__(
            self,
            onsets: tp.Union[tp.Iterable[Onset],
                             tp.Iterable[tp.Tuple[int, int]],
                             tp.Iterable[tp.Sequence[int]]],
            **kwargs
    ):
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


class IPolyphonicRhythm(IRhythm, metaclass=ABCMeta):

    class TrackNameError(Exception):
        """Thrown if there's something wrong with a track name"""
        pass

    class EquallyNamedTracksError(TrackNameError):
        """Thrown by set_tracks when given multiple tracks with same name"""
        pass

    class IllegalTrackName(TrackNameError):
        """Thrown by set_tracks if __validate_track_name__ returns False"""
        pass

    class Track(MonophonicRhythmBase):
        """Represents one track of a polyphonic rhythm

        A polyphonic rhythm consists of multiple monophonic rhythms; tracks. Each of those tracks is represented by one
        instance of this class. Note that -- although this class does implement IRhythm through MonophonicRhythmBase --
        it does not inherit the rhythm base state, because it does not extend RhythmBase. It is not necessary for Track
        to have base rhythm state, because this meta-data is already stored in the parent of the tracks; the
        PolyphonicRhythm. Base rhythm getters redirect to the polyphonic parent, setters will raise an exception when
        used.
        """

        def __init__(
                self,
                onsets: tp.Union[tp.Iterable[Onset],
                                 tp.Iterable[tp.Tuple[int, int]],
                                 tp.Iterable[tp.Sequence[int]]],
                track_name: str,
                parent: IRhythm = None
        ):
            """
            Creates a new rhythm track.

            :param onsets: An iterable returning an (absolute tick, velocity) tuple for each iteration. The tick
                           resolution should equal the parent's resolution. The onsets should be given in chronological
                           order.
            :param track_name: The name of this track. This can't be changed after instantiation.
            :param parent: The polyphonic rhythm which this track belongs to
            """

            super().__init__(onsets)
            self._track_name = str(track_name)           # type: str
            self._parent = parent                        # type: PolyphonicRhythm

        class ParentNotSetError(Exception):
            pass

        def get_parent(self):  # type: () -> PolyphonicRhythm
            """
            Returns the parent of this track. The parent is a polyphonic rhythm.

            :return: polyphonic rhythm which this track is parented to or None
            """

            return self._parent

        def set_parent(self, parent):  # type: (PolyphonicRhythm) -> None
            """
            Parents this track to the given polyphonic rhythm. Note that the tick resolution of the given parent will
            reflect the resolution of this track's onsets.

            :param parent: polyphonic rhythm
            :return: None
            """

            self._parent = parent

        def get_name(self):  # type: () -> str
            """
            Returns the name of this track. Note that there is a getter but not a setter for a track's name. The name of
            a track can not be changed after initialization.

            :return: track name
            """

            return self._track_name

        @property
        def parent(self):  # type: () -> PolyphonicRhythm
            """See Track.set_parent and Track.get_parent"""
            return self.get_parent()

        @parent.setter
        def parent(self, rhythm):  # type: (PolyphonicRhythm) -> None
            self.set_parent(rhythm)

        @property
        def name(self):
            """See Track.get_name. This property is read-only"""
            return self.get_name()

        def get_resolution(self):
            """
            Returns the resolution of parent in PPQN.

            :return: resolution of parent in PPQN and therefore
            :raises Track.ParentNotSetError
            """

            self._precondition_parent_set()
            return self._parent.get_resolution()

        def get_bpm(self):
            """
            Returns the tempo of parent in beats per minute.

            :return: bpm of parent
            :raises Track.ParentNotSetError
            """

            self._precondition_parent_set()
            return self._parent.get_bpm()

        def get_time_signature(self):
            """
            Returns the time signature of parent.

            :return: time signature of parent as a TimeSignature object
            :raises Track.ParentNotSetError
            """

            self._precondition_parent_set()
            return self._parent.get_time_signature()

        def get_duration_in_ticks(self):
            """
            Returns the duration of parent.

            :return: duration of parent in ticks
            :raises Track.ParentNotSetError
            """

            self._precondition_parent_set()
            return self._parent.get_duration_in_ticks()

        def set_resolution(self, resolution: int):
            """
            Can't set resolution for a single track. To set the resolution for all tracks, call parent.set_resolution.

            :param resolution: unused
            :return: None
            :raises Exception: always
            """

            raise Exception("Can't set resolution for single track, call parent.")

        def set_bpm(self, bpm: tp.Union[float, int]) -> None:
            """
            Can't set the bpm for a single track. To set the bpm for all tracks, call parent.set_bpm.

            :param bpm: unused
            :return: None
            :raises Exception: always
            """

            raise Exception("Can't set BPM of a single track, call parent.")

        def set_time_signature(self, time_signature: tp.Union[TimeSignature,
                                                              tp.Tuple[int, int],
                                                              tp.Sequence[int], None]) -> None:
            """
            Can't set the time signature for a single track. To set the time signature for all tracks, call
            parent.set_time_signature.

            :param time_signature: unused
            :return: None
            :raises Exception: always
            """

            raise Exception("Can't set time signature of a single track, call parent.")

        def set_duration_in_ticks(self, requested_duration: int) -> None:
            """
            Can't set the duration for a single track. To set the duration for all tracks, call
            parent.set_duration_in_ticks or parent.set_duration.

            :param requested_duration: unused
            :return: None
            :raises Exception: always
            """

            raise Exception("Can't set duration of a single track, call parent.")

        def _precondition_parent_set(self):
            if self._parent is None:
                raise self.ParentNotSetError

        # used internally by PolyphonicRhythm.set_resolution to rescale the onsets of this track
        def __rescale__(self, old_res: tp.Union[float, int], new_res: tp.Union[float, int]):
            self._onsets = tuple(onset.scale(old_res, new_res) for onset in self._onsets)

    @abstractmethod
    def set_tracks(self, tracks: tp.Iterable[Track], resolution: int) -> None:
        """
        Sets the tracks and updates the resolution of this rhythm. The given tracks will automatically be parented to
        this polyphonic rhythm. This method also resets the duration of the rhythm to the position of the last onset
        in the given tracks.

        :param tracks: iterator yielding PolyphonicRhythm.Track objects
        :param resolution: resolution of the onsets in the given tracks
        :return: None
        """

        raise NotImplementedError

    @abstractmethod
    def get_track_iterator(self) -> tp.Iterator[Track]:
        """
        Returns an iterator over the tracks of this polyphonic rhythm. Each iteration yields a PolyphonicRhythm.Track
        object.

        :return: iterator over this rhythm's tracks yielding PolyphonicRhythm.Track objects
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
            yield cls.Track(onsets=onsets, track_name=name)

    #####################################
    # Polyphonic rhythm representations #
    #####################################

    # TODO remove duplicate functionality (see IMonophonicRhythm.get_interval_histogram)
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


class PolyphonicRhythm(RhythmBase, IPolyphonicRhythm):

    def __init__(
            self, tracks: tp.Iterable[IPolyphonicRhythm.Track] = tuple(),
            resolution: int = 0,
            **kwargs,
    ):
        """
        Creates a new polyphonic rhythm with the given tracks.

        :param tracks: sequence of tracks
        :param resolution: the resolution of the given tracks, this parameter is unused if no tracks were given
        :param kwargs: Post-init keyword arguments. See RhythmBase.post_init.
        """

        super().__init__()
        self._tracks = OrderedDict()  # type: tp.Dict[str, PolyphonicRhythm.Track]

        if tracks:
            if resolution is None:
                raise TypeError("given tracks without resolution")
            self.set_tracks(tracks, resolution)

        self.post_init(**kwargs)

    def set_tracks(self, tracks: tp.Iterable[IPolyphonicRhythm.Track], resolution: int) -> None:
        """
        Sets the tracks and updates the resolution of this rhythm. The given tracks will automatically be parented to
        this polyphonic rhythm. This method also resets the duration of the rhythm to the position of the last onset
        in the given tracks.

        :param tracks: iterator yielding PolyphonicRhythm.Track objects
        :param resolution: resolution of the onsets in the given tracks
        :return: None
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
            t.set_parent(self)

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

    def get_track_iterator(self) -> tp.Iterator[IPolyphonicRhythm.Track]:
        """
        Returns an iterator over the tracks of this polyphonic rhythm. Each iteration yields a PolyphonicRhythm.Track
        object.

        :return: iterator over this rhythm's tracks yielding PolyphonicRhythm.Track objects
        """

        return iter(self._tracks.values())

    def get_track_by_name(self, track_name: str) -> tp.Union[IPolyphonicRhythm.Track, None]:
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

    def set_resolution(self, res: int):
        """
        Sets the resolution of this rhythm and rescales all onsets of all tracks.

        :param res: new resolution in ppqn
        :return: None
        """

        old_res = self.get_resolution()

        if old_res == res:
            return

        if res <= 0:
            raise ValueError("Expected resolution greater than zero but got %i" % res)

        if old_res <= 0:
            assert self.get_track_count() == 0, "tracks set without resolution"

        # rescale onsets to new resolution
        for track in self.get_track_iterator():
            track.__rescale__(old_res, res)

        # updates res property and duration (depends on rescaled onsets)
        super().set_resolution(res)

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
            track = PolyphonicRhythm.Track(onsets, track_name, self)
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


class RhythmLoop(PolyphonicRhythm):
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

        if self.get_track_count() == 0:
            return

        if not self.get_time_signature():
            # set duration freely if there is no time signature set (duration will be updated as soon as the time
            # signature is set)
            return super().set_duration_in_ticks(requested_duration)

        measure_duration = self.get_measure_duration("ticks")
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

    @IRhythm.Precondition.needs_time_signature
    def set_tracks(self, tracks: tp.Iterable[PolyphonicRhythm.Track], resolution: int) -> None:
        super().set_tracks(tracks, resolution)

    def set_time_signature(self, time_signature: tp.Union[TimeSignature,
                                                          tp.Tuple[int, int],
                                                          tp.Sequence[int], None]) -> None:
        """
        Sets the time signature of this rhythm loop and updates the duration if necessary to set the duration to a
        downbeat.

        :param time_signature: new time signature
        :return: None
        """

        if self.get_track_count() and not time_signature:
            raise Exception("Can't remove time signature from non-empty rhythm loop")

        super().set_time_signature(time_signature)

        if self.get_track_count():
            # update duration (snap to downbeat) if tracks have already been set
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
            self._load(midi_pattern, preserve_midi_duration)

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

        midi_track.append(midi.TrackNameEvent(text=self._name))  # track name
        midi_metronome = 24 if self._midi_metronome is None else self._midi_metronome
        midi_track.append(self._time_signature.to_midi_event(midi_metronome))  # time signature
        midi_track.append(midi.SetTempoEvent(bpm=self._bpm))  # tempo

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

    # TODO somehow expose this functionality so that you can create empty rhythm and afterwards load-in
    # the tracks from a midi pattern
    def _load(self, pattern: midi.Pattern, preserve_midi_duration: bool) -> None:
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
        ts_eot_event = None   # type: tp.Union[midi.EndOfTrackEvent, None]

        for msg in track:
            if isinstance(msg, midi.NoteOnEvent):
                midi_pitch = msg.get_pitch()  # type: int
                midi_key = mapping.find_by_pitch(midi_pitch)
                if midi_key not in track_data:
                    if midi_key is None:
                        print("Unknown midi key: %i (Mapping = %s)" % (midi_pitch, mapping.name))
                        continue
                    track_data[midi_key] = []
                onset = (msg.tick, msg.get_velocity())
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
                ts_eot_event = msg
                break

        if ts_midi_event is None:
            raise ValueError("Given pattern contains no time signature")

        time_signature = TimeSignature.from_midi_event(ts_midi_event)
        midi_metronome = ts_midi_event.get_metronome()  # type: int
        track_generator = (self.Track(onsets, key.abbreviation) for key, onsets in track_data.items())

        self._midi_metronome = midi_metronome
        self.set_time_signature(time_signature)
        self.set_bpm(bpm)
        self.set_tracks(track_generator, pattern.resolution)

        if preserve_midi_duration:
            ts_eot_event.tick()


class DistanceMeasure(object):
    """Abstract base class for distance measures"""

    def get_distance(self, obj_a, obj_b):
        raise NotImplementedError


class MonophonicRhythmDistanceMeasure(DistanceMeasure):
    """Abstract base class for monophonic rhythm distance measures

    This is an abstract base class for monophonic rhythm distance measures. It measures the distance
    between two MonophonicRhythm objects.
    """

    LENGTH_POLICIES = ['exact', 'multiple', 'fill']

    class UnknownLengthPolicy(Exception):
        def __init__(self, given_length_policy):
            super(MonophonicRhythmDistanceMeasure.UnknownLengthPolicy, self).__init__(
                "Given %s, please choose between: %s" % (
                    given_length_policy, MonophonicRhythmDistanceMeasure.LENGTH_POLICIES))

    def __init__(self, unit, length_policy):
        """
        Creates a new monophonic rhythm distance measure.

        :param unit: see internal_unit property
        :param length_policy: see documentation on length_policy property
        """

        self._len_policy = ''
        self._internal_unit = 0
        self._output_unit = 0

        self.length_policy = length_policy
        self.internal_unit = unit
        self.output_unit = unit

    @property
    def length_policy(self):
        """
        The length policy determines how permissive the monophonic rhythm similarity measure is with variable sized
        rhythm vector (N = onset count) or chain (N = duration) representations. Given two rhythm representations X and
        Y, the length policy should be one of:

            'exact': len(X) must equal len(Y)
            'multiple': len(X) must be a multiple of len(Y) or vice-versa
            'fill': len(X) and len(Y) must not be empty

        Implementations of MonophonicRhythmDistanceMeasure.get_distance will throw a ValueError if the representations
        of the given rhythm do not meet the requirements according to the length policy.
        """

        return self._len_policy

    @length_policy.setter
    def length_policy(self, length_policy):
        valid_policies = MonophonicRhythmDistanceMeasure.LENGTH_POLICIES
        if length_policy not in valid_policies:
            raise MonophonicRhythmDistanceMeasure.UnknownLengthPolicy(length_policy)
        self._len_policy = length_policy

    @property
    def internal_unit(self):
        """
        Unit used internally for distance computation. This unit is given to __get_iterable__
        """

        return self._internal_unit

    @internal_unit.setter
    def internal_unit(self, internal_unit):
        if internal_unit != 'ticks':
            convert_time(1, 1, internal_unit)
        self._internal_unit = internal_unit

    @property
    def output_unit(self):
        """
        The output unit is the unit of the distance returned by get_distance. E.g. when the output_unit is 'ticks', the
        distance returned by get_distance will be in 'ticks'.
        """

        return self._output_unit

    @output_unit.setter
    def output_unit(self, output_unit):
        if output_unit != 'ticks':
            # validates the given unit
            convert_time(1, 1, output_unit)
        self._output_unit = output_unit

    def get_distance(self, rhythm_a: IMonophonicRhythm, rhythm_b: IMonophonicRhythm):
        """
        Returns the distance between the given tracks.

        :param rhythm_a: monophonic rhythm to compare to monophonic rhythm b
        :param rhythm_b: monophonic rhythm to compare to monophonic rhythm a
        :return: distance between the given monophonic rhythms
        """

        internal_unit_in_ticks = self._internal_unit == 'ticks'
        output_unit_in_ticks = self._output_unit == 'ticks'
        res_a, res_b = rhythm_a.get_resolution(), rhythm_b.get_resolution()
        if (internal_unit_in_ticks or output_unit_in_ticks) and res_a != res_b:
            raise ValueError("%s unit set to 'ticks', but given rhythms have "
                             "different resolutions (%i != %i)" %
                             ("Internal" if internal_unit_in_ticks else "Output", res_a, res_b))
        internal_unit = res_a if internal_unit_in_ticks else self._internal_unit
        output_unit = res_a if output_unit_in_ticks else self._output_unit

        rhythms = [rhythm_a, rhythm_b]
        iterables = [self.__get_iterable__(t, internal_unit) for t in rhythms]
        cookies = [self.__get_cookie__(t, internal_unit) for t in rhythms]
        max_len = self._check_if_iterables_meet_len_policy(*iterables)
        distance = self.__compute_distance__(max_len, *(iterables + cookies))
        return convert_time(distance, internal_unit, output_unit, quantize=False)

    def __get_iterable__(self, rhythm: IMonophonicRhythm, unit):
        """
        Should prepare and return the rhythm representation on which the similarity measure is based. The returned
        vector's will be length policy checked.

        :param rhythm: the monophonic rhythm
        :param unit: the representation should be in the given unit
        :return: desired rhythm representation to use in __compute_distance__
        """

        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def __get_cookie__(self, rhythm: IMonophonicRhythm, unit):
        """
        The result of this method will be passed to __compute_distance__, both for rhythm a and rhythm b. By default,
        the cookie is the rhythm itself.

        :param rhythm: the monophonic rhythm
        :param unit: the unit given to __get_iterable__
        :return: cookie to use in __compute_distance__
        """

        return rhythm

    def __compute_distance__(self, max_len, iterable_a, iterable_b, cookie_a, cookie_b):
        """
        The result of this method is returned by get_distance. If that method is given two rhythms a and b, this method
        is given both the iterables of a and b and the cookies of a and b, returned by respectively __get_iterable__
        and __get_cookie__.

        :param max_len: max(len(iterable_a), len(iterable_b))
        :param iterable_a: the result of __get_iterable__, given rhythm a
        :param iterable_b: the result of __get_iterable__, given rhythm b
        :param cookie_a: the result of __get_cookie__, given rhythm a
        :param cookie_b: the result of __get_cookie__, given rhythm b
        :return: the distance between rhythm a and b
        """

        raise NotImplementedError

    def _check_if_iterables_meet_len_policy(self, iterable_a, iterable_b):
        if not iterable_a or not iterable_b:
            return

        l = [len(iterable_a), len(iterable_b)]

        if self.length_policy == "exact":
            if l[0] != l[1]:
                raise ValueError("When length policy is set to \"exact\", both iterables "
                                 "should have the same number of elements")

        elif self.length_policy == "multiple":
            if not all(x % l[0] == 0 or l[0] % x == 0 for x in l):
                raise ValueError("When length policy is set to \"multiple\", the length of the largest "
                                 "iterable should be a multiple of all the other iterable lengths")

        elif self.length_policy != "fill":
            raise ValueError("Unknown length policy: \"%s\"" % self.length_policy)

        return max(l)

    __measures__ = {}  # monophonic rhythm distance implementations by __friendly_name__

    @classmethod
    def get_measures(cls, friendly_name=True):
        """
        Returns an ordered dictionary containing implementations of MonophonicRhythmDistanceMeasure by name.

        :param friendly_name: when True, the name will be retrieved with __friendly_name__ instead of __name__
        :return: an ordered dictionary containing all subclasses of MonophonicRhythmDistanceMeasure by name
        """

        if len(MonophonicRhythmDistanceMeasure.__measures__) != MonophonicRhythmDistanceMeasure.__subclasses__():
            measures = OrderedDict()
            for tdm in cls.__subclasses__():
                name = tdm.__name__
                if friendly_name:
                    try:
                        # noinspection PyUnresolvedReferences
                        name = tdm.__friendly_name__
                    except AttributeError:
                        pass
                measures[name] = tdm
            cls.__measures__ = measures

        return MonophonicRhythmDistanceMeasure.__measures__

    @classmethod
    def get_measure_names(cls):
        measures = cls.get_measures()
        return tuple(measures.keys())

    @classmethod
    def get_measure_by_name(cls, measure_name):
        measures = cls.get_measures()
        try:
            return measures[measure_name]
        except KeyError:
            raise ValueError("No measure with name: '%s'" % measure_name)


class Quantizable(object):
    def __init__(self):
        self._quantize_enabled = False

    @property
    def quantize_enabled(self):
        return self._quantize_enabled

    @quantize_enabled.setter
    def quantize_enabled(self, quantize_enabled):
        self._quantize_enabled = bool(quantize_enabled)


@friendly_named_class("Hamming distance")
class HammingDistanceMeasure(MonophonicRhythmDistanceMeasure):
    """
    The hamming distance is based on the binary chains of the rhythms. The hamming distance is the sum of indexes
    where the binary rhythm chains do not match. The hamming distance is always an integer.
    """

    def __init__(self, unit='eighths', length_policy='multiple'):
        super(HammingDistanceMeasure, self).__init__(unit, length_policy)

    def __get_iterable__(self, rhythm: MonophonicRhythm, unit):
        return rhythm.get_binary(unit)

    def __compute_distance__(self, n, cx, cy, *cookies):  # cx = (binary) chain x
        hamming_distance, i = 0, 0
        while i < n:
            x = cx[i % len(cx)]
            y = cy[i % len(cy)]
            hamming_distance += x != y
            i += 1
        return hamming_distance


@friendly_named_class("Euclidean interval vector distance")
class EuclideanIntervalVectorDistanceMeasure(MonophonicRhythmDistanceMeasure, Quantizable):
    """
    The euclidean interval vector distance is the euclidean distance between the inter-onset vectors of the rhythms.
    """

    def __init__(self, unit='ticks', length_policy='exact', quantize=False):
        super(EuclideanIntervalVectorDistanceMeasure, self).__init__(unit, length_policy)
        self.quantize_enabled = quantize

    def __get_iterable__(self, rhythm: MonophonicRhythm, unit):
        return rhythm.get_post_note_inter_onset_intervals(unit, self.quantize_enabled)

    def __compute_distance__(self, n, vx, vy, *cookies):
        sum_squared_dt, i = 0, 0
        while i < n:
            dt = vx[i % len(vx)] - vy[i % len(vy)]
            sum_squared_dt += dt * dt
            i += 1
        return math.sqrt(sum_squared_dt)


@friendly_named_class("Interval difference vector distance")
class IntervalDifferenceVectorDistanceMeasure(MonophonicRhythmDistanceMeasure, Quantizable):
    """
    The interval difference vector distance is based on the interval difference vectors of the rhythms.
    """

    def __init__(self, unit='ticks', length_policy='fill', quantize=False, cyclic=True):
        super(IntervalDifferenceVectorDistanceMeasure, self).__init__(unit, length_policy)
        self.quantize_enabled = quantize
        self.cyclic = cyclic

    def __get_iterable__(self, rhythm: MonophonicRhythm, unit):
        return rhythm.get_interval_difference_vector(self.cyclic, unit, self.quantize_enabled)

    def __compute_distance__(self, n, vx, vy, *cookies):
        summed_fractions, i = 0, 0
        while i < n:
            x = float(vx[i % len(vx)])
            y = float(vy[i % len(vy)])
            numerator, denominator = (x, y) if x > y else (y, x)
            try:
                summed_fractions += numerator / denominator
            except ZeroDivisionError:
                return float('inf')
            i += 1
        return summed_fractions - n


@friendly_named_class("Swap distance")
class SwapDistanceMeasure(MonophonicRhythmDistanceMeasure, Quantizable):
    """
    The swap distance is the minimal number of swap operations required to transform one rhythm to another. A swap is an
    interchange of a one and a zero that are adjacent to each other in the binary representations of the rhythms.

    Although the concept of the swap distance is based on the rhythm's binary chain, this implementation uses the
    absolute onset times of the onsets. This makes it possible to work with floating point swap operations (0.x swap
    operation). Enable this by setting quantize to True in the constructor.
    """

    def __init__(self, unit='eighths', length_policy='multiple', quantize=False):
        super(SwapDistanceMeasure, self).__init__(unit, length_policy)
        self.quantize_enabled = quantize

    def __get_iterable__(self, rhythm: MonophonicRhythm, unit):
        return rhythm.get_onset_times(unit, self.quantize_enabled)

    def __get_cookie__(self, rhythm: MonophonicRhythm, unit):
        return int(math.ceil(rhythm.get_duration(unit)))

    def __compute_distance__(self, n, vx, vy, dur_x, dur_y):
        swap_distance, i, x_offset, y_offset = 0, 0, 0, 0
        while i < n:
            x_offset = i // len(vx) * dur_x
            y_offset = i // len(vy) * dur_y
            x = vx[i % len(vx)] + x_offset
            y = vy[i % len(vy)] + y_offset
            swap_distance += abs(x - y)
            i += 1
        return swap_distance


@friendly_named_class("Chronotonic distance")
class ChronotonicDistanceMeasure(MonophonicRhythmDistanceMeasure):
    """
    The chronotonic distance is the area difference (aka measure K) of the rhythm's chronotonic chains.
    """

    def __init__(self, unit='eighths', length_policy='multiple'):
        super(ChronotonicDistanceMeasure, self).__init__(unit, length_policy)

    def __get_iterable__(self, rhythm: MonophonicRhythm, unit):
        return rhythm.get_chronotonic_chain(unit)

    def __compute_distance__(self, n, cx, cy, *args):
        chronotonic_distance, i = 0, 0
        while i < n:
            x = cx[i % len(cx)]
            y = cy[i % len(cy)]
            # assuming that each pulse is a unit
            chronotonic_distance += abs(x - y)
            i += 1
        return chronotonic_distance


TRACK_WILDCARDS = ["*", "a*", "b*"]  # NOTE: Don't change the wildcard order or the code will break


def rhythm_pair_track_iterator(rhythm_a: IPolyphonicRhythm,
                               rhythm_b: IPolyphonicRhythm,
                               tracks: tp.Union[str, tp.Iterable[tp.Any]]):
    """
    Returns an iterator over the tracks of the given rhythms. Each iteration yields a track name and a pair of tracks
    with that track name; rhythm_a.get_track(name) and rhythm_b.get_track(name). This is yielded as a tuple:
        (track_name, (track_a, track_b))

    The tracks to iterate over can be specified with the 'tracks' argument:

        [iterable] - Iterate over tracks with the names in the given iterable
        '*'        - Iterate over all track names. If the rhythms contain tracks with the same name, this will just
                     result in one iteration. E.g. if both rhythms contain a track named 'foo', this will result in one
                     iteration yielding ('foo', (track_a, track_b))).
        'a*'      - Iterate over track names in rhythm a
        'b*'      - Iterate over track names in rhythm b

    :param rhythm_a: the first rhythm
    :param rhythm_b: the second rhythm
    :param tracks: either an iterable containing the track names, or one of the wildcards ['*', 'a*' or 'b*']
    :return: an iterator over the tracks of the given rhythms
    """

    try:
        wildcard_index = TRACK_WILDCARDS.index(tracks)
    except ValueError:
        wildcard_index = -1
        try:
            tracks = iter(tracks)
        except TypeError:
            raise ValueError("Excepted an iterable or "
                             "one of %s but got %s" % (TRACK_WILDCARDS, tracks))

    it_a, it_b = rhythm_a.get_track_iterator(), rhythm_b.get_track_iterator()

    if wildcard_index == -1:  # if given specific tracks
        for track_name in tracks:
            track_a = rhythm_a.get_track_by_name(track_name)
            track_b = rhythm_b.get_track_by_name(track_name)
            yield (track_name, (track_a, track_b))

    elif wildcard_index == 0:  # if '*' wildcard
        names = set()
        for track_a in it_a:
            name = track_a.name
            names.add(name)
            track_b = rhythm_b.get_track_by_name(name)
            names.add(name)
            yield (name, (track_a, track_b))
        for track_b in it_b:
            name = track_b.name
            if name in names:
                continue
            track_a = rhythm_a.get_track_by_name(name)
            yield (name, (track_a, track_b))

    elif wildcard_index == 1:  # if wildcard 'a*'
        for track_a in it_a:
            name = track_a.name
            track_b = rhythm_b.get_track_by_name(name)
            yield (name, (track_a, track_b))

    elif wildcard_index == 2:  # if wildcard 'b*'
        for track_b in it_b:
            name = track_b.name
            track_a = rhythm_a.get_track_by_name(name)
            yield (name, (track_a, track_b))

    else:
        assert False


class PolyphonicRhythmDistanceMeasure(DistanceMeasure):
    """Abstract base class for polyphonic rhythm distance measures

    This is an abstract base class for polyphonic rhythm distance measures. It measures the distance
    between two Rhythm objects.
    """

    def get_distance(self, rhythm_a: IPolyphonicRhythm, rhythm_b: IPolyphonicRhythm) -> tp.Union[float, int]:
        raise NotImplementedError


class SummedMonophonicRhythmDistance(PolyphonicRhythmDistanceMeasure):
    def __init__(self, track_distance_measure=HammingDistanceMeasure, tracks='a*', normalize=True):
        # type: (tp.Union[MonophonicRhythmDistanceMeasure, type], tp.Union[str, tp.Iterable[tp.Any]]) -> None
        self._tracks = []
        self.tracks = tracks
        self._track_distance_measure = None
        self.monophonic_measure = track_distance_measure
        self.normalize = normalize

    @property
    def monophonic_measure(self):  # type: () -> MonophonicRhythmDistanceMeasure
        """
        The distance measure used to compute the distance between the rhythm tracks; an instance of
        MonophonicRhythmDistanceMeasure.
        """

        return self._track_distance_measure

    @monophonic_measure.setter
    def monophonic_measure(
            self,
            track_distance_measure: tp.Union[MonophonicRhythmDistanceMeasure, tp.Type[MonophonicRhythmDistanceMeasure]]
    ):
        """
        Setter for the track distance measure. This should be either a MonophonicRhythmDistanceMeasure subclass or
        instance. When given a class, the measure will be initialized with no arguments.
        """

        if inspect.isclass(track_distance_measure) and \
                issubclass(track_distance_measure, MonophonicRhythmDistanceMeasure):
            track_distance_measure = track_distance_measure()
        elif not isinstance(track_distance_measure, MonophonicRhythmDistanceMeasure):
            raise ValueError("Expected a MonophonicRhythmDistanceMeasure subclass or "
                             "instance, but got '%s'" % track_distance_measure)
        self._track_distance_measure = track_distance_measure

    @property
    def tracks(self):
        """
        The tracks to iterate over when computing the distance. See rhythm_pair_track_iterator.
        """

        return self._tracks

    @tracks.setter
    def tracks(self, tracks):
        self._tracks = tracks

    def get_distance(self, rhythm_a, rhythm_b):
        """
        Returns the average track distance from the tracks of rhythm a and b. When the track distance can't be computed,
        duration of the longest rhythm is used as a distance. This can either happen when the a certain track doesn't
        exist in the two rhythms (e.g. 'snare' track in rhythm_a but not in rhythm_b) or when the track distance measure
        raises an exception when computing the distance.

        See RhythmSimilarityMeasure.tracks to specify which tracks should measured.

        :param rhythm_a: the first rhythm
        :param rhythm_b: the second rhythm
        :return: the average track distance of the two rhythms
        """

        measure = self.monophonic_measure
        unit = measure.output_unit
        duration = max(rhythm_a.get_duration(unit), rhythm_b.get_duration(unit))
        n_tracks, total_distance = 0, 0

        for name, tracks in rhythm_pair_track_iterator(rhythm_a, rhythm_b, self.tracks):
            distance = duration
            if None not in tracks:
                try:
                    distance = measure.get_distance(tracks[0], tracks[1])
                except ValueError:
                    pass
            total_distance += distance
            n_tracks += 1

        average_distance = float(total_distance) / n_tracks
        return average_distance / duration if self.normalize else average_distance


def create_rumba_rhythm(resolution=240):
    """
    Utility function that creates a one-bar rumba rhythm.

    :param resolution: rhythm resolution
    :return: rumba rhythm
    """

    track = PolyphonicRhythm.Track(((0, 127), (3, 127), (7, 127), (10, 127), (12, 127)), "main_track")
    rhythm = PolyphonicRhythm([track], 4, time_signature=(4, 4))
    rhythm.set_resolution(resolution)
    return rhythm

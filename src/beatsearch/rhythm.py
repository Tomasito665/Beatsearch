# coding=utf-8
import os
import enum
import inspect
import textwrap
import itertools
from abc import abstractmethod, ABCMeta
from io import IOBase
import typing as tp
from fractions import Fraction
from functools import wraps, total_ordering
from collections import OrderedDict, namedtuple, defaultdict
from beatsearch.utils import TupleView, friendly_named_class, most_common_element, sequence_product
import math
import numpy as np
import midi


class UnitError(Exception):
    pass


@total_ordering  # enable <, > <= and >= operators
class Unit(enum.Enum):
    OCTUPLE_WHOLE = Fraction(8, 1), ("octuple whole", "octuple", "large", "duplex longa", "maxima")
    QUADRUPLE_WHOLE = Fraction(4, 1), ("long", "longa")
    DOUBLE_WHOLE = Fraction(2, 1), ("double", "breve")
    WHOLE = Fraction(1, 1), ("whole", "semibreve")
    HALF = Fraction(1, 2), ("half", "minim")
    QUARTER = Fraction(1, 4), ("quarter", "crotchet")
    EIGHTH = Fraction(1, 8), ("eighth", "quaver")
    SIXTEENTH = Fraction(1, 16), ("sixteenth", "semiquaver")
    THIRTY_SECOND = Fraction(1, 32), ("thirty-second", "demisemiquaver")
    SIXTY_FOURTH = Fraction(1, 64), ("sixty-fourth", "hemidemisemiquaver")

    __by_note_names__ = dict()    # type: tp.Dict[str, Unit]
    __by_note_values__ = dict()   # type: tp.Dict[float, Unit]

    @classmethod
    def get(cls, query):  # type: (tp.Union[Fraction, str, float, Unit]) -> tp.Union[Unit, None]
        """Returns a unit given either its note value or one of its names

        Returns a Unit enum object given either:
            note value: its note value as a float (e.g., 1/4 for Unit.QUARTER)
            note name: one of its names (e.g., "quarter" or "crotchet" for Unit.QUARTER)
            Unit enum: the unit enumeration object itself (this will be returned)

        This method only returns None if the given query is None. In all other cases it will return a Unit enum object
        or raise UnitError.

        :param query: either a note value or a note name
        :return: unit enum object or None if query is None

        :raises UnitError: if unit not found
        """

        if query is None:
            return None

        if isinstance(query, str):
            try:
                return cls.__by_note_names__[query]
            except KeyError:
                raise UnitError("No unit named: %s" % query)
        elif not isinstance(query, Unit):
            if isinstance(query, Fraction):
                # we don't do the conversion to Fraction if not necessary
                # as it is quite an expensive operation
                fraction = query
            else:
                # unit with largest denominator (=64) is SIXTY_FOURTH
                fraction = Fraction(query).limit_denominator(64)
            try:
                return cls.__by_note_values__[fraction]
            except KeyError:
                raise UnitError("No unit with note value: %s (query=%s)" % (str(fraction), query))

        assert isinstance(query, Unit)
        return query

    @classmethod
    def check(cls, query: tp.Union[str, float]) -> None:
        """Tries to find a unit, raises UnitError if no such unit

        :param query: query (see get())
        :return: None

        :raises UnitError: if no unit found with given query
        """

        cls.get(query)

    def __init__(self, note_value: Fraction, note_names: tp.Tuple[str]):
        assert isinstance(note_names, tuple)
        self._note_value_float = float(note_value)

    def get_note_value(self) -> Fraction:
        """Returns the note value of this musical unit as a fraction

        :return: note value of this musical unit as a Fraction
        """

        return self.value[0]

    def get_note_names(self):
        """Returns the common names of this musical unit

        :return: names of this musical unit as a tuple of strings
        """

        return self.value[1]

    def convert(self, value, to_unit, quantize=False):
        # type: (tp.Union[int, float], Unit, bool) -> tp.Union[int, float]
        """Converts a value from this unit to another unit

        :param value: the value to convert
        :param to_unit: the musical unit to convert this value to
        :param quantize: if true, the converted value will be rounded
        :return: the converted value as a float (as an int if quantize is true)
        """

        from_note_value = self._note_value_float
        to_note_value = to_unit._note_value_float
        converted_value = value * (from_note_value / to_note_value)
        return round(converted_value) if quantize else converted_value

    def from_ticks(self, ticks: int, resolution: int, quantize=False) -> tp.Union[int, float]:
        """Converts the given ticks to this musical time unit

        :param ticks: tick value to convert
        :param resolution: tick resolution in PPQN
        :param quantize: if true, the returned value will be rounded
        :return: the given tick value in this time unit
        """

        quarter_value = ticks / int(resolution)
        return Unit.QUARTER.convert(quarter_value, self, quantize)

    def to_ticks(self, value: float, resolution: int) -> int:
        """Converts a value from this musical time unit to ticks

        :param value: value in this musical time unit
        :param resolution: tick resolution in PPQN
        :return: tick value
        """

        quarter_value = self.convert(value, self.QUARTER, False)
        return round(quarter_value * resolution)

    def __lt__(self, other):
        other_note_value = other._note_value_float if isinstance(other, Unit) else other
        return self._note_value_float < other_note_value


Unit.__by_note_names__ = dict((name, unit) for unit in Unit for name in unit.get_note_names())
Unit.__by_note_values__ = dict((unit.get_note_value(), unit) for unit in Unit)
UnitType = tp.Union[Unit, Fraction, str, float]


def parse_unit_argument(func: tp.Callable[[tp.Any], tp.Any]) -> tp.Callable[[tp.Any], tp.Any]:
    """Decorator that replaces the "unit" parameter with a Unit enum object

    Replaces the unit argument of the decorated function with Unit.get(unit). For example:

        @parse_unit_argument
        def foo(unit):
            return unit

        quarter = foo("quarter")  # returns Unit.QUARTER
        sixteenth = foo(1/16)  # returns Unit.SIXTEENTH

    :param func: function receiving a "unit" parameter
    :return: function receiving a Unit object parameter
    """

    func_parameters = inspect.signature(func).parameters

    try:
        unit_param_position = tuple(func_parameters.keys()).index("unit")
    except ValueError:
        raise ValueError("Functions decorated with parse_unit_argument should have a \"unit\" parameter")

    unit_param = func_parameters.get("unit")  # type: inspect.Parameter

    @wraps(func)
    def wrapper(*args, **kwargs):
        given_positional_param = len(args) > unit_param_position
        given_named_param = "unit" in kwargs

        if given_named_param:
            kwargs['unit'] = Unit.get(kwargs['unit'])
        elif given_positional_param:
            assert len(args) > unit_param_position
            unit = args[unit_param_position]
            unit = Unit.get(unit)
            args = itertools.chain(args[:unit_param_position], [unit], args[unit_param_position + 1:])
        else:
            kwargs['unit'] = Unit.get(unit_param.default)

        return func(*args, **kwargs)
    return wrapper


def rescale_tick(tick: int, old_res: int, new_res):
    """Rescales the given tick from one resolution to another

    :param tick:    tick value to rescale
    :param old_res: original tick resolution in PPQN
    :param new_res: new tick resolution in PPQN

    :return: rescaled tick value as an integer

    :raises ValueError: if one of the resolutions is equal or smaller than zero
    """

    if old_res <= 0 or new_res <= 0:
        raise ValueError("expected resolution greater than zero")

    return round(tick / old_res * new_res)


def convert_tick(tick: int, old_res: int, target: tp.Union[UnitType, int, None], quantize=False):
    """Utility function to either rescale a tick to another PPQN resolution or to convert it to a musical unit

    This function function has two behaviours:
        * rescale given tick to another PPQN       ->  When given an integer as "target" parameter, which is then
                                                       used as new PPQN resolution. The tick rescaling is done by
                                                       calling rescale_tick().
        * represent given tick in a musical unit   ->  When given a musical unit (see Unit.get) as "target" parameter.
                                                       This representation is done with Unit.from_tick().

    When the "target" parameter is None, this function will return the tick.

    :param tick:     tick value to convert
    :param old_res:  original tick resolution in PPQN
    :param target:   Either a musical unit (see Unit.get) or a new tick resolution in PPQN as an integer. When given
                     None, the same tick will be returned.
    :param quantize: When converting to a musical unit, when given true, the returned value will be rounded. This
                     parameter is ignored when converting to another resolution.

    :return: converted tick
    """

    if target is None:
        return tick

    if isinstance(target, int):
        return rescale_tick(tick, old_res, target)

    return Unit.get(target).from_ticks(tick, old_res, quantize)


class TimeSignature(object):
    """
    This class represents a musical time signature, consisting of a numerator and a denominator.
    """

    def __init__(self, numerator, denominator):
        self._numerator = numerator
        self._denominator = denominator
        self._beat_unit = Unit.get(Fraction(1, denominator))

    @property
    def numerator(self):
        return self._numerator

    @property
    def denominator(self):
        return self._denominator

    def get_beat_unit(self) -> Unit:
        return self._beat_unit

    def to_midi_event(self, metronome=24, thirty_seconds=8):
        return midi.TimeSignatureEvent(
            numerator=self.numerator,
            denominator=self.denominator,
            metronome=metronome,
            thirtyseconds=thirty_seconds
        )

    @parse_unit_argument
    def get_meter_tree(self, unit: UnitType = Unit.EIGHTH):
        """Returns a vector containing subdivision counts needed to construct a hierarchical meter tree structure

        A meter is representable as a tree structure, e.g. time signature 6/8 is represented by this tree:

                   ----- A ------           note value = 𝅗𝅥. (dotted half note)
              --- B ---      --- B ---      note value = ♩. (dotted quarter note)
             C    C    C    C    C    C     note value = 𝅘𝅥𝅮  (eighth note)

        In this tree, the A node splits up into two nodes and the B nodes into three nodes. Given that the tree may only
        split up into either only binary or only ternary subdivisions, this tree is represented by this list: [2, 3],
        because the first node has 2 subdivisions (A) and the second nodes have 3 subdivisions.

        This method will construct such a tree and return the subdivision list of the tree where the deepest node
        represents the given time unit. Only musical time units are allowed ("eighths", "quarters", etc., not "ticks").

        :param unit: the musical time unit of the deepest nodes in the tree
        :return: a tuple containing the subdivision counts needed to construct a hierarchical meter tree structure for
                 this time signature

        :raises ValueError: if given ticks or if this time signature is not divisible by the given time unit (e.g. a 6/8
                            time signature is not representable with multiples of quarters, but it is with multiples of
                            eighths or sixteenths)
        """

        n_units_per_beat = self.get_beat_unit().convert(1, unit, False)
        curr_branch = self.numerator
        divisions = []

        if math.isclose(n_units_per_beat, int(n_units_per_beat)):
            n_units_per_beat = int(n_units_per_beat)
        else:
            raise ValueError("Can't express %s time signature in \"%s\"" % (self, unit))

        while curr_branch > 1:
            for quotient in (2, 3):
                if curr_branch % quotient == 0:
                    divisions.append(quotient)
                    curr_branch /= quotient
                    break
            else:
                raise Exception("No context-sensitive meters allowed. Branch of %i units "
                                "not equally divisible into binary or ternary sub-units" % curr_branch)

        divisions.extend(itertools.repeat(2, n_units_per_beat - 1))
        return tuple(divisions)

    @parse_unit_argument
    def get_metrical_weights(self, unit=Unit.EIGHTH, root_weight=0):
        """Returns the metrical weights for a full measure of this time signature

        Constructs a hierarchical meter tree structure (see get_meter_tree) ands assigns a weight to each node. Then it
        flattens the tree and returns it as a list. The weight of a node is the weight of the parent node minus one. The
        weight of the root node is specified with the root_weight argument of this method.

        For example, given the "eighths" tree structure of a 4/4 time signature:

                         --------------- R -------------
                 ------ A ------                 ------ A ------        note value = 𝅝 (whole note)
             -- B --         -- B --         -- B --         -- B --    note value = ♩ (quarter note)
            C       C       C       C       C       C       C       C   note value = 𝅘𝅥𝅮 (eighth note)

        if root_weight is 0, then the weights of the R, A, B and C nodes are 0, -1, -2 and -3 respectively. This will
        yield these metrical weights: [0, -3, -2, -3, -1, -3, -2, -3].

        :param unit: step time unit
        :param root_weight: weight of first node in the
        :return: the metrical weights for a full measure of this time signature with the given time unit
        """

        subdivisions = self.get_meter_tree(unit)
        metrical_weights = [None] * sequence_product(subdivisions)
        n_branches = 1

        for n, curr_subdivision in enumerate(itertools.chain([1], subdivisions)):
            curr_subdivision_weight = root_weight - n
            n_branches *= curr_subdivision

            for ix in np.linspace(0, len(metrical_weights), n_branches, endpoint=False, dtype=int):
                if metrical_weights[ix] is None:
                    metrical_weights[ix] = curr_subdivision_weight

        return metrical_weights

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
                cls.check_resolution(rhythm)
                return f(rhythm, *args, **kwargs)
            return wrapper

        @classmethod
        def needs_time_signature(cls, f):
            @wraps(f)
            def wrapper(rhythm, *args, **kwargs):
                cls.check_time_signature(rhythm)
                return f(rhythm, *args, **kwargs)
            return wrapper

        @classmethod
        def check_resolution(cls, rhythm):
            if rhythm.get_resolution() == 0:
                raise cls.ResolutionNotSet

        @classmethod
        def check_time_signature(cls, rhythm):
            if not rhythm.get_time_signature():
                raise cls.TimeSignatureNotSet

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

    @parse_unit_argument
    def get_duration(self, unit: tp.Optional[UnitType] = None, ceil: bool = False) -> tp.Union[int, float]:
        """
        Returns the duration of this rhythm in the given musical time unit or in ticks if no unit is given.

        :param unit: time unit in which to return the duration or None to get the duration in ticks
        :param ceil: if True, the returned duration will be rounded up (ignored if unit is set to None)
        :return: duration of this rhythm in given unit or in ticks if no unit is given
        """

        duration_in_ticks = self.get_duration_in_ticks()

        if unit is None:
            return duration_in_ticks

        resolution = self.get_resolution()
        duration = unit.from_ticks(duration_in_ticks, resolution, False)
        return int(math.ceil(duration)) if ceil else duration

    @parse_unit_argument
    def set_duration(self, duration: tp.Union[int, float], unit: tp.Optional[UnitType] = None) -> None:
        """
        Sets the duration of this rhythm in the given time unit (in ticks if not unit given)

        :param duration: new duration in the given unit or in ticks if no unit provided
        :param unit: time unit of the given duration or None to set the duration in ticks
        :return: None
        """

        if unit is None:
            duration_in_ticks = round(duration)
        else:
            resolution = self.get_resolution()
            duration_in_ticks = unit.to_ticks(duration, resolution)

        self.set_duration_in_ticks(duration_in_ticks)

    @Precondition.needs_time_signature
    @parse_unit_argument
    def get_beat_duration(self, unit: tp.Optional[UnitType] = None) -> tp.Union[int, float]:
        # TODO change to pulse_duration
        """
        Returns the duration of one musical beat, based on the time signature.

        :param unit: musical unit in which to return the beat duration or None to get the beat duration in ticks
        :return: the duration of one beat in the given musical unit or in ticks if no unit is given
        :raises TimeSignatureNotSet: if no time signature has been set
        """

        time_signature = self.get_time_signature()
        beat_unit = time_signature.get_beat_unit()

        if unit is None:
            resolution = self.get_resolution()
            return beat_unit.to_ticks(1, resolution)

        return beat_unit.convert(1, unit, False)

    @Precondition.needs_time_signature
    @parse_unit_argument
    def get_measure_duration(self, unit: tp.Optional[UnitType] = None) -> tp.Union[int, float]:
        """
        Returns the duration of one musical measure, based on the time signature.

        :param unit: musical unit in which to return the measure duration or None to get the measure duration in ticks
        :return: the duration of one measure in the given unit or in ticks if no unit is given
        :raises TimeSignatureNotSet: if no time signature has been set
        """

        time_signature = self.get_time_signature()
        n_beats_per_measure = time_signature.numerator
        beat_unit = time_signature.get_beat_unit()

        if unit is None:
            resolution = self.get_resolution()
            return beat_unit.to_ticks(n_beats_per_measure, resolution)

        return beat_unit.convert(n_beats_per_measure, unit, False)

    def get_duration_in_measures(self):
        """
        Returns the duration of this rhythm in musical measures as a floating point number.

        :return: the duration of this rhythm in measures as a floating point number
        :raises TimeSignatureNotSet: if no time signature has been set
        """

        measure_duration = self.get_measure_duration(None)
        duration = self.get_duration(None)
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


class RhythmFactory(object, metaclass=ABCMeta):
    """Interface for rhythm factory utility classes"""

    def __init__(self):
        raise Exception("Can't instantiate RhythmFactory. It is a utility class and only contains static methods.")

    @classmethod
    @abstractmethod
    def from_string(
            cls,
            onset_string: str,
            time_signature: tp.Optional[tp.Union[tp.Tuple[int, int], TimeSignature]] = None,
            velocity: int = 100,
            resolution: int = 4,
            onset_character: str = "x",
            **kwargs) -> Rhythm:
        """Creates and returns a rhythm, given a string representation of its onsets"""

        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_binary_vector(
            cls,
            binary_vector: tp.Iterable[tp.Any],
            time_signature: tp.Optional[tp.Union[tp.Tuple[int, int], TimeSignature]] = None,
            velocity: int = 100,
            resolution: int = 4,
            **kwargs) -> Rhythm:
        """Creates and returns a rhythm, given a sequence representation of its onsets"""

        raise NotImplementedError

    @staticmethod
    def __string_to_binary_onset_vector__(onset_string, onset_char) -> tp.Tuple[bool, ...]:
        return tuple((char == onset_char) for char in onset_string)

    @staticmethod
    def __binary_vector_to_onsets__(binary_vector: tp.Sequence[bool], velocity: int) -> tp.Tuple[tp.Tuple[int, int]]:
        return tuple(filter(None, ((ix, velocity) if atom else None for ix, atom in enumerate(binary_vector))))


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
            new_dur = rescale_tick(old_dur, old_res, new_res)
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

        scaled_tick = rescale_tick(self.tick, resolution_from, resolution_to)
        return Onset(scaled_tick, self.velocity)


class MonophonicRhythm(Rhythm, metaclass=ABCMeta):
    """Monophonic rhythm interface

    Interface for monophonic rhythms.
    """

    @abstractmethod
    def get_onsets(self) -> tp.Tuple[Onset, ...]:
        """
        Returns the onsets within this rhythm as a tuple of onsets, where each onset is an instance of Onset.

        :return: the onsets within this rhythm as a tuple of Onset objects
        """

        raise NotImplementedError

    @abstractmethod
    def set_onsets(self, onsets: tp.Union[tp.Iterable[Onset], tp.Iterable[tp.Tuple[int, int]]]):
        """
        Sets the onsets of this rhythm.

        :param onsets: onsets as an iterable of (absolute tick, velocity) tuples or as Onset objects
        :return: None
        """

        raise NotImplementedError

    @property
    def onsets(self) -> tp.Tuple[Onset, ...]:
        """See MonophonicRhythm.get_onsets"""
        return self.get_onsets()

    @onsets.setter
    def onsets(self, onsets: tp.Union[tp.Iterable[Onset], tp.Iterable[tp.Tuple[int, int]]]):
        """See MonophonicRhythm.set_onsets"""
        self.set_onsets(onsets)

    @onsets.deleter
    def onsets(self):
        self.set_onsets([])

    def get_last_onset_tick(self) -> int:  # implements Rhythm.get_last_onset_tick
        try:
            return self.onsets[-1].tick
        except IndexError:
            return -1

    def get_onset_count(self) -> int:  # implements Rhythm.get_onset_count
        return len(self.onsets)

    # noinspection PyPep8Naming
    class create(RhythmFactory):  # not intended like a class but like a namespace for factory methods
        @classmethod
        def from_string(
                cls,
                onset_string: str,
                time_signature: tp.Optional[tp.Union[tp.Tuple[int, int], TimeSignature]] = None,
                velocity: int = 100,
                resolution: int = 4,
                onset_character="x",
                **kwargs):  # type: () -> MonophonicRhythmImpl
            """
            Creates a new monophonic rhythm from a string. Each character in the string will represent one tick and each
            onset character will represent an onset in the rhythm, e.g. "x--x---x--x-x---", given onset character "x".
            The length of the onset string will determine the duration of the rhythm.

            :param onset_string:   onset string where each onset character will result in an onset
            :param time_signature: time signature of the rhythm as a (numerator, denominator) tuple or TimeSignature obj
            :param velocity:       the velocity of the onsets as an integer, which will be the same for all onsets
            :param resolution:     resolution in pulses per quarter note (e.g., if resolution is set to 4, four
                                   characters in the onset string will represent one quarter note)
            :param onset_character: onset character (see onset_string)
            :param kwargs:         unused

            :return: monophonic rhythm object
            """

            return cls.from_binary_vector(
                binary_vector=cls.__string_to_binary_onset_vector__(onset_string, onset_character),
                time_signature=time_signature,
                velocity=velocity,
                resolution=resolution
            )

        @classmethod
        def from_binary_vector(
                cls,
                binary_vector: tp.Sequence[tp.Any],
                time_signature: tp.Optional[tp.Union[tp.Tuple[int, int], TimeSignature]] = None,
                velocity: int = 100,
                resolution: int = 4,
                **kwargs):  # type: () -> MonophonicRhythmImpl
            """
            Creates a new monophonic rhythm, given a binary chain (iterable). Each element in the iterable represents
            one tick. If the element is True, that will result in an onset, e.g. [1, 0, 1, 0, 1, 1, 1, 0].

            :param binary_vector:  sequence where each true-evaluated element will result in an onset
            :param time_signature: time signature of the rhythm as a (numerator, denominator) tuple or TimeSignature obj
            :param velocity:       the velocity of the onsets as an integer, which will be the same for all onsets
            :param resolution:     resolution in pulses per quarter note (e.g., if resolution is set to 4, four
                                   elements in the binary vector will represent one quarter note)
            :param kwargs:         unused

            :return: monophonic rhythm object
            """

            return MonophonicRhythmImpl(
                onsets=cls.__binary_vector_to_onsets__(binary_vector, velocity),
                duration_in_ticks=len(binary_vector), resolution=resolution,
                time_signature=time_signature
            )


class MonophonicRhythmBase(MonophonicRhythm, metaclass=ABCMeta):
    """Monophonic rhythm base class implementing MonophonicRhythm

    Abstract base class for monophonic rhythms. This class implements MonophonicRhythm.get_onsets, adding onset state
    to subclasses. Note that this class does not extend RhythmBase and therefore does NOT add rhythm base state like
    bpm, resolution, time signature, etc.

    This class inherits all monophonic rhythm representations from the MonophonicRhythmRepresentationsMixin class.
    """

    class OnsetsNotInChronologicalOrder(Exception):
        """Exception thrown when two adjacent onsets are not in chronological order"""
        def __init__(self, tick_a: int, tick_b: int):
            msg = "<..., %i, !%i!, ...>" % (tick_a, tick_b)
            super().__init__(msg)

    def __init__(
            self,
            onsets: tp.Union[tp.Iterable[Onset],
                             tp.Iterable[tp.Tuple[int, int]],
                             tp.Iterable[tp.Sequence[int]]] = None,
    ):
        """
        Creates a new monophonic rhythm from the given onsets. The onsets will be stored as MonophonicOnset.Onset
        named tuples.

        :param onsets: An iterable returning an (absolute tick, velocity) tuple for each iteration. The onsets should
                       be given in chronological order.
        :raises OnsetsNotInChronologicalOrder
        """

        self._onsets = tuple()  # type: tp.Tuple[Onset, ...]
        self.set_onsets(onsets)

    def get_onsets(self) -> tp.Tuple[Onset, ...]:
        """
        Returns the onsets within this rhythm as a tuple of onsets, where each onset is an instance of Onset.

        :return: the onsets within this rhythm as a tuple of MonophonicRhythmImpl.Onset objects
        """

        return self._onsets

    def set_onsets(self, onsets: tp.Union[tp.Iterable[Onset], tp.Iterable[tp.Tuple[int, int]]]):
        """
        Sets the onsets of this rhythm. The given onsets must be in chronological order. If they're not, this method
        will raise an OnsetsNotInChronologicalOrder exception.

        :param onsets: onsets as an iterable of (absolute tick, velocity) tuples or as Onset objects or false value to
                       remove the onsets
        :return: None
        :raises OnsetsNotInChronologicalOrder
        """

        if not onsets:
            self._onsets = tuple()
            return

        def validate_onsets_generator():
            prev_tick = -1
            for onset in onsets:
                try:
                    tick, velocity, *_ = onset
                except (TypeError, ValueError):
                    raise ValueError("onset should be iterable of at least two "
                                     "elements (tick, velocity) but got %s" % str(onset))
                if tick < prev_tick:
                    raise self.OnsetsNotInChronologicalOrder(prev_tick, tick)
                yield Onset(tick, velocity)
                prev_tick = tick

        self._onsets = tuple(validate_onsets_generator())

    # implements Rhythm.__rescale_onset_ticks__
    def __rescale_onset_ticks__(self, old_resolution: int, new_resolution: int):
        self._onsets = tuple(onset.scale(old_resolution, new_resolution) for onset in self._onsets)


class MonophonicRhythmImpl(RhythmBase, MonophonicRhythmBase):
    """Implements both rhythm base and monophonic rhythm base"""

    def __init__(
            self,
            onsets: tp.Union[tp.Iterable[Onset],
                             tp.Iterable[tp.Tuple[int, int]],
                             tp.Iterable[tp.Sequence[int]]] = None,
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

    def __init__(
            self, onsets: tp.Union[tp.Iterable[Onset],
                                   tp.Iterable[tp.Tuple[int, int]],
                                   tp.Iterable[tp.Sequence[int]]] = None,
            track_name: str = "", parent: Rhythm = None
    ):
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
        Returns an iterator over the tracks of this polyphonic rhythm. Each iteration yields a
        PolyphonicRhythmImpl.Track object.

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

    # noinspection PyPep8Naming
    class create(RhythmFactory):  # not intended like a class but like a namespace for factory methods
        @staticmethod
        def __track_name_generator(n: int) -> tp.Generator[str, None, None]:
            for i in range(n):
                yield "track %i" % n

        @classmethod
        def from_string(
                cls,
                input_string: str,
                time_signature: tp.Optional[tp.Union[tp.Tuple[int, int], TimeSignature]] = None,
                velocity: int = 100,
                resolution: int = 4,
                onset_character: str = "x",
                *_,
                track_separator_char: str = "\n",
                name_separator_char: str = ":",
                **kwargs):  # type: () -> PolyphonicRhythmImpl
            """
            Creates a new polyphonic rhythm from a string. The input string should contain one binary onset vector per
            track. The string should also provide the track names. The binary onset vector is a string where each onset
            character represents an onset. The tracks must be separated by the track separator character, which defaults
            to a new line.

            For example, given name separator ":", track separator "\n" and onset character "x", to create a simple
            rhythm with two tracks we could do:

                PolyphonicRhythm.create.from_string(textwrap.dedent(\"""
                    kick:   x---x---x---x---
                    snare:  ----x-------x-x-
                    hi-hat: x-xxx-xxx-xxx-xx
                \"""))

            :param input_string:         The input string contains information for all tracks, separated by the given
                                         track separator character. The track string is divided into the track name and
                                         the binary onset vector with the given name separator character. The binary
                                         onset vector is a string whose length determines the duration of the rhythm.
                                         Each onset character in the binary onset vector will result in an onset.
            :param time_signature:       time signature of the rhythm as a (numerator, denominator) tuple or a
                                         TimeSignature object
            :param velocity:             the velocity of the onsets as an integer, which will be the same for all onsets
            :param resolution:           resolution in pulses per quarter note (e.g., if resolution is set to 4, four
                                         characters in the binary onset onset string will represent one quarter note)
            :param onset_character:      onset character (see onset_string)
            :param track_separator_char: see input_string
            :param name_separator_char:  see input_string
            :param kwargs:               unused

            :return: polyphonic rhythm object
            """

            input_string = input_string.strip()
            track_names, track_onset_vectors = [], []

            for track_string in input_string.split(track_separator_char):
                track_name, onset_string = track_string.split(name_separator_char)
                track_onset_vectors.append(tuple((char == onset_character) for char in onset_string.strip()))
                track_names.append(track_name.strip())

            return cls.from_binary_vector(
                binary_vector_tracks=track_onset_vectors,
                time_signature=time_signature,
                velocity=velocity,
                resolution=resolution,
                track_names=track_names
            )

        @classmethod
        def from_binary_vector(
                cls,
                binary_vector_tracks: tp.Sequence[tp.Sequence[tp.Any]],
                time_signature: tp.Optional[tp.Union[tp.Tuple[int, int], TimeSignature]] = None,
                velocity: int = 100,
                resolution: int = 4,
                *_, track_names: tp.Sequence[str] = None,
                **kwargs):  # type: () -> PolyphonicRhythmImpl
            """
            Creates a new polyphonic rhythm from a sequence containing one binary onset vector per track. Track names
            are optional and are given to the track_names parameter.

            :param binary_vector_tracks: sequence holding one binary onset vector per track
            :param time_signature:       time signature of the rhythm as a (num, den) tuple or TimeSignature object
            :param velocity:             the velocity of the onsets as an integer, which will be the same for all onsets
            :param resolution:           resolution in pulses per quarter note (e.g., if resolution is set to 4, four
                                         characters in the onset string will represent one quarter note)
            :param track_names:          names of the tracks
            :param kwargs:               unused
            :return: polyphonic rhythm object
            """

            n_tracks = len(binary_vector_tracks)
            track_names = track_names or cls.__track_name_generator(n_tracks)

            def track_generator():
                for track_name, binary_vector in zip(track_names, binary_vector_tracks):
                    onsets = filter(None, ((ix, velocity) if atom else None for ix, atom in enumerate(binary_vector)))
                    yield Track(onsets, track_name)

            return PolyphonicRhythmImpl(track_generator(), time_signature=time_signature, resolution=resolution)

    #####################################
    # Polyphonic rhythm representations #
    #####################################

    # TODO remove duplicate functionality (see MonophonicRhythm.get_interval_histogram)
    # TODO adapt to new feature extraction API (create PolyphonicRhythmFeatureExtractor impl for this)
    # @parse_unit_argument
    # def get_interval_histogram(self, unit: tp.Optional[UnitType] = None) \
    #         -> tp.Tuple[tp.Iterable[int], tp.Iterable[int]]:
    #     """
    #     Returns the interval histogram of all the tracks combined.
    #
    #     :return: combined interval histogram of all the tracks in this rhythm
    #     """
    #
    #     intervals = []
    #
    #     for track in self.get_track_iterator():
    #         track_intervals = track.get_post_note_inter_onset_intervals(unit, quantize=True)
    #         intervals.extend(track_intervals)
    #
    #     histogram = np.histogram(intervals, tuple(range(min(intervals), max(intervals) + 2)))
    #     occurrences = histogram[0].tolist()
    #     bins = histogram[1].tolist()[:-1]
    #     return occurrences, bins


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


class FrequencyBand(enum.Enum):
    """Enumeration containing three drum sound frequency bands (low, mid and high)"""
    LOW = enum.auto()
    MID = enum.auto()
    HIGH = enum.auto()


class DecayTime(enum.Enum):
    """Enumeration containing three drum sound decay times (short, normal and long)"""
    SHORT = enum.auto()
    NORMAL = enum.auto()
    LONG = enum.auto()


class MidiDrumMapping(object, metaclass=ABCMeta):
    """Midi drum mapping interface

    Each MidiDrumMapping object represents a MIDI drum mapping and is a container for MidiDrumKey objects. It provides
    functionality for retrieval of these objects, based on either midi pitch, frequency band or key id.
    """

    class MidiDrumKey(object):
        """Struct-like class holding information about a single key within a MIDI drum mapping

        Holds information about the frequency band and the decay time of the drum sound it represents. Also stores the
        MIDI pitch ([0, 127]) which is used to produce this sound and an ID, which defaults to the MIDI pitch.
        """

        def __init__(self, midi_pitch: int, frequency_band: FrequencyBand,
                     decay_time: DecayTime, description: str, key_id: str = None):
            """Creates a new midi drum key

            :param midi_pitch:     the MIDI pitch as an integer in the range [0, 127] (the MIDI pitch has to be unique
                                   within the mapping this drum key belongs to)
            :param frequency_band: FrequencyBand enum object (LOW, MID or HIGH)
            :param decay_time:     DecayTime enum object (SHORT, NORMAL or LONG)
            :param description:    a small description (a few words, max 50 characters) of the sound of this drum sound
            :param key_id:         a unique (within the drum mapping) id for this key as a string (defaults to the midi
                                   pitch)

            :raises ValueError: if midi pitch not in range or if description exceeds the max number of characters
            :raises TypeError:  if given frequency band is not a FrequencyBand object or given decay time is not a
                                DecayTime object
            """

            midi_pitch = int(midi_pitch)
            description = str(description)
            key_id = str(midi_pitch if key_id is None else key_id)

            if not (0 <= midi_pitch <= 127):
                raise ValueError("expected midi pitch in range [0, 127]")
            if len(description) > 50:
                raise ValueError("description length should not exceed 50 characters")
            if not isinstance(frequency_band, FrequencyBand):
                raise TypeError
            if not isinstance(decay_time, DecayTime):
                raise TypeError

            self._data = (midi_pitch, frequency_band, decay_time, description, key_id)

        @property
        def midi_pitch(self) -> int:
            """The midi pitch of this midi drum key (read-only)"""
            return self._data[0]

        @property
        def frequency_band(self) -> FrequencyBand:
            """The frequency band (FrequencyBand enum object) of this drum key (read-only)"""
            return self._data[1]

        @property
        def decay_time(self) -> DecayTime:
            """The decay time (DecayTime enum object) of this drum key (read-only)"""
            return self._data[2]

        @property
        def description(self) -> str:
            """The description of this drum key as a string (read-only)"""
            return self._data[3]

        @property
        def id(self) -> str:
            """The id of this drum key as a string (read-only)"""
            return self._data[4]

        def __repr__(self):
            return "MidiDrumKey(%i, %s, %s, \"%s\", \"%s\")" % (
                self.midi_pitch, self.frequency_band.name, self.decay_time.name, self.description, self.id)

    @abstractmethod
    def get_name(self):
        """Returns the name of this drum mapping

        :return: name of this drum mapping as a string
        """

        raise NotImplementedError

    def get_keys(self) -> tp.Sequence[MidiDrumKey]:
        """Returns an immutable sequence containing all keys

        :return: an immutable sequence containing all the keys of this mapping as MidiDrumKey objects
        """

        raise NotImplementedError

    def get_key_by_midi_pitch(self, midi_pitch: int) -> tp.Union[MidiDrumKey, None]:
        """Returns the MidiDrumKey with the given midi pitch

        :param midi_pitch: midi pitch as an integer
        :return: MidiDrumKey object with the given midi pitch or None if no key found with given pitch
        """

        try:
            return next(key for key in self.get_keys() if key.midi_pitch == midi_pitch)
        except StopIteration:
            return None

    def get_key_by_id(self, key_id: str) -> tp.Union[MidiDrumKey, None]:
        """Returns the MidiDrumKey with the given key id

        :param key_id: key id of the midi drum key
        :return: MidiDrumKey object with the given key id or None if no key found with given key id
        """

        try:
            return next(key for key in self.get_keys() if key.id == key_id)
        except StopIteration:
            return None

    def get_keys_with_frequency_band(self, frequency_band: FrequencyBand) -> tp.Tuple[MidiDrumKey, ...]:
        """Returns the keys with the given frequency band

        :param frequency_band: FrequencyBand enum object (LOW, MID or HIGH)
        :return: a tuple containing the MidiDrumKey objects with the given frequency band or an empty tuple if nothing
                 found
        """

        return tuple(key for key in self.get_keys() if key.frequency_band == frequency_band)

    def get_keys_with_decay_time(self, decay_time: DecayTime) -> tp.Tuple[MidiDrumKey, ...]:
        """Returns the keys with the given decay time

        :param decay_time: DecayTime enum object (SHORT, NORMAL or LONG)
        :return: a tuple containing the MidiDrumKey objects with the given decay time or an empty tuple if nothing
                 found
        """

        return tuple(key for key in self.get_keys() if key.decay_time == decay_time)

    def __iter__(self) -> tp.Iterable[MidiDrumKey]:
        """Returns an iterator over the MidiDrumKey objects within this mapping

        :return: iterator yielding MidiDrumKey objects
        """

        return iter(self.get_keys())


class MidiDrumMappingImpl(MidiDrumMapping):
    """Midi drum mapping implementation

    This class is an implementation of the MidiDrumMapping interface. It adds mapping state and implements all retrieval
    functionality (get_key_by_midi_pitch, get_key_by_id, get_keys_with_frequency_band, get_keys_with_decay_time) with an
    execution time of O(1).
    """

    def __init__(self, name: str, keys: tp.Sequence[MidiDrumMapping.MidiDrumKey]):
        self._name = str(name)
        keys = tuple(keys)

        keys_by_midi_key = {}
        keys_by_id = {}
        keys_by_frequency_band = defaultdict(lambda: [])
        keys_by_decay_time = defaultdict(lambda: [])

        for k in keys:
            assert k.midi_pitch not in keys_by_midi_key, "multiple keys on midi key %i" % k.midi_pitch
            assert k.id not in keys_by_id, "multiple keys with id \"%s\"" % k.id

            keys_by_midi_key[k.midi_pitch] = k
            keys_by_id[k.id] = k
            keys_by_frequency_band[k.frequency_band].append(k)
            keys_by_decay_time[k.decay_time].append(k)

        self._keys = keys
        self._keys_by_midi_key = keys_by_midi_key
        self._keys_by_id = keys_by_id

        # copies a dict and converts the values to tuples
        solidify = lambda d: dict((item[0], tuple(item[1])) for item in d.items())

        self._keys_by_frequency_band = solidify(keys_by_frequency_band)
        self._keys_by_decay_time = solidify(keys_by_decay_time)

    # implements MidiDrumMapping.get_name
    def get_name(self):
        return self._name

    # implements MidiDrumMapping.get_key_by_midi_pitch with an execution time of O(1)
    def get_key_by_midi_pitch(self, midi_pitch: int) -> tp.Union[MidiDrumMapping.MidiDrumKey, None]:
        return self._keys_by_midi_key.get(midi_pitch, None)

    # implements MidiDrumMapping.get_key_by_id with an execution time of O(1)
    def get_key_by_id(self, key_id: str) -> tp.Union[MidiDrumMapping.MidiDrumKey, None]:
        return self._keys_by_id.get(key_id, None)

    # implements MidiDrumMapping.get_keys_with_frequency_band with an execution time of O(1)
    def get_keys_with_frequency_band(self, frequency_band: FrequencyBand) -> tp.Tuple[MidiDrumMapping.MidiDrumKey, ...]:
        return self._keys_by_frequency_band.get(frequency_band, tuple())

    # implements MidiDrumMapping.get_keys_with_decay_time with an execution time of O(1)
    def get_keys_with_decay_time(self, decay_time: DecayTime) -> tp.Tuple[MidiDrumMapping.MidiDrumKey, ...]:
        return self._keys_by_decay_time.get(decay_time, tuple())

    # implements MidiDrumMapping.get_keys with an execution time of O(1)
    def get_keys(self) -> tp.Sequence[MidiDrumMapping.MidiDrumKey]:
        return self._keys


class MidiDrumMappingGroup(MidiDrumMapping):
    def __init__(self, name: str, parent: MidiDrumMapping, midi_key_indices: tp.Sequence[int]):
        """Creates a new midi drum mapping group

        :param name: name of the midi drum mapping group
        :param parent: midi drum mapping containing the midi drum keys that this group is a selection of
        :param midi_key_indices: indices of the midi drum keys returned by parent.get_keys()
        """

        self._name = str(name)
        self._parent = parent
        self._key_view = TupleView(parent.get_keys(), midi_key_indices)

    def get_name(self) -> str:
        return self._name

    def get_keys(self) -> tp.Sequence[MidiDrumMapping.MidiDrumKey]:
        return self._key_view


class MidiDrumMappingReducer(object, metaclass=ABCMeta):
    def __init__(self, mapping: MidiDrumMapping):
        group_indices = defaultdict(lambda: [])

        for ix, key in enumerate(mapping):
            group_name = self.get_group_name(key)
            group_indices[group_name].append(ix)

        self._groups = dict((name, MidiDrumMappingGroup(
            name, mapping, indices)) for name, indices in group_indices.items())

    @staticmethod
    @abstractmethod
    def get_group_name(midi_key: MidiDrumMapping.MidiDrumKey) -> str:
        """Returns the name of the group, given the midi key

        :param midi_key: midi drum key
        :return: name of the group which the midi drum key belongs to
        """

        raise NotImplementedError

    def get_group(self, name: str) -> MidiDrumMappingGroup:
        """Returns the midi drum group with the given name

        :param name: name of the drum group
        :return: MidiDrumMappingGroup with the given name or None if no group found
        """

        return self._groups.get(name, None)

    def group_names(self):
        """Returns an iterator over the names of the groups within this reducer

        :return: iterator yielding the names of the groups in this reducer
        """

        return iter(self._groups.keys())


@friendly_named_class("Frequency-band mapping reducer")
class FrequencyBandMidiDrumMappingReducer(MidiDrumMappingReducer):
    @staticmethod
    def get_group_name(midi_key: MidiDrumMapping.MidiDrumKey) -> str:
        return midi_key.frequency_band.name


@friendly_named_class("Decay-time mapping reducer")
class DecayTimeMidiDrumMappingReducer(MidiDrumMappingReducer):
    @staticmethod
    def get_group_name(midi_key: MidiDrumMapping.MidiDrumKey) -> str:
        return midi_key.decay_time.name


@friendly_named_class("Unique-property combination reducer")
class UniquePropertyComboMidiDrumMappingReducer(MidiDrumMappingReducer):
    @staticmethod
    def get_group_name(midi_key: MidiDrumMapping.MidiDrumKey) -> str:
        return "%s.%s" % (midi_key.frequency_band.name, midi_key.decay_time.name)


def get_drum_mapping_reducer_implementation_names() -> tp.Tuple[str, ...]:
    """Returns a tuple containing the class names of all MidiDrumMappingReducer implementations"""
    return tuple(reducer.__name__ for reducer in MidiDrumMappingReducer.__subclasses__())


def get_drum_mapping_reducer_implementation_friendly_names() -> tp.Tuple[str, ...]:
    """Returns a tuple containing the friendly names of all MidiDrumMappingReducer implementations"""
    return tuple(getattr(reducer, "__friendly_name__") for reducer in MidiDrumMappingReducer.__subclasses__())


def get_drum_mapping_reducer_implementation(reducer_name: str, **kwargs) -> tp.Type[MidiDrumMappingReducer]:
    """Returns an implementation of MidiDrumMappingReducer

    Finds and returns a MidiDrumMappingReducer subclass, given its class name or friendly name. This method has an
    execution time of O(N).

    :param reducer_name: either the class name or the friendly name (if it is @friendly_named_class annotated) of
                         the reducer subclass
    :param kwargs:
        default - when given, this will be returned if nothing is found

    :return: subclass of MidiDrumMappingReducer with the given class name or friendly name
    :raises ValueError: if no subclass with the given name or friendly name (and no default is set)
    """

    for reducer in MidiDrumMappingReducer.__subclasses__():
        if reducer.__name__ == reducer_name:
            return reducer
        try:
            # noinspection PyUnresolvedReferences
            if reducer.__friendly_name__ == reducer_name:
                return reducer
        except AttributeError:
            continue

    try:
        return kwargs['default']
    except KeyError:
        raise ValueError("No MidiDrumMappingReducer found with class name or friendly name \"%s\"" % reducer_name)


GMDrumMapping = MidiDrumMappingImpl("GMDrumMapping", [
    MidiDrumMapping.MidiDrumKey(35, FrequencyBand.LOW, DecayTime.NORMAL, "Acoustic bass drum", key_id="abd"),
    MidiDrumMapping.MidiDrumKey(36, FrequencyBand.LOW, DecayTime.NORMAL, "Bass drum", key_id="bd1"),
    MidiDrumMapping.MidiDrumKey(37, FrequencyBand.MID, DecayTime.SHORT, "Side stick", key_id="sst"),
    MidiDrumMapping.MidiDrumKey(38, FrequencyBand.MID, DecayTime.NORMAL, "Acoustic snare", key_id="asn"),
    MidiDrumMapping.MidiDrumKey(39, FrequencyBand.MID, DecayTime.NORMAL, "Hand clap", key_id="hcl"),
    MidiDrumMapping.MidiDrumKey(40, FrequencyBand.MID, DecayTime.NORMAL, "Electric snare", key_id="esn"),
    MidiDrumMapping.MidiDrumKey(41, FrequencyBand.LOW, DecayTime.NORMAL, "Low floor tom", key_id="lft"),
    MidiDrumMapping.MidiDrumKey(42, FrequencyBand.HIGH, DecayTime.SHORT, "Closed hi-hat", key_id="chh"),
    MidiDrumMapping.MidiDrumKey(43, FrequencyBand.LOW, DecayTime.NORMAL, "High floor tom", key_id="hft"),
    MidiDrumMapping.MidiDrumKey(44, FrequencyBand.HIGH, DecayTime.NORMAL, "Pedal hi-hat", key_id="phh"),
    MidiDrumMapping.MidiDrumKey(45, FrequencyBand.MID, DecayTime.NORMAL, "Low tom", key_id="ltm"),
    MidiDrumMapping.MidiDrumKey(46, FrequencyBand.HIGH, DecayTime.LONG, "Open hi-hat", key_id="ohh"),
    MidiDrumMapping.MidiDrumKey(47, FrequencyBand.MID, DecayTime.NORMAL, "Low mid tom", key_id="lmt"),
    MidiDrumMapping.MidiDrumKey(48, FrequencyBand.MID, DecayTime.NORMAL, "High mid tom", key_id="hmt"),
    MidiDrumMapping.MidiDrumKey(49, FrequencyBand.HIGH, DecayTime.LONG, "Crash cymbal 1", key_id="cr1"),
    MidiDrumMapping.MidiDrumKey(50, FrequencyBand.MID, DecayTime.NORMAL, "High tom", key_id="htm"),
    MidiDrumMapping.MidiDrumKey(51, FrequencyBand.HIGH, DecayTime.LONG, "Ride cymbal 1", key_id="rc1"),
    MidiDrumMapping.MidiDrumKey(52, FrequencyBand.HIGH, DecayTime.LONG, "Chinese cymbal", key_id="chc"),
    MidiDrumMapping.MidiDrumKey(53, FrequencyBand.HIGH, DecayTime.LONG, "Ride bell", key_id="rbl"),
    MidiDrumMapping.MidiDrumKey(54, FrequencyBand.MID, DecayTime.NORMAL, "Tambourine", key_id="tmb"),
    MidiDrumMapping.MidiDrumKey(55, FrequencyBand.HIGH, DecayTime.LONG, "Splash cymbal", key_id="spl"),
    MidiDrumMapping.MidiDrumKey(56, FrequencyBand.MID, DecayTime.SHORT, "Cowbell", key_id="cwb"),
    MidiDrumMapping.MidiDrumKey(57, FrequencyBand.HIGH, DecayTime.LONG, "Crash cymbal 2", key_id="cr2"),
    MidiDrumMapping.MidiDrumKey(58, FrequencyBand.HIGH, DecayTime.LONG, "Vibraslap", key_id="vbs"),
    MidiDrumMapping.MidiDrumKey(59, FrequencyBand.HIGH, DecayTime.LONG, "Ride cymbal 2", key_id="rc2"),
    MidiDrumMapping.MidiDrumKey(60, FrequencyBand.MID, DecayTime.NORMAL, "Hi bongo", key_id="hbg"),
    MidiDrumMapping.MidiDrumKey(61, FrequencyBand.MID, DecayTime.NORMAL, "Low bongo", key_id="lbg"),
    MidiDrumMapping.MidiDrumKey(62, FrequencyBand.MID, DecayTime.NORMAL, "Muted high conga", key_id="mhc"),
    MidiDrumMapping.MidiDrumKey(63, FrequencyBand.MID, DecayTime.NORMAL, "Open high conga", key_id="ohc"),
    MidiDrumMapping.MidiDrumKey(64, FrequencyBand.MID, DecayTime.NORMAL, "Low conga", key_id="lcn"),
    MidiDrumMapping.MidiDrumKey(65, FrequencyBand.MID, DecayTime.NORMAL, "High timbale", key_id="htb"),
    MidiDrumMapping.MidiDrumKey(66, FrequencyBand.MID, DecayTime.NORMAL, "Low timbale", key_id="ltb"),
    MidiDrumMapping.MidiDrumKey(67, FrequencyBand.MID, DecayTime.NORMAL, "High agogo", key_id="hgo"),
    MidiDrumMapping.MidiDrumKey(68, FrequencyBand.MID, DecayTime.NORMAL, "Low agogo", key_id="lgo"),
    MidiDrumMapping.MidiDrumKey(69, FrequencyBand.HIGH, DecayTime.NORMAL, "Cabasa", key_id="cbs"),
    MidiDrumMapping.MidiDrumKey(70, FrequencyBand.HIGH, DecayTime.NORMAL, "Maracas", key_id="mcs"),
    MidiDrumMapping.MidiDrumKey(71, FrequencyBand.MID, DecayTime.NORMAL, "Short whistle", key_id="swh"),
    MidiDrumMapping.MidiDrumKey(72, FrequencyBand.MID, DecayTime.NORMAL, "Long whistle", key_id="lwh"),
    MidiDrumMapping.MidiDrumKey(73, FrequencyBand.MID, DecayTime.NORMAL, "Short guiro", key_id="sgr"),
    MidiDrumMapping.MidiDrumKey(74, FrequencyBand.MID, DecayTime.NORMAL, "Long guiro", key_id="lgr"),
    MidiDrumMapping.MidiDrumKey(75, FrequencyBand.MID, DecayTime.SHORT, "Claves", key_id="clv"),
    MidiDrumMapping.MidiDrumKey(76, FrequencyBand.MID, DecayTime.SHORT, "Hi wood block", key_id="hwb"),
    MidiDrumMapping.MidiDrumKey(77, FrequencyBand.MID, DecayTime.SHORT, "Low wood block", key_id="lwb"),
    MidiDrumMapping.MidiDrumKey(78, FrequencyBand.MID, DecayTime.NORMAL, "Muted cuica", key_id="mcu"),
    MidiDrumMapping.MidiDrumKey(79, FrequencyBand.MID, DecayTime.NORMAL, "Open cuica", key_id="ocu"),
    MidiDrumMapping.MidiDrumKey(80, FrequencyBand.MID, DecayTime.SHORT, "Muted triangle", key_id="mtr"),
    MidiDrumMapping.MidiDrumKey(81, FrequencyBand.MID, DecayTime.LONG, "Open triangle", key_id="otr")
])  # type: MidiDrumMapping


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

        measure_duration = int(self.get_measure_duration(None))
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
                 midi_mapping: MidiDrumMapping = GMDrumMapping,
                 midi_mapping_reducer_cls: tp.Optional[tp.Type[MidiDrumMappingReducer]] = None,
                 name: str = "", preserve_midi_duration: bool = False, **kwargs):
        super().__init__(**kwargs)
        # TODO resolution and other post_init props are not set.... call post_init from here?

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

        if midi_mapping_reducer_cls:
            mapping_reducer = midi_mapping_reducer_cls(midi_mapping)
        else:
            mapping_reducer = None

        self.set_name(name)
        self._midi_mapping = midi_mapping             # type: MidiDrumMapping
        self._midi_mapping_reducer = mapping_reducer  # type: tp.Union[MidiDrumMappingReducer, None]
        self._midi_metronome = -1                     # type: int
        self._prototype_midi_pitches = dict()         # type: tp.Dict[str, int]

        # loads the tracks and sets the bpm, time signature, midi metronome and resolution
        if midi_pattern:
            self.load_midi_pattern(midi_pattern, preserve_midi_duration)

    @property
    def midi_mapping(self):
        """The midi mapping.

        The MIDI mapping is used when parsing the MIDI data to create the track names. This is a read-only property.
        """
        return self._midi_mapping

    @property
    def midi_mapping_reducer(self):
        """The mapping reducer class.

        The MIDI drum mapping reducer class is the class of the mapping reducer used to parse the MIDI data and create
        the tracks of this rhythm. This is a read-only property.
        """

        mapping_reducer = self._midi_mapping_reducer
        if not mapping_reducer:
            return None
        return mapping_reducer.__class__

    def as_midi_pattern(self, note_length: int = 0,
                        midi_channel: int = 9, midi_format: int = 0,
                        midi_keys: tp.Optional[tp.Dict[str, int]] = None) -> midi.Pattern:
        """
        Converts this rhythm to a MIDI pattern.

        :param note_length: note duration in ticks
        :param midi_channel: NoteOn/NoteOff events channel (defaults to 9, which is the default for drum sounds)
        :param midi_format: midi format
        :param midi_keys: optional, dictionary holding the MIDI keys per track name
        :return: MIDI pattern
        """

        midi_track = midi.Track(tick_relative=False)  # create track and add metadata events

        midi_track.append(midi.TrackNameEvent(text=self.get_name()))  # track name
        midi_metronome = 24 if self._midi_metronome is None else self._midi_metronome
        midi_track.append(self.get_time_signature().to_midi_event(midi_metronome))  # time signature

        if self.bpm:
            midi_track.append(midi.SetTempoEvent(bpm=self.bpm))  # tempo

        midi_keys = midi_keys or self._prototype_midi_pitches

        # add note events
        for track in self.get_track_iterator():
            pitch = midi_keys[track.name]
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
        Checks if the given track name is a valid MidiDrumKey id according to this loop's midi mapping.

        :param track_name: track name to check
        :return: a message telling that the track name is not a valid midi key id or an empty string if the track name
                 is ok
        """

        mapping_reducer = self._midi_mapping_reducer

        if mapping_reducer:
            group_names = tuple(mapping_reducer.group_names())
            if track_name in group_names:
                return ""
            return "No group called \"%s\"" % track_name

        mapping = self._midi_mapping
        if mapping.get_key_by_id(track_name):
            return ""
        return "No midi key found with id \"%s\" in %s" % (track_name, mapping.get_name())

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

        if self._midi_mapping_reducer:
            get_track_name = self._midi_mapping_reducer.get_group_name
        else:
            get_track_name = lambda m_key: m_key.id

        pattern.make_ticks_abs()
        track = list(itertools.chain(*pattern))  # merge all tracks into one
        track = midi.Track(sorted(track, key=lambda event: event.tick))  # sort in chronological order

        bpm = 0                                      # type: tp.Union[float]
        midi_note_events = defaultdict(lambda: [])   # type: tp.Dict[str, tp.List[midi.NoteEvent]]
        ts_midi_event = None                         # type: tp.Union[midi.TimeSignatureEvent, None]
        eot_event = None                             # type: tp.Union[midi.EndOfTrackEvent, None]

        for msg in track:
            if isinstance(msg, midi.NoteOnEvent):
                midi_pitch = msg.get_pitch()  # type: int
                mapping_key = mapping.get_key_by_midi_pitch(midi_pitch)
                if mapping_key is None:
                    print("Unknown midi key: %i (Mapping = %s)" % (midi_pitch, mapping.get_name()))
                    continue
                track_name = get_track_name(mapping_key)
                midi_note_events[track_name].append(msg)
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
        track_data = OrderedDict()  # type: tp.Dict[str, tp.Tuple[Track, int]]

        for t_name, events in sorted(midi_note_events.items()):
            most_common_midi_pitch = most_common_element(tuple(e.get_pitch() for e in events))
            onsets = ((int(e.tick), int(e.get_velocity())) for e in events)
            track_data[t_name] = Track(onsets, t_name), most_common_midi_pitch

        self._midi_metronome = midi_metronome
        self.set_time_signature(time_signature)
        self.set_bpm(bpm)
        self.set_tracks((entry[0] for entry in track_data.values()), pattern.resolution)
        self._prototype_midi_pitches = dict(tuple((t_name, entry[1]) for t_name, entry in track_data.items()))

        if preserve_midi_duration:
            self.set_duration_in_ticks(eot_event.tick)


def create_rumba_rhythm(resolution=240, polyphonic=True):
    """
    Utility function that creates a one-bar rumba rhythm.

    :param resolution: rhythm resolution
    :param polyphonic: when true, a polyphonic rhythm will be returned (kick, snare and hi-hat), when false only the
                       snare is returned (which plays the rumba clave pattern)
    :return: monophonic/polyphonic rhythm object
    """

    if polyphonic:
        rhythm = PolyphonicRhythm.create.from_string(textwrap.dedent("""
            kick:   ---x--x----x--x-
            snare:  --x-x---x--x---x
            hi-hat: x-x-xxxx-xxxx-xx
        """), TimeSignature(4, 4))
    else:
        rhythm = MonophonicRhythm.create.from_string("--x-x---x--x---x", TimeSignature(4, 4))

    rhythm.set_resolution(resolution)
    return rhythm


__all__ = [
    # Rhythm classes
    'Rhythm', 'MonophonicRhythm', 'PolyphonicRhythm',
    'RhythmLoop', 'MidiRhythm',

    # Time unit
    'Unit', 'UnitType', 'UnitError', 'parse_unit_argument', 'rescale_tick', 'convert_tick',

    # Misc
    'Onset', 'Track', 'TimeSignature', 'GMDrumMapping', 'create_rumba_rhythm',

    # MIDI drum mapping
    'MidiDrumMapping', 'GMDrumMapping', 'FrequencyBand', 'DecayTime',
    'MidiDrumMappingReducer', 'FrequencyBandMidiDrumMappingReducer',
    'DecayTimeMidiDrumMappingReducer', 'UniquePropertyComboMidiDrumMappingReducer',
    'get_drum_mapping_reducer_implementation_names',
    'get_drum_mapping_reducer_implementation_friendly_names',
    'get_drum_mapping_reducer_implementation'
]

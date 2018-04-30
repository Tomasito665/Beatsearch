# coding=utf-8
import os
import math
import enum
import uuid
import pickle
import logging
import inspect
import textwrap
import itertools
import numpy as np
import typing as tp
from io import IOBase
from fractions import Fraction
from abc import abstractmethod, ABCMeta
from functools import wraps, total_ordering
from collections import OrderedDict, namedtuple, defaultdict
from beatsearch.utils import (
    TupleView,
    friendly_named_class,
    most_common_element,
    sequence_product,
    FileInfo,
    get_midi_files_in_directory,
    make_dir_if_not_exist
)
import midi  # after beatsearch import


LOGGER = logging.getLogger(__name__)


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
        numerator = int(numerator)
        denominator = int(denominator)

        if numerator < 1:
            raise ValueError("Expected numerator equal or greater than 1 but got %i" % numerator)

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

        n_beat_divisions = math.log2(n_units_per_beat)
        assert math.isclose(n_beat_divisions, int(n_beat_divisions)), \
            "expected number of steps in a beat to be an exact base-2 logarithm of %i" % n_units_per_beat
        n_beat_divisions = int(n_beat_divisions)

        divisions.extend(itertools.repeat(2, n_beat_divisions))
        return tuple(divisions)

    @parse_unit_argument
    def get_salience_profile(self, unit: UnitType = Unit.EIGHTH, kind: str = "hierarchical", root_weight: int = 0):
        """Returns the metrical weights for a full measure of this time signature

        Returns the metrical weights for a full measure of this time signature with a step size of the given unit. When
        equal_upbeats is false, the salience profile will be based on a fully hierarchical metrical structure of the
        meter. Otherwise, if equal_upbeats is true, all beat weights will be equal except for the downbeat, which will
        have a greater weight.

        This method constructs a hierarchical meter tree structure (see get_meter_tree) and assigns a weight to each
        node. Then it flattens the tree and returns it as a list. The weight of a node is the weight of its parent node
        minus one. The weight of the root node is specified with the root_weight argument of this method. This way of
        computing the salience profile corresponds to the method proposed by H.C. Longuet-Higgins & C.S. Lee in their
        work titled "The Rhythmic Interpretation of Monophonic Music".

        This method can create three kinds of salience profiles, depending on the given "kind" parameter.

            'hierarchical'   A fully hierarchical salience profile.

            'equal_upbeats'  A salience profile in which every beat is equally weighted except for the downbeat, which
                             is heavier. The steps within the beats are fully hierarchical. This salience profile is
                             used by Maria A.G. Witek et al in their work titled "Syncopation, Body-Movement and
                             Pleasure in Groove Music".

            'equal_beats'    A salience profile in which every beat (both upbeats and downbeats have the same weight) is
                             equally weighted. The steps within the beats are fully hierarchical.

        :param unit: step time unit
        :param kind: one of {'hierarchical', 'equal_upbeats', 'equal_beats'}, see main method description for more info
        :param root_weight: weight of first node in the
        :return: the metrical weights for a full measure of this time signature with the given time unit
        """

        try:
            f = {
                'hierarchical': self.__get_salience_profile_full_hierarchical,
                'equal_upbeats': self.__get_salience_profile_with_equal_upbeats,
                'equal_beats': self.__get_salience_profile_with_equal_beats
            }[kind]
        except KeyError:
            raise ValueError("unknown kind: '%s', should be one of either "
                             "hierarchical, equal_upbeats or equal_beats" % kind)

        return f(unit, root_weight)

    @parse_unit_argument
    def get_natural_duration_map(self, unit: UnitType, trim_to_pulse: bool = True) -> tp.List[int]:
        """Returns the maximum note durations on each metrical position as multiples of the given unit

        Returns a list containing the maximum note duration initiated at each metrical position. The returned durations
        are expressed as multiples as the given unit.

        :param unit: step size as a musical unit
        :param trim_to_pulse: when true, the durations won't exceed the duration of one pulse
        :return: the maximum note durations on each metrical position as a list
        """

        if trim_to_pulse:
            pulse_duration = self.get_beat_unit().convert(1, unit, True)
            get_value = lambda depth, n_siblings, n_nodes, length, n_levels: min(length // n_nodes, pulse_duration)
        else:
            get_value = lambda depth, n_siblings, n_nodes, length, n_levels: length // n_nodes

        return self.construct_flat_meter_tree(unit, get_value)

    @parse_unit_argument
    def construct_flat_meter_tree(self, unit: UnitType, get_value: tp.Callable[[int, int, int, int, int], tp.Any]):
        """Utility function to create a one dimensional representation of a meter tree structure

        Creates a hierarchical meter tree structure of this time signature with the given step size and returns a one
        dimensional vector representation of it. The values of the returned vector are obtained with the given get_value
        callable. This callable will receive the following positional parameters:

                - depth: the number of levels away from the root node
                - n_siblings: the sibling count per node (including that node) on this depth
                - n_nodes: number of nodes on this depth
                - length: the length of the one dimensional vector ("width" of the tree) (constant)
                - n_levels: the number of levels in the tree ("height" of the tree) (constant)

        :param unit: step size as a musical unit
        :param get_value: this callable will be used to populate the returned vector, receiving these positional
                          parameters: (depth, n_siblings, n_nodes, length, n_levels)

        :return: one dimensional vector representation of a the hierarchical meter of this time signature as a list
        """

        # given get_value function receives: branch_ix, subdivision, n_branches, n_steps
        assert unit <= self.get_beat_unit(), "can't represent this time signature in %s" % str(unit)

        n_steps = self.get_beat_unit().convert(self.numerator, unit, True)
        subdivisions = self.get_meter_tree(unit)

        if not subdivisions:
            assert self.get_beat_unit().convert(self.numerator, unit, True) == 1
            return [get_value(0, 1, 1, 1, 0)]

        assert sequence_product(subdivisions) == n_steps, \
            "if the product of %s is not %i, something is broken :(" % (str(subdivisions), n_steps)

        n_levels = len(subdivisions)
        meter_map = [None] * n_steps
        n_branches = 1

        for n, curr_subdivision in enumerate(itertools.chain([1], subdivisions)):
            n_branches *= curr_subdivision
            value = get_value(n, curr_subdivision, n_branches, n_steps, n_levels)

            for ix in np.linspace(0, n_steps, n_branches, endpoint=False, dtype=int):
                if meter_map[ix] is None:
                    meter_map[ix] = value

        return meter_map

    def __get_salience_profile_full_hierarchical(self, unit: Unit, root_weight: int):
        return self.construct_flat_meter_tree(
            unit,
            lambda depth, *_: root_weight - depth
        )

    def __get_salience_profile_with_equal_beats(self, unit: Unit, root_weight: int):
        # get the fully hierarchical salience profile of one beat
        one_beat_ts = TimeSignature(1, self.denominator)
        one_beat_weights = one_beat_ts.__get_salience_profile_full_hierarchical(unit, root_weight=root_weight - 1)

        # repeat the one-beat salience profile to fill one measure
        return one_beat_weights * self.numerator

    def __get_salience_profile_with_equal_upbeats(self, unit: Unit, root_weight: int):
        salience_profile = self.__get_salience_profile_with_equal_beats(unit, root_weight - 1)
        # noinspection PyTypeChecker
        salience_profile[0] = root_weight
        return salience_profile

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

    class BadFormat(Exception):
        pass

    @classmethod
    @abstractmethod
    def from_string(
            cls,
            onset_string: str,
            time_signature: tp.Optional[tp.Union[tp.Tuple[int, int], TimeSignature]] = None,
            velocity: int = 100,
            unit: UnitType = Unit.SIXTEENTH,
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
            unit: UnitType = Unit.SIXTEENTH,
            **kwargs) -> Rhythm:
        """Creates and returns a rhythm, given a sequence representation of its onsets"""

        raise NotImplementedError

    @staticmethod
    def __string_to_binary_onset_vector__(onset_string, onset_char) -> tp.Tuple[bool, ...]:
        return tuple((char == onset_char) for char in onset_string)

    @staticmethod
    def __binary_vector_to_onsets__(binary_vector: tp.Sequence[bool], velocity: int) -> tp.Tuple[tp.Tuple[int, int]]:
        return tuple(filter(None, ((ix, velocity) if atom else None for ix, atom in enumerate(binary_vector))))

    @staticmethod
    @parse_unit_argument
    def __check_and_return_resolution__(unit: UnitType):
        resolution = Unit.QUARTER.convert(1, unit, True)
        if resolution <= 0:
            raise ValueError("Unit must be equal or smaller than %s" % str(Unit.QUARTER))
        return resolution


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
        Tries to set the duration of this rhythm to the requested duration and returns the actual new duration. If the
        position of this rhythm's last note is X, the duration of the rhythm can't be less than X + 1. If the requested
        duration is less than X + 1, the duration will be set to X + 1.

        :param requested_duration: new duration in ticks
        :return: the new duration
        """

        last_onset_position = self.get_last_onset_tick()
        self._duration_in_ticks = max(last_onset_position + 1, int(requested_duration))
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
                unit: UnitType = Unit.SIXTEENTH,
                onset_character="x",
                **kwargs):  # type: () -> MonophonicRhythmImpl
            """
            Creates a new monophonic rhythm from a string. Each character in the string will represent one tick and each
            onset character will represent an onset in the rhythm, e.g. "x--x---x--x-x---", given onset character "x".
            The length of the onset string will determine the duration of the rhythm.

            :param onset_string:   onset string where each onset character will result in an onset
            :param time_signature: time signature of the rhythm as a (numerator, denominator) tuple or TimeSignature obj
            :param velocity:       the velocity of the onsets as an integer, which will be the same for all onsets
            :param unit:           step size as a musical unit (e.g., if unit is set to Unit.EIGHTH (or 1/8 or "eighth")
                                   one character will represent one eighth note)
            :param onset_character: onset character (see onset_string)
            :param kwargs:         unused

            :return: monophonic rhythm object
            """

            return cls.from_binary_vector(
                binary_vector=cls.__string_to_binary_onset_vector__(onset_string, onset_character),
                time_signature=time_signature,
                velocity=velocity,
                unit=unit
            )

        @classmethod
        def from_binary_vector(
                cls,
                binary_vector: tp.Sequence[tp.Any],
                time_signature: tp.Optional[tp.Union[tp.Tuple[int, int], TimeSignature]] = None,
                velocity: int = 100,
                unit: UnitType = Unit.SIXTEENTH,
                **kwargs):  # type: () -> MonophonicRhythmImpl
            """
            Creates a new monophonic rhythm, given a binary chain (iterable). Each element in the iterable represents
            one tick. If the element is True, that will result in an onset, e.g. [1, 0, 1, 0, 1, 1, 1, 0].

            :param binary_vector:  sequence where each true-evaluated element will result in an onset
            :param time_signature: time signature of the rhythm as a (numerator, denominator) tuple or TimeSignature obj
            :param velocity:       the velocity of the onsets as an integer, which will be the same for all onsets
            :param unit:           step size as a musical unit (e.g., if unit is set to Unit.EIGHTH (or 1/8 or "eighth")
                                   one element in the binary vector will represent one eighth note)
            :param kwargs:         unused

            :return: monophonic rhythm object
            """

            resolution = cls.__check_and_return_resolution__(unit)
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
        PolyphonicRhythmImpl.Track object. The order in which the tracks of this rhythm are yielded is
        always the same for each iterator returned by this method.

        :return: iterator over this rhythm's tracks yielding PolyphonicRhythmImpl.Track objects
        """

        raise NotImplementedError

    def __getitem__(self, track_name: str):
        """
        Returns the track with the given name. Raises a ValueError if no track with given name.

        :param track_name: name of the track
        :return: track with given name
        """

        track = self.get_track_by_name(track_name)
        if track is None:
            raise ValueError
        return track

    @abstractmethod
    def get_track_by_name(self, track_name: str):
        """
        Returns the track with the given name or None if this rhythm has no track with the given name.

        :param track_name: track name
        :return: Track object or None
        """

        raise NotImplementedError

    @abstractmethod
    def get_track_names(self) -> tp.Tuple[str, ...]:
        """
        Returns a tuple containing the names of the tracks in this rhythm. The order in which the names are returned is
        the same as the order in which the tracks are yielded by the track iterator returned by
        :meth:`beatsearch.rhythm.PolyphonicRhythm.get_track_iterator`.

        :return: tuple containing the track names
        """

        raise NotImplementedError

    @abstractmethod
    def get_track_count(self):
        """
        Returns the number of tracks within this rhythm.

        :return: number of tracks
        """

        raise NotImplementedError

    @abstractmethod
    def clear_tracks(self):
        """
        Clears all tracks.

        :return: None
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
                unit: UnitType = Unit.SIXTEENTH,
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
            :param unit:                 step size as a musical unit (e.g., if unit is set to Unit.EIGHTH (or 1/8 or
                                         "eighth") one element in the binary vector will represent one eighth note)
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
                unit=unit,
                track_names=track_names
            )

        @classmethod
        def from_binary_vector(
                cls,
                binary_vector_tracks: tp.Sequence[tp.Sequence[tp.Any]],
                time_signature: tp.Optional[tp.Union[tp.Tuple[int, int], TimeSignature]] = None,
                velocity: int = 100,
                unit: UnitType = Unit.SIXTEENTH,
                *_, track_names: tp.Sequence[str] = None,
                **kwargs):  # type: () -> PolyphonicRhythmImpl
            """
            Creates a new polyphonic rhythm from a sequence containing one binary onset vector per track. Track names
            are optional and are given to the track_names parameter.

            :param binary_vector_tracks: sequence holding one binary onset vector per track
            :param time_signature:       time signature of the rhythm as a (num, den) tuple or TimeSignature object
            :param velocity:             the velocity of the onsets as an integer, which will be the same for all onsets
            :param unit:                 step size as a musical unit (e.g., if unit is set to Unit.EIGHTH (or 1/8 or
                                         "eighth") one character will represent one eighth note)
            :param track_names:          names of the tracks
            :param kwargs:               unused
            :return: polyphonic rhythm object
            """

            resolution = cls.__check_and_return_resolution__(unit)
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
        Returns an iterator over the tracks of this polyphonic rhythm. Each iteration yields a
        PolyphonicRhythmImpl.Track object. The order in which the tracks of this rhythm are yielded is
        always the same for each iterator returned by this method.

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

    def get_track_names(self) -> tp.Tuple[str, ...]:
        """
        Returns a tuple containing the names of the tracks in this rhythm. The order in which the names are returned is
        the same as the order in which the tracks are yielded by the track iterator returned by
        :meth:`beatsearch.rhythm.PolyphonicRhythm.get_track_iterator`.

        :return: tuple containing the track names
        """

        return tuple(self._tracks.keys())

    def get_track_count(self) -> int:
        """
        Returns the number of tracks within this rhythm.

        :return: number of tracks
        """

        return len(self._tracks)

    def clear_tracks(self) -> None:
        """
        Clears all tracks.

        :return: None
        """

        self._tracks.clear()

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
    LOW = 0
    MID = 1
    HIGH = 2


class DecayTime(enum.Enum):
    """Enumeration containing three drum sound decay times (short, normal and long)"""
    SHORT = 0
    NORMAL = 1
    LONG = 2


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


class MidiDrumMapping(object, metaclass=ABCMeta):
    """Midi drum mapping interface

    Each MidiDrumMapping object represents a MIDI drum mapping and is a container for MidiDrumKey objects. It provides
    functionality for retrieval of these objects, based on either midi pitch, frequency band or key id.
    """

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

    def __getitem__(self, item: tp.Union[int, str]) -> MidiDrumKey:
        """Returns the midi drum key with the given midi pitch or id

        :param item: either the midi or the key id
        :return: midi drum key
        :raises KeyError: if this mapping contains no midi drum key with given id or pitch
        """

        if isinstance(item, int):
            midi_drum_key = self.get_key_by_midi_pitch(item)
        elif isinstance(item, str):
            midi_drum_key = self.get_key_by_id(item)
        else:
            raise TypeError("expected either an int (midi pitch) or a str (key id) but got %s" % item)

        if not midi_drum_key:
            raise KeyError("no midi drum key with id or midi pitch %s" % item)

        return midi_drum_key


class MidiDrumMappingImpl(MidiDrumMapping):
    """Midi drum mapping implementation

    This class is an implementation of the MidiDrumMapping interface. It adds mapping state and implements all retrieval
    functionality (get_key_by_midi_pitch, get_key_by_id, get_keys_with_frequency_band, get_keys_with_decay_time) with an
    execution time of O(1).
    """

    def __init__(self, name: str, keys: tp.Sequence[MidiDrumKey]):
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
    def get_key_by_midi_pitch(self, midi_pitch: int) -> tp.Union[MidiDrumKey, None]:
        return self._keys_by_midi_key.get(midi_pitch, None)

    # implements MidiDrumMapping.get_key_by_id with an execution time of O(1)
    def get_key_by_id(self, key_id: str) -> tp.Union[MidiDrumKey, None]:
        return self._keys_by_id.get(key_id, None)

    # implements MidiDrumMapping.get_keys_with_frequency_band with an execution time of O(1)
    def get_keys_with_frequency_band(self, frequency_band: FrequencyBand) -> tp.Tuple[MidiDrumKey, ...]:
        return self._keys_by_frequency_band.get(frequency_band, tuple())

    # implements MidiDrumMapping.get_keys_with_decay_time with an execution time of O(1)
    def get_keys_with_decay_time(self, decay_time: DecayTime) -> tp.Tuple[MidiDrumKey, ...]:
        return self._keys_by_decay_time.get(decay_time, tuple())

    # implements MidiDrumMapping.get_keys with an execution time of O(1)
    def get_keys(self) -> tp.Sequence[MidiDrumKey]:
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

    def get_keys(self) -> tp.Sequence[MidiDrumKey]:
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
    def get_group_name(midi_key: MidiDrumKey) -> str:
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
    def get_group_name(midi_key: MidiDrumKey) -> str:
        return midi_key.frequency_band.name


@friendly_named_class("Decay-time mapping reducer")
class DecayTimeMidiDrumMappingReducer(MidiDrumMappingReducer):
    @staticmethod
    def get_group_name(midi_key: MidiDrumKey) -> str:
        return midi_key.decay_time.name


@friendly_named_class("Unique-property combination reducer")
class UniquePropertyComboMidiDrumMappingReducer(MidiDrumMappingReducer):
    @staticmethod
    def get_group_name(midi_key: MidiDrumKey) -> str:
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


def create_drum_mapping(name: str, keys: tp.Sequence[MidiDrumKey]) -> MidiDrumMapping:
    """
    Utility function to create a new MIDI drum mapping.

    :param name: name of the drum mapping
    :param keys: drum mappings as a sequence of :class:`beatsearch.rhythm.MidiDrumMapping.MidiDrumKey` objects
    :return: midi drum mapping
    """

    return MidiDrumMappingImpl(name, keys)


GMDrumMapping = create_drum_mapping("GMDrumMapping", [
    MidiDrumKey(35, FrequencyBand.LOW, DecayTime.NORMAL, "Acoustic bass drum", key_id="abd"),
    MidiDrumKey(36, FrequencyBand.LOW, DecayTime.NORMAL, "Bass drum", key_id="bd1"),
    MidiDrumKey(37, FrequencyBand.MID, DecayTime.SHORT, "Side stick", key_id="sst"),
    MidiDrumKey(38, FrequencyBand.MID, DecayTime.NORMAL, "Acoustic snare", key_id="asn"),
    MidiDrumKey(39, FrequencyBand.MID, DecayTime.NORMAL, "Hand clap", key_id="hcl"),
    MidiDrumKey(40, FrequencyBand.MID, DecayTime.NORMAL, "Electric snare", key_id="esn"),
    MidiDrumKey(41, FrequencyBand.LOW, DecayTime.NORMAL, "Low floor tom", key_id="lft"),
    MidiDrumKey(42, FrequencyBand.HIGH, DecayTime.SHORT, "Closed hi-hat", key_id="chh"),
    MidiDrumKey(43, FrequencyBand.LOW, DecayTime.NORMAL, "High floor tom", key_id="hft"),
    MidiDrumKey(44, FrequencyBand.HIGH, DecayTime.NORMAL, "Pedal hi-hat", key_id="phh"),
    MidiDrumKey(45, FrequencyBand.MID, DecayTime.NORMAL, "Low tom", key_id="ltm"),
    MidiDrumKey(46, FrequencyBand.HIGH, DecayTime.LONG, "Open hi-hat", key_id="ohh"),
    MidiDrumKey(47, FrequencyBand.MID, DecayTime.NORMAL, "Low mid tom", key_id="lmt"),
    MidiDrumKey(48, FrequencyBand.MID, DecayTime.NORMAL, "High mid tom", key_id="hmt"),
    MidiDrumKey(49, FrequencyBand.HIGH, DecayTime.LONG, "Crash cymbal 1", key_id="cr1"),
    MidiDrumKey(50, FrequencyBand.MID, DecayTime.NORMAL, "High tom", key_id="htm"),
    MidiDrumKey(51, FrequencyBand.HIGH, DecayTime.LONG, "Ride cymbal 1", key_id="rc1"),
    MidiDrumKey(52, FrequencyBand.HIGH, DecayTime.LONG, "Chinese cymbal", key_id="chc"),
    MidiDrumKey(53, FrequencyBand.HIGH, DecayTime.LONG, "Ride bell", key_id="rbl"),
    MidiDrumKey(54, FrequencyBand.MID, DecayTime.NORMAL, "Tambourine", key_id="tmb"),
    MidiDrumKey(55, FrequencyBand.HIGH, DecayTime.LONG, "Splash cymbal", key_id="spl"),
    MidiDrumKey(56, FrequencyBand.MID, DecayTime.SHORT, "Cowbell", key_id="cwb"),
    MidiDrumKey(57, FrequencyBand.HIGH, DecayTime.LONG, "Crash cymbal 2", key_id="cr2"),
    MidiDrumKey(58, FrequencyBand.HIGH, DecayTime.LONG, "Vibraslap", key_id="vbs"),
    MidiDrumKey(59, FrequencyBand.HIGH, DecayTime.LONG, "Ride cymbal 2", key_id="rc2"),
    MidiDrumKey(60, FrequencyBand.MID, DecayTime.NORMAL, "Hi bongo", key_id="hbg"),
    MidiDrumKey(61, FrequencyBand.MID, DecayTime.NORMAL, "Low bongo", key_id="lbg"),
    MidiDrumKey(62, FrequencyBand.MID, DecayTime.NORMAL, "Muted high conga", key_id="mhc"),
    MidiDrumKey(63, FrequencyBand.MID, DecayTime.NORMAL, "Open high conga", key_id="ohc"),
    MidiDrumKey(64, FrequencyBand.MID, DecayTime.NORMAL, "Low conga", key_id="lcn"),
    MidiDrumKey(65, FrequencyBand.MID, DecayTime.NORMAL, "High timbale", key_id="htb"),
    MidiDrumKey(66, FrequencyBand.MID, DecayTime.NORMAL, "Low timbale", key_id="ltb"),
    MidiDrumKey(67, FrequencyBand.MID, DecayTime.NORMAL, "High agogo", key_id="hgo"),
    MidiDrumKey(68, FrequencyBand.MID, DecayTime.NORMAL, "Low agogo", key_id="lgo"),
    MidiDrumKey(69, FrequencyBand.HIGH, DecayTime.NORMAL, "Cabasa", key_id="cbs"),
    MidiDrumKey(70, FrequencyBand.HIGH, DecayTime.NORMAL, "Maracas", key_id="mcs"),
    MidiDrumKey(71, FrequencyBand.MID, DecayTime.NORMAL, "Short whistle", key_id="swh"),
    MidiDrumKey(72, FrequencyBand.MID, DecayTime.NORMAL, "Long whistle", key_id="lwh"),
    MidiDrumKey(73, FrequencyBand.MID, DecayTime.NORMAL, "Short guiro", key_id="sgr"),
    MidiDrumKey(74, FrequencyBand.MID, DecayTime.NORMAL, "Long guiro", key_id="lgr"),
    MidiDrumKey(75, FrequencyBand.MID, DecayTime.SHORT, "Claves", key_id="clv"),
    MidiDrumKey(76, FrequencyBand.MID, DecayTime.SHORT, "Hi wood block", key_id="hwb"),
    MidiDrumKey(77, FrequencyBand.MID, DecayTime.SHORT, "Low wood block", key_id="lwb"),
    MidiDrumKey(78, FrequencyBand.MID, DecayTime.NORMAL, "Muted cuica", key_id="mcu"),
    MidiDrumKey(79, FrequencyBand.MID, DecayTime.NORMAL, "Open cuica", key_id="ocu"),
    MidiDrumKey(80, FrequencyBand.MID, DecayTime.SHORT, "Muted triangle", key_id="mtr"),
    MidiDrumKey(81, FrequencyBand.MID, DecayTime.LONG, "Open triangle", key_id="otr")
])  # type: MidiDrumMapping


class RhythmLoop(PolyphonicRhythmImpl):
    """Rhythm loop with a name and duration always snapped to a downbeat"""

    def __init__(self, name: str = "", **kwargs):
        super().__init__(**kwargs)
        self._name = ""
        self.set_name(name)

    # noinspection PyPep8Naming
    class create(PolyphonicRhythm.create):  # not intended like a class but like a namespace for factory methods
        @staticmethod
        def __track_name_generator(n: int) -> tp.Generator[str, None, None]:
            for i in range(n):
                yield "track %i" % n

        @classmethod
        def from_polyphonic_rhythm(cls, rhythm: PolyphonicRhythm, title: str = "untitled"):
            """
            Creates a rhythm loop from a polyphonic rhythm. Note that, as rhythm loops must have a duration of a
            multiple of one measure, the duration of the loop might be different from the duration of the given rhythm,
            to adjust to the nearest downbeat.

            :param rhythm: polyphonic rhythm
            :param title: rhythm loop title
            :return: rhythm loop object
            """

            return RhythmLoop(
                title,
                tracks=rhythm.get_track_iterator(),
                resolution=rhythm.get_resolution(),
                bpm=rhythm.get_bpm(),
                time_signature=rhythm.get_time_signature()
            )

        @classmethod
        def from_string(
                cls,
                input_string: str,
                time_signature: tp.Optional[tp.Union[tp.Tuple[int, int], TimeSignature]] = (4, 4),
                velocity: int = 100,
                unit: UnitType = Unit.SIXTEENTH,
                onset_character: str = "x",
                *_,
                title_underline_character: str = "=",
                newline_character: str = "\n",
                track_name_separator_character: str = ":",
                **kwargs):  # type: () -> RhythmLoop
            """
            Creates a new rhythm loop from a multi-line string. The input string should contain the rhythm title and the
            binary onset vectors (one per track). The rhythm title must appear on the first line and must be underlined.
            For every track, a name and the binary onset vector must be given, separated by the track name separator
            character.

            Note that :class:`beatsearch.rhythm.RhythmLoop` must have a duration of an exact multiple of one measure.
            This method will raise a ValueError if the duration of the binary onset vectors does not meet that
            requirement (e.g., if given time signature 4/4 and a quaver step size, the length of the onset vector
            strings must be 8=one measure, 16=two measures, 24=three measures, etc).

            For example, given title underline character '=', onset character 'x' and track name separator
            character ':', to create a simple rhythm loop titled "Cha cha cha", we could do:

                PolyphonicRhythm.create.from_string(textwrap.dedent(\"""
                           Cha cha cha
                    ==========================
                    cowbell:  x-x-x-x-x-x-x-x-
                    stick:    --x--x----x--x--
                    tom:      ------xx------xx
                    kick:     ---x------------
                \"""))

            :param input_string:         The multi-line input string contains the loop title on the first line, the
                                         title underline on the second line and track information on the remaining
                                         lines. Each track is represented by a track string. Track strings are divided
                                         into two parts: the track name and the binary onset vector whose length
                                         determines the duration of the rhythm. Each onset character in the binary onset
                                         will result in an onset. Drum loops must have a duration of a multiple of one
                                         measure.
            :param time_signature:       time signature of the rhythm as a (numerator, denominator) tuple or a
                                         TimeSignature object, defaults to (4, 4)
            :param velocity:             the velocity of the onsets as an integer, which will be the same for all onsets
            :param unit:                 step size as a musical unit (e.g., if unit is set to Unit.EIGHTH (or 1/8 or
                                         "eighth") one element in the binary vector will represent one eighth note)
            :param onset_character:      onset character (see onset_string)
            :param title_underline_character: the title must be underlined with this character (defaults to '=')
            :param newline_character:    newline character (defaults to '\n')
            :param track_name_separator_character: the track string is split into the track name and binary onset vector
                                                   on this character (defaults to ':')
            :param kwargs:               unused

            :return: drum loop object
            """

            input_string = input_string.strip()
            input_string_lines = input_string.split(newline_character)

            if len(input_string_lines) < 2:
                raise cls.BadFormat()

            title = input_string_lines[0].strip()
            title_underline = input_string_lines[1].strip()

            if not all(c == title_underline_character for c in title_underline):
                raise cls.BadFormat()

            polyphonic_rhythm = PolyphonicRhythm.create.from_string(
                newline_character.join(input_string_lines[2:]),
                time_signature=time_signature,
                velocity=velocity,
                unit=unit,
                onset_character=onset_character,
                **kwargs
            )

            return cls.from_polyphonic_rhythm(polyphonic_rhythm, title)

        @classmethod
        def from_binary_vector(
                cls,
                binary_vector_tracks: tp.Sequence[tp.Sequence[tp.Any]],
                time_signature: tp.Optional[tp.Union[tp.Tuple[int, int], TimeSignature]] = None,
                velocity: int = 100,
                unit: UnitType = Unit.SIXTEENTH,
                *_, track_names: tp.Sequence[str] = None, title: str = "untitled",
                **kwargs):  # type: () -> PolyphonicRhythmImpl
            """
            Creates a new polyphonic rhythm from a sequence containing one binary onset vector per track. Track names
            are optional and are given to the track_names parameter.

            :param binary_vector_tracks: sequence holding one binary onset vector per track
            :param time_signature:       time signature of the rhythm as a (num, den) tuple or TimeSignature object
            :param velocity:             the velocity of the onsets as an integer, which will be the same for all onsets
            :param unit:                 step size as a musical unit (e.g., if unit is set to Unit.EIGHTH (or 1/8 or
                                         "eighth") one character will represent one eighth note)
            :param track_names:          names of the tracks
            :param title:                rhythm loop title
            :param kwargs:               unused
            :return: polyphonic rhythm object
            """

            polyphonic_rhythm = PolyphonicRhythm.create.from_binary_vector(
                binary_vector_tracks,
                time_signature=time_signature,
                velocity=velocity,
                unit=unit,
                track_names=track_names
            )

            return cls.from_polyphonic_rhythm(polyphonic_rhythm, title)

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

        self.set_name(name)
        self._midi_pattern = None              # type: tp.Union[midi.Pattern, None]
        self._midi_mapping = midi_mapping      # type: MidiDrumMapping
        self._midi_mapping_reducer = None      # type: tp.Union[MidiDrumMappingReducer, None]
        self._midi_metronome = -1              # type: int
        self._prototype_midi_pitches = dict()  # type: tp.Dict[str, int]

        # Note: we set the mapping (and reducer) before loading the midi pattern to avoid loading twice
        self.set_midi_drum_mapping(midi_mapping)
        self.set_midi_drum_mapping_reducer(midi_mapping_reducer_cls)

        # loads the tracks and sets the bpm, time signature, midi metronome and resolution
        if midi_pattern:
            self.load_midi_pattern(midi_pattern, preserve_midi_duration)

    @property
    def midi_mapping(self):
        """The midi mapping.

        The MIDI mapping is used when parsing the MIDI data to create the track names. This is a read-only property.
        """
        return self._midi_mapping

    def set_midi_drum_mapping(self, drum_mapping: MidiDrumMapping) -> None:
        """Sets the MIDI drum mapping and resets the tracks accordingly.

        :param drum_mapping: midi drum mapping
        :return: None
        """

        if not isinstance(drum_mapping, MidiDrumMapping):
            raise TypeError("expected MidiDrumMapping but got %s" % str(drum_mapping))

        self._midi_mapping = drum_mapping
        mapping_reducer = self.get_midi_drum_mapping_reducer()

        # updates midi drum mapping reducer and reloads the tracks
        self.set_midi_drum_mapping_reducer(mapping_reducer)

    def get_midi_drum_mapping(self) -> MidiDrumMapping:
        """Returns the current MIDI drum mapping

        :return: MIDI drum mapping object
        """

        return self._midi_mapping

    def get_midi_drum_mapping_reducer(self) -> tp.Union[tp.Type[MidiDrumMapping], None]:
        """
        Returns the current MIDI drum mapping reducer class or None if no mapping reducer has been set.

        :return: MIDI mapping reducer or None if no reducer set
        """

        mapping_reducer = self._midi_mapping_reducer
        return mapping_reducer.__class__ if mapping_reducer else None

    def set_midi_drum_mapping_reducer(self, mapping_reducer_cls: tp.Union[tp.Type[MidiDrumMappingReducer], None]):
        """
        Sets the MIDI drum mapping reducer and reloads the tracks. If no
         The rhythm duration will remain unchanged.

        :param mapping_reducer_cls: MIDI drum mapping reducer class or None to remove the mapping reducer
        :return: None
        """

        mapping = self._midi_mapping

        if mapping_reducer_cls:
            mapping_reducer = mapping_reducer_cls(mapping)
        else:
            mapping_reducer = None

        prev_resolution = self.get_resolution()
        prev_tick_duration = self.get_duration_in_ticks()

        self._midi_mapping_reducer = mapping_reducer
        self.__reload_midi_pattern(False)

        self.set_resolution(prev_resolution)
        self.set_duration_in_ticks(prev_tick_duration)

    @property
    def midi_drum_mapping_reducer(self) -> tp.Union[tp.Type[MidiDrumMappingReducer], None]:
        """The MIDI drum mapping reducer class. Setting this property will reset the tracks of this rhythm. Set this
        property to None for no MIDI drum mapping reducer."""

        return self.get_midi_drum_mapping_reducer()

    @midi_drum_mapping_reducer.setter
    def midi_drum_mapping_reducer(self, mapping_reducer_cls: tp.Union[tp.Type[MidiDrumMappingReducer], None]) -> None:
        self.set_midi_drum_mapping_reducer(mapping_reducer_cls)

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
        Loads a midi pattern and sets this rhythm's tracks, time signature, bpm and duration. The given midi pattern
        must have a resolution property and can't have more than one track containing note events. The midi events map
        to rhythm properties like this:

        * :class:`midi.NoteOnEvent`, adds an onset to this rhythm
        * :class:`midi.TimeSignatureEvent`, set the time signature of this rhythm (required)
        * :class:`midi.SetTempoEvent`, sets the bpm of this rhythm
        * :class:`midi.EndOfTrackEvent`, sets the duration of this rhythm (only if preserve_midi_duration is true)

        The `EndOfTrackEvent` is required if the `preserve_midi_duration` is set to `True`. If preserve_midi_duration is
        `False`, the duration of this rhythm will be set to the first downbeat after the last note position.

        :param pattern: the midi pattern to load
        :param preserve_midi_duration: when true, the duration will be set to the position of the midi EndOfTrackEvent,
                                       otherwise it will be set to the first downbeat after the last note position
        :return: None
        """

        if not isinstance(pattern, midi.Pattern):
            raise TypeError("expected a midi.Pattern but got %s" % str(pattern))

        self._midi_pattern = pattern
        ret = self.__reload_midi_pattern(preserve_midi_duration)
        assert ret

    def __reload_midi_pattern(self, preserve_midi_duration: bool):
        # resets the tracks of this rhythm according to the current midi pattern, midi mapping and mapping reducer,
        # returns False if no midi pattern has been loaded yet
        pattern = self._midi_pattern

        if not pattern:
            return False

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
                    LOGGER.warning("Skipping unknown midi key: %i (Mapping = %s)" % (midi_pitch, mapping.get_name()))
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
                # NOTE: Although this is the last event of the track, there might be other tracks coming, so don't put
                # a break here (e.g. with midi format 1, where meta data and notes are divided into separate tracks)
                eot_event = msg

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

        return True


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


class MidiRhythmCorpus(object):
    DEFAULT_RHYTHM_RESOLUTION = 120
    DEFAULT_MIDI_MAPPING_REDUCER = None

    _RHYTHM_DATA_RHYTHM = 0
    _RHYTHM_DATA_FILE_INFO = 1

    _PICKLE_DATA_ID_KEY = "id"
    _PICKLE_DATA_RESOLUTION_KEY = "res"
    _PICKLE_DATA_RHYTHM_DATA_KEY = "rhythm_data"
    _PICKLE_DATA_MAPPING_REDUCER_NAME_KEY = "mapping_reducer"

    class MidiCorpusStateError(Exception):
        pass

    class BadCacheFormatError(Exception):
        pass

    def __init__(self, path: tp.Optional[tp.Union[IOBase, str]] = None, **kwargs):
        """Creates and optionally loads a MIDI rhythm corpus

        Calling this constructor with the path parameter set to ...
            ... a directory is equivalent to calling :meth:`beatsearch.rhythm.MidiRhythmCorpus.load_from_directory` on
                an unload MidiRhythmCorpus.
            ... a file path is equivalent to calling :meth:`beatsearch.rhythm.MidiRhythmCorpus.load_from_cache_file` on
                an unload MidiRhythmCorpus.

        :param path: when given a directory path or a file path, this corpus will automatically load using either
                     :meth:`beatsearch.rhythm.MidiRhythmCorpus.load_from_directory` or
                     :meth:`beatsearch.rhythm.MidiRhythmCorpus.load_from_cache_file` respectively

        :param kwargs:
            rhythm_resolution: rhythm resolution in PPQN (immediately overwritten if loading from cache)
            midi_mapping_reducer: MIDI drum mapping reducer class (immediately overwritten if loading from cache)
        """

        self._rhythm_resolution = None     # type: tp.Union[int, None]
        self._midi_mapping_reducer = None  # type: tp.Union[tp.Type[MidiDrumMappingReducer], None]
        self._rhythm_data = None           # type: tp.Union[tp.Tuple[tp.Tuple[MidiRhythm, FileInfo], ...], None]
        self._id = None                    # type: tp.Union[uuid.UUID, None]

        # calling setters
        self.rhythm_resolution = kwargs.get("rhythm_resolution", self.DEFAULT_RHYTHM_RESOLUTION)
        self.midi_mapping_reducer = kwargs.get("midi_mapping_reducer", self.DEFAULT_MIDI_MAPPING_REDUCER)

        # load corpus
        if isinstance(path, str) and os.path.isdir(path):
            self.load_from_directory(path)
        elif path:
            for arg_name in ("rhythm_resolution", "midi_mapping_reducer"):
                if arg_name in kwargs:
                    LOGGER.debug("Ignoring named parameter %s. Loading corpus from cache.")
            self.load_from_cache_file(path)

    def load_from_directory(self, midi_root_dir: str):
        """Loads this MIDI corpus from a MIDI root directory

        Recursively scans the given directory for MIDI files and loads one rhythm per MIDI file.

        :param midi_root_dir: MIDI root directory
        :return: None
        """

        if self.has_loaded():
            raise self.MidiCorpusStateError("corpus has already loaded")

        if not os.path.isdir(midi_root_dir):
            raise IOError("no such directory: %s" % midi_root_dir)

        self._rhythm_data = tuple(self.__lazy_load_rhythm_data_from_directory(
            midi_root_dir=midi_root_dir,
            resolution=self.rhythm_resolution,
            mapping_reducer=self.midi_mapping_reducer
        ))

        self._id = uuid.uuid4()

    def unload(self):
        """Unloads this rhythm corpus

        This method won't have any effect if the corpus has not loaded.

        :return: None
        """

        if not self.has_loaded():
            return

        self._rhythm_data = None
        self._id = None

    @staticmethod
    def __lazy_load_rhythm_data_from_directory(
            midi_root_dir: str,
            resolution: int,
            mapping_reducer: tp.Optional[tp.Type[MidiDrumMappingReducer]]
    ) -> tp.Generator[tp.Tuple[MidiRhythm, FileInfo], None, None]:

        for f_path in get_midi_files_in_directory(midi_root_dir):
            f_path = f_path.replace("\\", "/")

            try:
                rhythm = MidiRhythm(f_path, midi_mapping_reducer_cls=mapping_reducer)
                rhythm.set_resolution(resolution)
                LOGGER.info("%s: OK" % f_path)
            except (TypeError, ValueError) as e:
                LOGGER.warning("%s: ERROR, %s" % (f_path, str(e)))
                continue

            m_time = os.path.getmtime(f_path)
            file_info = FileInfo(path=f_path, modified_time=m_time)
            yield rhythm, file_info

    def load_from_cache_file(self, cache_fpath: tp.Union[IOBase, str]):
        """Loads this MIDI corpus from a serialized pickle file

        Loads a MIDI corpus from a serialized pickle file created with previously created with
        :meth:`beatsearch.rhythm.MidiRhythmCorpus.save_to_cache_file`.

        :param cache_fpath: path to the serialized pickle file
        :return: None
        """

        if self.has_loaded():
            raise self.MidiCorpusStateError("corpus has already loaded")

        if isinstance(cache_fpath, str):
            with open(cache_fpath, "rb") as pickle_file:
                unpickled_data = pickle.load(pickle_file)
        else:
            unpickled_data = pickle.load(cache_fpath)

        try:
            rhythm_resolution = unpickled_data[self._PICKLE_DATA_RESOLUTION_KEY]
            mapping_reducer_name = unpickled_data[self._PICKLE_DATA_MAPPING_REDUCER_NAME_KEY]
            rhythm_data = unpickled_data[self._PICKLE_DATA_RHYTHM_DATA_KEY]
            rhythm_id = unpickled_data[self._PICKLE_DATA_ID_KEY]
        except KeyError:
            raise ValueError("Midi root directory cache file has bad format: %s" % cache_fpath)

        if mapping_reducer_name:
            mapping_reducer = get_drum_mapping_reducer_implementation(mapping_reducer_name)
        else:
            mapping_reducer = None

        self.rhythm_resolution = rhythm_resolution
        self.midi_mapping_reducer = mapping_reducer
        self._rhythm_data = rhythm_data
        self._id = rhythm_id

    def save_to_cache_file(self, cache_file: tp.Union[IOBase, str], overwrite=False):
        """Serializes this MIDI corpus to a pickle file

        :param cache_file: either an opened file handle in binary-write mode or a file path
        :param overwrite: when True, no exception will be raised if a file path is given which already exists
        :return: None
        """

        if not self.has_loaded():
            raise self.MidiCorpusStateError("can't save a corpus that hasn't loaded yet")

        resolution = self.rhythm_resolution
        mapping_reducer_name = self.midi_mapping_reducer.__name__ if self.midi_mapping_reducer else ""

        pickle_data = {
            self._PICKLE_DATA_RESOLUTION_KEY: resolution,
            self._PICKLE_DATA_MAPPING_REDUCER_NAME_KEY: mapping_reducer_name,
            self._PICKLE_DATA_RHYTHM_DATA_KEY: self._rhythm_data,
            self._PICKLE_DATA_ID_KEY: self._id
        }

        if isinstance(cache_file, str):
            if os.path.isfile(cache_file) and not overwrite:
                raise RuntimeError("there's already a file with path: %s" % cache_file)
            with open(cache_file, "wb") as cache_file:
                pickle.dump(pickle_data, cache_file)
        else:
            pickle.dump(pickle_data, cache_file)

    def has_loaded(self):
        """Returns whether this corpus has already loaded

        Returns whether this rhythm corpus has already been loaded. This will return true after a successful call to
        load().

        :return: True if this corpus has already loaded: False otherwise
        """

        return self._rhythm_data is not None

    def is_up_to_date(self, midi_root_dir: str):
        """Returns whether the rhythms in this corpus are fully up to date with the MIDI contents of the given directory

        Recursively scans the given directory for MIDI files and checks whether the files are the identical (both
        file names and file modification timestamps) to the files that were used to create this corpus.

        :param midi_root_dir: midi root directory that was used to create this corpus
        :return: True if up to date; False otherwise
        """

        if not os.path.isdir(midi_root_dir):
            raise IOError("no such directory: %s" % midi_root_dir)

        for file_info in (entry[self._RHYTHM_DATA_FILE_INFO] for entry in self._rhythm_data):
            fpath = file_info.path

            # rhythm data is not up to date if either the file doesn't exist anymore or the file has been modified
            if not os.path.isfile(fpath) or os.path.getmtime(fpath) != file_info.modified_time:
                return False

        n_cached_midi_files = len(self._rhythm_data)  # "cached" referring to the rhythms in this MidiRhythmCorpus obj
        n_actual_midi_files = sum(bool(fpath) for fpath in get_midi_files_in_directory(midi_root_dir))

        # won't be equal if new MIDI files have been added
        return n_cached_midi_files == n_actual_midi_files

    def export_as_midi_files(self, directory: str, **kwargs):
        """Converts all rhythms in this corpus to MIDI patterns and saves them to the given directory

        :param directory: directory to save the MIDI files to
        :param kwargs: named arguments given to :meth:`beatsearch.rhythm.MidiRhythm.as_midi_pattern`
        :return: None
        """

        make_dir_if_not_exist(directory)

        for entry in self._rhythm_data:
            rhythm = entry[self._RHYTHM_DATA_RHYTHM]
            file_info = entry[self._RHYTHM_DATA_FILE_INFO]
            fname = os.path.basename(file_info.path)
            fpath = os.path.join(directory, fname)
            pattern = rhythm.as_midi_pattern(**kwargs)
            midi.write_midifile(fpath, pattern)


    @property
    def rhythm_resolution(self):
        """The resolution in PPQN

        Tick resolution in PPQN (pulses-per-quarter-note) of the rhythms within this corpus. This property will become
        a read-only property after the corpus has loaded.

        :return: resolution in PPQN of the rhythms in this corpus
        """

        return self._rhythm_resolution

    @rhythm_resolution.setter
    def rhythm_resolution(self, resolution: tp.Union[int, None]):
        if self.has_loaded():
            raise self.MidiCorpusStateError("corpus has already been loaded, making rhythm_resolution read-only")

        if resolution is None:
            self._rhythm_resolution = None
            return

        resolution = int(resolution)
        if resolution <= 0:
            raise ValueError("resolution should be greater than zero")

        self._rhythm_resolution = resolution

    @property
    def midi_mapping_reducer(self) -> tp.Union[tp.Type[MidiDrumMappingReducer], None]:
        """The MIDI drum mapping reducer

        The MIDI drum mapping reducer applied to the rhythms in this corpus. Note that setting this property is an
        expensive operation, as it will iterate over every rhythm to reset its tracks according to the new mapping
        reducer.
        """

        return self._midi_mapping_reducer

    @midi_mapping_reducer.setter
    def midi_mapping_reducer(self, midi_mapping_reducer: tp.Union[tp.Type[MidiDrumMappingReducer], None]):
        if midi_mapping_reducer is not None and not issubclass(midi_mapping_reducer, MidiDrumMappingReducer):
            raise TypeError("expected a MidiDrumMappingReducer subclass or None but got '%s'" % midi_mapping_reducer)

        if self.has_loaded():
            for rhythm in self:
                rhythm.set_midi_drum_mapping_reducer(midi_mapping_reducer)

        self._midi_mapping_reducer = midi_mapping_reducer

    @property
    def id(self):
        """The id of this rhythm corpus

        The UUID id of this rhythm corpus. This is a read-only property.
        """

        return self._id

    def __getitem__(self, i):
        """Returns the i-th rhythm if i is an integer or the rhythm with the given name if i is a string"""
        if isinstance(i, int):
            return self._rhythm_data[i][self._RHYTHM_DATA_RHYTHM]
        elif isinstance(i, str):
            # TODO: Make this O(1)
            try:
                return next(rhythm for rhythm in self if rhythm.name == i)
            except StopIteration:
                raise KeyError("No rhythm named: %s" % i)
        else:
            raise TypeError("Please provide either an integer for "
                            "indexing by rhythm index or a string for "
                            "indexing by rhythm name.")

    def __len__(self):
        """Returns the number of rhythms within this corpus"""
        return len(self._rhythm_data)

    def __iter__(self):
        """Returns an iterator over the rhythms within this corpus"""
        return iter(data_entry[self._RHYTHM_DATA_RHYTHM] for data_entry in self._rhythm_data)


__all__ = [
    # Rhythm classes
    'Rhythm', 'MonophonicRhythm', 'PolyphonicRhythm',
    'RhythmLoop', 'MidiRhythm',

    # Time unit
    'Unit', 'UnitType', 'UnitError', 'parse_unit_argument', 'rescale_tick', 'convert_tick',

    # Misc
    'Onset', 'Track', 'TimeSignature', 'GMDrumMapping', 'create_rumba_rhythm', 'MidiRhythmCorpus',

    # MIDI drum mapping
    'MidiDrumMapping', 'GMDrumMapping', 'create_drum_mapping', 'MidiDrumKey', 'FrequencyBand', 'DecayTime',
    'MidiDrumMappingReducer', 'FrequencyBandMidiDrumMappingReducer',
    'DecayTimeMidiDrumMappingReducer', 'UniquePropertyComboMidiDrumMappingReducer',
    'get_drum_mapping_reducer_implementation_names',
    'get_drum_mapping_reducer_implementation_friendly_names',
    'get_drum_mapping_reducer_implementation'
]

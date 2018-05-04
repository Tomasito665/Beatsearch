import math
import enum
import types
import logging
import itertools
import numpy as np
import typing as tp
from fractions import Fraction
from collections import OrderedDict, defaultdict
from abc import ABCMeta, abstractmethod
from beatsearch.utils import Quantizable, minimize_term_count, TupleView
from beatsearch.rhythm import Rhythm, MonophonicRhythm, PolyphonicRhythm, \
    Unit, UnitType, parse_unit_argument, convert_tick


LOGGER = logging.getLogger(__name__)

######################################
# Feature extractor abstract classes #
######################################


class FeatureExtractor(object, metaclass=ABCMeta):
    @abstractmethod
    def process(self, obj) -> tp.Any:
        """Computes and returns a feature
        Computes a feature of the given object and returns it.

        :param obj: object of which to compute the feature
        :return: the feature value
        """

        raise NotImplementedError


class RhythmFeatureExtractor(FeatureExtractor, metaclass=ABCMeta):
    @abstractmethod
    def get_pre_processors(self):  # type: () -> tp.Tuple[RhythmFeatureExtractor, ...]
        """Returns the preprocessors of this rhythm feature extractor

        Preprocessors are also instances of RhythmFeatureExtractor. When calling process(), the preprocessors will
        process the given rhythm and their results will be passed down to __process__.

        :return: tuple with this rhythm feature extractor's preprocessors
        """

        raise NotImplementedError

    @abstractmethod
    def get_unit(self) -> tp.Union[Unit, None]:
        """Returns the musical unit of this preprocessor

        :return: musical unit of this preprocessor as a string
        """

        raise NotImplementedError

    @abstractmethod
    def set_unit(self, unit: tp.Optional[UnitType]) -> None:
        """Sets the musical unit of this preprocessor

        :return:
        """

        raise NotImplementedError

    def process(self, rhythm: Rhythm) -> tp.Any:
        """Computes and returns a rhythm feature

        :param rhythm: rhythm of which to compute the feature
        :return: the rhythm feature value
        """

        pre_processor_results = list(pre_processor.process(rhythm) for pre_processor in self.get_pre_processors())
        ret = self.__process__(rhythm, pre_processor_results)
        return tuple(ret) if isinstance(ret, types.GeneratorType) else ret

    @abstractmethod
    def __process__(self, rhythm: Rhythm, pre_processor_results: tp.List[tp.Any]):
        """Computes and returns a rhythm feature

        If this method is implemented as a generator, it will be evaluated by process() and returned as a tuple.

        :param rhythm:                 rhythm of which to compute the feature
        :param pre_processor_results:  results of the preprocessors, which process the given rhythm
                                       immediately before the call to this method

        :return: the rhythm feature value
        """

        raise NotImplementedError

    @staticmethod
    def init_pre_processors():
        """
        Override this method to add preprocessors to the feature extractor. Preprocessors are RhythmFeatureExtractorBase
        objects themselves and their results will be passed down to __process__ as the last arguments.

        :return: an iterable of RhythmFeatureExtractorBase instances
        """

        return []

    ##############
    # properties #
    ##############

    @property
    def pre_processors(self):  # type: () -> tp.Tuple[RhythmFeatureExtractor, ...]
        """See get_pre_processors. This is a read-only property."""
        return self.get_pre_processors()

    @property
    def unit(self) -> tp.Union[Unit, None]:
        """See get_unit and set_unit"""
        return self.get_unit()

    @unit.setter
    def unit(self, unit: tp.Optional[UnitType]):
        self.set_unit(unit)


class RhythmFeatureExtractorBase(RhythmFeatureExtractor, metaclass=ABCMeta):
    def __init__(self, unit: tp.Optional[UnitType] = Unit.EIGHTH):
        self._pre_processors = tuple(self.init_pre_processors())  # type: tp.Tuple[RhythmFeatureExtractorBase, ...]
        self.__tick_to_unit__ = None

        self._unit = None  # type: tp.Union[Unit, None]
        self.unit = unit  # calling setter, which also sets the unit of the preprocessors

    def get_pre_processors(self) -> tp.Tuple[RhythmFeatureExtractor, ...]:
        """Returns the preprocessors, whose result is passed to __process__.

        Preprocessors are also rhythm feature extractors. Their unit is linked to this extractor's unit.
        """

        return self._pre_processors

    def get_unit(self) -> tp.Union[Unit, None]:
        """Returns the time unit of this feature extractor"""
        return self._unit

    @parse_unit_argument
    def set_unit(self, unit: tp.Optional[UnitType]) -> None:
        """Sets the time unit of this feature extractor

        Sets the time unit of this rhythm feature extractor. This method will also set the unit of the preprocessors.

        :param unit: new unit
        :return: None
        """

        for pre_processor in self._pre_processors:
            pre_processor.set_unit(unit)

        self._unit = unit


class QuantizableRhythmFeatureExtractorMixin(RhythmFeatureExtractor, Quantizable, metaclass=ABCMeta):
    def __init__(self, quantize_enabled):
        self._quantize_enabled = None
        self.set_quantize_enabled(quantize_enabled)

    def set_quantize_enabled(self, quantize_enabled: bool):
        quantize_enabled = bool(quantize_enabled)
        for feature_extractor in self.pre_processors:
            if not isinstance(feature_extractor, Quantizable):
                continue
            feature_extractor.set_quantize_enabled(quantize_enabled)
        self._quantize_enabled = quantize_enabled

    def is_quantize_enabled(self) -> bool:
        return self._quantize_enabled


class MonophonicRhythmFeatureExtractor(RhythmFeatureExtractorBase, metaclass=ABCMeta):
    @abstractmethod
    def __process__(self, rhythm: MonophonicRhythm, pre_processor_results: tp.List[tp.Any]):
        """Computes and returns a monophonic rhythm feature
        Computes a feature of the given monophonic rhythm and returns it.

        :param rhythm: monophonic rhythm of which to compute the feature
        :return: the monophonic rhythm feature value
        """

        raise NotImplementedError


class PolyphonicRhythmFeatureExtractor(RhythmFeatureExtractorBase, metaclass=ABCMeta):
    @abstractmethod
    def __process__(
            self,
            rhythm: PolyphonicRhythm,
            pre_processor_results: tp.List[tp.Any]
    ):
        """Computes and returns a polyphonic rhythm feature
        Computes a feature of the given polyphonic rhythm and returns it.

        :param rhythm: polyphonic rhythm of which to compute the feature
        :param unit: concrete time unit
        :return: the polyphonic rhythm feature value
        """

        raise NotImplementedError


####################################################
# Generic rhythm feature extractor implementations #
####################################################


#######################################################
# Monophonic rhythm feature extractor implementations #
#######################################################


class BinaryOnsetVector(MonophonicRhythmFeatureExtractor):
    def __process__(self, rhythm: MonophonicRhythm, _):
        """
        Returns the binary representation of the note onsets of the given rhythm where each step where a note onset
        happens is denoted with a 1; otherwise with a 0. The given resolution is the resolution in PPQ (pulses per
        quarter note) of the binary vector.

        :param rhythm: the monophonic rhythm from which to compute the binary onset vector
        :return: the binary representation of the note onsets in the given rhythm
        """

        Rhythm.Precondition.check_resolution(rhythm)

        unit = self.unit
        n_steps = rhythm.get_duration(unit, ceil=True)
        binary_string = [0] * n_steps

        if unit is None:
            get_onset_position = lambda t: t
        else:
            resolution = rhythm.get_resolution()
            get_onset_position = lambda t: unit.from_ticks(t, resolution, True)

        for onset in rhythm.get_onsets():
            onset_position = get_onset_position(onset.tick)
            try:
                binary_string[onset_position] = 1
            except IndexError:
                pass  # when quantization pushes note after end of rhythm

        return binary_string


class NoteVector(MonophonicRhythmFeatureExtractor, QuantizableRhythmFeatureExtractorMixin):
    """
    This feature extractor predicts the durations of the onsets in the given rhythm. It computes the so-called note
    vector, which is a sequence containing musical events. Musical events can be of three types:

        - NOTE          a sounding note
        - TIED_NOTE     a non-sounding note
        - REST          a rest

    Events are expressed as tuples containing three elements:

        - e_type    the type of the event (NOTE, TIED_NOTE or REST)
        - e_dur     the duration of the event
        - e_pos     the position of the event
    """

    NOTE = "N"
    """Note event"""

    TIED_NOTE = "T"
    """Tied note event"""

    REST = "R"
    """Rest event"""

    _GET_REST_OR_TIED_NOTE = {
        # With support for tied notes
        True: lambda prev_e: NoteVector.TIED_NOTE if prev_e == NoteVector.NOTE else NoteVector.REST,
        # Without support for tied notes
        False: lambda _: NoteVector.REST
    }

    def __init__(self, unit: UnitType = Unit.EIGHTH, tied_notes: bool = True, cyclic: bool = True):
        super().__init__(unit)
        self._tied_notes = None
        self._cyclic = None

        # call setters
        self.tied_notes = tied_notes
        self.cyclic = cyclic

    def set_unit(self, unit: tp.Optional[UnitType]) -> None:
        if unit is None:
            raise ValueError("NoteVector does not support tick-based computation")
        super().set_unit(unit)

    @staticmethod
    def init_pre_processors():
        yield BinaryOnsetVector()

    @property
    def tied_notes(self):
        return self._tied_notes

    @tied_notes.setter
    def tied_notes(self, tied_notes: bool):
        self._tied_notes = bool(tied_notes)

    @property
    def cyclic(self):
        return self._cyclic

    @cyclic.setter
    def cyclic(self, cyclic: bool):
        self._cyclic = bool(cyclic)

    def __process__(self, rhythm: MonophonicRhythm, pre_processor_results: tp.List[tp.Any]):
        Rhythm.Precondition.check_time_signature(rhythm)
        time_sig = rhythm.get_time_signature()

        natural_duration_map = time_sig.get_natural_duration_map(self.unit, trim_to_pulse=True)
        duration_pool = sorted(set(natural_duration_map))  # NOTE The rest of this method depends on this being sorted
        binary_vector = pre_processor_results[0]

        tied_note_support = self.tied_notes
        get_rest_or_tied_note = self._GET_REST_OR_TIED_NOTE[tied_note_support]

        step = 0
        step_count = len(binary_vector)
        note_vector = []  # type: tp.List[tp.Tuple[str, int, int]]

        while step < step_count:
            metrical_position = step % len(natural_duration_map)
            natural_duration = natural_duration_map[metrical_position]
            is_onset = binary_vector[step]

            # Compute the duration till the next note trimmed at the natural duration (which itself is pulse-trimmed)
            max_duration = next((j for j in range(1, natural_duration) if binary_vector[step + j]), natural_duration)

            if is_onset:
                # Get the maximum available duration that fits the max duration
                note_duration = next(d for d in reversed(duration_pool) if d <= max_duration)
                note_vector.append((self.NOTE, note_duration, step))

                step += note_duration
                continue

            rests = tuple(minimize_term_count(max_duration, duration_pool, assume_sorted=True))
            prev_e_type = note_vector[-1][0] if note_vector else self.REST

            for rest_duration in rests:
                curr_e_type = get_rest_or_tied_note(prev_e_type)
                note_vector.append((curr_e_type, rest_duration, step))
                prev_e_type = curr_e_type
                step += rest_duration

        # Adjust heading rest to tied note if necessary
        if self.cyclic and note_vector and note_vector[0][0] == self.REST:
            last_e_type = note_vector[-1][0]
            note_vector[0] = (get_rest_or_tied_note(last_e_type), *note_vector[0][1:])

        return note_vector


class IOIVector(MonophonicRhythmFeatureExtractor, QuantizableRhythmFeatureExtractorMixin):
    class Mode(enum.Enum):
        PRE_NOTE = enum.auto()
        POST_NOTE = enum.auto()

    def __init__(self, unit: tp.Optional[UnitType] = Unit.EIGHTH, mode: Mode = Mode.POST_NOTE, quantize_enabled=True):
        MonophonicRhythmFeatureExtractor.__init__(self, unit)
        QuantizableRhythmFeatureExtractorMixin.__init__(self, quantize_enabled)
        self._mode = None
        self.mode = mode  # call setter

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode: Mode):
        self._mode = mode

    def __process__(self, rhythm: MonophonicRhythm, pre_processor_results: tp.List[tp.Any]):
        """
        Returns the time difference between the notes in the given rhythm. The elements of the vector will depend on
        this IOIVector extractor's mode property:

        PRE_NOTE
            Time difference between the current note and the previous note. The first note will return the time
            difference with the start of the rhythm.

            For example, given the Rumba Clave rhythm:
              X--X---X--X-X---
              0  3   4  3 2

        POST_NOTE
            Time difference between the current note and the next note. The last note will return the time difference
            with the end (duration) of the rhythm.

            For example, given the Rumba Clave rhythm:
              X--X---X--X-X---
              3  4   3  2 4

        :param rhythm: the monophonic rhythm from which to compute the inter-onset interval vector
        :return: inter-onset interval vector of the given rhythm
        """

        Rhythm.Precondition.check_resolution(rhythm)

        # TODO Implement both modes generically here

        if self.mode == self.Mode.PRE_NOTE:
            return self._pre_note_ioi(rhythm)
        elif self.mode == self.Mode.POST_NOTE:
            return self._post_note_ioi(rhythm)
        else:
            raise RuntimeError("Unknown mode: \"%s\"" % self.mode)

    def _pre_note_ioi(self, rhythm: MonophonicRhythm):
        quantize = self.is_quantize_enabled()
        resolution = rhythm.get_resolution()
        unit = self.get_unit()
        current_tick = 0
        intervals = []

        for onset in rhythm.get_onsets():
            delta_tick = onset.tick - current_tick
            intervals.append(convert_tick(delta_tick, resolution, unit, quantize))
            current_tick = onset.tick

        return intervals

    # TODO Add cyclic option to include the offset in the last onset's interval
    def _post_note_ioi(self, rhythm: MonophonicRhythm):
        quantize = self.is_quantize_enabled()
        resolution = rhythm.get_resolution()
        unit = self.get_unit()
        intervals = []
        onset_positions = itertools.chain((onset.tick for onset in rhythm.get_onsets()), [rhythm.duration_in_ticks])
        last_onset_tick = -1

        for onset_tick in onset_positions:
            if last_onset_tick < 0:
                last_onset_tick = onset_tick
                continue

            delta_in_ticks = onset_tick - last_onset_tick
            delta_in_units = convert_tick(delta_in_ticks, resolution, unit, quantize)
            intervals.append(delta_in_units)
            last_onset_tick = onset_tick

        return intervals


class IOIHistogram(MonophonicRhythmFeatureExtractor):
    def __init__(self, unit: tp.Optional[UnitType] = Unit.EIGHTH):
        super().__init__(unit)

    @staticmethod
    def init_pre_processors():
        return [IOIVector(mode=IOIVector.Mode.POST_NOTE, quantize_enabled=True)]

    def __process__(self, rhythm: MonophonicRhythm, pre_processor_results: tp.List[tp.Any]):
        """
        Returns the number of occurrences of the inter-onset intervals of the notes of the given rhythm in ascending
        order. The inter onset intervals are computed in POST_NOTE mode.

        For example, given the Rumba Clave rhythm, with inter-onset vector [3, 4, 3, 2, 4]:
            (
                [1, 2, 2],  # occurrences
                [2, 3, 4]   # bins (interval durations)
            )


        :return: an (occurrences, bins) tuple
        """

        ioi_vector = pre_processor_results[0]
        histogram = np.histogram(ioi_vector, tuple(range(min(ioi_vector), max(ioi_vector) + 2)))
        occurrences = histogram[0].tolist()
        bins = histogram[1].tolist()[:-1]
        return occurrences, bins


class BinarySchillingerChain(MonophonicRhythmFeatureExtractor):
    def __init__(self, unit: tp.Optional[UnitType] = Unit.EIGHTH, values=(1, 0)):
        super().__init__(unit)
        self._values = None
        self.values = values  # calls setter

    @staticmethod
    def init_pre_processors():
        yield BinaryOnsetVector()

    @property
    def values(self):
        """The two values of the schillinger chain returned by process

        Binary vector to be used in the schillinger chain returned by process(). E.g. when set to ('a', 'b'), the
        schillinger chain returned by process() will consist of 'a' and 'b'.
        """

        return self._values

    @values.setter
    def values(self, values):
        values = iter(values)
        self._values = next(values), next(values)

    def __process__(self, rhythm: MonophonicRhythm, pre_processor_results: tp.List[tp.Any]):
        """
        Returns the Schillinger notation of this rhythm where each onset is a change of a "binary note".

        For example, given the Rumba Clave rhythm and with values (0, 1):
          X--X---X--X-X---
          0001111000110000

        However, when given the values (1, 0), the schillinger chain will be the opposite:
          X--X---X--X-X---
          1110000111001111

        :return: Schillinger rhythm vector as a list
        """

        values = self._values
        chain = list(pre_processor_results[0])
        i, value_i = 0, 1

        while i < len(chain):
            if chain[i] == 1:
                value_i = 1 - value_i
            chain[i] = values[value_i]
            i += 1

        return chain


class ChronotonicChain(MonophonicRhythmFeatureExtractor):
    @staticmethod
    def init_pre_processors():
        yield BinaryOnsetVector()

    def __process__(self, rhythm: MonophonicRhythm, pre_processor_results: tp.List[tp.Any]):
        """
        Returns the chronotonic chain representation of the given rhythm.

        For example, given the Rumba Clave rhythm:
          X--X---X--X-X---
          3334444333224444

        :return: the chronotonic chain as a list
        """

        chain = pre_processor_results[0]  # binary onset vector
        i, delta = 0, 0

        while i < len(chain):
            if chain[i] == 1:
                j = i + 1
                while j < len(chain) and chain[j] == 0:
                    j += 1
                delta = j - i
            chain[i] = delta
            i += 1

        return chain


class IOIDifferenceVector(MonophonicRhythmFeatureExtractor, QuantizableRhythmFeatureExtractorMixin):
    def __init__(self, unit: tp.Optional[UnitType] = Unit.EIGHTH, quantize_enabled=True, cyclic=True):
        MonophonicRhythmFeatureExtractor.__init__(self, unit)
        QuantizableRhythmFeatureExtractorMixin.__init__(self, quantize_enabled)
        self._cyclic = None
        self.cyclic = cyclic  # call setters

    @staticmethod
    def init_pre_processors():
        yield IOIVector(mode=IOIVector.Mode.POST_NOTE)

    @property
    def cyclic(self):
        """Cyclic behaviour, see process documentation"""
        return self._cyclic

    @cyclic.setter
    def cyclic(self, cyclic):
        self._cyclic = bool(cyclic)

    def __process__(self, rhythm: MonophonicRhythm, pre_processor_results: tp.List[tp.Any]):
        """
        Returns the interval difference vector (aka difference of rhythm vector) of the given rhythm. Per note, this is
        the difference between the current onset interval and the next onset interval. So, if N is the number of onsets,
        the returned vector will have a length of N - 1. This is different with cyclic rhythms, where the last onset's
        interval is compared with the first onset's interval. In this case, the length will be N.
        The inter-onset interval vector is computed in POST_NOTE mode.

        For example, given the POST_NOTE inter-onset interval vector for the Rumba clave:
          [3, 4, 3, 2, 4]

        The interval difference vector would be:
           With cyclic set to False: [4/3, 3/4, 2/3, 4/2]
           With cyclic set to True:  [4/3, 3/4, 2/3, 4/2, 3/4]

        :param rhythm: monophonic rhythm of which to compute the inter-onset difference vector
        :return: interval difference vector of the given rhythm
        """

        vector, i = pre_processor_results[0], 0

        if self.cyclic:
            vector.append(vector[0])

        while i < len(vector) - 1:
            try:
                vector[i] = vector[i + 1] / float(vector[i])
            except ZeroDivisionError:
                vector[i] = float('inf')
            i += 1

        vector.pop()
        return vector


class OnsetPositionVector(MonophonicRhythmFeatureExtractor, QuantizableRhythmFeatureExtractorMixin):
    def __init__(self, unit: tp.Optional[UnitType] = Unit.EIGHTH, quantize_enabled=True):
        MonophonicRhythmFeatureExtractor.__init__(self, unit)
        QuantizableRhythmFeatureExtractorMixin.__init__(self, quantize_enabled)

    def __process__(self, rhythm: MonophonicRhythm, pre_processor_results: tp.List[tp.Any]):
        """
        Returns the absolute onset times of the notes in the given rhythm.

        :return: a list with the onset times of this rhythm
        """

        Rhythm.Precondition.check_resolution(rhythm)
        resolution = rhythm.get_resolution()
        quantize = self.is_quantize_enabled()
        unit = self.get_unit()

        return [convert_tick(onset[0], resolution, unit, quantize) for onset in rhythm.get_onsets()]


_SyncType = tp.Tuple[int, int, int]
"""Syncopation type for type hinting

Syncopations are expressed as a tuple of three elements:
    - syncopation strength
    - position of syncopated note
    - position of rest
"""


class SyncopationVector(MonophonicRhythmFeatureExtractor):
    def __init__(self, unit: UnitType = Unit.EIGHTH, salience_profile_type: str = "equal_upbeats"):
        super().__init__(unit)
        self._salience_prf_type = ""
        self.salience_profile_type = salience_profile_type

    @staticmethod
    def init_pre_processors():
        yield NoteVector()

    def set_unit(self, unit: tp.Optional[UnitType]) -> None:
        if unit is None:
            raise ValueError("SyncopationVector does not support tick-based computation")
        super().set_unit(unit)

    @property
    def salience_profile_type(self) -> str:
        """
        The type of salience profile to be used for syncopation detection. This must be one of: ['hierarchical',
        'equal_upbeats', 'equal_beats']. See :meth:`beatsearch.rhythm.TimeSignature.get_salience_profile` for more info.
        """

        return self._salience_prf_type

    @salience_profile_type.setter
    def salience_profile_type(self, salience_profile_type: str):
        self._salience_prf_type = str(salience_profile_type)

    def __process__(self, rhythm: MonophonicRhythm, pre_processor_results: tp.List[tp.Any]) -> tp.Iterable[_SyncType]:
        """
        Extracts the syncopations from the given monophonic rhythm. The syncopations are computed with the method
        proposed by H.C. Longuet-Higgins and C. S. Lee in their work titled: "The Rhythmic Interpretation of
        Monophonic Music".

        The syncopations are returned as a sequence of tuples containing three elements:
            - the syncopation strength
            - the syncopated note position
            - the position of the rest against which the note is syncopated

        :param rhythm: rhythm from which to extract the syncopations
        :return: the syncopations as (syncopation strength, note position, rest position) tuples
        """

        Rhythm.Precondition.check_time_signature(rhythm)

        note_vector = pre_processor_results[0]
        time_signature = rhythm.get_time_signature()
        metrical_weights = time_signature.get_salience_profile(self.unit, kind=self.salience_profile_type)

        for n, [curr_event_type, _, curr_step] in enumerate(note_vector):
            next_event_type, next_event_duration, next_step = note_vector[(n + 1) % len(note_vector)]

            # only iterate over [note, rest or tied note] pairs
            if not (curr_event_type == NoteVector.NOTE and next_event_type in (NoteVector.TIED_NOTE, NoteVector.REST)):
                continue

            note_weight = metrical_weights[curr_step % len(metrical_weights)]
            rest_weight = metrical_weights[next_step % len(metrical_weights)]

            if rest_weight >= note_weight:
                syncopation_strength = rest_weight - note_weight
                yield syncopation_strength, curr_step, next_step


class SyncopatedOnsetRatio(MonophonicRhythmFeatureExtractor):
    def __init__(self, unit: UnitType = Unit.EIGHTH, ret_fraction: bool = False):
        super().__init__(unit)
        self._ret_fraction = None
        self.ret_fraction = ret_fraction  # calling setter

    @staticmethod
    def init_pre_processors():
        # NOTE: Order is important (__process__ depends on it)
        yield BinaryOnsetVector()
        yield SyncopationVector()

    @property
    def ret_fraction(self) -> bool:
        return self._ret_fraction

    @ret_fraction.setter
    def ret_fraction(self, ret_fraction):
        self._ret_fraction = bool(ret_fraction)

    def __process__(self, rhythm: MonophonicRhythm, pre_processor_results: tp.List[tp.Any]):
        """
        Returns the number of syncopated onsets over the total number of onsets. The syncopations are computed
        SyncopationVector.

        :param rhythm: rhythm for which to compute the syncopated onset ratio
        :return: either if ret_fraction is True a Fraction object, otherwise a float
        """

        binary_onset_vector = pre_processor_results[0]
        syncopation_vector = pre_processor_results[1]

        # NOTE: We use binary onset vector and don't call rhythm.get_onset_count because the onset count might be
        # different for different units (onsets might merge with large unit like Unit.QUARTER)
        n_onsets = sum(binary_onset_vector)
        n_syncopated_onsets = len(syncopation_vector)
        assert n_syncopated_onsets <= n_onsets

        if self.ret_fraction:
            return Fraction(n_syncopated_onsets, n_onsets)
        else:
            return n_syncopated_onsets / float(n_onsets)


class MeanSyncopationStrength(MonophonicRhythmFeatureExtractor):
    @staticmethod
    def init_pre_processors():
        yield SyncopationVector()

    def __process__(self, rhythm: MonophonicRhythm, pre_processor_results: tp.List[tp.Any]):
        """
        Returns the average syncopation strength per step. The step size depends on the unit (see set_unit). The
        syncopations are computed with SyncopationVector.

        :param rhythm: rhythm of which to compute the average syncopation strength
        :return: average syncopation strength per step
        """

        syncopation_vector = pre_processor_results[0]
        total_syncopation_strength = sum(info[1] for info in syncopation_vector)
        n_steps = rhythm.get_duration(self.unit, ceil=True)

        try:
            return total_syncopation_strength / n_steps
        except ZeroDivisionError:
            return 0


class OnsetDensity(MonophonicRhythmFeatureExtractor):
    def __init__(self, unit: tp.Optional[UnitType] = Unit.EIGHTH):
        super().__init__(unit)

    @staticmethod
    def init_pre_processors():
        yield BinaryOnsetVector()

    def __process__(self, rhythm: MonophonicRhythm, pre_processor_results: tp.List[tp.Any]):
        """
        Computes the onset density of the given rhythm. The onset density is the number of onsets over the number of
        positions in the binary onset vector of the given rhythm.

        :param rhythm: monophonic rhythm to compute the onset density of
        :return: onset density of the given rhythm
        """

        binary_vector = pre_processor_results[0]
        n_onsets = sum(binary_vector)  # onsets are ones, non-onsets are zeros
        return float(n_onsets) / len(binary_vector)


class MonophonicTensionVector(MonophonicRhythmFeatureExtractor):
    """
    This feature extractor computes the monophonic tension of a rhythm. If E(i) is the i-th event in the note vector of
    the given rhythm (see :class:`beatsearch.feature_extraction.NoteVector`, it is said that the tension during event
    E(i) equals the metrical weight of the starting position of E(i) for rests and sounding notes. If E(i) is a tied
    note (non-sounding note), then the tension of E(i) equals the tension of E(i-1).
    """

    def __init__(self, unit: UnitType = Unit.EIGHTH,
                 salience_profile_type: str = "equal_upbeats", normalize: bool = False):
        super().__init__(unit)
        self._salience_prf_type = None
        self._normalize = None

        self.salience_profile_type = salience_profile_type
        self.normalize = normalize

    @staticmethod
    def init_pre_processors():
        yield NoteVector(tied_notes=True)

    @property
    def cyclic(self) -> bool:
        note_vector = self.get_pre_processors()[0]  # type: NoteVector
        return note_vector.cyclic

    @cyclic.setter
    def cyclic(self, cyclic: bool):
        note_vector = self.get_pre_processors()[0]  # type: NoteVector
        note_vector.cyclic = cyclic

    @property
    def salience_profile_type(self) -> str:
        return self._salience_prf_type

    @salience_profile_type.setter
    def salience_profile_type(self, salience_profile_type: str):
        self._salience_prf_type = salience_profile_type

    @property
    def normalize(self) -> bool:
        """When set to True, the tension will have a range of [0, 1]"""

        return self._normalize

    @normalize.setter
    def normalize(self, normalize: bool):
        self._normalize = bool(normalize)

    def __process__(self, rhythm: MonophonicRhythm, pre_processor_results: tp.List[tp.Any]):
        time_sig = rhythm.get_time_signature()
        salience_profile = time_sig.get_salience_profile(self.unit, self.salience_profile_type)

        note_vector = pre_processor_results[0]  # type: tp.Sequence[tp.Tuple[str, int, int]]
        tension_per_event = []                  # type: tp.List[int]
        prev_e_tension = None                   # type: tp.Optional[int]

        if self.normalize:
            assert max(salience_profile) == 0
            salience_range = abs(min(salience_profile))
            assert salience_range > 0
            normalizer = 1.0 / salience_range
        else:
            normalizer = 1.0

        for event_index, [e_type, _, e_pos] in enumerate(note_vector):
            if e_type == NoteVector.TIED_NOTE:
                e_tension = prev_e_tension
            else:
                metrical_pos = e_pos % len(salience_profile)
                e_tension = salience_profile[metrical_pos] * -1 * normalizer

            tension_per_event.append(e_tension)
            prev_e_tension = e_tension

        if tension_per_event and tension_per_event[0] is None:
            assert self.cyclic
            tension_per_event[0] = tension_per_event[-1]

        # Expand each tension element N times where N is the duration of the corresponding event (= e[1])
        return tuple(itertools.chain(*(itertools.repeat(t, e[1]) for t, e in zip(tension_per_event, note_vector))))


class MonophonicTension(MonophonicRhythmFeatureExtractor):
    # TODO Add documentation

    def __init__(self, unit: UnitType = Unit.EIGHTH, salience_profile_type: str = "hierarchical"):
        super().__init__(unit)
        self._instr_weights = dict()
        self.salience_profile_type = salience_profile_type  # type: str

    @staticmethod
    def init_pre_processors():
        tension_vec = MonophonicTensionVector(normalize=True)
        yield tension_vec

    def set_instrument_weights(self, weights: tp.Dict[str, float]):
        tension_vec = self.get_pre_processors()[0]  # type: PolyphonicTensionVector
        tension_vec.set_instrument_weights(weights)

    def get_instrument_weights(self) -> tp.Dict[str, float]:
        tension_vec = self.get_pre_processors()[0]  # type: PolyphonicTensionVector
        return tension_vec.get_instrument_weights()

    def set_unit(self, unit: tp.Optional[UnitType]) -> None:
        if unit is None:
            raise ValueError("MonophonicTension does not support tick-based computation")
        super().set_unit(unit)

    def __process__(self, rhythm: MonophonicRhythm, pre_processor_results: tp.List[tp.Any]):
        tension_vec = pre_processor_results[0]
        return math.sqrt(sum(t * t for t in tension_vec))


#######################################################
# Polyphonic rhythm feature extractor implementations #
#######################################################


class MultiChannelMonophonicRhythmFeatureVector(PolyphonicRhythmFeatureExtractor):
    class MonophonicFeatureExtractorNotSet(Exception):
        pass

    def __init__(
            self, unit: tp.Optional[UnitType] = None,
            mono_extractor_cls: tp.Optional[tp.Type[MonophonicRhythmFeatureExtractor]] = None,
            *mono_extractor_args,
            **mono_extractor_kwargs
    ):
        # TODO if possible, "inherit" the default unit of the mono extractor
        self._mono_extractor = None  # type: tp.Union[MonophonicRhythmFeatureExtractor, None]
        super().__init__(unit)

        if mono_extractor_cls:
            mono_extractor = mono_extractor_cls(unit, *mono_extractor_args, **mono_extractor_kwargs)
            self.monophonic_extractor = mono_extractor

    @property
    def monophonic_extractor(self):
        return self._mono_extractor

    @monophonic_extractor.setter
    def monophonic_extractor(self, monophonic_feature_extractor: MonophonicRhythmFeatureExtractor):
        monophonic_feature_extractor.set_unit(self.unit)  # TODO Shouldn't this one adapt to the mono extractor's unit?
        self._mono_extractor = monophonic_feature_extractor

    def set_unit(self, unit: tp.Optional[UnitType]) -> None:
        super().set_unit(unit)
        if self._mono_extractor:
            self._mono_extractor.set_unit(unit)

    def __process__(self, rhythm: PolyphonicRhythm, pre_processor_results: tp.List[tp.Any]):
        if not self._mono_extractor:
            raise self.MonophonicFeatureExtractorNotSet("Please set the mono_extractor property")
        return OrderedDict(self.__compute_features_per_track(rhythm))

    def __compute_features_per_track(self, rhythm: PolyphonicRhythm) -> tp.Generator[tp.Tuple[str, tp.Any], None, None]:
        for track in rhythm.get_track_iterator():
            yield track.get_name(), self._mono_extractor.process(track)


class PolyphonicSyncopationVector(PolyphonicRhythmFeatureExtractor):
    KEEP_HEAVIEST = "keep_heaviest"
    KEEP_FIRST = "keep_first"
    KEEP_LAST = "keep_last"
    KEEP_ALL = "keep_all"

    _PRE_PROCESSOR_BIN_VECTOR = 0
    _PRE_PROCESSOR_SYNC_VECTOR = 1

    _NESTED_SYNCOPATION_FILTERS = {
        KEEP_HEAVIEST: lambda nested_syncopations: [max(nested_syncopations, key=lambda s: s[0])],
        KEEP_FIRST: lambda nested_syncopations: [min(nested_syncopations, key=lambda s: s[1])],
        KEEP_LAST: lambda nested_syncopations: [max(nested_syncopations, key=lambda s: s[1])],
        KEEP_ALL: lambda nested_syncopations: nested_syncopations
    }

    def __init__(
            self,
            unit: UnitType = Unit.EIGHTH,
            instr_weighting_function: tp.Callable[[str, tp.Set[str]], int] = lambda *_: 0,
            salience_profile_type: str = "equal_upbeats",
            interrupted_syncopations: bool = True,
            nested_syncopations: str = KEEP_HEAVIEST
    ):
        super().__init__(unit)
        self._instr_weighting_f = None     # type: tp.Callable[[str, tp.Set[str]], int]
        self._only_uninterrupted_sync = None      # type: bool
        self._nested_sync_strategy = ""    # type: str

        self.instrumentation_weight_function = instr_weighting_function
        self.salience_profile_type = salience_profile_type
        self.only_uninterrupted_syncopations = interrupted_syncopations
        self.nested_syncopations = nested_syncopations

    @classmethod
    def init_pre_processors(cls):
        mc_bin_vector = MultiChannelMonophonicRhythmFeatureVector(Unit.EIGHTH)
        mc_sync_vector = MultiChannelMonophonicRhythmFeatureVector(Unit.EIGHTH)
        pre_processors = [mc_bin_vector, mc_sync_vector]

        mc_bin_vector.monophonic_extractor = BinaryOnsetVector()
        mc_sync_vector.monophonic_extractor = SyncopationVector()

        assert isinstance(pre_processors[cls._PRE_PROCESSOR_BIN_VECTOR].monophonic_extractor, BinaryOnsetVector)
        assert isinstance(pre_processors[cls._PRE_PROCESSOR_SYNC_VECTOR].monophonic_extractor, SyncopationVector)

        return pre_processors

    def set_unit(self, unit: tp.Optional[UnitType]) -> None:
        if unit is None:
            raise ValueError("PolyphonicSyncopationVector does not support tick-based computation")
        super().set_unit(unit)

    @property
    def salience_profile_type(self) -> str:
        """
        The type of salience profile to be used for syncopation detection. This must be one of: ['hierarchical',
        'equal_upbeats', 'equal_beats']. See :meth:`beatsearch.rhythm.TimeSignature.get_salience_profile` for more info.
        """

        mc_sync_vector = self.get_pre_processors()[self._PRE_PROCESSOR_SYNC_VECTOR]
        sync_vector = mc_sync_vector.monophonic_extractor  # type: SyncopationVector
        return sync_vector.salience_profile_type

    @salience_profile_type.setter
    def salience_profile_type(self, salience_profile_type: str):
        mc_sync_vector = self.get_pre_processors()[self._PRE_PROCESSOR_SYNC_VECTOR]
        sync_vector = mc_sync_vector.monophonic_extractor  # type: SyncopationVector
        sync_vector.salience_profile_type = salience_profile_type

    @property
    def only_uninterrupted_syncopations(self) -> bool:
        """
        Setting this property to True causes this feature extractor to only find uninterrupted syncopations. A
        syncopation is said to be interrupted if an other instrument plays a note during the syncopation. Note that
        setting this property to True will make syncopations containing nested syncopations undetectable, effectively
        ignoring the nested_syncopations property.
        """

        return self._only_uninterrupted_sync

    @only_uninterrupted_syncopations.setter
    def only_uninterrupted_syncopations(self, only_uninterrupted_sync: bool):
        self._only_uninterrupted_sync = only_uninterrupted_sync

    @property
    def nested_syncopations(self) -> str:
        """
        This property determines the way in which nested syncopations are handled. Two syncopations are said to be
        nested if one syncopation starts during the other. Note that if only_uninterrupted_syncopations is set to True,
        there won't be any nested syncopations detected, effectively ignoring this property.

        Nested syncopations can be handled in four different ways:

            - keep_heaviest:  only the syncopation with the highest syncopation strength remains
            - keep_first:     only the first syncopation remains (the less nested syncopation)
            - keep_last:      only the last syncopation remains (the most nested syncopation)
            - keep_all:       all syncopations remain

        Suppose you have a rhythm with three instruments, instrument A, B and C. Then suppose these three nested
        syncopations:

            Instrument A (strength=1): ....:...<|>...
            Instrument B (strength=2): <---:----|>...
            Instrument C (strength=5): ....:<---|>...

            Legend:
                < = syncopated note        - = pending syncopation     > = closing note (end of syncopation)
                | = crotchet pulse         : = quaver pulse

        From these three syncopations:

            - keep_heaviest:  only syncopation C remains
            - keep_first:     only syncopation B remains
            - keep_last:      only syncopation A remains
            - keep_all:       syncopation A, B and C remain
        """

        return self._nested_sync_strategy

    @nested_syncopations.setter
    def nested_syncopations(self, nested_syncopations: str):
        if nested_syncopations not in self._NESTED_SYNCOPATION_FILTERS:
            legal_opts = list(self._NESTED_SYNCOPATION_FILTERS.keys())
            raise ValueError("Unknown nested_syncopations option: %s. "
                             "Please choose between: %s" % (nested_syncopations, legal_opts))

        self._nested_sync_strategy = nested_syncopations

    @property
    def instrumentation_weight_function(self) -> tp.Callable[[str, tp.Set[str]], int]:
        """
        The instrumentation weight function is used to compute the instrumentation weights of the syncopations. This
        function receives two positional parameters:

            syncopated_instrument: the name of the instrument that plays the syncopated note as a string
            closing_instruments:   the names of the instruments which syncopated_instrument is syncopated against (empty
                                   set if the syncopation is against a rest)

        The instrumentation weight function must return the weight as an integer. When this property is set to None, the
        instrumentation weight will equal zero for all syncopations.
        """

        return self._instr_weighting_f

    @instrumentation_weight_function.setter
    def instrumentation_weight_function(self, instr_weighting_function: tp.Callable[[str, tp.Set[str]], int]):
        assert callable(instr_weighting_function)
        self._instr_weighting_f = instr_weighting_function

    def __process__(self, rhythm: PolyphonicRhythm, pre_processor_results: tp.List[tp.Any]):
        """
        Finds the polyphonic syncopations and their syncopation strengths. This is an adaption to the method proposed by
        M. Witek et al in their worked titled "Syncopation, Body-Movement and Pleasure in Groove Music". This method is
        implemented in terms of the monophonic syncopation feature extractor. The monophonic syncopations are found per
        instrument. They are upgraded to polyphonic syncopations by adding an instrumentation weight. The syncopations
        are then filtered based on the properties 'only_uninterrupted_syncopations' and 'nested_syncopations'.

        :param rhythm: rhythm of which to compute the polyphonic syncopation vector
        :return: polyphonic syncopations as a tuple
        """

        binary_onset_vectors = pre_processor_results[self._PRE_PROCESSOR_BIN_VECTOR]  # type: OrderedDict
        mono_sync_vectors = pre_processor_results[self._PRE_PROCESSOR_SYNC_VECTOR]    # type: OrderedDict
        instruments = rhythm.get_track_names()

        # If the rhythm doesn't contain any monophonic syncopations, it won't contain any polyphonic syncopations either
        if all(len(sync_vector) == 0 for sync_vector in mono_sync_vectors):
            return []

        syncopations = []  # type: tp.List[_SyncType]
        only_uninterrupted = self.only_uninterrupted_syncopations
        get_instr_weight = self.instrumentation_weight_function or (lambda *_: 0)

        for curr_instr, sync_vector in mono_sync_vectors.items():
            other_instruments = set(other_instr for other_instr in instruments if other_instr != curr_instr)

            # Iterate over all monophonic syncopations played by the current instrument "upgrade" these to polyphonic
            # syncopations by adding instrumentation weights
            for mono_syncopation in sync_vector:
                mono_sync_strength, note_position, rest_position = mono_syncopation

                # Instruments that play a note on the rest position of the monophonic syncopation, hence "close" the
                # syncopation. Whatever instrument is syncopated is said to be syncopated against these instruments
                sync_closing_instruments = set(
                    other_instr for other_instr in other_instruments
                    if binary_onset_vectors[other_instr][rest_position]
                )

                # Compute the instrumentation weight with the instrumentation weight
                instrumentation_weight = get_instr_weight(curr_instr, sync_closing_instruments)
                sync_strength = mono_sync_strength + instrumentation_weight
                polyphonic_syncopation = sync_strength, note_position, rest_position

                # No need to check if this syncopation is interrupted if we aren't doing anything with it anyway
                if not only_uninterrupted:
                    syncopations.append(polyphonic_syncopation)
                    continue

                # Check if this syncopation is interrupted by any onset (on any other instrument)
                for other_instr in other_instruments:
                    other_instr_bin_vec = binary_onset_vectors[other_instr]
                    # Window of binary onset vector of the other instrument during the syncopations
                    binary_window = (other_instr_bin_vec[i] for i in range(note_position + 1, rest_position))
                    if any(onset for onset in binary_window):
                        is_interrupted = True
                        break
                else:  # yes, you see that right, that is a for-else block right there ;-)
                    is_interrupted = False

                # Add the syncopation only if it is uninterrupted
                if not is_interrupted:
                    syncopations.append(polyphonic_syncopation)

        # Sort the syncopations on position for the retrieval of nested syncopation groups
        syncopations = sorted(syncopations, key=lambda sync: sync[1])
        n_syncopations = len(syncopations)

        # Retrieve the filter for nested syncopation groups depending on the current nested sync strategy
        nested_sync_strat = self._nested_sync_strategy
        nested_sync_filter = self._NESTED_SYNCOPATION_FILTERS[nested_sync_strat]

        # Keep track of the indices of the heading and tailing syncopation of the groups
        group_head_sync_i = 0
        group_tail_sync_i = 0

        # Iterate over "syncopation groups", which are sets of nested syncopations (one starting during another one)
        while group_head_sync_i < n_syncopations:
            group_tail_pos = syncopations[group_head_sync_i][2]

            # Stretch the tail of the group so that it includes all nested syncopations
            # NOTE: This assumes that the syncopations are sorted by their note positions
            for curr_sync_i in range(group_head_sync_i, n_syncopations):
                note_pos, rest_pos = syncopations[curr_sync_i][1:]
                if note_pos > group_tail_pos:
                    break
                group_tail_pos = max(group_tail_pos, rest_pos)
                group_tail_sync_i = curr_sync_i

            # Create a view of the syncopations in this group
            nested_sync_group = TupleView.range_view(syncopations, group_head_sync_i, group_tail_sync_i + 1)

            # Handle the nested syncopations and yield the ones which survive the filter
            for syncopation in nested_sync_filter(nested_sync_group):
                yield syncopation

            # Update the syncopation group bounds
            group_head_sync_i = group_tail_sync_i + 1
            group_tail_sync_i = group_head_sync_i


class PolyphonicSyncopationVectorWitek(PolyphonicRhythmFeatureExtractor):
    def __init__(self, unit: UnitType = Unit.EIGHTH, salience_profile_type: str = "hierarchical"):
        super().__init__(unit)
        self.salience_profile_type = salience_profile_type  # type: str
        self._f_get_instrumentation_weight = None
        # type: tp.Optional[tp.Callable[[tp.Set[str], tp.Set[str]], tp.Union[int, float]]]

    @staticmethod
    def init_pre_processors():
        multi_channel_note_vector = MultiChannelMonophonicRhythmFeatureVector(Unit.EIGHTH)
        multi_channel_note_vector.monophonic_extractor = NoteVector()
        yield multi_channel_note_vector

    def set_unit(self, unit: tp.Optional[UnitType]) -> None:
        if unit is None:
            raise ValueError("PolyphonicSyncopationVector does not support tick-based computation")
        super().set_unit(unit)

    def set_instrumentation_weight_function(
            self,
            func: tp.Union[tp.Callable[[tp.Set[str], tp.Set[str]], tp.Union[int, float]], None]
    ):
        if func and not callable(func):
            raise TypeError

        self._f_get_instrumentation_weight = func or None

    def get_instrumentation_weight_function(self) -> \
            tp.Union[tp.Callable[[tp.Set[str], tp.Set[str]], tp.Union[int, float]], None]:
        return self._f_get_instrumentation_weight

    # noinspection PyUnusedLocal
    @staticmethod
    def default_instrumentation_weight_function(
            syncopated_instruments: tp.Set[str],
            other_instruments: tp.Set[str]
    ) -> int:
        n_other_instruments = len(other_instruments)
        return min(3 - n_other_instruments, 0)

    def __process__(self, rhythm: PolyphonicRhythm, pre_processor_results: tp.List[tp.Any]):
        """
        Finds the polyphonic syncopations and their syncopation strengths. This is an implementation of the method
        proposed by Maria A.G. Witek et al in their work titled "Syncopation, Body-Movement and Pleasure in Groove
        Music".

        The definition of syncopation, as proposed in the work of Witek, goes as follows:

            If N is a note that precedes a rest, R, and R has a metric weight greater than or equal to N, then the
            pair (N, R) is said to constitute a monophonic syncopation. If N is a note on a certain instrument that
            precedes a note on a different instrument, Ndi, and Ndi has a metric weight greater than or equal to N,
            then the pair (N, Ndi) is said to constitute a polyphonic syncopation.

        This definition is used to find the syncopations. Then, the syncopation strengths are computed with this
        formula:

            S = Ndi - N + I

        where S is the degree of syncopation, Ndi is the metrical weight of the note succeeding the syncopated note, N
        is the metrical weight of the syncopated note and I is the instrumentation weight. The instrumentation weight
        depends on the relation of the instruments involved in the syncopation. This relation depends on two factors:

            - the number of instruments involved in the syncopation
            - the type of instruments involved in the syncopation

        As there is no formula given by Witek for how to compute this value, the computation of this value is up to the
        owner of this feature extractor. The function to compute the weight can be set through the
        set_instrumentation_weight_function method and should be a callable receiving two arguments:

            - the names of the tracks (instruments) that play a syncopated note
            - the names of the tracks (instruments) that the note is syncopated against (empty if syncopated against a
              rest)

        The syncopations are returned as a sequence of three-element tuples containing:
            - the degree of syncopation (syncopation strength)
            - position of the syncopated note(s)
            - position of the note(s)/rest(s) against which the note(s) are syncopated

        The syncopations are returned (position, syncopation strength) tuples.

        NOTE: the formula in the Witek's work is different: S = N - Ndi + I. I suspect that it is a typo, as examples
        in the same work show that the formula, S = Ndi - N + I, is used.

        :param rhythm: the polyphonic rhythm of which to compute the polyphonic syncopation vector
        :return: syncopations as (syncopation strength, syncopated event position, other event position) tuples
        """

        Rhythm.Precondition.check_time_signature(rhythm)
        note_vectors = pre_processor_results[0]

        if not note_vectors:
            return

        time_signature = rhythm.get_time_signature()
        salience_profile = time_signature.get_salience_profile(self.unit, kind=self.salience_profile_type)
        get_instrumentation_weight = self._f_get_instrumentation_weight or self.default_instrumentation_weight_function

        # A dictionary containing (instrument, event) tuples by event position
        instrument_event_pairs_by_position = defaultdict(lambda: set())
        for instrument, events in note_vectors.items():
            # NoteVector with ret_positions set to True adds event position as 3rd element (= e[2])
            # NoteVector returns the position
            for e in events:
                instrument_event_pairs_by_position[e[2]].add((instrument, e))

        # Positions that contain an event (either a rest or a note) in ascending order
        dirty_positions = sorted(instrument_event_pairs_by_position.keys())
        assert dirty_positions[0] == 0, "first step should always contain an event (either a rest or a note)"

        # Iterate over the positions that contain events
        for i_position, curr_step in enumerate(dirty_positions):
            next_step = dirty_positions[(i_position + 1) % len(dirty_positions)]
            curr_metrical_weight = salience_profile[curr_step % len(salience_profile)]
            next_metrical_weight = salience_profile[next_step % len(salience_profile)]

            # Syncopation not possible on this current metrical position pair
            if curr_metrical_weight > next_metrical_weight:
                continue

            curr_instrument_event_pairs = instrument_event_pairs_by_position[curr_step]
            next_instrument_event_pairs = instrument_event_pairs_by_position[next_step]

            curr_sounding_inst_set = set(inst for inst, evt in curr_instrument_event_pairs if evt[0] == NoteVector.NOTE)
            next_sounding_inst_set = set(inst for inst, evt in next_instrument_event_pairs if evt[0] == NoteVector.NOTE)

            if len(curr_sounding_inst_set) < 0:
                continue

            # Detect a syncopation if there is at least one instrument that plays a sounding note current position and a
            # rest (or a tied note) at next position
            if any(inst not in next_sounding_inst_set for inst in curr_sounding_inst_set):
                instrumentation_weight = get_instrumentation_weight(curr_sounding_inst_set, next_sounding_inst_set)
                syncopation_degree = next_metrical_weight - curr_metrical_weight + instrumentation_weight
                yield syncopation_degree, curr_step, next_step


class PolyphonicTensionVector(PolyphonicRhythmFeatureExtractor):
    """
    This feature extractor computes the monophonic tension vector per track. It multiplies the monophonic tensions with
    instrument weights and then returns the sum. Instrument weights can be set with
    :meth:`beatsearch.feature_extraction.PolyphonicTensionVector.set_instrument_weights`.
    """

    def __init__(self, unit: UnitType = Unit.EIGHTH,
                 salience_profile_type: str = "hierarchical", normalize: bool = False):
        super().__init__(unit)
        self._instr_weights = dict()
        self.salience_profile_type = salience_profile_type  # type: str
        self.normalize = normalize

    @staticmethod
    def init_pre_processors():
        mc_tension_vec = MultiChannelMonophonicRhythmFeatureVector(Unit.EIGHTH)
        mc_tension_vec.monophonic_extractor = MonophonicTensionVector()
        yield mc_tension_vec

    @property
    def salience_profile_type(self) -> str:
        mc_tension_vec = self.get_pre_processors()[0]           # type: MultiChannelMonophonicRhythmFeatureVector
        mono_tension_vec = mc_tension_vec.monophonic_extractor  # type: MonophonicTensionVector
        return mono_tension_vec.salience_profile_type

    @salience_profile_type.setter
    def salience_profile_type(self, salience_profile_type: str):
        mc_tension_vec = self.get_pre_processors()[0]           # type: MultiChannelMonophonicRhythmFeatureVector
        mono_tension_vec = mc_tension_vec.monophonic_extractor  # type: MonophonicTensionVector
        mono_tension_vec.salience_profile_type = salience_profile_type

    @property
    def normalize(self):
        """When set to True, the tension will have a range of [0, 1]"""
        mc_tension_vec = self.get_pre_processors()[0]           # type: MultiChannelMonophonicRhythmFeatureVector
        mono_tension_vec = mc_tension_vec.monophonic_extractor  # type: MonophonicTensionVector
        return mono_tension_vec.normalize

    @normalize.setter
    def normalize(self, normalize: bool):
        mc_tension_vec = self.get_pre_processors()[0]           # type: MultiChannelMonophonicRhythmFeatureVector
        mono_tension_vec = mc_tension_vec.monophonic_extractor  # type: MonophonicTensionVector
        mono_tension_vec.normalize = normalize

    def set_instrument_weights(self, instrument_weights: tp.Dict[str, float]):
        """
        Sets the instrument weights. This will clear the previous weights. The instrument weights must be given as a
        dictionary containing the weights by instrument name. The weights must be given as a float (or something
        interpretable as a float). The instrument names must be given as a strings.

        :param instrument_weights: dictionary containing weights by instrument name
        :return: None
        """

        self._instr_weights.clear()

        for instr, weight in instrument_weights.items():
            instr = str(instr)
            weight = float(weight)
            self._instr_weights[instr] = weight

    def get_instrument_weights(self) -> tp.Dict[str, float]:
        """
        Returns a new dictionary containing the weights per instrument. Weights are floating point numbers between 0
        and 1. Instruments are instrument names (track names).

        :return: dictionary containing weights per instrument
        """

        return {**self._instr_weights}

    def set_unit(self, unit: tp.Optional[UnitType]) -> None:
        if unit is None:
            raise ValueError("PolyphonicTensionVector does not support tick-based computation")
        super().set_unit(unit)

    def __process__(self, rhythm: PolyphonicRhythm, pre_processor_results: tp.List[tp.Any]):
        weights = self.get_instrument_weights()
        mono_tension_vectors = pre_processor_results[0]  # type: tp.Dict[str, tp.Sequence[int]]
        default_weight = 1.0 / rhythm.get_track_count()
        tension_vectors_scaled = []

        for instr, tension_vec in mono_tension_vectors.items():
            try:
                w = weights[instr]
            except KeyError:
                LOGGER.warning("Weight unknown for: %s (defaulting to %.2f)" % (instr, default_weight))

                # We add the default weight to the weights dictionary for normalization. Note that we aren't actually
                # affecting the weights of this extractor because get_instrument_weights returns a hard copy of the
                # weights dictionary
                w = weights[instr] = default_weight

            scaled_tension_vec = tuple(t * w for t in tension_vec)
            tension_vectors_scaled.append(scaled_tension_vec)

        if self.normalize:
            # We can assume that the max mono tension = 1.0 (because it's normalized)
            normalizer = 1.0 / sum(weights.values())
        else:
            normalizer = 1.0

        return tuple(sum(col) * normalizer for col in zip(*tension_vectors_scaled))


class PolyphonicTension(PolyphonicRhythmFeatureExtractor):
    # TODO Add documentation

    def __init__(self, unit: UnitType = Unit.EIGHTH, salience_profile_type: str = "hierarchical"):
        super().__init__(unit)
        self._instr_weights = dict()
        self.salience_profile_type = salience_profile_type  # type: str

    @staticmethod
    def init_pre_processors():
        tension_vec = PolyphonicTensionVector(normalize=True)
        yield tension_vec

    def set_instrument_weights(self, weights: tp.Dict[str, float]):
        tension_vec = self.get_pre_processors()[0]  # type: PolyphonicTensionVector
        tension_vec.set_instrument_weights(weights)

    def get_instrument_weights(self) -> tp.Dict[str, float]:
        tension_vec = self.get_pre_processors()[0]  # type: PolyphonicTensionVector
        return tension_vec.get_instrument_weights()

    def set_unit(self, unit: tp.Optional[UnitType]) -> None:
        if unit is None:
            raise ValueError("PolyphonicTension does not support tick-based computation")
        super().set_unit(unit)

    def __process__(self, rhythm: PolyphonicRhythm, pre_processor_results: tp.List[tp.Any]):
        tension_vec = pre_processor_results[0]
        return math.sqrt(sum(t * t for t in tension_vec))


__all__ = [
    # Feature extractor abstract base classes (or interfaces)
    'FeatureExtractor', 'RhythmFeatureExtractor',
    'MonophonicRhythmFeatureExtractor', 'PolyphonicRhythmFeatureExtractor',

    # Monophonic rhythm feature extractor implementations
    'BinaryOnsetVector', 'NoteVector', 'IOIVector', 'IOIHistogram', 'BinarySchillingerChain', 'ChronotonicChain',
    'IOIDifferenceVector', 'OnsetPositionVector', 'SyncopationVector', 'SyncopatedOnsetRatio',
    'MeanSyncopationStrength', 'OnsetDensity', 'MonophonicTensionVector',

    # Polyphonic rhythm feature extractor implementations
    'MultiChannelMonophonicRhythmFeatureVector', 'PolyphonicSyncopationVector', 'PolyphonicSyncopationVectorWitek',
    'PolyphonicTensionVector'
]

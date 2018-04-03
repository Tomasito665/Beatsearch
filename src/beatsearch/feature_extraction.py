import enum
import itertools
import numpy as np
import typing as tp
from fractions import Fraction
from abc import ABCMeta, abstractmethod
from beatsearch.utils import Quantizable
from beatsearch.rhythm import Rhythm, MonophonicRhythm, PolyphonicRhythm, \
    Unit, UnitType, parse_unit_argument, convert_tick


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

    def process(self, rhythm: Rhythm) -> object:
        """Computes and returns a rhythm feature

        :param rhythm: rhythm of which to compute the feature
        :return: the rhythm feature value
        """

        pre_processor_results = list(pre_processor.process(rhythm) for pre_processor in self.get_pre_processors())
        return self.__process__(rhythm, pre_processor_results)

    @abstractmethod
    def __process__(self, rhythm: Rhythm, pre_processor_results: tp.List[tp.Any]):
        """Computes and returns a rhythm feature

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
            unit: tp.Union[int, str],
            *pre_processor_results: tp.List[tp.Any]
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


class IOIVector(MonophonicRhythmFeatureExtractor, QuantizableRhythmFeatureExtractorMixin):
    class Mode(enum.Enum):
        PRE_NOTE = enum.auto()
        POST_NOTE = enum.auto()

    # noinspection PyShadowingBuiltins
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


class SyncopationVector(MonophonicRhythmFeatureExtractor):
    def __init__(self, unit: UnitType = Unit.EIGHTH, equal_upbeat_salience_profile: bool = False):
        super().__init__(unit)
        self.equal_upbeat_salience_profile = equal_upbeat_salience_profile

    @staticmethod
    def init_pre_processors():
        yield BinaryOnsetVector()

    def set_unit(self, unit: tp.Optional[UnitType]) -> None:
        if unit is None:
            raise ValueError("SyncopationVector does not support tick-based computation")
        super().set_unit(unit)

    def __process__(self, rhythm: MonophonicRhythm, pre_processor_results: tp.List[tp.Any]):
        """
        Extracts the syncopations from the given monophonic rhythm. The syncopations are computed with the method
        proposed by H.C. Longuet-Higgins and C. S. Lee in their work titled: "The Rhythmic Interpretation of
        Monophonic Music".

        The syncopations are returned as two-dimensional tuples. If the given rhythm contains N syncopations, the
        returned tuples will have a length of N. Each element of the returned tuple contains two elements:
            - the position of the syncopation in the binary onset vector of the rhythm
            - the syncopation strength

        :param rhythm: rhythm from which to extract the syncopations
        :return: a two-dimensional tuple containing the onset positions in the first dimension and the syncopation
                 strengths in the second dimension
        """

        Rhythm.Precondition.check_time_signature(rhythm)

        binary_vector = pre_processor_results[0]
        time_signature = rhythm.get_time_signature()
        metrical_weights = time_signature.get_salience_profile(
            self.unit, equal_upbeats=self.equal_upbeat_salience_profile)

        def get_syncopations():
            for step, curr_step_is_onset in enumerate(binary_vector):
                next_step = (step + 1) % len(binary_vector)
                next_step_is_onset = binary_vector[next_step]

                # iterate only over note-rest pairs
                if not curr_step_is_onset or next_step_is_onset:
                    continue

                note_weight = metrical_weights[step]
                rest_weight = metrical_weights[next_step]

                if note_weight <= rest_weight:
                    syncopation_strength = rest_weight - note_weight
                    yield step, syncopation_strength

        return tuple(get_syncopations())


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


__all__ = [
    # Feature extractor abstract base classes (or interfaces)
    'FeatureExtractor', 'RhythmFeatureExtractor',
    'MonophonicRhythmFeatureExtractor', 'PolyphonicRhythmFeatureExtractor',

    # Monophonic rhythm feature extractor implementations
    'BinaryOnsetVector', 'IOIVector', 'IOIHistogram', 'BinarySchillingerChain', 'ChronotonicChain',
    'IOIDifferenceVector', 'OnsetPositionVector', 'SyncopationVector', 'SyncopatedOnsetRatio',
    'MeanSyncopationStrength', 'OnsetDensity'
]

import enum
import math
import itertools
import numpy as np
import typing as tp
from abc import ABCMeta, abstractmethod
from beatsearch.utils import Quantizable
from beatsearch.rhythm import Rhythm, MonophonicRhythm, PolyphonicRhythm, Unit, convert_time

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
    def get_unit(self) -> str:
        """Returns the musical unit of this preprocessor

        :return: musical unit of this preprocessor as a string
        """

        raise NotImplementedError

    @abstractmethod
    def set_unit(self, unit: str) -> None:
        """Sets the musical unit of this preprocessor

        :return:
        """

        raise NotImplementedError

    def process(self, rhythm: Rhythm) -> tp.Any:
        """Computes and returns a rhythm feature
        Computes a feature of the given rhythm and returns it.

        :param rhythm: rhythm of which to compute the feature
        :return: the rhythm feature value
        """

        unit = self.get_unit()
        concrete_unit = rhythm.get_resolution() if unit == "ticks" else unit
        pre_processor_results = list(pre_processor.process(rhythm) for pre_processor in self.get_pre_processors())
        return self.__process__(rhythm, concrete_unit, pre_processor_results)

    @abstractmethod
    def __process__(self, rhythm: Rhythm, unit: tp.Union[int, str], pre_processor_results: tp.List[tp.Any]):
        """Computes and returns a rhythm feature
        Computes a feature of the given rhythm and returns it.

        :param rhythm: rhythm of which to compute the feature
        :param unit: concrete unit ("ticks" unit will be converted to rhythm's resolution)
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
    def unit(self) -> str:
        """See get_unit and set_unit"""
        return self.get_unit()

    @unit.setter
    def unit(self, unit: str):
        self.set_unit(unit)


class RhythmFeatureExtractorBase(RhythmFeatureExtractor, metaclass=ABCMeta):
    def __init__(self, unit="eighths"):
        self._pre_processors = tuple(self.init_pre_processors())  # type: tp.Tuple[RhythmFeatureExtractorBase, ...]
        self._unit = None
        self.unit = unit  # calling setter, which also sets the unit of the preprocessors

    def get_pre_processors(self) -> tp.Tuple[RhythmFeatureExtractor, ...]:
        """Returns the preprocessors, whose result is passed to __process__.

        Preprocessors are also rhythm feature extractors. Their unit is linked to this extractor's unit.
        """

        return self._pre_processors

    def get_unit(self):
        """Returns the time unit of this feature extractor"""
        return self._unit

    def set_unit(self, unit: str):
        """Sets the time unit of this feature extractor

        Sets the time unit of this rhythm feature extractor. This method will also set the unit of the preprocessors.

        :param unit: new time unit as a string
        :return: None
        """

        unit = str(unit)

        if not Unit.exists(unit) and unit != "ticks":
            raise Unit.UnknownTimeUnit(unit)

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
    def __process__(
            self,
            rhythm: MonophonicRhythm,
            unit: tp.Union[int, str],
            pre_processor_results: tp.List[tp.Any]
    ):
        """Computes and returns a monophonic rhythm feature
        Computes a feature of the given monophonic rhythm and returns it.

        :param rhythm: monophonic rhythm of which to compute the feature
        :param unit: concrete time unit
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


#####################################
# Feature extractor implementations #
#####################################


class BinaryOnsetVector(MonophonicRhythmFeatureExtractor):
    def __process__(self, rhythm: MonophonicRhythm, unit: tp.Union[int, str], *_):
        """
        Returns the binary representation of the note onsets of the given rhythm where each step where a note onset
        happens is denoted with a 1; otherwise with a 0. The given resolution is the resolution in PPQ (pulses per
        quarter note) of the binary vector.

        :param rhythm: the monophonic rhythm from which to compute the binary onset vector
        :return: the binary representation of the note onsets in the given rhythm
        """

        Rhythm.Precondition.check_resolution(rhythm)

        resolution = rhythm.get_resolution()
        duration = rhythm.get_duration(unit)
        binary_string = [0] * int(math.ceil(duration))

        for onset in rhythm.get_onsets():
            pulse = convert_time(onset.tick, resolution, unit, quantize=True)
            try:
                binary_string[pulse] = 1
            except IndexError:
                pass  # when quantization pushes note after end of rhythm

        return binary_string


class IOIVector(MonophonicRhythmFeatureExtractor, QuantizableRhythmFeatureExtractorMixin):
    class Mode(enum.Enum):
        PRE_NOTE = enum.auto()
        POST_NOTE = enum.auto()

    # noinspection PyShadowingBuiltins
    def __init__(self, unit="eighths", mode: Mode = Mode.POST_NOTE, quantize_enabled=True):
        MonophonicRhythmFeatureExtractor.__init__(self, unit)
        QuantizableRhythmFeatureExtractorMixin.__init__(self, quantize_enabled)
        self._mode = None
        self.mode = mode  # call setter

    @property
    def mode(self):
        return self._mode

    # noinspection PyShadowingBuiltins
    @mode.setter
    def mode(self, mode: Mode):
        self._mode = mode

    def __process__(
            self,
            rhythm: MonophonicRhythm,
            unit: tp.Union[int, str],
            pre_processor_results: tp.List[tp.Any]
    ):
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
            return self._pre_note_ioi(rhythm, unit)
        elif self.mode == self.Mode.POST_NOTE:
            return self._post_note_ioi(rhythm, unit)
        else:
            raise RuntimeError("Unknown mode: \"%s\"" % self.mode)

    def _pre_note_ioi(self, rhythm: MonophonicRhythm, unit: tp.Union[int, str]):
        resolution = rhythm.get_resolution()
        current_tick = 0
        intervals = []
        quantize = self.is_quantize_enabled()

        for onset in rhythm.get_onsets():
            delta_tick = onset.tick - current_tick
            intervals.append(convert_time(delta_tick, resolution, unit, quantize=quantize))
            current_tick = onset.tick

        return intervals

    # TODO Add cyclic option to include the offset in the last onset's interval
    def _post_note_ioi(self, rhythm: MonophonicRhythm, unit: tp.Union[int, str]):
        intervals = []
        onset_positions = itertools.chain((onset.tick for onset in rhythm.get_onsets()), [rhythm.duration_in_ticks])
        last_onset_tick = -1
        quantize = self.is_quantize_enabled()

        for onset_tick in onset_positions:
            if last_onset_tick < 0:
                last_onset_tick = onset_tick
                continue

            delta_in_ticks = onset_tick - last_onset_tick
            delta_in_units = convert_time(delta_in_ticks, rhythm.resolution, unit, quantize=quantize)

            intervals.append(delta_in_units)
            last_onset_tick = onset_tick

        return intervals


class IOIHistogram(MonophonicRhythmFeatureExtractor):
    def __init__(self, unit="eighths"):
        super().__init__(unit)

    @staticmethod
    def init_pre_processors():
        return [IOIVector(mode=IOIVector.Mode.POST_NOTE, quantize_enabled=True)]

    def __process__(
            self,
            rhythm: MonophonicRhythm,
            unit: tp.Union[int, str],
            pre_processor_results: tp.List[tp.Any]
    ):
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
    def __init__(self, unit="eighths", values=(1, 0)):
        super().__init__(unit)
        self._values = None
        self.values = values  # calls setter

    @staticmethod
    def init_pre_processors():
        yield BinaryOnsetVector()

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        values = iter(values)
        self._values = next(values), next(values)

    def __process__(
            self,
            rhythm: MonophonicRhythm,
            unit: tp.Union[int, str],
            pre_processor_results: tp.List[tp.Any]
    ):
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

    def __process__(
            self,
            rhythm: MonophonicRhythm,
            unit: tp.Union[int, str],
            pre_processor_results: tp.List[tp.Any]
    ):
        """
        Returns the chronotonic chain representation of the given rhythm.

        For example, given the Rumba Clave rhythm:
          X--X---X--X-X---
          3334444333224444

        :param unit: unit
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
    def __init__(self, unit="eighths", quantize_enabled=True, cyclic=True):
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

    def __process__(
            self,
            rhythm: MonophonicRhythm,
            unit: tp.Union[int, str],
            pre_processor_results: tp.List[tp.Any]
    ):
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
        :param unit: time unit
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
    def __init__(self, unit="eighths", quantize_enabled=True):
        MonophonicRhythmFeatureExtractor.__init__(self, unit)
        QuantizableRhythmFeatureExtractorMixin.__init__(self, quantize_enabled)

    def __process__(
            self,
            rhythm: MonophonicRhythm,
            unit: tp.Union[int, str],
            pre_processor_results: tp.List[tp.Any]
    ):
        """
        Returns the absolute onset times of the notes in the given rhythm.

        :param unit: the unit of the onset times
        :return: a list with the onset times of this rhythm
        """

        Rhythm.Precondition.check_resolution(rhythm)
        resolution = rhythm.get_resolution()
        quantize = self.is_quantize_enabled()
        return [convert_time(onset[0], resolution, unit, quantize) for onset in rhythm.get_onsets()]


class SyncopationVector(MonophonicRhythmFeatureExtractor):
    @staticmethod
    def init_pre_processors():
        yield BinaryOnsetVector()

    def set_unit(self, unit: str):
        if unit == "ticks":
            raise ValueError("SyncopationVector only supports musical units, not ticks")
        super().set_unit(unit)

    def __process__(
            self,
            rhythm: MonophonicRhythm,
            unit: tp.Union[int, str],
            pre_processor_results: tp.List[tp.Any]
    ):
        """
        Extracts the syncopations from the given monophonic rhythm. The syncopations are computed with the method
        proposed by H.C. Longuet-Higgins and C. S. Lee in their work titled: "The Rhythmic Interpretation of
        Monophonic Music".

        The syncopations are returned as a two-dimensional tuple. If the given rhythm contains N syncopations, the
        returned tuple will have a length of N. Each element of the returned tuple contains two elements:
            - the position of the syncopation in the binary onset vector of the rhythm
            - the syncopation strength

        :param rhythm: rhythm to extract the syncopations from
        :param unit: time unit
        :return: a two-dimensional tuple containing the onset positions in the first dimension and the syncopation
                 strengths in the second dimension
        """

        Rhythm.Precondition.check_time_signature(rhythm)

        binary_vector = pre_processor_results[0]
        time_signature = rhythm.get_time_signature()
        metrical_weights = time_signature.get_metrical_weights(unit)

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


class OnsetDensity(MonophonicRhythmFeatureExtractor):
    def __init__(self, unit="eighths"):
        super().__init__(unit)

    @staticmethod
    def init_pre_processors():
        yield BinaryOnsetVector()

    def __process__(
            self,
            rhythm: MonophonicRhythm,
            unit: tp.Union[int, str],
            pre_processor_results: tp.List[tp.Any]
    ):
        """
        Computes the onset density of the given rhythm. The onset density is the number of onsets over the number of
        positions in the binary onset vector of the given rhythm.

        :param rhythm: monophonic rhythm to compute the onset density of
        :param unit: time unit
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
    'IOIDifferenceVector', 'OnsetPositionVector', 'SyncopationVector', 'OnsetDensity'
]

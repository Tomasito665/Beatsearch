import enum
import math
import itertools
import numpy as np
import typing as tp
from abc import ABCMeta, abstractmethod
from beatsearch.rhythm import Rhythm, MonophonicRhythm, PolyphonicRhythm, Unit, convert_time

######################################
# Feature extractor abstract classes #
######################################


class FeatureExtractor(object, metaclass=ABCMeta):
    @abstractmethod
    def process(self, obj):
        """Computes and returns a feature
        Computes a feature of the given object and returns it.

        :param obj: object of which to compute the feature
        :return: the feature value
        """

        raise NotImplementedError


class RhythmFeatureExtractor(FeatureExtractor, metaclass=ABCMeta):
    def __init__(self, unit="eighths"):
        self._unit = None
        self.unit = unit

    def get_unit(self):
        """Returns the time unit of this feature extractor"""
        return self._unit

    def set_unit(self, unit: str):
        """Sets the time unit of this feature extractor

        :param unit: new time unit as a string
        :return: None
        """

        unit = str(unit)
        if not Unit.exists(unit) and unit != "ticks":
            raise Unit.UnknownTimeUnit(unit)
        self._unit = unit

    @property
    def unit(self):
        """The time unit of this feature extractor

        This time unit will be used when computing the feature.
        """

        return self.get_unit()

    @unit.setter
    def unit(self, unit: str):  # setter
        self.set_unit(unit)

    def process(self, rhythm: Rhythm):
        """Computes and returns a rhythm feature
        Computes a feature of the given rhythm and returns it.

        :param rhythm: rhythm of which to compute the feature
        :return: the rhythm feature value
        """

        unit = self._unit
        return self.__process__(rhythm, unit if unit != "ticks" else rhythm.get_resolution())

    @abstractmethod
    def __process__(self, rhythm: Rhythm, unit: tp.Union[int, str]):
        """Computes and returns a rhythm feature
        Computes a feature of the given rhythm and returns it.

        :param rhythm: rhythm of which to compute the feature
        :param unit: concrete unit ("ticks" unit will be converted to rhythm's resolution)
        :return: the rhythm feature value
        """

        raise NotImplementedError


class MonophonicRhythmFeatureExtractor(RhythmFeatureExtractor, metaclass=ABCMeta):
    @abstractmethod
    def __process__(self, rhythm: MonophonicRhythm, unit: tp.Union[int, str]):
        """Computes and returns a monophonic rhythm feature
        Computes a feature of the given monophonic rhythm and returns it.

        :param rhythm: monophonic rhythm of which to compute the feature
        :param unit: concrete time unit
        :return: the monophonic rhythm feature value
        """

        raise NotImplementedError


class PolyphonicRhythmFeatureExtractor(RhythmFeatureExtractor, metaclass=ABCMeta):
    @abstractmethod
    def __process__(self, rhythm: PolyphonicRhythm, unit: tp.Union[int, str]):
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
    def __process__(self, rhythm: MonophonicRhythm, unit: tp.Union[int, str]):
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


class IOIVector(MonophonicRhythmFeatureExtractor):
    class Mode(enum.Enum):
        PRE_NOTE = enum.auto()
        POST_NOTE = enum.auto()

    # noinspection PyShadowingBuiltins
    def __init__(self, unit="eighths", mode: Mode = Mode.POST_NOTE, quantize=True):
        super().__init__(unit)
        self._mode = None
        self.quantize = quantize
        self.mode = mode

    @property
    def mode(self):
        return self._mode

    # noinspection PyShadowingBuiltins
    @mode.setter
    def mode(self, mode: Mode):
        self._mode = mode

    def __process__(self, rhythm: MonophonicRhythm, unit: tp.Union[int, str]):
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

        for onset in rhythm.get_onsets():
            delta_tick = onset.tick - current_tick
            intervals.append(convert_time(delta_tick, resolution, unit, quantize=self.quantize))
            current_tick = onset.tick

        return intervals

    # TODO Add cyclic option to include the offset in the last onset's interval
    def _post_note_ioi(self, rhythm: MonophonicRhythm, unit: tp.Union[int, str]):
        intervals = []
        onset_positions = itertools.chain((onset.tick for onset in rhythm.get_onsets()), [rhythm.duration_in_ticks])
        last_onset_tick = -1

        for onset_tick in onset_positions:
            if last_onset_tick < 0:
                last_onset_tick = onset_tick
                continue

            delta_in_ticks = onset_tick - last_onset_tick
            delta_in_units = convert_time(delta_in_ticks, rhythm.resolution, unit, quantize=self.quantize)

            intervals.append(delta_in_units)
            last_onset_tick = onset_tick

        return intervals


class IOIHistogram(MonophonicRhythmFeatureExtractor):
    def __init__(self, unit="eighths"):
        self._ioi_vector_extractor = IOIVector(unit, mode=IOIVector.Mode.POST_NOTE)
        self._ioi_vector_extractor.quantize = True
        super().__init__(unit)  # NOTE: super constructor must be called after self._ioi_vector_extractor definition

    def set_unit(self, unit: str):
        super().set_unit(unit)
        self._ioi_vector_extractor.set_unit(unit)

    def __process__(self, rhythm: MonophonicRhythm, unit: tp.Union[int, str]):
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

        ioi_vector_extractor = self._ioi_vector_extractor
        assert ioi_vector_extractor.unit == self.unit

        ioi_vector = ioi_vector_extractor.process(rhythm)
        histogram = np.histogram(ioi_vector, tuple(range(min(ioi_vector), max(ioi_vector) + 2)))
        occurrences = histogram[0].tolist()
        bins = histogram[1].tolist()[:-1]
        return occurrences, bins


class BinarySchillingerChain(MonophonicRhythmFeatureExtractor):
    def __init__(self, unit="eighths", values=(1, 0)):
        self._values = None
        self._binary_vector_extractor = BinaryOnsetVector(unit)
        self.values = values
        super().__init__(unit)  # super constructor must be called after self._binary_vector_extractor definition

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        values = iter(values)
        self._values = next(values), next(values)

    def set_unit(self, unit: str):
        self._binary_vector_extractor.unit = unit
        super().set_unit(unit)

    def __process__(self, rhythm: MonophonicRhythm, unit: tp.Union[int, str]):
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

        binary_vector_extractor = self._binary_vector_extractor
        assert binary_vector_extractor.unit == self.unit

        values = self._values
        chain = self._binary_vector_extractor.process(rhythm)
        i, value_i = 0, 1

        while i < len(chain):
            if chain[i] == 1:
                value_i = 1 - value_i
            chain[i] = values[value_i]
            i += 1

        return chain


class ChronotonicChain(MonophonicRhythmFeatureExtractor):
    def __init__(self, unit="eighths"):
        super().__init__(unit)
        self._binary_vector_extractor = BinaryOnsetVector(unit)

    def __process__(self, rhythm: MonophonicRhythm, unit: tp.Union[int, str]):
        """
        Returns the chronotonic chain representation of the given rhythm.

        For example, given the Rumba Clave rhythm:
          X--X---X--X-X---
          3334444333224444

        :param unit: unit
        :return: the chronotonic chain as a list
        """

        binary_vector_extractor = self._binary_vector_extractor
        assert binary_vector_extractor.unit == self.unit

        chain = binary_vector_extractor.process(rhythm)
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


class IOIDifferenceVector(MonophonicRhythmFeatureExtractor):
    def __init__(self, unit="eighths", quantize=True, cyclic=True):
        self._ioi_vector_extractor = IOIVector(unit, mode=IOIVector.Mode.POST_NOTE, quantize=quantize)
        super().__init__(unit)  # super constructor must be called after self._ioi_vector_extractor definition
        self._cyclic = None
        # calling setters
        self.cyclic = cyclic
        self.quantize = quantize

    @property
    def quantize(self):
        return self._ioi_vector_extractor.quantize

    @quantize.setter
    def quantize(self, quantize):
        self._ioi_vector_extractor.quantize = bool(quantize)

    @property
    def cyclic(self):
        """Cyclic behaviour, see process documentation"""
        return self._cyclic

    @cyclic.setter
    def cyclic(self, cyclic):
        self._cyclic = bool(cyclic)

    def set_unit(self, unit: str):
        super().set_unit(unit)
        self._ioi_vector_extractor.set_unit(unit)

    def __process__(self, rhythm: MonophonicRhythm, unit: tp.Union[int, str]):
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

        ioi_vector_extractor = self._ioi_vector_extractor
        assert ioi_vector_extractor.unit == self.unit
        assert ioi_vector_extractor.quantize == self.quantize

        vector, i = ioi_vector_extractor.process(rhythm), 0

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


class OnsetPositionVector(MonophonicRhythmFeatureExtractor):
    def __init__(self, unit="eighths", quantize=True):
        super().__init__(unit)
        self.quantize = quantize

    def __process__(self, rhythm: MonophonicRhythm, unit: tp.Union[int, str]):
        """
        Returns the absolute onset times of the notes in the given rhythm.

        :param unit: the unit of the onset times
        :return: a list with the onset times of this rhythm
        """

        Rhythm.Precondition.check_resolution(rhythm)
        resolution = rhythm.get_resolution()
        quantize = self.quantize
        return [convert_time(onset[0], resolution, unit, quantize) for onset in rhythm.get_onsets()]


class SyncopationVector(MonophonicRhythmFeatureExtractor):
    def __init__(self, unit="eighths"):
        self._binary_vector_extractor = BinaryOnsetVector(unit)
        super().__init__(unit)  # super constructor must be called after self._binary_vector_extractor definition

    def set_unit(self, unit: str):
        if unit == "ticks":
            raise ValueError("SyncopationVector only supports musical units, not ticks")
        super().set_unit(unit)
        self._binary_vector_extractor.set_unit(unit)

    def __process__(self, rhythm: MonophonicRhythm, unit: tp.Union[int, str]):
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

        assert self._binary_vector_extractor.unit == self.unit
        binary_vector = self._binary_vector_extractor.process(rhythm)
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

import inspect
import math
import typing as tp
from collections import OrderedDict
from abc import abstractmethod, ABCMeta
from beatsearch.rhythm import MonophonicRhythm, MonophonicRhythmImpl, PolyphonicRhythm, Unit
from beatsearch.feature_extraction import BinaryOnsetVector, IOIVector, \
    IOIDifferenceVector, OnsetPositionVector, ChronotonicChain
from beatsearch.utils import friendly_named_class, Quantizable


class DistanceMeasure(object):
    """Abstract base class for distance measures"""

    def get_distance(self, obj_a, obj_b):
        raise NotImplementedError


class MonophonicRhythmDistanceMeasure(DistanceMeasure, metaclass=ABCMeta):
    """Abstract base class for monophonic rhythm distance measures

    This is an abstract base class for monophonic rhythm distance measures. It measures the distance
    between two MonophonicRhythmImpl objects.
    """

    LENGTH_POLICIES = ["exact", "multiple", "fill"]
    DEFAULT_LENGTH_POLICY = LENGTH_POLICIES[0]

    class UnknownLengthPolicy(Exception):
        def __init__(self, given_length_policy):
            super(MonophonicRhythmDistanceMeasure.UnknownLengthPolicy, self).__init__(
                "Given %s, please choose between: %s" % (
                    given_length_policy, MonophonicRhythmDistanceMeasure.LENGTH_POLICIES))

    class LengthPolicyNotMet(Exception):
        MESSAGES = {
            'exact': "When length policy is set to \"exact\", both iterables should have the same number of elements",
            'multiple': "When length policy is set to \"multiple\", the length of the largest "
                        "iterable should be a multiple of all the other iterable lengths"
        }

        def __init__(self, length_policy):
            msg = self.MESSAGES[length_policy]
            super().__init__(msg)

    def __init__(self, unit="ticks", length_policy=DEFAULT_LENGTH_POLICY):
        """
        Creates a new monophonic rhythm distance measure.

        :param unit: see internal_unit property
        :param length_policy: see documentation on length_policy property
        """

        self._len_policy = ''
        self._unit = 0

        self.length_policy = length_policy
        self.set_unit(unit)

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
        valid_policies = self.LENGTH_POLICIES
        if length_policy not in valid_policies:
            raise self.UnknownLengthPolicy(length_policy)
        self._len_policy = length_policy

    def set_unit(self, unit: str):
        """Sets the time unit

        This time unit is used internally for distance computation..

        :param unit: new time unit as a string
        :return: None
        """

        unit = str(unit)
        if not Unit.exists(unit) and unit != "ticks":
            raise Unit.UnknownTimeUnit(unit)
        self._unit = unit

    def get_unit(self):
        """Returns the time unit

        This time unit is used internally for distance computation.

        :return: time unit as a string
        """

        return self._unit

    @property
    def unit(self):
        """
        Unit used internally for distance computation.
        """

        return self.get_unit()

    @unit.setter
    def unit(self, unit):
        self.set_unit(unit)

    def get_distance(self, rhythm_a: MonophonicRhythm, rhythm_b: MonophonicRhythm):
        """
        Returns the distance between the given tracks.

        :param rhythm_a: monophonic rhythm to compare to monophonic rhythm b
        :param rhythm_b: monophonic rhythm to compare to monophonic rhythm a
        :return: distance between the given monophonic rhythms
        """

        unit_in_ticks = self._unit == "ticks"
        res_a, res_b = rhythm_a.get_resolution(), rhythm_b.get_resolution()

        if unit_in_ticks and res_a != res_b:
            raise ValueError("%s unit set to 'ticks', but given rhythms have "
                             "different resolutions (%i != %i)" %
                             ("Internal" if unit_in_ticks else "Output", res_a, res_b))

        rhythms = [rhythm_a, rhythm_b]
        iterables = [self.__get_iterable__(t) for t in rhythms]
        cookies = [self.__get_cookie__(t) for t in rhythms]

        if not self.__class__.check_if_iterables_meet_len_policy(self.length_policy, *iterables):
            raise self.LengthPolicyNotMet(self.length_policy)

        return self.__compute_distance__(max(len(i) for i in iterables), *(iterables + cookies))

    @abstractmethod
    def __get_iterable__(self, rhythm: MonophonicRhythm):
        """
        Should prepare and return the rhythm representation on which the similarity measure is based. The returned
        vector's will be length policy checked. The returned vector should be in the current unit.

        :param rhythm: the monophonic rhythm
        :return: desired rhythm representation to use in __compute_distance__
        """

        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def __get_cookie__(self, rhythm: MonophonicRhythm):
        """
        The result of this method will be passed to __compute_distance__, both for rhythm a and rhythm b. By default,
        the cookie is the rhythm itself.

        :param rhythm: the monophonic rhythm
        :return: cookie to use in __compute_distance__
        """

        return rhythm

    @staticmethod  # TODO enforce python 3.3
    @abstractmethod
    def __compute_distance__(max_len, iterable_a, iterable_b, cookie_a, cookie_b):
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

    @classmethod
    def check_if_iterables_meet_len_policy(cls, length_policy, iterable_a, iterable_b):
        """
        Checks whether the given iterables meet a certain duration policy. Duration policies are:
            'exact'     - met when all iterables have the exact same length and are not empty
            'multiple'  - met when the length of the largest iterable is a multiple of all the other iterable lengths
            'fill'      - met when any of the chains is empty

        :param length_policy: one of {'exact', 'multiple' or 'fill'}
        :param iterable_a: iterable
        :param iterable_b: iterable
        :return: True if the two iterables meet the length policy, False if not
        :raises ValueError if given unknown length policy
        """

        l = [len(iterable_a), len(iterable_b)]

        if length_policy == "exact":
            return l[0] == l[1]
        elif length_policy == "multiple":
            return all(x % l[0] == 0 or l[0] % x == 0 for x in l)
        elif length_policy == "fill":
            return l[0] > 0 and l[1] > 0

        raise cls.UnknownLengthPolicy(length_policy)

    __measures_by_friendly_name__ = {}  # monophonic rhythm distance implementations by __friendly_name__
    __measures_by_class_name__ = {}  # monophonic rhythm distance implementations by __name__

    @classmethod
    def get_measures(cls, friendly_name=True):
        """
        Returns an ordered dictionary containing implementations of MonophonicRhythmDistanceMeasure by name.

        :param friendly_name: when True, the name will be retrieved with __friendly_name__ instead of __name__
        :return: an ordered dictionary containing all subclasses of MonophonicRhythmDistanceMeasure by name
        """

        if len(MonophonicRhythmDistanceMeasure.__measures_by_friendly_name__) != \
                len(MonophonicRhythmDistanceMeasure.__subclasses__()):
            by_friendly_name = OrderedDict()
            by_class_name = OrderedDict()
            for dm in cls.__subclasses__():  # Distance Measure
                name = dm.__name__
                if friendly_name:
                    try:
                        # noinspection PyUnresolvedReferences
                        name = dm.__friendly_name__
                    except AttributeError:
                        pass
                by_friendly_name[name] = dm
                by_class_name[dm.__name__] = dm
            cls.__measures_by_friendly_name__ = by_friendly_name
            cls.__measures_by_class_name__ = by_class_name

        return MonophonicRhythmDistanceMeasure.__measures_by_friendly_name__

    @classmethod
    def get_measure_names(cls):
        measures = cls.get_measures()
        return tuple(measures.keys())

    @classmethod
    def get_measure_by_name(cls, measure_name):
        """
        Returns a subclass given its name. This can either be its friendly name or its class name.

        :param measure_name: either its friendly name or its class name
        :return: the monophonic rhythm distance measure
        """

        measures = cls.get_measures()

        try:
            return measures[measure_name]
        except KeyError:
            by_class_name = cls.__measures_by_class_name__
            try:
                return by_class_name[measure_name]
            except KeyError:
                raise ValueError("No measure with class or friendly name: '%s'" % measure_name)


@friendly_named_class("Hamming distance")
class HammingDistanceMeasure(MonophonicRhythmDistanceMeasure):
    """
    The hamming distance is based on the binary chains of the rhythms. The hamming distance is the sum of indexes
    where the binary rhythm chains do not match. The hamming distance is always an integer.
    """

    def __init__(self, unit="eighths", length_policy="multiple"):
        self._binary_onset_vector = BinaryOnsetVector(unit)
        super().__init__(unit, length_policy)

    def set_unit(self, unit: str):
        super().set_unit(unit)
        self._binary_onset_vector.set_unit(unit)

    def __get_iterable__(self, rhythm: MonophonicRhythmImpl):
        assert self._binary_onset_vector.unit == self.unit
        return self._binary_onset_vector.process(rhythm)

    @staticmethod
    def __compute_distance__(n, cx, cy, *cookies):
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

    def __init__(self, unit="ticks", length_policy="exact", quantize=False):
        self._ioi_vector = IOIVector(unit, mode=IOIVector.Mode.POST_NOTE, quantize_enabled=quantize)
        super().__init__(unit, length_policy)  # super constructor must be called after self._ioi_vector definition
        self.quantize_enabled = quantize

    def set_unit(self, unit: str):
        super().set_unit(unit)
        self._ioi_vector.set_unit(unit)

    def set_quantize_enabled(self, quantize_enabled):
        self._ioi_vector.set_quantize_enabled(quantize_enabled)

    def is_quantize_enabled(self) -> bool:
        return self._ioi_vector.is_quantize_enabled()

    def __get_iterable__(self, rhythm: MonophonicRhythmImpl):
        assert self._ioi_vector.unit == self.unit
        return self._ioi_vector.process(rhythm)

    @staticmethod
    def __compute_distance__(n, vx, vy, *cookies):
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

    def __init__(self, unit="ticks", length_policy="fill", quantize=False, cyclic=True):
        self._ioi_diff_vector = IOIDifferenceVector(unit, cyclic)
        super().__init__(unit, length_policy)
        self.quantize_enabled = quantize
        self.cyclic = cyclic  # call setter

    def set_unit(self, unit: str):
        super().set_unit(unit)
        self._ioi_diff_vector.set_unit(unit)

    @property
    def cyclic(self):
        return self._ioi_diff_vector.cyclic

    @cyclic.setter
    def cyclic(self, cyclic):
        self._ioi_diff_vector.cyclic = cyclic

    def set_quantize_enabled(self, quantize_enabled: bool):
        self._ioi_diff_vector.set_quantize_enabled(quantize_enabled)

    def is_quantize_enabled(self) -> bool:
        return self._ioi_diff_vector.is_quantize_enabled()

    def __get_iterable__(self, rhythm: MonophonicRhythmImpl):
        assert self._ioi_diff_vector.unit == self.unit
        return self._ioi_diff_vector.process(rhythm)

    @staticmethod
    def __compute_distance__(n, vx, vy, *cookies):
        summed_fractions, i = 0, 0
        while i < n:
            x = float(vx[i % len(vx)])
            y = float(vy[i % len(vy)])
            numerator, denominator = (x, y) if x > y else (y, x)
            try:
                summed_fractions += numerator / denominator
            except ZeroDivisionError:
                return float("inf")
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

    def __init__(self, unit="eighths", length_policy="multiple", quantize=False):
        self._onset_position_vector = OnsetPositionVector(unit, quantize)
        super().__init__(unit, length_policy)  # super constructor must be called after _onset_position_vector attr def
        self.quantize_enabled = quantize

    def set_unit(self, unit: str):
        super().set_unit(unit)
        self._onset_position_vector.set_unit(unit)

    def set_quantize_enabled(self, quantize_enabled: bool):
        self._onset_position_vector.set_quantize_enabled(quantize_enabled)

    def is_quantize_enabled(self) -> bool:
        return self._onset_position_vector.is_quantize_enabled()

    def __get_iterable__(self, rhythm: MonophonicRhythmImpl):
        assert self._onset_position_vector.unit == self.unit
        return self._onset_position_vector.process(rhythm)

    def __get_cookie__(self, rhythm: MonophonicRhythmImpl):
        return int(math.ceil(rhythm.get_duration(self.unit)))

    @staticmethod
    def __compute_distance__(n, vx, vy, dur_x, dur_y):
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

    def __init__(self, unit="eighths", length_policy="multiple"):
        self._chronotonic_vector = ChronotonicChain(unit)
        super().__init__(unit, length_policy)  # super constructor must be called after _chronotonic_vector attr def

    def set_unit(self, unit: str):
        super().set_unit(unit)
        self._chronotonic_vector.set_unit(unit)

    def __get_iterable__(self, rhythm: MonophonicRhythmImpl):
        assert self._chronotonic_vector.unit == self.unit
        return self._chronotonic_vector.process(rhythm)

    @staticmethod
    def __compute_distance__(n, cx, cy, *args):
        chronotonic_distance, i = 0, 0
        while i < n:
            x = cx[i % len(cx)]
            y = cy[i % len(cy)]
            # assuming that each pulse is a unit
            chronotonic_distance += abs(x - y)
            i += 1
        return chronotonic_distance


TRACK_WILDCARDS = ["*", "a*", "b*"]  # NOTE: Don't change the wildcard order or the code will break


def rhythm_pair_track_iterator(rhythm_a: PolyphonicRhythm,
                               rhythm_b: PolyphonicRhythm,
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


class PolyphonicRhythmDistanceMeasure(DistanceMeasure, metaclass=ABCMeta):
    """Abstract base class for polyphonic rhythm distance measures

    This is an abstract base class for polyphonic rhythm distance measures. It measures the distance
    between two Rhythm objects.
    """

    @abstractmethod
    def get_distance(self, rhythm_a: PolyphonicRhythm, rhythm_b: PolyphonicRhythm) -> tp.Union[float, int]:
        raise NotImplementedError


class SummedMonophonicRhythmDistance(PolyphonicRhythmDistanceMeasure):
    def __init__(self, track_distance_measure=HammingDistanceMeasure, tracks="a*", normalize=True):
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
        unit = measure.unit
        duration = max(rhythm_a.get_duration(unit), rhythm_b.get_duration(unit))
        n_tracks, total_distance = 0, 0

        for name, tracks in rhythm_pair_track_iterator(rhythm_a, rhythm_b, self.tracks):
            distance = duration
            if None not in tracks:
                try:
                    distance = measure.get_distance(tracks[0], tracks[1])
                except MonophonicRhythmDistanceMeasure.LengthPolicyNotMet:
                    pass
            total_distance += distance
            n_tracks += 1

        average_distance = float(total_distance) / n_tracks
        return average_distance / duration if self.normalize else average_distance

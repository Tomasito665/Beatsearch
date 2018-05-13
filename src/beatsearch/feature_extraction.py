import math
import types
import inspect
import anytree
import logging
import itertools
import numpy as np
import typing as tp
from collections import defaultdict
from abc import ABCMeta, abstractmethod
from beatsearch.utils import QuantizableMixin, minimize_term_count, TupleView, iterable_to_str
from beatsearch.rhythm import Rhythm, MonophonicRhythm, PolyphonicRhythm, \
    Unit, UnitType, parse_unit_argument, convert_tick

LOGGER = logging.getLogger(__name__)

######################################
# Feature extractor abstract classes #
######################################


_ImmutablePrimitiveType = tp.Union[bool, int, float, tuple, str, frozenset]
_FeatureType = tp.Union[_ImmutablePrimitiveType, np.ndarray]
_FtrExtrProcessRetType = tp.Union[_FeatureType, tp.Generator[_FeatureType, None, None]]


class FeatureExtractor(object, metaclass=ABCMeta):
    __registered_auxiliary_extractors__: tp.List         # Auxiliary feature extractors in registration order
    __registered_auxiliary_extractor_ids__: tp.Set[int]  # Keep extractor obj ids in set for O(1) registered(extr) check
    __preconditions__ = tuple()

    class FactorError(Exception):
        pass

    class FeatureObjectError(Exception):
        pass

    def process(self, obj, pre_computed_auxiliary_features=tuple()):
        # type: (tp.Any, tp.Iterable[tp.Tuple[FeatureExtractorAuxiliaryFeature, _FeatureType]]) -> _FeatureType
        """Computes and returns a feature

        Computes a feature of the given object and returns it.

        :param obj: object of which to compute the feature
        :param pre_computed_auxiliary_features: pre-computed auxiliary features as a dictionary holding the pre-computed
                                                auxiliary features by :class:`beatsearch.feature_extraction.
                                                FeatureExtractorAuxiliaryFeature` objects
        :return: the feature value
        """

        # Construct complete auxiliary tree
        auxiliary_tree_root = self.construct_full_auxiliary_tree(obj)
        root_auxiliary_desc = auxiliary_tree_root.get_description()

        # Create iterator to yield over the auxiliary tree per depth level in reversed order (so, the first iteration
        # only yields auxiliaries with independent feature extractors)
        reversed_depth_order_auxiliary_ier = reversed([  # type: tp.Iterable[FeatureExtractorAuxiliaryFeature]
            *anytree.LevelOrderIter(auxiliary_tree_root)
        ])

        # Keep a cache of already-computed feature auxiliaries (include the given pre-computed aux features)
        feature_cache = {aux.get_description(): ftr for aux, ftr in pre_computed_auxiliary_features}

        for auxiliary in reversed_depth_order_auxiliary_ier:
            description = auxiliary.get_description()   # type: _FeatureAuxiliaryDescType
            child_descriptions = (child.get_description() for child in auxiliary.children)

            # We don't compute the feature if we already did somewhere down the auxiliary tree
            if description in feature_cache:
                assert all((desc in feature_cache) for desc in child_descriptions), \
                    "processed auxiliary but not it's children (children should be already " \
                    "processed because iteration is from leaves to root)"
                continue

            # Compute the feature and add it to our cache (the last iteration will process the main feature)
            feature_cache[description] = auxiliary.process_feature(feature_cache[desc] for desc in child_descriptions)

        assert root_auxiliary_desc in feature_cache
        return feature_cache[root_auxiliary_desc]

    def construct_full_auxiliary_tree(self, obj: tp.Any, root_auxiliary=None):
        # type: (tp.Any, tp.Optional[FeatureExtractorAuxiliaryFeature]) -> FeatureExtractorAuxiliaryFeature
        """
        Constructs the complete feature auxiliary tree containing all auxiliaries needed to compute this feature for the
        given object. The following statements are true for the tree:

            - the root node represents this feature extractor (highest abstraction level)
            - the children of a node are its direct feature auxiliaries
            - leaf nodes represent auxiliaries with an independent extractor (auxiliaries on lowest abstraction level)

        :return: the root node of the tree
        """

        if root_auxiliary:
            if not isinstance(root_auxiliary, FeatureExtractorAuxiliaryFeature):
                raise TypeError("Expected FeatureExtractorAuxiliaryFeature but got a '%s'" % root_auxiliary)
            if not isinstance(root_auxiliary.extractor, self.__class__):
                raise ValueError("Root auxiliary extractor must be "
                                 "a '%s' (not a '%s')" % (self.__class__, type(root_auxiliary.extractor)))
            if root_auxiliary.obj is not obj:
                raise ValueError("Root auxiliary obj must be identical to the obj given to this method")
        else:
            root_auxiliary = FeatureExtractorAuxiliaryFeature(obj, self)

        # Independent feature extractors don't have any auxiliary feature extractors and thus
        if self.is_independent():
            assert not tuple(self.__get_auxiliaries__(obj))
            return root_auxiliary

        try:
            auxiliary_extractor_ids = self.__registered_auxiliary_extractor_ids__
        except AttributeError:
            auxiliary_extractor_ids = set()

        for aux in self.__get_auxiliaries__(obj):
            aux.parent = root_auxiliary
            extractor = aux.extractor  # type: FeatureExtractor
            assert id(extractor) in auxiliary_extractor_ids, \
                "Auxiliary extractor '%s' was never registered as an auxiliary extractor. You " \
                "might want to consider doing so in %s.__init__." % (str(extractor), self.__class__)
            extractor.construct_full_auxiliary_tree(aux.obj, aux)

        return root_auxiliary

    @abstractmethod
    def __process__(self, obj: tp.Any, auxiliary_features: tp.Tuple[_FeatureType, ...]) -> _FtrExtrProcessRetType:
        """
        Computes this feature, given the main object and a tuple containing the auxiliary features. The auxiliary
        features are computed with the extractors returned by
        :meth:`beatsearch.feature_extraction.FeatureExtractor.get_auxiliary_feature_extractors`.

        :param obj: main object of which to compute the feature
        :param auxiliary_features: tuple containing the results of the auxiliary feature extractors
        :return: the feature value
        """

        raise NotImplementedError

    def __check_preconditions__(self, obj: tp.Any):
        """
        Calls the preconditions on the given object.

        :param obj: object to pass to the preconditions
        :return: None
        """

        for precondition in self.__preconditions__:
            try:
                precondition(obj)
            except Exception as e:
                LOGGER.error("Unmet precondition: %s" % str(e))
                raise e

    def __get_auxiliaries__(self, obj: tp.Any):  # type: () -> tp.Iterable[FeatureExtractorAuxiliaryFeature]
        """
        Returns the auxiliary features needed to extract the main feature from the given object. Auxiliaries are
        returned as :class:`beatsearch.feature_extraction.FeatureExtractorAuxiliaryFeature` objects. By default, this
        method returns one auxiliary per registered auxiliary feature extractor in the same order as the extractors were
        registered (see :meth:`beatsearch.feature_extraction.FeatureExtractor.register_auxiliary_extractor`).

        :param obj: main object to process
        :return: iterable containing the auxiliaries or an empty sequence if this feature extractor is independent
        """

        return (FeatureExtractorAuxiliaryFeature(obj, extractor) for extractor in self.get_auxiliary_extractors())

    def register_auxiliary_extractor(self, extractor):  # type: (FeatureExtractor) -> FeatureExtractor
        """Registers the given feature extractor as an auxiliary feature extractor

        :param extractor: auxiliary feature extractor
        :return: given feature extractor

        :raises TypeError: if given extractor is not a FeatureExtractor object
        :raises ValueError: if given feature extractor has already been registered
        """

        if not isinstance(extractor, FeatureExtractor):
            LOGGER.warning("Registering an auxiliary feature extractor of type '%s'" % type(extractor))

        try:
            registered_extractors = self.__registered_auxiliary_extractors__
            registered_extractor_ids = self.__registered_auxiliary_extractor_ids__
        except AttributeError:
            registered_extractors = self.__registered_auxiliary_extractors__ = list()
            registered_extractor_ids = self.__registered_auxiliary_extractor_ids__ = set()

        extractor_id = id(extractor)

        if extractor_id in registered_extractor_ids:
            raise ValueError("Already registered '%s' as an auxiliary feature extractor of '%s'" % (extractor, self))

        self.__setup_auxiliary_extractor__(extractor)
        registered_extractors.append(extractor)
        registered_extractor_ids.add(extractor_id)

        return extractor

    def __setup_auxiliary_extractor__(self, extractor):  # type: (FeatureExtractor) -> None
        """
        This method is called when a new auxiliary feature is about to be registered. Override this method to setup new
        auxiliary extractors.

        :param extractor: auxiliary feature extractor
        :return: None
        """

        pass

    def is_independent(self) -> bool:
        """
        Returns whether this feature extractor is independent. A feature extractor is considered to be independent if it
        has not registered any auxiliary feature extractors and therefore doesn't depend on any other extractors.

        :return: True if this extractor is independent
        """

        auxiliary_extractors = self.get_auxiliary_extractors(None)
        return not bool(auxiliary_extractors)

    def get_auxiliary_extractors(self, extractor_type=None):  # type: (tp.Optional[tp.Type]) -> tp.Iterator
        """Returns an iterator over this extractor's auxiliary extractors

        Returns an iterator over the auxiliary feature extractors registered with :meth:`beatsearch.feature_extraction.
        FeatureExtractor.register_auxiliary_extractor`. When given an extractor type, only the extractors returning
        `isinstance(extr, extractor_type)` will be returned.

        :param extractor_type: if given, only extractors of this exact type will be returned
        :return: iterator over registered auxiliary feature extractors
        """

        try:
            extractors = self.__registered_auxiliary_extractors__
        except AttributeError:
            extractors = list()

        if extractor_type:
            if not inspect.isclass(extractor_type):
                raise TypeError("Expected a class but got a '%s'" % type(extractor_type))
            return iter(filter(lambda extractor: isinstance(extractor, extractor_type), extractors))

        return iter(extractors)

    @abstractmethod
    def __get_factors__(self) -> tp.Iterable[tp.Hashable]:
        """Returns the factors that (may) affect the feature value returned by __process__

        The assumption is made that if two extractors of the same type have equal factors, the result of calling
        :meth:`beatsearch.feature_extraction.FeatureExtractor.process` with the same object will yield identical
        results. The process() function takes advantage of this so that it can use cached features rather than
        re-computing the same feature down the auxiliary feature extractor tree.

        :return: iterable containing all factors
        """

        raise NotImplementedError

    def __eq__(self, other) -> bool:
        """
        The equality check returns true if the other object is also a feature extractor of the same type and will return
        the same feature value, if given the same rhythm. You may assume that for two extractors `a` and `b` and given
        that `a == b` return true, the following statement will also be true: `a.process(stuff) == b.process(stuff)`.

        Returns true if the other object is of the same feature extractor type and it has identical factors (see
        :meth:`beatsearch.feature_extraction.FeatureExtractor.__get_factors__`).

        :param other: other object
        :return: true if other object is also a feature extractor of the same type and has the same settings
        """

        if other is self:
            return True

        if not isinstance(other, self.__class__):
            return False

        own_factors = tuple(self.__get_factors__())
        other_factors = tuple(self.__get_factors__())
        return own_factors == other_factors


_FeatureAuxiliaryDescType = tp.Tuple[tp.Any, tp.Type[FeatureExtractor], tp.Tuple[tp.Hashable]]


class FeatureExtractorAuxiliaryFeature(anytree.NodeMixin):
    """
    Represents one node within the auxiliary feature extractor tree returned by:
        :meth:`beatsearch.feature_extraction.FeatureExtractor.get_auxiliary_feature_extractor_tree`

    Every node has the following properties:

        - parent:       the parent node
        - extractor:    the feature extractor needed to process this auxiliary
        - obj:          the object of which the extractor is expected to compute the feature
        - result:       the computed feature
    """

    def __init__(self, obj: tp.Any, extractor: FeatureExtractor):
        if not isinstance(extractor, FeatureExtractor):
            raise TypeError("Expected a FeatureExtractor but got a '%s'" % type(extractor))
        self._obj = obj
        self._extractor = extractor
        self._factors = tuple(extractor.__get_factors__())

    @property
    def obj(self) -> tp.Any:
        return self._obj

    @property
    def extractor(self) -> FeatureExtractor:
        return self._extractor

    @property
    def name(self) -> str:
        return self.extractor.__class__.__name__

    def process_feature(self, auxiliary_features: tp.Iterable[_FeatureType]) -> _FeatureType:
        """
        Process the auxiliary feature. If the auxiliary extractor factors changed during the lifetime of this object, an
        :class:`beatsearch.feature_extraction.FeatureExtractor.FactorError` exception will be raise. Similarly, if the
        object's state changed, a :class:`beatsearch.feature_extraction.FeatureExtractor.FeatureObjectError` is raised.

        :return: result of calling :meth:`beatsearch.feature_extraction.FeatureExtractor.__process__` on this
                 self.extractor, given self.obj (it will convert the result to a tuple in case that __process__ returns
                 a generator)

        :raises: FactorError if extractor factors changed

        :raises: FeatureObjectError if object state changed
        """

        obj, extractor = self.obj, self.extractor
        expected_factors, actual_factors = self._factors, tuple(extractor.__get_factors__())

        if actual_factors != expected_factors:
            raise FeatureExtractor.FactorError("Factors of '%s' changed, making "
                                               "the auxiliary extractor unusable" % str(extractor))

        extractor.__check_preconditions__(obj)
        auxiliary_features = auxiliary_features if isinstance(auxiliary_features, tuple) else tuple(auxiliary_features)
        feature = extractor.__process__(self.obj, auxiliary_features)
        return tuple(feature) if isinstance(feature, types.GeneratorType) else feature

    def __eq__(self, other):
        """
        Returns true if both the extractor and the main object are equal. If so, you may assume that the results of
        calling process() on both extractors will yield identical results.
        """

        if not isinstance(other, FeatureExtractorAuxiliaryFeature):
            return False
        return self.get_description() == other.get_description()

    def __repr__(self) -> str:
        obj = self.obj
        extractor_name = self.extractor.__class__.__name__
        human_readable_factors = iterable_to_str(self._factors, nested_iterables=True, boundary_chars="()")

        try:
            obj_name = obj.name
        except AttributeError:
            try:
                obj_name = obj.__class__.__name__
            except AttributeError:
                obj_name = str(obj)

        return "%s <%s> : %s" % (extractor_name, obj_name, human_readable_factors)

    def get_description(self) -> tp.Tuple[tp.Any, tp.Type[FeatureExtractor], tp.Tuple[tp.Hashable]]:
        """Returns a description of this auxiliary as a (obj, extractor type, extractor factors) tuple"""
        return self.obj, self._extractor.__class__, self._factors


_FeatExtrGetAuxFeatureExtrRetType = tp.Union[tp.Sequence[FeatureExtractor], tp.Generator[FeatureExtractor, None, None]]


#############################################################
# Rhythm-related feature extraction base classes and mixins #
#############################################################


class RhythmFeatureExtractor(FeatureExtractor, metaclass=ABCMeta):
    """Rhythm feature extractor interface"""

    @abstractmethod
    def get_unit(self) -> tp.Union[Unit, None]:
        """Returns the musical unit of this rhythm feature extractor

        :return: musical unit of this rhythm feature extractor or None for tick based computation
        """

        raise NotImplementedError

    @abstractmethod
    def set_unit(self, unit: tp.Optional[UnitType]) -> None:
        """Sets the musical unit of this rhythm feature extractor and of all its auxiliary rhythm feature extractors

        :return: musical unit of this rhythm feature extractor
        :raises ValueError if given None when called on a rhythm feature extractor which doesn't support
                tick-based computation
        """

        raise NotImplementedError

    @classmethod
    def supports_tick_based_computation(cls) -> bool:
        """
        Returns whether this rhythm feature extractor supports tick-based computation. Implementations can disable tick-
        based computation by adding a __tick_based_computation_support__ class property and setting it to False. If
        tick-based computation is supported, the feature extractor will accept None as a valid unit; if not, a
        ValueError will be raised when trying to set the unit to None.

        :return: whether this rhythm feature extractor supports tick-based computation
        """

        try:
            # noinspection PyUnresolvedReferences
            return cls.__tick_based_computation_support__
        except AttributeError:
            return True

    def process(self, rhythm, pre_computed_auxiliary_features=tuple()) -> _FeatureType:
        if not isinstance(rhythm, Rhythm):
            raise TypeError("Expected a rhythm but got a '%s'" % type(rhythm))
        return super().process(rhythm, pre_computed_auxiliary_features)

    ##############
    # properties #
    ##############

    @property
    def unit(self) -> tp.Union[Unit, None]:
        """See get_unit and set_unit"""
        return self.get_unit()

    @unit.setter
    def unit(self, unit: tp.Optional[UnitType]):
        self.set_unit(unit)

    ######################################################
    # Empty re-implementations (for Rhythm type-hinting) #
    ######################################################

    def __get_auxiliaries__(self, rhythm: Rhythm) -> tp.Optional[tp.Iterable[FeatureExtractorAuxiliaryFeature]]:
        return super().__get_auxiliaries__(rhythm)

    @abstractmethod
    def __process__(self, rhythm: Rhythm, aux_fts: tp.Tuple[_FeatureType, ...]) -> _FtrExtrProcessRetType:
        raise NotImplementedError


class MonophonicRhythmFeatureExtractor(RhythmFeatureExtractor, metaclass=ABCMeta):
    def process(self, rhythm, pre_computed_auxiliary_features=tuple()) -> _FeatureType:
        if not isinstance(rhythm, MonophonicRhythm):
            raise TypeError("Expected a monophonic rhythm but got a '%s'" % type(rhythm))
        return super().process(rhythm, pre_computed_auxiliary_features)

    ################################################################
    # Empty re-implementations (for MonophonicRhythm type-hinting) #
    ################################################################

    def __get_auxiliaries__(self, rhythm: MonophonicRhythm) -> \
            tp.Optional[tp.Iterable[FeatureExtractorAuxiliaryFeature]]:
        return super().__get_auxiliaries__(rhythm)

    @abstractmethod
    def __process__(self, rhythm: MonophonicRhythm, aux_fts: tp.Tuple[_FeatureType, ...]) -> _FtrExtrProcessRetType:
        raise NotImplementedError


class PolyphonicRhythmFeatureExtractor(RhythmFeatureExtractor, metaclass=ABCMeta):
    def process(self, rhythm, pre_computed_auxiliary_features=tuple()) -> _FeatureType:
        if not isinstance(rhythm, PolyphonicRhythm):
            raise TypeError("Expected a polyphonic rhythm but got a '%s'" % type(rhythm))
        return super().process(rhythm, pre_computed_auxiliary_features)

    ################################################################
    # Empty re-implementations (for PolyphonicRhythm type-hinting) #
    ################################################################

    def __get_auxiliaries__(self, rhythm: PolyphonicRhythm) -> \
            tp.Optional[tp.Iterable[FeatureExtractorAuxiliaryFeature]]:
        return super().__get_auxiliaries__(rhythm)

    @abstractmethod
    def __process__(self, rhythm: PolyphonicRhythm, aux_fts: tp.Tuple[_FeatureType, ...]) -> _FtrExtrProcessRetType:
        raise NotImplementedError


_DEFAULT_UNIT = Unit.EIGHTH


class RhythmFeatureExtractorBase(RhythmFeatureExtractor, metaclass=ABCMeta):
    __tick_based_computation_support__ = True

    def __init__(self, unit: tp.Optional[UnitType], aux_to: tp.Optional[FeatureExtractor] = None):
        """
        Creates a new rhythm feature extractor. When given a master feature extractor, this rhythm feature extractor
        will automatically register itself as an auxiliary feature extractor to the given master extractor.

        :param unit: musical unit or None for tick-based computation
        :param aux_to: the feature extractor that this extractor is an auxiliary extractor to
        """

        self._unit = None  # type: tp.Union[Unit, None]
        self.set_unit(unit)

        if aux_to:
            aux_to.register_auxiliary_extractor(self)

    def get_unit(self) -> tp.Union[Unit, None]:
        unit = self._unit
        for extractor in self.get_auxiliary_extractors(self.__class__):
            assert extractor.get_unit() == unit
        return unit

    @parse_unit_argument
    def set_unit(self, unit: tp.Optional[UnitType]) -> None:
        if unit is None and not self.supports_tick_based_computation():
            raise ValueError("'%s' doesn't support tick-based computation")
        for extractor in self.get_auxiliary_extractors(RhythmFeatureExtractor):
            extractor.set_unit(unit)
        self._unit = unit

    def __setup_auxiliary_extractor__(self, extractor):
        if isinstance(extractor, RhythmFeatureExtractor):
            extractor.set_unit(self.unit)


class QuantizableRhythmFeatureExtractorMixin(RhythmFeatureExtractor, QuantizableMixin, metaclass=ABCMeta):
    """This class adds support for quantization to a feature extractor and automatically binds the 'quantize' property
    to the quantizable auxiliary feature extractors"""

    def __on_quantize_set__(self, quantize: bool):
        for extractor in self.get_auxiliary_extractors(QuantizableMixin):
            extractor.quantize = quantize


class MonophonicRhythmFeatureExtractorBase(
    RhythmFeatureExtractorBase,
    MonophonicRhythmFeatureExtractor, metaclass=ABCMeta
):
    """Abstract base class for monophonic rhythm feature extractors"""

    def __init__(self, unit: tp.Optional[UnitType], aux_to: tp.Optional[FeatureExtractor] = None):
        RhythmFeatureExtractorBase.__init__(self, unit, aux_to)


class PolyphonicRhythmFeatureExtractorBase(
    RhythmFeatureExtractorBase,
    PolyphonicRhythmFeatureExtractor, metaclass=ABCMeta
):
    """Abstract base class for polyphonic rhythm feature extractors"""

    def __init__(self, unit: tp.Optional[UnitType], aux_to: tp.Optional[FeatureExtractor] = None):
        RhythmFeatureExtractorBase.__init__(self, unit, aux_to)


####################################################
# Generic rhythm feature extractor implementations #
####################################################

# TODO Create generic rhythm feature for salience profile

#######################################################
# Monophonic rhythm feature extractor implementations #
#######################################################


class BinaryOnsetVector(MonophonicRhythmFeatureExtractorBase):
    """
    Computes the binary representation of the note onsets of the given rhythm where each step where a note onset
    happens is denoted with a 1; otherwise with a 0. The given resolution is the resolution in PPQ (pulses per
    quarter note) of the binary vector.

    This is an independent feature extractor.
    """

    __tick_based_computation_support__ = True
    __preconditions__ = (Rhythm.Precondition.check_resolution,)
    __get_factors__ = lambda e: (e.unit,)

    def __init__(self, unit: tp.Optional[UnitType] = _DEFAULT_UNIT, **kw):
        super().__init__(unit, **kw)

    def __process__(self, rhythm: MonophonicRhythm, aux_fts: tp.Tuple[_FeatureType, ...]) -> _FtrExtrProcessRetType:
        unit = self.get_unit()
        n_steps = rhythm.get_duration(unit, ceil=True)

        if unit is None:
            get_step = lambda tick: tick
        else:
            resolution = rhythm.get_resolution()
            get_step = lambda tick: unit.from_ticks(tick, resolution, True)

        onset_step_indices = iter(get_step(onset.tick) for onset in rhythm.get_onsets())
        next_onset_step_index = next(onset_step_indices, -1)
        curr_step = 0

        while curr_step < n_steps:
            if next_onset_step_index == curr_step:
                # There might be more than one onset on the current step due to quantization, which is why we loop till
                # the onset index has actually moved from the current step
                while next_onset_step_index == curr_step:
                    # Set next step to -1 if there are no more onsets left
                    next_onset_step_index = next(onset_step_indices, -1)
                yield 1
            else:
                yield 0
            curr_step += 1


class NoteVector(MonophonicRhythmFeatureExtractorBase, QuantizableRhythmFeatureExtractorMixin):
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

    __tick_based_computation_support__ = False

    __preconditions__ = (
        Rhythm.Precondition.check_time_signature,
        Rhythm.Precondition.check_duration_is_aligned_with_pulse
    )

    __get_factors__ = lambda e: (e.unit, e.tied_notes, e.cyclic)

    # Events
    NOTE = "N"          # Note event
    TIED_NOTE = "T"     # Tied-note event
    REST = "R"          # Rest event

    _GET_REST_OR_TIED_NOTE = {
        # With support for tied notes
        True: lambda prev_e: NoteVector.TIED_NOTE if prev_e == NoteVector.NOTE else NoteVector.REST,
        # Without support for tied notes
        False: lambda _: NoteVector.REST
    }

    def __init__(self, unit: UnitType = _DEFAULT_UNIT, tied_notes: bool = True, cyclic: bool = True, **kw):
        super().__init__(unit, **kw)

        self._bin_onset_vec = BinaryOnsetVector(aux_to=self)  # type: BinaryOnsetVector
        self._tied_notes = None  # type: bool
        self._cyclic = None  # type: bool

        self.tied_notes = tied_notes
        self.cyclic = cyclic

    def __process__(self, rhythm: MonophonicRhythm, aux_fts: tp.Tuple[_FeatureType, ...]) -> _FtrExtrProcessRetType:
        time_sig = rhythm.get_time_signature()

        natural_duration_map = time_sig.get_natural_duration_map(self.unit, trim_to_pulse=True)
        duration_pool = sorted(set(natural_duration_map))  # NOTE The rest of this method depends on this being sorted
        binary_vector = aux_fts[0]  # type: tp.Tuple[int]

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
            # NOTE: If the duration is not an exact multiple of a beat (or exact multiple of a measure if we were using
            # natural durations with trim_to_pulse set to False) this will eventually raise an index error on
            # binary_vector[step + j]. That's NoteVector has the check_duration_is_aligned_with_pulse precondition.
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
            # noinspection PyTypeChecker
            note_vector[0] = (get_rest_or_tied_note(last_e_type), *note_vector[0][1:])

        return tuple(note_vector)  # TODO <- find a nicer way of converting to tuple

    ##############
    # Properties #
    ##############

    @property
    def tied_notes(self) -> bool:
        return self._tied_notes

    @tied_notes.setter
    def tied_notes(self, tied_notes: bool):
        self._tied_notes = bool(tied_notes)

    @property
    def cyclic(self) -> bool:
        return self._cyclic

    @cyclic.setter
    def cyclic(self, cyclic: bool):
        self._cyclic = bool(cyclic)


class IOIVector(MonophonicRhythmFeatureExtractorBase, QuantizableRhythmFeatureExtractorMixin):
    """
    Computes the time difference between the notes in the given rhythm (Inter Onset Intervals). The elements of the
    vector will depend on this IOIVector extractor's mode property:

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
    """

    __tick_based_computation_support__ = True
    __preconditions__ = (Rhythm.Precondition.check_resolution,)
    __get_factors__ = lambda e: (e.unit, e.mode, e.quantize)

    # IOI Vector modes
    PRE_NOTE = "pre_note"
    POST_NOTE = "post_note"

    def __init__(
            self, unit: tp.Optional[UnitType] = _DEFAULT_UNIT,
            mode: str = POST_NOTE, quantize: bool = True, **kw
    ):
        super().__init__(unit, **kw)
        self._mode = None
        self.mode = mode
        self.quantize = quantize

    def __process__(self, rhythm: MonophonicRhythm, aux_fts: tp.Tuple[_FeatureType, ...]) -> _FtrExtrProcessRetType:
        # TODO Implement both modes generically here
        if self.mode == self.PRE_NOTE:
            return self._pre_note_ioi(rhythm)
        elif self.mode == self.POST_NOTE:
            return self._post_note_ioi(rhythm)
        else:
            raise RuntimeError("Unknown mode: \"%s\"" % self.mode)

    ##############
    # Properties #
    ##############

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode: str):
        if mode not in [self.PRE_NOTE, self.POST_NOTE]:
            raise ValueError("Unknown IOI vector mode: '%s'" % mode)
        self._mode = mode

    ###################
    # Private methods #
    ###################

    def _pre_note_ioi(self, rhythm: MonophonicRhythm) -> tp.Generator[tp.Union[int, float], None, None]:
        quantize = self.quantize
        resolution = rhythm.get_resolution()
        unit = self.get_unit()
        current_tick = 0

        for onset in rhythm.get_onsets():
            delta_tick = onset.tick - current_tick
            yield convert_tick(delta_tick, resolution, unit, quantize)
            current_tick = onset.tick

    # TODO Add cyclic option to include the offset in the last onset's interval
    def _post_note_ioi(self, rhythm: MonophonicRhythm) -> tp.Generator[tp.Union[int, float], None, None]:
        quantize = self.quantize
        resolution = rhythm.get_resolution()
        unit = self.get_unit()
        onset_positions = itertools.chain((onset.tick for onset in rhythm.get_onsets()), [rhythm.duration_in_ticks])
        last_onset_tick = -1

        for onset_tick in onset_positions:
            if last_onset_tick < 0:
                last_onset_tick = onset_tick
                continue

            delta_in_ticks = onset_tick - last_onset_tick
            yield convert_tick(delta_in_ticks, resolution, unit, quantize)
            last_onset_tick = onset_tick


class IOIHistogram(MonophonicRhythmFeatureExtractorBase):
    """
    Computes the number of occurrences of the inter-onset intervals of the notes of the given rhythm in ascending
    order. The inter onset intervals are computed in POST_NOTE mode.

    For example, given the Rumba Clave rhythm, with inter-onset vector [3, 4, 3, 2, 4]:
        (
            [1, 2, 2],  # occurrences
            [2, 3, 4]   # bins (interval durations)
        )
    """

    __tick_based_computation_support__ = True
    __preconditions__ = ()
    __get_factors__ = lambda e: (e.unit,)

    def __init__(self, unit: tp.Optional[UnitType] = _DEFAULT_UNIT, **kw):
        super().__init__(unit, **kw)
        self._ioi_vector = IOIVector(mode=IOIVector.POST_NOTE, quantize=True, aux_to=self)

    def __process__(self, rhythm: MonophonicRhythm, aux_fts: tp.Tuple[_FeatureType, ...]) -> _FtrExtrProcessRetType:
        ioi_vector = aux_fts[0]
        histogram = np.histogram(ioi_vector, tuple(range(min(ioi_vector), max(ioi_vector) + 2)))
        occurrences = histogram[0].tolist()
        bins = histogram[1].tolist()[:-1]
        return occurrences, bins


class BinarySchillingerChain(MonophonicRhythmFeatureExtractorBase):
    """
    Returns the Schillinger notation of this rhythm where each onset is a change of a "binary note".

    For example, given the Rumba Clave rhythm and with values (0, 1):
      X--X---X--X-X---
      0001111000110000

    However, when given the values (1, 0), the schillinger chain will be the opposite:
      X--X---X--X-X---
      1110000111001111
    """

    __tick_based_computation_support__ = True
    __preconditions__ = ()
    __get_factors__ = lambda e: (e.unit, e.values)

    def __init__(self, unit: tp.Optional[UnitType] = _DEFAULT_UNIT, values=(1, 0), **kw):
        super().__init__(unit, **kw)
        self._bin_onset_vec = BinaryOnsetVector(aux_to=self)
        self._values = None
        self.values = values  # calls setter

    def __process__(self, rhythm: MonophonicRhythm, aux_fts: tp.Tuple[_FeatureType, ...]) -> _FtrExtrProcessRetType:
        values = self._values
        binary_onset_vector = aux_fts[0]
        i, value_i = 0, 1

        while i < len(binary_onset_vector):
            if binary_onset_vector[i] == 1:
                value_i = 1 - value_i
            yield values[value_i]
            i += 1

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


class ChronotonicChain(MonophonicRhythmFeatureExtractorBase):
    """
    Computes the chronotonic chain representation of the given rhythm.

    For example, given the Rumba Clave rhythm:
      X--X---X--X-X---
      3334444333224444
    """

    __tick_based_computation_support__ = True
    __preconditions__ = ()
    __get_factors__ = lambda e: (e.unit,)

    def __init__(self, unit: tp.Optional[UnitType] = _DEFAULT_UNIT, **kw):
        super().__init__(unit, **kw)
        self._bin_onset_vec = BinaryOnsetVector(aux_to=self)

    def __process__(self, rhythm: MonophonicRhythm, aux_fts: tp.Tuple[_FeatureType, ...]) -> _FtrExtrProcessRetType:
        chain = list(aux_fts[0])  # binary onset vector
        i, delta = 0, 0

        while i < len(chain):
            if chain[i] == 1:
                j = i + 1
                while j < len(chain) and chain[j] == 0:
                    j += 1
                delta = j - i
            chain[i] = delta
            i += 1

        return tuple(chain)


class IOIDifferenceVector(MonophonicRhythmFeatureExtractorBase, QuantizableRhythmFeatureExtractorMixin):
    """
    Computes the interval difference vector (aka difference of rhythm vector) of the given rhythm. Per note, this is
    the difference between the current onset interval and the next onset interval. So, if N is the number of onsets,
    the returned vector will have a length of N - 1. This is different with cyclic rhythms, where the last onset's
    interval is compared with the first onset's interval. In this case, the length will be N.
    The inter-onset interval vector is computed in POST_NOTE mode.

    For example, given the POST_NOTE inter-onset interval vector for the Rumba clave:
      [3, 4, 3, 2, 4]

    The interval difference vector would be:
       With cyclic set to False: [4/3, 3/4, 2/3, 4/2]
       With cyclic set to True:  [4/3, 3/4, 2/3, 4/2, 3/4]
    """

    __tick_based_computation_support__ = True
    __preconditions__ = ()
    __get_factors__ = lambda e: (e.unit, e.quantize, e.cyclic)

    def __init__(self, unit: tp.Optional[UnitType] = _DEFAULT_UNIT, quantize: bool = True, cyclic: bool = True, **kw):
        super().__init__(unit, **kw)
        self._ioi_vector = IOIVector(mode=IOIVector.POST_NOTE, aux_to=self)
        self._cyclic = None
        self.cyclic = cyclic
        self.quantize = quantize

    def __process__(self, rhythm: MonophonicRhythm, aux_fts: tp.Tuple[_FeatureType, ...]) -> _FtrExtrProcessRetType:
        vector = list(aux_fts[0])
        if self.cyclic:
            vector.append(vector[0])

        i = 0
        while i < len(vector) - 1:
            try:
                vector[i] = vector[i + 1] / float(vector[i])
            except ZeroDivisionError:
                vector[i] = math.inf
            i += 1

        vector.pop()
        return tuple(vector)  # TODO Find nicer way of making immutable

    ##############
    # Properties #
    ##############

    @property
    def cyclic(self):
        """Cyclic behaviour, see process documentation"""
        return self._cyclic

    @cyclic.setter
    def cyclic(self, cyclic):
        self._cyclic = bool(cyclic)


class OnsetPositionVector(MonophonicRhythmFeatureExtractorBase, QuantizableRhythmFeatureExtractorMixin):
    """Finds the absolute onset times of the notes in the given rhythm"""

    __tick_based_computation_support__ = True
    __preconditions__ = (Rhythm.Precondition.check_resolution,)
    __get_factors__ = lambda e: (e.unit, e.quantize)

    def __init__(self, unit: tp.Optional[UnitType] = _DEFAULT_UNIT, quantize: bool = True, **kw):
        super().__init__(unit, **kw)
        self.quantize = quantize

    def __process__(self, rhythm: MonophonicRhythm, aux_fts: tp.Tuple[_FeatureType, ...]) -> _FtrExtrProcessRetType:
        resolution = rhythm.get_resolution()
        quantize = self.quantize
        unit = self.get_unit()
        return (convert_tick(onset[0], resolution, unit, quantize) for onset in rhythm.get_onsets())


_SyncType = tp.Tuple[int, int, int]
"""Syncopation type for type hinting

Syncopations are expressed as a tuple of three elements:
    - syncopation strength
    - position of syncopated note
    - position of rest
"""


class SyncopationVector(MonophonicRhythmFeatureExtractorBase):
    """
    Finds the syncopations in a monophonic rhythm. The syncopations are computed with the method proposed by H.C.
    Longuet-Higgins and C. S. Lee in their work titled: "The Rhythmic Interpretation of Monophonic Music".

    The syncopations are returned as a sequence of tuples containing three elements:
        - the syncopation strength
        - the syncopated note position
        - the position of the rest against which the note is syncopated
    """

    __tick_based_computation_support__ = False
    __preconditions__ = (Rhythm.Precondition.check_time_signature,)
    __get_factors__ = lambda e: (e.unit, e.salience_profile_type)

    def __init__(self, unit: UnitType = _DEFAULT_UNIT, salience_profile_type: str = "equal_upbeats", **kw):
        super().__init__(unit, **kw)
        self._note_vector = NoteVector(aux_to=self)
        self._salience_prf_type = ""
        self.salience_profile_type = salience_profile_type

    def __process__(self, rhythm: MonophonicRhythm, aux_fts: tp.Tuple[_FeatureType, ...]) -> _FtrExtrProcessRetType:
        note_vector = aux_fts[0]
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

    ##############
    # Properties #
    ##############

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


class SyncopatedOnsetRatio(MonophonicRhythmFeatureExtractorBase):
    """
    Computes the number of syncopated onsets over the total number of onsets. The syncopations are computed with
    :class:`beatsearch.feature_extraction.SyncopationVector`.
    """

    __tick_based_computation_support__ = False
    __preconditions__ = (Rhythm.Precondition.check_time_signature,)
    __get_factors__ = lambda e: (e.unit, e.ret_fraction)

    def __init__(self, unit: UnitType = _DEFAULT_UNIT, ret_fraction: bool = False, **kw):
        super().__init__(unit, **kw)
        self._binary_onset_vec = BinaryOnsetVector(aux_to=self)
        self._sync_vector = SyncopationVector(aux_to=self)
        self._ret_fraction = None
        self.ret_fraction = ret_fraction  # calling setter

    def __process__(self, rhythm: MonophonicRhythm, aux_fts: tp.Tuple[_FeatureType, ...]) -> _FtrExtrProcessRetType:
        binary_onset_vector = aux_fts[0]
        syncopation_vector = aux_fts[1]

        # NOTE: We use binary onset vector and don't call rhythm.get_onset_count because the onset count might be
        # different for different units (onsets might merge with large unit like Unit.QUARTER)
        n_onsets = sum(binary_onset_vector)
        n_syncopated_onsets = len(syncopation_vector)
        assert n_syncopated_onsets <= n_onsets

        if self.ret_fraction:
            return n_syncopated_onsets, n_onsets
        else:
            return n_syncopated_onsets / float(n_onsets)

    ##############
    # Properties #
    ##############

    @property
    def ret_fraction(self) -> bool:
        """When this is set to true, process() will return a (numerator, denominator) instead of a float"""
        return self._ret_fraction

    @ret_fraction.setter
    def ret_fraction(self, ret_fraction):
        self._ret_fraction = bool(ret_fraction)


class MeanSyncopationStrength(MonophonicRhythmFeatureExtractorBase):
    """
    Computes the average syncopation strength per step. The step size depends on the unit (see set_unit). The
    syncopations are computed with SyncopationVector.
    """

    __tick_based_computation_support__ = False
    __preconditions__ = (Rhythm.Precondition.check_time_signature,)
    __get_factors__ = lambda e: (e.unit,)

    def __init__(self, unit: UnitType = _DEFAULT_UNIT, **kw):
        super().__init__(unit, **kw)
        self._sync_vector = SyncopationVector(aux_to=self)

    def __process__(self, rhythm: MonophonicRhythm, aux_fts: tp.Tuple[_FeatureType, ...]) -> _FtrExtrProcessRetType:
        syncopation_vector = aux_fts[0]
        total_syncopation_strength = sum(info[1] for info in syncopation_vector)
        n_steps = rhythm.get_duration(self.unit, ceil=True)

        try:
            return total_syncopation_strength / n_steps
        except ZeroDivisionError:
            return 0


class OnsetDensity(MonophonicRhythmFeatureExtractorBase):
    """
    Computes the onset density of the given rhythm. The onset density is the number of onsets over the number of
    positions in the binary onset vector of the given rhythm.
    """

    __tick_based_computation_support__ = False
    __preconditions__ = ()
    __get_factors__ = lambda e: (e.unit,)

    def __init__(self, unit: tp.Optional[UnitType] = _DEFAULT_UNIT, **kw):
        super().__init__(unit, **kw)
        self._bin_onset_vec = BinaryOnsetVector(aux_to=self)

    def __process__(self, rhythm: MonophonicRhythm, aux_fts: tp.Tuple[_FeatureType, ...]) -> _FtrExtrProcessRetType:
        binary_vector = aux_fts[0]
        n_onsets = sum(binary_vector)  # onsets are ones, non-onsets are zeros
        return float(n_onsets) / len(binary_vector)


class MonophonicOnsetLikelihoodVector(MonophonicRhythmFeatureExtractorBase):
    """
    Computes the likelihood of a step being an onset for every step in the given monophonic rhythm. The onset
    likelihoods are ranged from 0 to 1. An onset likelihood of 0 means that there's no evidence -- according to the
    model used by this extractor -- that the corresponding step will contain an onset. An onset likelihood of 1 means
    that it is very likely that the corresponding step will contain an onset.

    The likelihoods are based on predictions made on different levels. Each level has a window size corresponding to a
    musical unit. For each level, the prediction is that the current group will equal the last group. For example:

    Rhythm            x o x o x o x o x o x x x o o o

    Predictions
                1/16 |?|x|o|x|o|x|o|x|o|x|o|x:x|o|o|o|
                1/8  |? ?|x o|x o|x o|x o|x o|x x|x o|
                1/4  |? ? ? ?|x o x o|x o x o|x x x o|
                1/2  |? ? ? ? ? ? ? ?|x o x o x o x o|
                1/1  |? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ?|

    The groups containing question marks are groups that have no antecedents. The predictions of these uncertain groups
    are handled according to the "priors" property.

        - cyclic:       it is assumed that the rhythm is a loop, leaving no uncertain groups
        - optimistic:   predictions are set to the ground truth (predictions of uncertain groups are always correct)
        - pessimistic:  predictions are set to the negative ground truth (predictions of uncertain groups are always
                        incorrect)
    """

    __tick_based_computation_support__ = False
    __preconditions__ = (Rhythm.Precondition.check_time_signature,)
    __get_factors__ = lambda e: (e.unit, e.priors)

    CYCLIC = "cyclic"
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"

    def __init__(self, unit: UnitType = _DEFAULT_UNIT, priors: str = CYCLIC, **kw):
        super().__init__(unit, **kw)
        self._bin_onset_vec = BinaryOnsetVector(unit, aux_to=self)
        self._priors = None   # type: str
        self.priors = priors  # calling setter

    def __process__(self, rhythm: MonophonicRhythm, aux_fts: tp.Tuple[_FeatureType, ...]) -> _FtrExtrProcessRetType:
        n_steps = rhythm.get_duration(self.unit, ceil=True)
        binary_onset_vector = aux_fts[0]  # type: tp.Tuple[int]
        assert len(binary_onset_vector) == n_steps, \
            "expected binary onset vector to have %i steps, not %i" % (n_steps, len(binary_onset_vector))

        # Retrieve the window sizes and their weights
        time_signature = rhythm.get_time_signature()
        natural_duration_map = time_signature.get_natural_duration_map(self.unit, trim_to_pulse=False)
        window_sizes = sorted(w_size for w_size in set(natural_duration_map) if w_size < n_steps)
        window_sizes_sum = sum(window_sizes)
        window_weights = {w_size: (w_size / window_sizes_sum) for w_size in window_sizes}
        priors = self.priors

        if priors == self.OPTIMISTIC:
            get_prior_pred = lambda _step, _w_size: binary_onset_vector[_step]
        elif priors == self.PESSIMISTIC:
            get_prior_pred = lambda _step, _w_size: int(not binary_onset_vector[_step])
        elif priors == self.CYCLIC:
            get_prior_pred = lambda _step, _w_size: binary_onset_vector[(_step - _w_size) % n_steps]
        else:
            assert False, "Unknown priors value: %s" % priors

        # Iterate over the steps of the binary onset vector and make predictions about whether the step has an onset.
        # Predictions are made on different levels and hit/err counts are kept for window prediction confidence
        # coefficients (if a prediction with a certain window size has already made many errors in the current window,
        # its confidence coefficient will decrease)
        hit_count_prev_windows = {w_size: w_size for w_size in window_sizes}
        err_count_curr_windows = {w_size: 0 for w_size in window_sizes}

        for step, is_onset in enumerate(binary_onset_vector):
            # We keep the onset likelihoods in a dictionary per window size
            onset_likelihoods = []

            for w_size in window_sizes:
                if step % w_size == 0:
                    hit_count_prev_windows[w_size] = w_size - err_count_curr_windows[w_size]
                    err_count_curr_windows[w_size] = 0

                if step < w_size:
                    # If we don't have previous knowledge about this step on this window size, we
                    onset_prediction = get_prior_pred(step, w_size)
                else:
                    onset_prediction = binary_onset_vector[step - w_size]

                # The confidence of the prediction is the number of correctly predicted steps in the previous window
                # minus the number of incorrectly predicted steps in the current window, normalized by the window size
                prediction_confidence = (hit_count_prev_windows[w_size] - err_count_curr_windows[w_size]) / w_size

                err_count_curr_windows[w_size] += int(is_onset != onset_prediction)
                onset_likelihoods.append(prediction_confidence * window_weights[w_size] * onset_prediction)

            yield sum(onset_likelihoods)

    ##############
    # Properties #
    ##############

    @property
    def priors(self) -> tp.Union[str, tp.Tuple[int]]:
        """The way that uncertain groups are handled. One of ['cyclic', 'optimistic', 'pessimistic']. See
        :class:`beatsearch.feature_extraction.MonophonicOnsetLikelihoodVector` for more info."""
        return self._priors

    @priors.setter
    def priors(self, priors: tp.Union[str, tp.Iterable[int]]):
        legal_priors = (self.CYCLIC, self.OPTIMISTIC, self.PESSIMISTIC)
        priors = str(priors)
        if priors not in legal_priors:
            raise ValueError("Unknown priors value: %s (choose between: %s)" % (priors, str(legal_priors)))
        self._priors = priors


class MonophonicVariabilityVector(MonophonicRhythmFeatureExtractorBase):
    """
    Computes the variability of each step of the given monophonic rhythm. First, the onset likelihood vector is computed
    with :class:`beatsearch.feature_extraction.MonophonicOnsetLikelihoodVector`. The variability of step N is the
    absolute difference of the probability of N being an onset and whether there was actually an onset on step N.
    """

    __tick_based_computation_support__ = False
    __preconditions__ = (Rhythm.Precondition.check_time_signature,)
    __get_factors__ = lambda e: (e.unit, e.priors)

    def __init__(self, unit: UnitType = _DEFAULT_UNIT, priors: str = MonophonicOnsetLikelihoodVector.CYCLIC, **kw):
        super().__init__(unit, **kw)
        self._mono_onset_likelihood_vec = MonophonicOnsetLikelihoodVector(aux_to=self)
        self._bin_onset_vec = BinaryOnsetVector(aux_to=self)
        self.priors = priors

    def __process__(self, rhythm: MonophonicRhythm, aux_fts: tp.Tuple[_FeatureType, ...]) -> _FtrExtrProcessRetType:
        predicted_onsets = aux_fts[0]  # binary onset likelihood vector
        actual_onsets = aux_fts[1]     # binary onset vector
        return (abs(a_onset - p_onset) for (p_onset, a_onset) in zip(predicted_onsets, actual_onsets))

    ##############
    # Properties #
    ##############

    @property
    def priors(self) -> str:
        """See :meth:`beatsearch.feature_extraction.MonophonicOnsetLikelihoodVector.priors`"""
        return self._mono_onset_likelihood_vec.priors

    @priors.setter
    def priors(self, priors: str):
        self._mono_onset_likelihood_vec.priors = priors


class MonophonicMetricalTensionVector(MonophonicRhythmFeatureExtractorBase):
    """
    This feature extractor computes the monophonic metrical tension of a rhythm. If E(i) is the i-th event in the note
    vector of the given rhythm (see :class:`beatsearch.feature_extraction.NoteVector`, it is said that the tension
    during event E(i) equals the metrical weight of the starting position of E(i) for rests and sounding notes. If E(i)
    is a tied note (non-sounding note), then the tension of E(i) equals the tension of E(i-1).
    """

    __tick_based_computation_support__ = False
    __preconditions__ = (Rhythm.Precondition.check_time_signature,)
    __get_factors__ = lambda e: (e.unit, e.salience_profile_type, e.normalize, e.cyclic)

    def __init__(
            self, unit: UnitType = _DEFAULT_UNIT,
            salience_profile_type: str = "equal_upbeats",
            normalize: bool = False,
            cyclic: bool = True, **kw
    ):
        super().__init__(unit, **kw)

        self._note_vec = NoteVector(tied_notes=True, cyclic=cyclic, aux_to=self)
        self._salience_prf_type = None
        self._normalize = None

        self.salience_profile_type = salience_profile_type
        self.normalize = normalize

    def __process__(self, rhythm: MonophonicRhythm, aux_fts: tp.Tuple[_FeatureType, ...]) -> _FtrExtrProcessRetType:
        time_sig = rhythm.get_time_signature()
        salience_profile = time_sig.get_salience_profile(self.unit, self.salience_profile_type)

        note_vector = aux_fts[0]  # type: tp.Sequence[tp.Tuple[str, int, int]]
        tension_per_event = []  # type: tp.List[int]
        prev_e_tension = None  # type: tp.Optional[int]

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

    ##############
    # Properties #
    ##############

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

    @property
    def cyclic(self) -> bool:
        return self._note_vec.cyclic

    @cyclic.setter
    def cyclic(self, cyclic: bool):
        self._note_vec.cyclic = cyclic


class MonophonicMetricalTensionMagnitude(MonophonicRhythmFeatureExtractorBase):
    """Computes the magnitude of the monophonic metrical tension vector (the euclidean distance to the zero vector)"""

    __tick_based_computation_support__ = False
    __preconditions__ = (Rhythm.Precondition.check_time_signature,)
    __get_factors__ = lambda e: (e.unit, e.salience_profile_type, e.normalize, e.cyclic)

    def __init__(self, unit: UnitType = _DEFAULT_UNIT, salience_profile_type: str = "equal_upbeats", **kw):
        super().__init__(unit, **kw)
        self._tension_vec = MonophonicMetricalTensionVector(normalize=True, aux_to=self)
        self.salience_profile_type = salience_profile_type  # type: str

    def __process__(self, rhythm: MonophonicRhythm, aux_fts: tp.Tuple[_FeatureType, ...]) -> _FtrExtrProcessRetType:
        return math.sqrt(sum(t * t for t in aux_fts[0]))

    ##############
    # Properties #
    ##############

    @property
    def salience_profile_type(self) -> str:
        return self._tension_vec.salience_profile_type

    @salience_profile_type.setter
    def salience_profile_type(self, salience_profile_type: str):
        self._tension_vec.salience_profile_type = salience_profile_type

    @property
    def normalize(self) -> bool:
        """When set to True, the magnitude will have a range of [0, sqrt(N)] where N is the step count of the given
        rhythm with this extractor's unit"""

        return self._tension_vec.normalize

    @normalize.setter
    def normalize(self, normalize: bool):
        self._tension_vec.normalize = normalize

    @property
    def cyclic(self) -> bool:
        return self._tension_vec.cyclic

    @cyclic.setter
    def cyclic(self, cyclic: bool):
        self._tension_vec.cyclic = cyclic


#######################################################
# Polyphonic rhythm feature extractor implementations #
#######################################################


class MultiTrackMonoFeature(PolyphonicRhythmFeatureExtractor):
    """
    This class can be used to "upgrade" a monophonic rhythm feature extractor to a polyphonic one. Instances of this
    class have an underlying monophonic rhythm feature extractor. When processing a polyphonic rhythm, this class will
    process the monophonic feature per track and return it like a (feature_track_a, feature_track_b, etc.) tuple. The
    length of the tuple equals the number of tracks in the polyphonic rhythm. The features are returned in the same
    order in which the tracks are returned by :meth:`beatsearch.rhythm.PolyphonicRhythm.get_track_iterator`. All
    attributes/methods but `__process__` and `__get_auxiliaries__` are forwarded to the underlying monophonic feature
    extractor.
    """

    def __init__(self, mono_extr_cls: tp.Type[MonophonicRhythmFeatureExtractorBase],
                 *args, aux_to: tp.Optional[FeatureExtractor]=None, **kwargs):
        """
        Creates a new multi-track monophonic feature.

        :param mono_extr_cls: monophonic extractor class
        :param args:          named arguments, forwarded to the constructor of the monophonic extractor
        :param kwargs:        named arguments, forwarded to the constructor of the monophonic extractor
        """

        if not inspect.isclass(mono_extr_cls):
            raise TypeError("Expected a class but got a '%s'" % type(mono_extr_cls))

        # Instantiate the underlying monophonic rhythm feature extractor and explicitly register it as an auxiliary
        # extractor (not passing aux_to=self to mono extractor so that we don't have to rely on its constructor actually
        # auto-registering itself)
        mono_extr_obj = mono_extr_cls(*args, **kwargs)  # type: MonophonicRhythmFeatureExtractor
        self.register_auxiliary_extractor(mono_extr_obj)

        # NOTE: Order is important here, __mono_extractor__ attribute assignment must go after auxiliary registration.
        # The PolyphonicRhythmFeatureExtractor.register_auxiliary_extractor method will otherwise try to assign its
        # private state for keeping auxiliary extractors, not find it and forward it to the mono extractor, which
        # results in the mono extractor registering itself as an auxiliary extractor)
        self.__mono_extractor__ = mono_extr_obj

        # Auto-register if given a master extractor
        if aux_to:
            aux_to.register_auxiliary_extractor(self)

    def __get_auxiliaries__(self, rhythm: PolyphonicRhythm) -> \
            tp.Optional[tp.Iterable[FeatureExtractorAuxiliaryFeature]]:
        for track in rhythm.get_track_iterator():
            yield FeatureExtractorAuxiliaryFeature(track, self.__mono_extractor__)

    def __process__(self, rhythm: PolyphonicRhythm, aux_fts: tp.Tuple[_FeatureType, ...]) -> _FtrExtrProcessRetType:
        return aux_fts

    def __get_factors__(self) -> tp.Iterable[tp.Hashable]:
        mono_extractor = self.__mono_extractor__
        return mono_extractor.__class__, mono_extractor.__get_factors__()

    def get_unit(self) -> tp.Union[Unit, None]:
        return self.__mono_extractor__.get_unit()

    def set_unit(self, unit: tp.Optional[UnitType]) -> None:
        self.__mono_extractor__.set_unit(unit)

    def __setattr__(self, name: str, value: tp.Any) -> None:
        if self.__is_mono_extractor_already_set__():
            # still setting things up in the constructor
            setattr(self.__mono_extractor__, name, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name) -> tp.Any:
        if self.__is_mono_extractor_already_set__():
            # still setting things up in the constructor
            return getattr(self.__mono_extractor__, name)
        else:
            return super().__getattribute__(name)

    def __is_mono_extractor_already_set__(self):
        try:
            super().__getattribute__("__mono_extractor__")
            return True
        except AttributeError:
            return False


class PolyphonicSyncopationVector(PolyphonicRhythmFeatureExtractorBase):
    """
    Finds the polyphonic syncopations and their syncopation strengths. This is an adaption to the method proposed by
    M. Witek et al in their worked titled "Syncopation, Body-Movement and Pleasure in Groove Music". This method is
    implemented in terms of the monophonic syncopation feature extractor. The monophonic syncopations are found per
    instrument. They are upgraded to polyphonic syncopations by adding an instrumentation weight. The syncopations
    are then filtered based on the properties 'only_uninterrupted_syncopations' and 'nested_syncopations'.
    """

    __tick_based_computation_support__ = False
    __preconditions__ = (Rhythm.Precondition.check_time_signature,)
    __get_factors__ = lambda e: (
        e.unit, e.instrumentation_weight_function, e.salience_profile_type,
        e.only_uninterrupted_syncopations, e.nested_syncopations
    )

    KEEP_HEAVIEST = "keep_heaviest"
    KEEP_FIRST = "keep_first"
    KEEP_LAST = "keep_last"
    KEEP_ALL = "keep_all"

    _NESTED_SYNCOPATION_FILTERS = {
        KEEP_HEAVIEST: lambda nested_syncopations: [max(nested_syncopations, key=lambda s: s[0])],
        KEEP_FIRST: lambda nested_syncopations: [min(nested_syncopations, key=lambda s: s[1])],
        KEEP_LAST: lambda nested_syncopations: [max(nested_syncopations, key=lambda s: s[1])],
        KEEP_ALL: lambda nested_syncopations: nested_syncopations
    }

    def __init__(
            self,
            unit: UnitType = _DEFAULT_UNIT,
            instr_weighting_function: tp.Callable[[str, tp.Set[str]], int] = lambda *_: 0,
            salience_profile_type: str = "equal_upbeats",
            interrupted_syncopations: bool = True,
            nested_syncopations: str = KEEP_HEAVIEST, **kw
    ):
        super().__init__(unit, **kw)
        self._mt_bin_onset_vec_extr = MultiTrackMonoFeature(BinaryOnsetVector, aux_to=self)
        self._mt_sync_vec_extr = MultiTrackMonoFeature(SyncopationVector, aux_to=self)

        self._instr_weighting_f = None  # type: tp.Callable[[str, tp.Set[str]], int]
        self._only_uninterrupted_sync = None  # type: bool
        self._nested_sync_strategy = ""  # type: str

        self.instrumentation_weight_function = instr_weighting_function
        self.salience_profile_type = salience_profile_type
        self.only_uninterrupted_syncopations = interrupted_syncopations
        self.nested_syncopations = nested_syncopations

    def __process__(self, rhythm: PolyphonicRhythm, aux_fts: tp.Tuple[_FeatureType, ...]) -> _FtrExtrProcessRetType:
        binary_onset_vectors = aux_fts[0]  # type: tp.Tuple[tp.Tuple[int, ...]]
        mono_sync_vectors = aux_fts[1]     # type: tp.Tuple[tp.Tuple[_SyncType, ...]]
        n_instruments = rhythm.get_track_count()
        instrument_names = rhythm.get_track_names()

        # If the rhythm doesn't contain any monophonic syncopations, it won't contain any polyphonic syncopations either
        if all(len(sync_vector) == 0 for sync_vector in mono_sync_vectors):
            return tuple()

        syncopations = []  # type: tp.List[_SyncType]
        only_uninterrupted = self.only_uninterrupted_syncopations
        get_instr_weight = self.instrumentation_weight_function or (lambda *_: 0)

        for curr_instr, [curr_instr_name, sync_vector] in enumerate(zip(instrument_names, mono_sync_vectors)):
            other_instruments = set(instr for instr in range(n_instruments) if instr != curr_instr)

            # Iterate over all monophonic syncopations played by the current instrument "upgrade" these to polyphonic
            # syncopations by adding instrumentation weights
            for mono_syncopation in sync_vector:
                mono_sync_strength, note_position, rest_position = mono_syncopation

                # Instruments that play a note on the rest position of the monophonic syncopation, hence "close" the
                # syncopation. Whatever instrument is syncopated is said to be syncopated against these instruments
                sync_closing_instrument_names = set(
                    instrument_names[other_instr] for other_instr in other_instruments
                    if binary_onset_vectors[other_instr][rest_position]
                )

                # Compute the instrumentation weight, given the syncopated instrument and the closing instruments
                instrumentation_weight = get_instr_weight(curr_instr_name, sync_closing_instrument_names)
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

    ##############
    # Properties #
    ##############

    @property
    def salience_profile_type(self) -> str:
        """
        The type of salience profile to be used for syncopation detection. This must be one of: ['hierarchical',
        'equal_upbeats', 'equal_beats']. See :meth:`beatsearch.rhythm.TimeSignature.get_salience_profile` for more info.
        """

        return self._mt_sync_vec_extr.salience_profile_type

    @salience_profile_type.setter
    def salience_profile_type(self, salience_profile_type: str):
        self._mt_sync_vec_extr.salience_profile_type = salience_profile_type  # type: SyncopationVector

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


class PolyphonicSyncopationVectorWitek(PolyphonicRhythmFeatureExtractorBase):
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
    """

    __tick_based_computation_support__ = False
    __preconditions__ = (Rhythm.Precondition.check_time_signature,)
    __get_factors__ = lambda e: (e.unit, e.salience_profile_type, e.get_instrumentation_weight_function())

    # noinspection PyUnusedLocal
    @staticmethod
    def default_instr_weight_function(
            syncopated_instruments: tp.Set[str],
            other_instruments: tp.Set[str]
    ) -> int:
        """The default instrumentation weight function"""
        n_other_instruments = len(other_instruments)
        return min(3 - n_other_instruments, 0)

    InstrumentationWeightFunctionType = tp.Optional[tp.Callable[[tp.Set[str], tp.Set[str]], tp.Union[int, float]]]
    _mt_note_vector: tp.Union[NoteVector]  # Enable type hinting for MultiTrackMonoFeature<NoteVector>

    def __init__(self, unit: UnitType = _DEFAULT_UNIT, salience_profile_type: str = "equal_upbeats",
                 instrumentation_weight_function: InstrumentationWeightFunctionType = None, **kw):
        super().__init__(unit, **kw)
        self._mt_note_vector = MultiTrackMonoFeature(NoteVector, aux_to=self)
        self._f_get_instrumentation_weight = None           # type: tp.Optional[self.InstrumentationWeightFunctionType]
        self.salience_profile_type = salience_profile_type  # type: str
        self.set_instrumentation_weight_function(instrumentation_weight_function)

    def __process__(self, rhythm: PolyphonicRhythm, aux_fts: tp.Tuple[_FeatureType, ...]) -> _FtrExtrProcessRetType:
        note_vectors = aux_fts[0]

        if not note_vectors:
            return tuple()

        instrument_names = rhythm.get_track_names()
        time_signature = rhythm.get_time_signature()
        salience_profile = time_signature.get_salience_profile(self.unit, kind=self.salience_profile_type)
        get_instrumentation_weight = self._f_get_instrumentation_weight or self.default_instr_weight_function

        # A dictionary containing (instrument_index, event) tuples by event position
        instr_event_pairs_by_position = defaultdict(lambda: set())
        for instr_i, events in enumerate(note_vectors):
            # NoteVector with ret_positions set to True adds event position as 3rd element (= e[2])
            # NoteVector returns the position
            for e in events:
                instr_event_pairs_by_position[e[2]].add((instr_i, e))

        # Positions that contain an event (either a rest or a note) in ascending order
        dirty_positions = sorted(instr_event_pairs_by_position.keys())
        assert dirty_positions[0] == 0, "first step should always contain an event (either a rest or a note)"

        # Iterate over the positions that contain events
        for i_position, curr_step in enumerate(dirty_positions):
            next_step = dirty_positions[(i_position + 1) % len(dirty_positions)]
            curr_metrical_weight = salience_profile[curr_step % len(salience_profile)]
            next_metrical_weight = salience_profile[next_step % len(salience_profile)]

            # Syncopation not possible on this current metrical position pair
            if curr_metrical_weight > next_metrical_weight:
                continue

            curr_instr_event_pairs = instr_event_pairs_by_position[curr_step]
            next_instr_event_pairs = instr_event_pairs_by_position[next_step]

            curr_sounding_instruments = set(inst for inst, evt in curr_instr_event_pairs if evt[0] == NoteVector.NOTE)
            next_sounding_instruments = set(inst for inst, evt in next_instr_event_pairs if evt[0] == NoteVector.NOTE)

            if len(curr_sounding_instruments) < 0:
                continue

            # Detect a syncopation if there is at least one instrument that plays a sounding note on the current
            # position and a rest (or a tied note) on the next position
            if any(inst not in next_sounding_instruments for inst in curr_sounding_instruments):
                curr_sounding_instrument_names = set(instrument_names[i] for i in curr_sounding_instruments)
                next_sounding_instrument_names = set(instrument_names[i] for i in next_sounding_instruments)
                instrumentation_weight = get_instrumentation_weight(
                    curr_sounding_instrument_names, next_sounding_instrument_names)
                syncopation_degree = next_metrical_weight - curr_metrical_weight + instrumentation_weight
                yield syncopation_degree, curr_step, next_step

    ##############
    # Properties #
    ##############

    def set_instrumentation_weight_function(self, func: tp.Union[InstrumentationWeightFunctionType, None]):
        if func and not callable(func):
            raise TypeError

        self._f_get_instrumentation_weight = func or None

    def get_instrumentation_weight_function(self) -> tp.Union[InstrumentationWeightFunctionType, None]:
        return self._f_get_instrumentation_weight


class PolyphonicTensionVector(PolyphonicRhythmFeatureExtractorBase):
    """
    This feature extractor computes the monophonic tension vector per track. It multiplies the monophonic tensions with
    instrument weights and then returns the sum. Instrument weights can be set with
    :meth:`beatsearch.feature_extraction.PolyphonicTensionVector.set_instrument_weights`.
    """

    __tick_based_computation_support__ = False
    __preconditions__ = (Rhythm.Precondition.check_time_signature,)
    __get_factors__ = lambda e: (
        e.unit, e.salience_profile_type, e.normalize,
        tuple(e.get_instrument_weights().items())  # <- convert to item tuple to make it hashable
    )

    _mt_tension_vec: MonophonicTensionVector  # Enable type hinting for MultiTrackMonoFeature<MonophonicTensionVector>

    def __init__(self, unit: UnitType = _DEFAULT_UNIT,
                 salience_profile_type: str = "hierarchical", normalize: bool = False):
        super().__init__(unit)
        self._mt_tension_vec = MultiTrackMonoFeature(MonophonicTensionVector, aux_to=self)
        self._instr_weights = dict()
        self.salience_profile_type = salience_profile_type  # type: str
        self.normalize = normalize

    def __process__(self, rhythm: PolyphonicRhythm, aux_fts: tp.Tuple[_FeatureType, ...]):
        instrument_names = rhythm.get_track_names()
        weights = self.get_instrument_weights()
        mono_tension_vectors = aux_fts[0]
        default_weight = 1.0 / rhythm.get_track_count()
        tension_vectors_scaled = []

        for instr_i, tension_vec in enumerate(mono_tension_vectors):
            instr_name = instrument_names[instr_i]

            try:
                w = weights[instr_name]
            except KeyError:
                LOGGER.warning("Weight unknown for: %s (defaulting to %.2f)" % (instr_name, default_weight))

                # We add the default weight to the weights dictionary for normalization. Note that we aren't actually
                # affecting the weights of this extractor because get_instrument_weights returns a hard copy of the
                # weights dictionary
                w = weights[instr_name] = default_weight

            scaled_tension_vec = tuple(t * w for t in tension_vec)
            tension_vectors_scaled.append(scaled_tension_vec)

        if self.normalize:
            # We can assume that the max mono tension is 1.0, because it's normalized
            normalizer = 1.0 / sum(weights.values())
        else:
            normalizer = 1.0

        return tuple(sum(col) * normalizer for col in zip(*tension_vectors_scaled))

    ##################
    # Unique methods #
    ##################

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

    ##############
    # Properties #
    # ############

    @property
    def salience_profile_type(self) -> str:
        return self._mt_tension_vec.salience_profile_type

    @salience_profile_type.setter
    def salience_profile_type(self, salience_profile_type: str):
        self._mt_tension_vec.salience_profile_type = salience_profile_type

    @property
    def normalize(self):
        """When set to True, the tension will have a range of [0, 1]"""
        return self._mt_tension_vec.normalize

    @normalize.setter
    def normalize(self, normalize: bool):
        self._mt_tension_vec.normalize = normalize


class PolyphonicTension(PolyphonicRhythmFeatureExtractorBase):
    # TODO Add documentation

    __tick_based_computation_support__ = False
    __preconditions__ = (Rhythm.Precondition.check_time_signature,)
    __get_factors__ = lambda e: (
        e.unit, e.salience_profile_type, e.normalize,
        tuple(e.get_instrument_weights().items())  # <- convert to item tuple to make it hashable
    )

    def __init__(self, unit: UnitType = _DEFAULT_UNIT, salience_profile_type: str = "hierarchical"):
        super().__init__(unit)
        self._poly_tension_vec = PolyphonicTensionVector(normalize=True)
        self._instr_weights = dict()
        self.salience_profile_type = salience_profile_type  # type: str

    def __process__(self, rhythm: PolyphonicRhythm, aux_fts: tp.Tuple[_FeatureType, ...]) -> _FtrExtrProcessRetType:
        tension_vec = aux_fts[0]
        return math.sqrt(sum(t * t for t in tension_vec))

    def set_instrument_weights(self, weights: tp.Dict[str, float]):
        self._poly_tension_vec.set_instrument_weights(weights)

    def get_instrument_weights(self) -> tp.Dict[str, float]:
        return self._poly_tension_vec.get_instrument_weights()


__all__ = [
    # Feature extractor abstract base classes (or interfaces)
    'FeatureExtractor', 'RhythmFeatureExtractor', 'MonophonicRhythmFeatureExtractor',
    'PolyphonicRhythmFeatureExtractor', 'FeatureExtractorAuxiliaryFeature',

    # Monophonic rhythm feature extractor implementations
    'BinaryOnsetVector', 'NoteVector', 'IOIVector', 'IOIHistogram', 'BinarySchillingerChain', 'ChronotonicChain',
    'IOIDifferenceVector', 'OnsetPositionVector', 'SyncopationVector', 'SyncopatedOnsetRatio',
    'MeanSyncopationStrength', 'OnsetDensity', 'MonophonicOnsetLikelihoodVector', 'MonophonicVariabilityVector',
    'MonophonicMetricalTensionVector', 'MonophonicMetricalTensionMagnitude',

    # Polyphonic rhythm feature extractor implementations
    'MultiTrackMonoFeature', 'PolyphonicSyncopationVector', 'PolyphonicSyncopationVectorWitek',
    'PolyphonicTensionVector'
]

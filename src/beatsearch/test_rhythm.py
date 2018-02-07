import enum
import numpy as np
import typing as tp
from functools import wraps, partial
from unittest import TestCase, main
from unittest.mock import MagicMock, patch
from abc import ABCMeta, abstractmethod

# rhythm interfaces
from beatsearch.rhythm import Rhythm, MonophonicRhythm, PolyphonicRhythm

# rhythm abstract base classes
from beatsearch.rhythm import RhythmBase, MonophonicRhythmBase, SlavedRhythmBase

# concrete rhythm implementations
from beatsearch.rhythm import MonophonicRhythmImpl, PolyphonicRhythmImpl, Track, RhythmLoop, MidiRhythm

# misc
from beatsearch.rhythm import TimeSignature, Onset, MidiMapping, MidiKey, \
    MonophonicRhythmRepresentationsMixin, OnsetsNotInChronologicalOrder
import midi


#####################
# Rhythm Interfaces #
#####################


class TestRhythmInterface(TestCase):
    def test_not_instantiable(self):
        self.assertRaises(Exception, Rhythm)


class TestMonophonicRhythmInterface(TestCase):
    def test_not_instantiable(self):
        self.assertRaises(Exception, MonophonicRhythm)


class TestPolyphonicRhythmInterface(TestCase):
    def test_not_instantiable(self):
        self.assertRaises(Exception, PolyphonicRhythm)


################################
# Rhythm abstract base classes #
################################


class TestMonophonicRhythmBase(TestCase):
    def test_not_instantiable(self):
        self.assertRaises(Exception, RhythmBase)


##########################
# Rhythm implementations #
##########################


class RhythmMockedSetters(Rhythm, metaclass=ABCMeta):
    def __init__(self):
        self.set_bpm = MagicMock()
        self.set_resolution = MagicMock()
        self.set_time_signature = MagicMock()
        self.set_duration_in_ticks = MagicMock()


class RhythmMockedGetters(Rhythm, metaclass=ABCMeta):
    def __init__(self):
        self.get_bpm = MagicMock(return_value=123)
        self.get_resolution = MagicMock(return_value=123)
        self.get_time_signature = MagicMock(return_value=(5, 4))
        self.get_duration_in_ticks = MagicMock(return_value=123)


class RhythmMockedSettersAndGetters(RhythmMockedSetters, RhythmMockedGetters, metaclass=ABCMeta):
    def __init__(self):
        RhythmMockedSetters.__init__(self)
        RhythmMockedGetters.__init__(self)


class RhythmMockedPostInit(Rhythm, metaclass=ABCMeta):
    def __init__(self):
        self.post_init = MagicMock()


class ITestRhythmBase(object):
    @classmethod
    @abstractmethod
    def get_rhythm_class(cls) -> tp.Type[RhythmBase]:
        """Returns the concrete RhythmBase implementation class"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_mocked_setters_mixin(cls):
        """Returns the mixin used to mock the setters of the class returned by get_rhythm_class"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_mocked_getters_mixin(cls):
        """Returns the mixin used to mock the getters of the class returned by get_rhythm_class"""
        raise NotImplementedError

    @classmethod
    def get_mocked_post_init_mixin(cls):
        """Returns the mixin used to mock the post_init function of the class returned by get_rhythm_class"""
        return RhythmMockedPostInit

    @classmethod
    def get_mocked_setters_and_getters_mixin(cls):
        """Returns the mixin used to mock the setters and the getters of the class returned by get_rhythm_class"""
        return RhythmMockedSettersAndGetters

    # TODO For each test needing one of these mixed-in classes, a new class is made. This is horrifically inefficient...
    # however, when lazy initializing them as a class variable, they're not re-initialized down to more than one level
    # of inheritance of ITestRhythmBase

    @classmethod
    def get_rhythm_class_with_mocked_setters(cls) -> tp.Type[RhythmMockedSetters]:
        """Returns a class inheriting from the RhythmBase implementation and from the RhythmMockedSetters mixin"""

        rhythm_class = cls.get_rhythm_class()
        mocked_setters_mixin = cls.get_mocked_setters_mixin()

        # noinspection PyAbstractClass
        class RhythmImplWithMockedSetters(rhythm_class, mocked_setters_mixin):
            def __init__(self, **kw):
                super().__init__(**kw)  # init rhythm_class
                # Note: mocked setters mixin initialization is after super initialization, which
                # means that setters called from within the super constructor are not mocked
                mocked_setters_mixin.__init__(self)

        return RhythmImplWithMockedSetters

    @classmethod
    def get_rhythm_class_with_mocked_getters(cls) -> tp.Type[RhythmMockedGetters]:
        """Returns a class inheriting from the RhythmBase implementation and from the RhythmMockedGetters mixin"""

        rhythm_class = cls.get_rhythm_class()
        mocked_getters_mixin = cls.get_mocked_getters_mixin()

        # noinspection PyAbstractClass
        class RhythmImplWithMockedGetters(rhythm_class, mocked_getters_mixin):
            def __init__(self, **kw):
                super().__init__(**kw)
                mocked_getters_mixin.__init__(self)

        return RhythmImplWithMockedGetters

    @classmethod
    def get_rhythm_class_with_mocked_post_init(cls) -> tp.Type[RhythmMockedPostInit]:
        """Returns a class inheriting from the RhythmBase implementation and from the RhythmMockedPostInit mixin"""

        rhythm_class = cls.get_rhythm_class()
        mocked_post_init_mixin = cls.get_mocked_post_init_mixin()

        # noinspection PyAbstractClass
        class RhythmImplWithMockedPostInit(rhythm_class, mocked_post_init_mixin):
            def __init__(self, **kw):
                mocked_post_init_mixin.__init__(self)  # first because post_init is called by super().__init__
                super().__init__(**kw)

        return RhythmImplWithMockedPostInit

    @classmethod
    def get_rhythm_class_with_mocked_setters_and_getters(cls) -> tp.Type[RhythmMockedSettersAndGetters]:
        """
        Returns a class inheriting from the RhythmBase implementation and from the RhythmMockedSettersAndGetters mixin
        """

        rhythm_class = cls.get_rhythm_class()
        mocked_setters_and_getters = cls.get_mocked_setters_and_getters_mixin()

        # noinspection PyAbstractClass
        class RhythmImplWithMockedSettersAndGetters(rhythm_class, mocked_setters_and_getters):
            def __init__(self, **kw):
                super().__init__(**kw)
                mocked_setters_and_getters.__init__(self)

        return RhythmImplWithMockedSettersAndGetters


# noinspection PyPep8Naming
class inject_rhythm(object):
    def __init__(self):
        raise TypeError("inject_rhythm is used as a namespace and it shouldn't be initialised")

    class Mock(enum.Enum):
        NO_MOCK = enum.auto()
        MOCK_GETTERS = enum.auto()
        MOCK_SETTERS = enum.auto()
        MOCK_POST_INIT = enum.auto()
        MOCK_GETTERS_AND_SETTERS = enum.auto()

    @classmethod
    def __get_rhythm_class(cls, test: ITestRhythmBase, mock_type: Mock):
        if mock_type is cls.Mock.NO_MOCK:
            return test.get_rhythm_class()
        elif mock_type is cls.Mock.MOCK_GETTERS:
            return test.get_rhythm_class_with_mocked_getters()
        elif mock_type is cls.Mock.MOCK_SETTERS:
            return test.get_rhythm_class_with_mocked_setters()
        elif mock_type is cls.Mock.MOCK_POST_INIT:
            return test.get_rhythm_class_with_mocked_post_init()
        elif mock_type is cls.Mock.MOCK_GETTERS_AND_SETTERS:
            return test.get_rhythm_class_with_mocked_setters_and_getters()
        else:
            raise ValueError("Unknown mock type: %s" % str(mock_type))

    # noinspection PyMethodParameters
    def __create_decorator(mock_type: Mock, **constructor_kwargs):
        def decorator(func: tp.Callable[[ITestRhythmBase, Rhythm, tp.Any], tp.Any]):
            @wraps(func)
            def wrapper(self: ITestRhythmBase, *args, **kwargs):
                rhythm_class = inject_rhythm.__get_rhythm_class(self, mock_type)
                rhythm_obj = rhythm_class(**constructor_kwargs)
                return func(self, rhythm_obj, *args, **kwargs)
            return wrapper
        return decorator

    no_mock = partial(__create_decorator, mock_type=Mock.NO_MOCK)
    mocked_getters = partial(__create_decorator, mock_type=Mock.MOCK_GETTERS)
    mocked_setters = partial(__create_decorator, mock_type=Mock.MOCK_SETTERS)
    mocked_setters_and_getters = partial(__create_decorator, mock_type=Mock.MOCK_GETTERS_AND_SETTERS)
    mocked_post_init = partial(__create_decorator, mock_type=Mock.MOCK_POST_INIT)

    @staticmethod
    def type(func: tp.Callable[[ITestRhythmBase, tp.Type[Rhythm], tp.Any], tp.Any]):
        @wraps(func)
        def wrapper(self: ITestRhythmBase, *args, **kwargs):
            rhythm_class = inject_rhythm.__get_rhythm_class(self, inject_rhythm.Mock.NO_MOCK)
            return func(self, rhythm_class, *args, **kwargs)
        return wrapper


class FakeRhythmBaseImplementation(RhythmBase):
    def __init__(self, **kwargs):
        RhythmBase.__init__(self)
        self.post_init(**kwargs)

    def __rescale_onset_ticks__(self, old_resolution: int, new_resolution: int) -> None:
        pass

    def get_last_onset_tick(self) -> int:
        return -1

    def get_onset_count(self) -> int:
        return 0


class TestRhythmBase(ITestRhythmBase, TestCase):
    @classmethod
    def get_rhythm_class(cls) -> tp.Type[RhythmBase]:
        return FakeRhythmBaseImplementation

    @classmethod
    def get_mocked_setters_mixin(cls):
        return RhythmMockedSetters

    @classmethod
    def get_mocked_getters_mixin(cls):
        return RhythmMockedGetters

    #############
    # post_init #
    #############

    @inject_rhythm.mocked_setters()
    def test_post_init_calls_bpm_setter(self, rhythm):
        rhythm.post_init(bpm=123)
        rhythm.set_bpm.assert_called_once_with(123)

    @inject_rhythm.mocked_setters()
    def test_post_init_calls_resolution_setter(self, rhythm):
        rhythm.post_init(resolution=123)
        rhythm.set_resolution.assert_called_once_with(123)

    @inject_rhythm.mocked_setters()
    def test_post_init_calls_time_signature_setter(self, rhythm):
        rhythm.post_init(time_signature=(5, 4))
        rhythm.set_time_signature.assert_called_once_with((5, 4))

    @inject_rhythm.mocked_setters()
    def test_post_init_calls_duration_in_ticks_setter(self, rhythm):
        rhythm.post_init(duration_in_ticks=123)
        rhythm.set_duration_in_ticks.assert_called_once_with(123)

    @inject_rhythm.mocked_setters()
    def test_post_init_calls_duration_in_ticks_setter_when_given_duration_alias(self, rhythm):
        rhythm.post_init(duration=123)
        rhythm.set_duration_in_ticks.assert_called_once_with(123)

    @inject_rhythm.mocked_setters()
    def test_post_init_doesnt_call_setters_when_given_anything(self, rhythm):
        rhythm.post_init()
        rhythm.set_bpm.assert_not_called()
        rhythm.set_resolution.assert_not_called()
        rhythm.set_time_signature.assert_not_called()
        rhythm.set_duration_in_ticks.assert_not_called()

    #########################################
    # properties call corresponding setters #
    #########################################

    @inject_rhythm.mocked_setters()
    def test_bpm_prop_calls_setter(self, rhythm):
        rhythm.bpm = 123
        rhythm.set_bpm.assert_called_once_with(123)

    @inject_rhythm.mocked_setters()
    def test_resolution_prop_calls_setter(self, rhythm):
        rhythm.resolution = 123
        rhythm.set_resolution.assert_called_once_with(123)

    @inject_rhythm.mocked_setters()
    def test_time_signature_prop_calls_setter(self, rhythm):
        rhythm.time_signature = (5, 4)
        rhythm.set_time_signature.assert_called_once_with((5, 4))

    @inject_rhythm.mocked_setters()
    def test_duration_in_ticks_prop_calls_setter(self, rhythm):
        rhythm.duration_in_ticks = 123
        rhythm.set_duration_in_ticks.assert_called_once_with(123)

    #########################################
    # properties call corresponding getters #
    #########################################

    @inject_rhythm.mocked_getters()
    def test_bpm_prop_calls_getter(self, rhythm):
        self.assertEqual(rhythm.bpm, rhythm.get_bpm.return_value)
        rhythm.get_bpm.assert_called_once()

    @inject_rhythm.mocked_getters()
    def test_resolution_prop_calls_getter(self, rhythm):
        self.assertEqual(rhythm.resolution, rhythm.get_resolution.return_value)
        rhythm.get_resolution.assert_called_once()

    @inject_rhythm.mocked_getters()
    def test_time_signature_prop_calls_getter(self, rhythm):
        self.assertEqual(rhythm.time_signature, rhythm.get_time_signature.return_value)
        rhythm.get_time_signature.assert_called_once()

    @inject_rhythm.mocked_getters()
    def test_duration_in_ticks_prop_calls_getter(self, rhythm):
        self.assertEqual(rhythm.duration_in_ticks, rhythm.get_duration_in_ticks.return_value)
        rhythm.get_duration_in_ticks.assert_called_once()

    ############################
    # getters and setters work #
    ############################

    @inject_rhythm.no_mock()
    def test_bpm_setter_getter(self, rhythm):
        rhythm.set_duration_in_ticks(123)
        self.assertEqual(rhythm.get_duration_in_ticks(), 123)

    @inject_rhythm.no_mock()
    def test_resolution_setter_getter(self, rhythm):
        rhythm.set_resolution(123)
        self.assertEqual(rhythm.get_resolution(), 123)

    @inject_rhythm.no_mock()
    def test_time_signature_setter_getter_with_ts_object(self, rhythm):
        numerator, denominator = 5, 4
        rhythm.set_time_signature(TimeSignature(numerator, denominator))
        ret = rhythm.get_time_signature()
        self.assertIsInstance(ret, TimeSignature)
        self.assertEqual(ret.numerator, numerator)
        self.assertEqual(ret.denominator, denominator)

    @inject_rhythm.no_mock()
    def test_time_signature_setter_getter_with_tuple(self, rhythm):
        numerator, denominator = 5, 4
        rhythm.set_time_signature((numerator, denominator))
        ret = rhythm.get_time_signature()
        self.assertIsInstance(ret, TimeSignature)
        self.assertEqual(ret.numerator, numerator)
        self.assertEqual(ret.denominator, denominator)

    @inject_rhythm.no_mock()
    def test_duration_in_ticks_setter_getter(self, rhythm):
        rhythm.set_duration_in_ticks(123)
        self.assertEqual(rhythm.get_duration_in_ticks(), 123)

    ####################################################################
    # test constructor sets properties when given as keyword arguments #
    ####################################################################

    @inject_rhythm.mocked_post_init(bpm=123, resolution=234, time_signature=(5, 4), duration_in_ticks=345, duration=456)
    def test_constructor_calls_post_init_with_correct_arguments(self, rhythm):
        rhythm.post_init.assert_called_once_with(
            bpm=123,
            resolution=234,
            time_signature=(5, 4),
            duration_in_ticks=345,
            duration=456
        )

    @inject_rhythm.mocked_post_init()
    def test_constructor_calls_post_init_with_no_arguments(self, rhythm):
        rhythm.post_init.assert_called_with()

    ##############################
    # test beat/measure duration #
    ##############################

    @inject_rhythm.type
    def test_beat_duration(self, rhythm_class):
        resolution = 16

        def create_rhythm(time_signature):
            r = rhythm_class()
            r.get_resolution = MagicMock(return_value=resolution)
            r.get_time_signature = MagicMock(return_value=time_signature)
            return r

        rhythm_info = {
            "quarter_beat_unit": [create_rhythm(TimeSignature(5, 4)), resolution],
            "eighth_beat_unit": [create_rhythm(TimeSignature(5, 8)), resolution // 2]
        }

        for test_name, [rhythm, expected_beat_duration] in rhythm_info.items():
            with self.subTest(name=test_name):
                self.assertEqual(rhythm.get_beat_duration(), expected_beat_duration)

    @inject_rhythm.type
    def test_measure_duration(self, rhythm_class):
        ppq = 16

        def do_subtest(name, time_signature, expected_measure_duration):
            r = rhythm_class()
            r.get_resolution = MagicMock(return_value=ppq)
            r.get_time_signature = MagicMock(return_value=time_signature)
            with self.subTest(name=name):
                self.assertEqual(r.get_measure_duration(), expected_measure_duration)

        test_info = [
            ("4/4", TimeSignature(4, 4), ppq * 4),
            ("4/8", TimeSignature(4, 8), ppq * 2),
            ("5/4", TimeSignature(5, 4), ppq * 5),
            ("5/8", TimeSignature(5, 8), ppq * 2.5)
        ]

        for args in test_info:
            do_subtest(*args)

    #############################
    # test duration in measures #
    # ###########################

    @inject_rhythm.no_mock()
    def test_duration_in_measures(self, rhythm):
        expected_duration_in_measures = 8
        measure_duration = 123

        rhythm.get_time_signature = MagicMock(TimeSignature)
        rhythm.get_measure_duration = MagicMock(return_value=measure_duration)
        rhythm.get_duration_in_ticks = MagicMock(return_value=measure_duration * expected_duration_in_measures)
        rhythm.get_duration = MagicMock(return_value=measure_duration * expected_duration_in_measures)

        actual_duration_in_measures = rhythm.get_duration_in_measures()
        self.assertIsInstance(actual_duration_in_measures, float)
        self.assertEqual(actual_duration_in_measures, expected_duration_in_measures)


class FakeOnset(object):
    def __init__(self, tick, velocity):
        self.tick = tick
        self.velocity = velocity

    def __iter__(self):
        return self.tick, self.velocity

    def __getitem__(self, item):
        dictionary = {'tick': self.tick, 'velocity': self.velocity}
        try:
            return dictionary[item]
        except KeyError:
            return self.__iter__()[item]

    def __eq__(self, other):
        return self.tick == other[0] and self.velocity == other[1]


def mock_onset(tick=0, velocity=0):
    onset_mock = FakeOnset(tick, velocity)
    return onset_mock


class TestMonophonicRhythmImplementationsMixin(TestCase):

    class MockedMonophonicRhythmWithRhythmRepresentationsMixin(MonophonicRhythmRepresentationsMixin,
                                                               FakeRhythmBaseImplementation,
                                                               RhythmMockedGetters):
        def __init__(self):
            super().__init__()
            FakeRhythmBaseImplementation.__init__(self)
            RhythmMockedGetters.__init__(self)
            self.get_onsets = MagicMock()

        def get_onsets(self): pass

        def set_onsets(self, onsets): pass

    @classmethod
    def _get_rhythm_mock_with_23_rumba_clave_onsets(cls, resolution):
        rhythm_mock = cls.MockedMonophonicRhythmWithRhythmRepresentationsMixin()
        assert resolution >= 4, "the 2/3 claves pattern is is not representable with a resolution smaller than 4"

        onset_positions = (
            int(resolution / 4.0 * 0),
            int(resolution / 4.0 * 3),
            int(resolution / 4.0 * 7),
            int(resolution / 4.0 * 10),
            int(resolution / 4.0 * 12)
        )

        mocked_onsets = tuple(mock_onset(tick, 100) for tick in onset_positions)

        rhythm_mock.get_resolution.return_value = resolution
        rhythm_mock.get_onsets.return_value = mocked_onsets
        rhythm_mock.get_duration_in_ticks.return_value = int(resolution * 4)

        return rhythm_mock

    def test_binary(self):
        rhythm = self._get_rhythm_mock_with_23_rumba_clave_onsets(4)
        expected_binary_ticks = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]
        actual_binary_ticks = rhythm.get_binary(unit="ticks")
        np.testing.assert_array_equal(actual_binary_ticks, expected_binary_ticks)

    def test_pre_note_inter_onset_intervals(self):
        rhythm = self._get_rhythm_mock_with_23_rumba_clave_onsets(4)
        expected_ioi_chain = [0, 3, 4, 3, 2]
        actual_ioi_chain = rhythm.get_pre_note_inter_onset_intervals(unit="ticks")
        np.testing.assert_array_equal(actual_ioi_chain, expected_ioi_chain)

    def test_post_note_inter_onset_intervals(self):
        rhythm = self._get_rhythm_mock_with_23_rumba_clave_onsets(4)
        expected_ioi_chain = [3, 4, 3, 2, 4]
        actual_ioi_chain = rhythm.get_post_note_inter_onset_intervals(unit="ticks")
        np.testing.assert_array_equal(actual_ioi_chain, expected_ioi_chain)

    def test_interval_histogram(self):
        rhythm = self._get_rhythm_mock_with_23_rumba_clave_onsets(4)
        expected_histogram = (
            [1, 2, 2],  # occurrences
            [2, 3, 4]   # intervals
        )
        actual_histogram = rhythm.get_interval_histogram(unit="ticks")
        np.testing.assert_array_equal(expected_histogram, actual_histogram)

    def test_schillinger_chain(self):
        rhythm = self._get_rhythm_mock_with_23_rumba_clave_onsets(4)
        expected_chain = [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1]
        actual_chain = rhythm.get_binary_schillinger_chain(unit="ticks", values=(1, 0))
        np.testing.assert_array_equal(actual_chain, expected_chain)

    def test_chronotonic_chain(self):
        rhythm = self._get_rhythm_mock_with_23_rumba_clave_onsets(4)
        expected_chain = [3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 2, 2, 4, 4, 4, 4]
        actual_chain = rhythm.get_chronotonic_chain(unit="ticks")
        np.testing.assert_array_equal(actual_chain, expected_chain)

    def test_interval_difference_vector(self):
        rhythm = self._get_rhythm_mock_with_23_rumba_clave_onsets(4)
        expected_vector_non_cyclic = [4 / 3, 3 / 4, 2 / 3, 4 / 2]
        expected_vector_cyclic = expected_vector_non_cyclic + [3 / 4]

        actual_vector_cyclic = rhythm.get_interval_difference_vector(cyclic=True, unit="ticks")
        actual_vector_non_cyclic = rhythm.get_interval_difference_vector(cyclic=False, unit="ticks")

        with self.subTest(cyclic=True):
            np.testing.assert_array_almost_equal(actual_vector_cyclic, expected_vector_cyclic)

        with self.subTest(cyclic=False):
            np.testing.assert_array_almost_equal(actual_vector_non_cyclic, expected_vector_non_cyclic)

    def test_onset_times(self):
        rhythm = self._get_rhythm_mock_with_23_rumba_clave_onsets(4)
        expected_onset_times = [0, 3, 7, 10, 12]
        actual_onset_times = rhythm.get_onset_times(unit="ticks")
        np.testing.assert_array_almost_equal(actual_onset_times, expected_onset_times)


class TestMonophonicRhythmImpl(TestRhythmBase):
    class MonophonicRhythmMockedSettersMixin(MonophonicRhythm, RhythmMockedSetters, metaclass=ABCMeta):
        def __init__(self):
            MonophonicRhythm.__init__(self)
            RhythmMockedSetters.__init__(self)
            self.set_onsets = MagicMock()

    class MonophonicRhythmMockedGettersMixin(MonophonicRhythm, RhythmMockedGetters, metaclass=ABCMeta):
        def __init__(self):
            MonophonicRhythm.__init__(self)
            RhythmMockedGetters.__init__(self)
            self.get_onsets = MagicMock(return_value=((1, 2), (3, 4), (4, 5)))

    @classmethod
    def get_rhythm_class(cls) -> tp.Type[RhythmBase]:
        return MonophonicRhythmImpl

    @classmethod
    def get_mocked_setters_mixin(cls):
        return cls.MonophonicRhythmMockedSettersMixin

    @classmethod
    def get_mocked_getters_mixin(cls):
        return cls.MonophonicRhythmMockedGettersMixin

    ######################################################################
    # inherits the rhythm representations from the representations mixin #
    ######################################################################

    @inject_rhythm.type
    def test_inherits_methods_from_monophonic_rhythm_representations_mixin(self, rhythm_class):
        self.assertTrue(issubclass(rhythm_class, MonophonicRhythmRepresentationsMixin),
                        "all rhythm implementations should extend the "
                        "MonophonicRhythmRepresentationsMixin but %s doesn't" % str(rhythm_class))

    ###################################
    # onsets prop <-> getters/setters #
    ###################################

    @inject_rhythm.mocked_setters()
    def test_onsets_prop_calls_setter(self, rhythm):
        onsets = (1, 2), (3, 4), (4, 5)
        rhythm.onsets = onsets
        rhythm.set_onsets.assert_called_once_with(onsets)

    @inject_rhythm.mocked_getters()
    def test_onsets_prop_calls_getter(self, rhythm):
        self.assertEqual(rhythm.onsets, rhythm.get_onsets.return_value)

    #################
    # onsets getter #
    #################

    @inject_rhythm.no_mock()
    def test_onsets_setter_raises_onsets_not_in_chronological_order(self, rhythm):
        onsets = ((1, 0), (2, 0), (3, 0), (2, 0), (4, 0))
        self.assertRaises(OnsetsNotInChronologicalOrder, rhythm.set_onsets, onsets)

    @inject_rhythm.no_mock()
    def test_onsets_setter_raises_value_error_on_invalid_input(self, rhythm):
        onset_inputs = {
            'on_missing_tick': {'velocity': 10},
            'on_string': "this is not a valid onset chain",
            'on_negative_tick': [(-3, 10), (0, 20)]
        }

        for test_name, onsets in onset_inputs.items():
            with self.subTest(name=test_name):
                self.assertRaises(ValueError, rhythm.set_onsets, onsets)

    ########################
    # onsets setter/getter #
    ########################

    @inject_rhythm.no_mock()
    def test_onsets_setter_getter_with_onset_objects(self, rhythm):
        onset_input = (mock_onset(0, 80), mock_onset(5, 14), mock_onset(6, 40), mock_onset(8, 95))
        expected_onsets = np.array([(0, 80), (5, 14), (6, 40), (8, 95)], dtype=Onset)
        rhythm.set_onsets(onset_input)
        actual_onsets = rhythm.get_onsets()
        self.assertIsNot(actual_onsets, onset_input, "set_onsets should deep copy/convert the given onsets")
        np.testing.assert_array_equal(actual_onsets, expected_onsets)

    @inject_rhythm.no_mock()
    def test_onsets_setter_getter_with_mixed_onset_formats(self, rhythm):
        onset_input = ([0, 80], (5, 14, 14), mock_onset(6, 40), iter([8, 95]), {'velocity': 23, 'tick': 10})
        rhythm.set_onsets(onset_input)
        expected_onsets = np.array([(0, 80), (5, 14), (6, 40), (8, 95), (10, 23)], dtype=Onset)
        actual_onsets = rhythm.get_onsets()
        self.assertIsNot(actual_onsets, expected_onsets, "set_onsets should deep copy/convert the given onsets")
        np.testing.assert_array_equal(actual_onsets, expected_onsets)

    ###################
    # onset rescaling #
    ###################

    @inject_rhythm.no_mock()
    def test_rescale_onsets_scales_onsets(self, rhythm):
        res_from, res_to = 16, 32
        original_onsets = np.array([(0, 80), (3, 80), (7, 80), (10, 80)], dtype=Onset)
        expected_scaled_onsets = np.array([(0, 80), (6, 80), (14, 80), (20, 80)], dtype=Onset)

        rhythm.onsets = original_onsets
        rhythm.__rescale_onset_ticks__(res_from, res_to)
        actual_scaled_onsets = rhythm.onsets

        np.testing.assert_array_equal(actual_scaled_onsets, expected_scaled_onsets)

    ###############
    # onset count #
    ###############

    @inject_rhythm.no_mock()
    def test_onset_count_zero_when_no_onsets(self, rhythm):
        self.assertEqual(0, rhythm.get_onset_count())

    @inject_rhythm.no_mock()
    def test_onset_count_with_onsets(self, rhythm):
        rhythm.set_onsets((mock_onset(1, 80), mock_onset(2, 80), mock_onset(3, 80), mock_onset(4, 80), mock_onset(5, 80), mock_onset(6, 80)))
        self.assertEqual(rhythm.get_onset_count(), 6)

    #######################
    # last onset position #
    #######################

    @inject_rhythm.no_mock()
    def test_last_onset_tick_minus_one_when_no_onsets(self, rhythm):
        self.assertEqual(rhythm.get_last_onset_tick(), -1)

    @inject_rhythm.no_mock()
    def test_last_onset_tick_with_onsets(self, rhythm):
        rhythm.set_onsets((mock_onset(12, 80), mock_onset(43, 80), mock_onset(56, 80), mock_onset(98, 80), mock_onset(101, 80), mock_onset(106, 80)))
        self.assertEqual(rhythm.get_last_onset_tick(), 106)


class FakeSlavedRhythmBaseImplementation(SlavedRhythmBase):
    def __init__(self, **kw):
        SlavedRhythmBase.__init__(self, **kw)

    def __rescale_onset_ticks__(self, old_resolution: int, new_resolution: int) -> None:
        pass

    def get_last_onset_tick(self) -> int:
        return -1

    def get_onset_count(self) -> int:
        return 0


class TestSlavedRhythm(TestRhythmBase):
    class SlavedRhythmMockedSettersMixin(RhythmMockedSetters, metaclass=ABCMeta):
        def __init__(self):
            RhythmMockedSetters.__init__(self)
            self.set_parent = MagicMock()

    class SlavedRhythmMockedGettersMixin(RhythmMockedGetters, metaclass=ABCMeta):
        def __init__(self):
            RhythmMockedGetters.__init__(self)
            self.get_parent = MagicMock()

    @classmethod
    def get_rhythm_class(cls) -> tp.Type[RhythmBase]:
        # noinspection PyTypeChecker
        return FakeSlavedRhythmBaseImplementation

    @classmethod
    def get_mocked_setters_mixin(cls):
        return cls.SlavedRhythmMockedSettersMixin

    @classmethod
    def get_mocked_getters_mixin(cls):
        return cls.SlavedRhythmMockedGettersMixin

    ###########################################
    # setters raise ParentPropertyAccessError #
    ###########################################

    @inject_rhythm.no_mock()
    def test_bpm_setter_getter(self, rhythm):
        self.assertRaises(SlavedRhythmBase.ParentPropertyAccessError, rhythm.set_bpm, 0)

    @inject_rhythm.no_mock()
    def test_resolution_setter_getter(self, rhythm):
        self.assertRaises(SlavedRhythmBase.ParentPropertyAccessError, rhythm.set_resolution, 0)

    @inject_rhythm.no_mock()
    def test_time_signature_setter_getter_with_ts_object(self, rhythm):
        self.assertRaises(SlavedRhythmBase.ParentPropertyAccessError, rhythm.set_time_signature, 0)

    @inject_rhythm.no_mock()
    def test_time_signature_setter_getter_with_tuple(self, rhythm):
        self.assertRaises(SlavedRhythmBase.ParentPropertyAccessError, rhythm.set_time_signature, 0)

    @inject_rhythm.no_mock()
    def test_duration_in_ticks_setter_getter(self, rhythm):
        self.assertRaises(SlavedRhythmBase.ParentPropertyAccessError, rhythm.set_duration_in_ticks, 0)

    ##############################
    # getters redirect to parent #
    ##############################

    @inject_rhythm.no_mock()
    def test_bpm_getter_redirects_to_parent(self, rhythm):
        parent = MagicMock(Rhythm)
        parent.bpm = 123
        parent.get_bpm.return_value = parent.bpm
        rhythm.parent = parent
        self.assertEqual(rhythm.get_bpm(), parent.bpm)

    @inject_rhythm.no_mock()
    def test_resolution_getter_redirects_to_parent(self, rhythm):
        parent = MagicMock(Rhythm)
        parent.resolution = 123
        parent.get_resolution.return_value = parent.resolution
        rhythm.parent = parent
        self.assertEqual(rhythm.get_resolution(), parent.resolution)

    @inject_rhythm.no_mock()
    def test_time_signature_getter_redirects_to_parent(self, rhythm):
        parent = MagicMock(Rhythm)
        parent.time_signature = (5, 4)
        parent.get_time_signature.return_value = parent.time_signature
        rhythm.parent = parent
        self.assertEqual(rhythm.get_time_signature(), parent.time_signature)

    @inject_rhythm.no_mock()
    def test_duration_in_ticks_getter_redirects_to_parent(self, rhythm):
        parent = MagicMock(Rhythm)
        parent.duration_in_ticks = 123
        parent.get_duration_in_ticks.return_value = parent.duration_in_ticks
        rhythm.parent = parent
        self.assertEqual(rhythm.get_duration_in_ticks(), parent.duration_in_ticks)

    ###################
    # parent property #
    ###################

    @inject_rhythm.mocked_setters(parent="I am your father")
    def test_constructor_sets_parent(self, rhythm):
        self.assertEqual(rhythm.get_parent(), "I am your father")

    @inject_rhythm.mocked_setters()
    def test_parent_prop_calls_setter(self, rhythm):
        rhythm.parent = "I am your father"
        rhythm.set_parent.assert_called_once_with("I am your father")

    @inject_rhythm.mocked_getters()
    def test_parent_prop_calls_getter(self, rhythm):
        rhythm.get_parent.return_value = "I am your father"
        self.assertEqual(rhythm.parent, "I am your father")

    #######################################################################
    # post_init tests disabled, slaved rhythms have no post_init function #
    #######################################################################

    def test_constructor_calls_post_init_with_correct_arguments(self, rhythm=None):
        pass

    def test_constructor_calls_post_init_with_no_arguments(self, rhythm=None):
        pass

    def test_post_init_calls_bpm_setter(self, rhythm=None):
        pass

    def test_post_init_calls_duration_in_ticks_setter(self, rhythm=None):
        pass

    def test_post_init_calls_duration_in_ticks_setter_when_given_duration_alias(self, rhythm=None):
        pass

    def test_post_init_calls_resolution_setter(self, rhythm=None):
        pass

    def test_post_init_calls_time_signature_setter(self, rhythm=None):
        pass

    def test_post_init_doesnt_call_setters_when_given_anything(self, rhythm=None):
        pass


class TestTrack(TestSlavedRhythm):
    @classmethod
    def get_rhythm_class(cls) -> tp.Type[Rhythm]:
        return Track

    @inject_rhythm.no_mock(track_name="kick")
    def test_track_name_equals_constructor_argument(self, rhythm):
        self.assertEqual(rhythm.get_name(), "kick")

    @inject_rhythm.no_mock()
    def test_track_name_prop_calls_getter(self, rhythm):
        rhythm.get_name = MagicMock(return_value="Juan")
        self.assertEqual(rhythm.name, "Juan")

    @inject_rhythm.no_mock()
    def test_track_name_read_only(self, rhythm):
        self.assertRaises(AttributeError, setattr, rhythm, "name", "Juan")


class TestPolyphonicRhythmImpl(TestRhythmBase):

    @classmethod
    def get_rhythm_class(cls) -> tp.Type[RhythmBase]:
        return PolyphonicRhythmImpl

    @inject_rhythm.mocked_post_init(bpm=123, resolution=234, time_signature=(5, 4), duration_in_ticks=345, duration=456)
    def test_constructor_calls_post_init_with_correct_arguments(self, rhythm):
        rhythm.post_init.assert_called_once_with(
            bpm=123,
            time_signature=(5, 4),
            duration_in_ticks=345,
            duration=456
            # resolution should not be passed to post_init, it should be handled at the PolyphonicRhythmImpl level,
            # not at the base level
        )

    @staticmethod
    def create_fake_named_track(track_name):
        track = MagicMock(Track)
        track.name = track_name
        track.get_name.return_value = track_name
        return track

    ###############################################
    # set_tracks/track_iterator/get_track_by_name #
    ###############################################

    @inject_rhythm.mocked_setters()
    def test_set_tracks_adds_all_tracks(self, rhythm):
        kick, snare, tom = tuple(self.create_fake_named_track(name) for name in ["kick", "snare", "tom"])
        rhythm.__get_track_naming_error__ = MagicMock(return_value="")
        rhythm.set_tracks((kick, snare, tom), 16)
        actual_tracks = set(rhythm.get_track_iterator())
        self.assertIn(kick, actual_tracks)
        self.assertIn(snare, actual_tracks)
        self.assertIn(tom, actual_tracks)

    @inject_rhythm.mocked_setters()
    def test_set_tracks_adds_all_tracks_same_order_as_iterator_returns(self, rhythm):
        kick, snare, tom = tuple(self.create_fake_named_track(name) for name in ["kick", "snare", "tom"])
        rhythm.__get_track_naming_error__ = MagicMock(return_value="")
        rhythm.set_tracks((kick, snare, tom), 16)
        actual_kick, actual_snare, actual_tom = rhythm.get_track_iterator()
        self.assertIs(actual_kick, kick)
        self.assertIs(actual_snare, snare)
        self.assertIs(actual_tom, tom)

    @inject_rhythm.mocked_setters()
    def test_set_tracks_adds_all_tracks_same_order_as_iterator_returns(self, rhythm):
        kick, snare, tom = tuple(self.create_fake_named_track(name) for name in ["kick", "snare", "tom"])
        rhythm.__get_track_naming_error__ = MagicMock(return_value="")
        rhythm.set_tracks((kick, snare, tom), 16)
        self.assertIs(rhythm.get_track_by_name("kick"), kick)
        self.assertIs(rhythm.get_track_by_name("snare"), snare)
        self.assertIs(rhythm.get_track_by_name("tom"), tom)

    @inject_rhythm.mocked_setters()
    def test_set_tracks_parents_tracks_to_rhythm(self, rhythm):
        tracks = tuple(self.create_fake_named_track(name) for name in ["kick", "snare", "tom"])
        rhythm.__get_track_naming_error__ = MagicMock(return_value="")
        rhythm.set_tracks(tracks, 16)

        for track in rhythm.get_track_iterator():
            with self.subTest(name="track_%s" % track.name):
                self.assertIs(track.parent, rhythm)

    @inject_rhythm.mocked_setters()
    def test_set_tracks_sets_resolution(self, rhythm):
        rhythm.set_tracks([], 128)
        rhythm.set_resolution.assert_called_once_with(128)

    @inject_rhythm.no_mock()
    def test_set_tracks_updates_duration_to_last_onset_tick(self, rhythm):
        rhythm.get_last_onset_tick = MagicMock(return_value=123)
        rhythm.set_tracks([], 0)
        self.assertEqual(rhythm.get_duration_in_ticks(), 123)

    #########################
    # track name validation #
    #########################

    @inject_rhythm.mocked_setters()
    def test_set_tracks_raises_equally_named_tracks_error(self, rhythm):
        track_a = self.create_fake_named_track("i_am_fake")
        track_b = self.create_fake_named_track("i_am_fake")
        rhythm.__get_track_naming_error__ = MagicMock(return_value="")
        self.assertRaises(PolyphonicRhythm.EquallyNamedTracksError, rhythm.set_tracks, (track_a, track_b), 16)

    @inject_rhythm.no_mock()
    def test_set_tracks_raises_illegal_track_name(self, rhythm):
        track = self.create_fake_named_track("")
        error = "that name is illegal"
        rhythm.__get_track_naming_error__ = MagicMock(return_value=error)
        self.assertRaisesRegex(PolyphonicRhythm.IllegalTrackName, error, rhythm.set_tracks, [track], 0)

    ###############
    # track count #
    ###############

    @inject_rhythm.mocked_setters()
    def test_track_count(self, rhythm):
        tracks = tuple(self.create_fake_named_track(name) for name in ["kick", "snare", "tom", "hi-hat", "crash"])
        rhythm.__get_track_naming_error__ = MagicMock(return_value="")
        rhythm.set_tracks(tracks, 0)
        self.assertEqual(rhythm.get_track_count(), 5)

    ##################
    # rescale onsets #
    ##################

    @inject_rhythm.no_mock()
    def test_set_resolution_calls_rescaling(self, rhythm):
        rhythm.get_last_onset_tick = MagicMock(return_value=0)
        rhythm.__rescale_onset_ticks__ = MagicMock()
        rhythm.__get_track_naming_error__ = MagicMock(return_value="")
        rhythm.set_tracks(tuple(self.create_fake_named_track(name) for name in ["kick", "snare", "tom"]), 16)
        rhythm.set_resolution(32)
        rhythm.__rescale_onset_ticks__.assert_called_once_with(16, 32)

    ###############
    # onset count #
    ###############

    @inject_rhythm.mocked_setters()
    def test_onset_count_sum_of_track_onset_count(self, rhythm):
        track_a, track_b = MagicMock(Track), MagicMock(Track)
        track_a.get_onset_count.return_value = 4
        track_b.get_onset_count.return_value = 7
        rhythm.__get_track_naming_error__ = MagicMock(return_value="")
        rhythm.set_tracks((track_a, track_b), 0)
        self.assertEqual(rhythm.get_onset_count(), 11)

    @inject_rhythm.no_mock()
    def test_onset_count_zero_if_no_tracks(self, rhythm):
        self.assertEqual(rhythm.get_onset_count(), 0)


class TestRhythmLoop(TestPolyphonicRhythmImpl):

    class RhythmLoopMockedSettersMixin(RhythmMockedSetters, metaclass=ABCMeta):
        def __init__(self):
            RhythmMockedSetters.__init__(self)
            self.set_name = MagicMock()

    class RhythmLoopMockedGettersMixin(RhythmMockedGetters, metaclass=ABCMeta):
        def __init__(self):
            RhythmMockedGetters.__init__(self)
            self.get_name = MagicMock()

    @classmethod
    def get_rhythm_class(cls) -> tp.Type[RhythmBase]:
        return RhythmLoop

    @classmethod
    def get_mocked_setters_mixin(cls):
        return cls.RhythmLoopMockedSettersMixin

    @classmethod
    def get_mocked_getters_mixin(cls):
        return cls.RhythmLoopMockedGettersMixin

    #################
    # name property #
    #################

    @inject_rhythm.mocked_setters()
    def test_name_prop_calls_setter(self, rhythm):
        assert rhythm.name != "Jorge"
        rhythm.name = "Jorge"
        rhythm.set_name.assert_called_once_with("Jorge")

    @inject_rhythm.mocked_getters()
    def test_name_prop_calls_getter(self, rhythm):
        assert rhythm.name != "Jorge"
        rhythm.get_name.return_value = "Jorge"
        self.assertEqual(rhythm.name, "Jorge")

    @inject_rhythm.no_mock()
    def test_name_setter_getter(self, rhythm):
        assert rhythm.get_name() != "Jorge"
        rhythm.set_name("Jorge")
        self.assertEqual(rhythm.get_name(), "Jorge")

    @inject_rhythm.no_mock(name="Jorge")
    def test_name_getter_returns_constructor_arg(self, rhythm):
        self.assertEqual(rhythm.get_name(), "Jorge")

    #################################
    # duration snapping to downbeat #
    #################################

    @inject_rhythm.type
    def _create_rhythm_mocked_ts_and_res(self, rhythm_class, time_signature=False, resolution=False):
        rhythm = rhythm_class()
        if time_signature is not False:
            rhythm.get_time_signature = MagicMock(return_value=time_signature)
        if resolution is not False:
            rhythm.get_resolution = MagicMock(return_value=resolution)
        return rhythm

    def test_duration_in_ticks_setter_permissive_when_no_downbeat_can_be_computed(self):
        rhythms = {
            'no_ts-no_res': self._create_rhythm_mocked_ts_and_res(time_signature=None, resolution=0),
            'no_ts': self._create_rhythm_mocked_ts_and_res(time_signature=None, resolution=16),
            'no_res': self._create_rhythm_mocked_ts_and_res(time_signature=TimeSignature(4, 4), resolution=0)
        }

        for test_name, rhythm in rhythms.items():
            with self.subTest(name=test_name):
                with patch.object(RhythmBase, "set_duration_in_ticks") as mock:
                    rhythm.set_duration_in_ticks(123)
                    mock.assert_called_once_with(123)

    def test_duration_in_ticks_setter_snaps_given_duration_to_next_downbeat_when_downbeat_can_be_computed(self):
        def do_subtest(name, measure_duration, given_duration, expected_duration, ts=None, res=16):
            rhythm = self._create_rhythm_mocked_ts_and_res(time_signature=ts or TimeSignature(4, 4), resolution=res)
            rhythm.get_measure_duration = MagicMock(return_value=measure_duration)
            rhythm.set_duration_in_ticks(given_duration)
            with self.subTest(name=name):
                self.assertEqual(rhythm.get_duration_in_ticks(), expected_duration)

        subtest_info = [
            ("zero-duration-not-changed", 64, 0, 0),
            ("duration-snaps-to-first-rounded-downbeat", 64, 33, 64),
            ("duration-snaps-to-first-non-rounded-downbeat", 64, 1, 64),
            ("duration-snaps-to-nth-rounded-downbeat", 64, 225, 256),
            ("duration-snaps-to-nth-non-rounded-downbeat", 64, 193, 256),
            ("duration-at-downbeat-not-changed", 64, 192, 192)
        ]

        for args in subtest_info:
            do_subtest(*args)

    def test_non_zero_duration_is_updated_as_soon_as_downbeat_is_computable(self):
        with_ts_without_res = self._create_rhythm_mocked_ts_and_res(time_signature=TimeSignature(4, 4))
        with_res_without_ts = self._create_rhythm_mocked_ts_and_res(resolution=16)

        for r in [with_ts_without_res, with_res_without_ts]:
            r.set_duration_in_ticks = MagicMock()
            r.get_duration_in_ticks = MagicMock(return_value=16)  # <- non-zero duration

        with self.subTest(name="completing_with_resolution"):
            with_ts_without_res.set_resolution(16)
            with_ts_without_res.set_duration_in_ticks.assert_called_once()

        with self.subTest(name="completing_with_time_signature"):
            with_res_without_ts.set_time_signature(TimeSignature(4, 4))
            with_res_without_ts.set_duration_in_ticks.assert_called_once()


class TestMidiRhythm(TestRhythmLoop):
    @classmethod
    def get_rhythm_class(cls) -> tp.Type[RhythmBase]:
        return MidiRhythm

    ###############################################
    # loading from midi pattern integration tests #
    ###############################################

    # noinspection SpellCheckingInspection
    @staticmethod
    def _create_midi_time_signature_mock_event(numerator, denominator, metronome=24, thirtyseconds=8, tick=0):
        ts_event = MagicMock(midi.TimeSignatureEvent)
        ts_event.get_numerator.return_value = numerator
        ts_event.get_denominator.return_value = denominator
        ts_event.get_thirtyseconds.return_value = thirtyseconds
        ts_event.get_metronome.return_value = metronome
        ts_event.tick = tick
        return ts_event

    @staticmethod
    def _create_midi_set_tempo_mock_event(bpm, tick=0):
        bpm_event = MagicMock(midi.SetTempoEvent)
        bpm_event.get_bpm.return_value = bpm
        bpm_event.tick = tick
        return bpm_event

    @staticmethod
    def _create_midi_note_on_mock_events(notes):
        for rel_tick, pitch, velocity in notes:
            event = MagicMock(midi.NoteOnEvent)
            event.tick = rel_tick
            event.get_velocity.return_value = velocity
            event.get_pitch.return_value = pitch
            yield event

    @staticmethod
    def _create_midi_key_mock(name, pitch, abbreviation):
        midi_key = MagicMock(MidiKey)
        midi_key.pitch = pitch
        midi_key.name = name
        midi_key.abbreviation = abbreviation
        return midi_key

    ##########################################
    # from/to midi pattern integration tests #
    ##########################################

    @inject_rhythm.mocked_setters(midi_mapping=MagicMock(MidiMapping))
    def test_load_midi_pattern(self, rhythm):
        kick, snare = 36, 38

        expected_ts = TimeSignature(6, 8)
        expected_bpm = 180
        expected_kick_onsets = np.array([(0, 80), (3, 40), (7, 80), (12, 80)], dtype=Onset)
        expected_snare_onsets = np.array([(2, 80), (10, 60)], dtype=Onset)
        expected_resolution = 16

        kick_key_mock = self._create_midi_key_mock("kick", kick, "kck")
        snare_key_mock = self._create_midi_key_mock("snare", snare, "snr")

        assert isinstance(rhythm.midi_mapping, MagicMock), \
            "something went wrong in the constructor, midi_mapping was not set"
        rhythm.midi_mapping.find_by_pitch = lambda pitch: kick_key_mock if pitch == kick else snare_key_mock

        tick_ix, velocity_ix = 0, 1

        midi_track = midi.Track([
            self._create_midi_time_signature_mock_event(expected_ts.numerator, expected_ts.denominator),
            self._create_midi_set_tempo_mock_event(expected_bpm),
            midi.NoteOnEvent(tick=0, pitch=kick, velocity=expected_kick_onsets[0][velocity_ix]),
            midi.NoteOnEvent(tick=2, pitch=snare, velocity=expected_snare_onsets[0][velocity_ix]),
            midi.NoteOnEvent(tick=1, pitch=kick, velocity=expected_kick_onsets[1][velocity_ix]),
            midi.NoteOnEvent(tick=4, pitch=kick, velocity=expected_kick_onsets[2][velocity_ix]),
            midi.NoteOnEvent(tick=3, pitch=snare, velocity=expected_snare_onsets[1][velocity_ix]),
            midi.NoteOnEvent(tick=2, pitch=kick, velocity=expected_kick_onsets[3][velocity_ix])
        ])

        pattern = midi.Pattern([midi_track], resolution=expected_resolution)
        rhythm.load_midi_pattern(pattern)
        kick_track, snare_track = tuple(rhythm.get_track_iterator())

        actual_kick_onsets = kick_track.get_onsets()
        actual_snare_onsets = snare_track.get_onsets()

        with self.subTest("track_onsets_set_correctly"):
            np.testing.assert_array_equal(actual_kick_onsets, expected_kick_onsets)
            np.testing.assert_array_equal(actual_snare_onsets, expected_snare_onsets)

        with self.subTest("track_onsets_tick_positions_are_unsigned_integers"):
            for kick_onset, snare_onset in zip(actual_kick_onsets, actual_snare_onsets):
                self.assertIsInstance(kick_onset[tick_ix], np.uint32,
                                      "kick onset tick position \"%s\" is not an int" % str(kick_onset[tick_ix]))
                self.assertIsInstance(snare_onset[tick_ix], np.uint32,
                                      "snare onset tick position \"%s\" is not an int" % str(snare_onset[tick_ix]))

        with self.subTest("track_names_set_correctly"):
            self.assertEqual(kick_track.get_name(), kick_key_mock.abbreviation)
            self.assertEqual(snare_track.get_name(), snare_key_mock.abbreviation)

        with self.subTest("bpm_set_correctly"):
            rhythm.set_bpm.assert_called_once_with(float(180))

        with self.subTest("time_signature_set_correctly"):
            try:
                rhythm.set_time_signature.assert_called_once_with(TimeSignature(6, 8))
            except AssertionError:
                rhythm.set_time_signature.assert_called_once_with((6, 8))

        with self.subTest("resolution_set_correctly"):
            rhythm.set_resolution.assert_called_once_with(expected_resolution)

    @inject_rhythm.mocked_getters(midi_mapping=MagicMock(MidiMapping))
    def test_as_midi_pattern(self, rhythm):
        kick, snare = 36, 38

        kick_key_mock = self._create_midi_key_mock("kick", kick, "kck")
        snare_key_mock = self._create_midi_key_mock("snare", snare, "snr")

        assert isinstance(rhythm.midi_mapping, MagicMock), \
            "something went wrong in the constructor, midi_mapping was not set"
        rhythm.midi_mapping.find_by_abbreviation = \
            lambda abbr: kick_key_mock if abbr == kick_key_mock.abbreviation else snare_key_mock

        kick_onsets = (mock_onset(0, 80), mock_onset(3, 40), mock_onset(7, 80), mock_onset(12, 80))
        snare_onsets = (mock_onset(2, 80), mock_onset(10, 60))
        time_signature = TimeSignature(6, 8)
        resolution = 16
        bpm = 180

        # noinspection PyTypeChecker
        kick_track = Track(kick_onsets, kick_key_mock.abbreviation, rhythm)
        # noinspection PyTypeChecker
        snare_track = Track(snare_onsets, snare_key_mock.abbreviation, rhythm)

        rhythm.get_track_iterator = MagicMock(return_value=iter((kick_track, snare_track)))
        rhythm.get_resolution.return_value = resolution
        rhythm.get_time_signature.return_value = time_signature
        rhythm.get_bpm.return_value = bpm
        rhythm.get_name.return_value = "Jorge"

        midi_pattern = rhythm.as_midi_pattern(note_length=0)

        self.assertEqual(midi_pattern.resolution, resolution)
        self.assertEqual(len(midi_pattern), 1)

        midi_track = midi_pattern[0]
        actual_track_name_event, actual_time_signature_event, \
            actual_tempo_event, *actual_note_events, actual_eot_event = midi_track

        self.assertIsInstance(actual_track_name_event, midi.TrackNameEvent)
        self.assertEqual(actual_track_name_event.text, "Jorge")

        self.assertIsInstance(actual_time_signature_event, midi.TimeSignatureEvent)
        self.assertEqual(actual_time_signature_event.numerator, time_signature.numerator)
        self.assertEqual(actual_time_signature_event.denominator, time_signature.denominator)

        self.assertIsInstance(actual_tempo_event, midi.SetTempoEvent)
        self.assertEqual(actual_tempo_event.bpm, midi.SetTempoEvent(bpm=float(bpm)).bpm)

        self.assertIsInstance(actual_eot_event, midi.EndOfTrackEvent)

        expected_note_events = (
            midi.NoteOnEvent(tick=0, pitch=kick, velocity=kick_onsets[0].velocity),
            midi.NoteOffEvent(tick=0, pitch=kick, velocity=0),

            midi.NoteOnEvent(tick=2, pitch=snare, velocity=snare_onsets[0].velocity),
            midi.NoteOffEvent(tick=0, pitch=snare, velocity=0),

            midi.NoteOnEvent(tick=1, pitch=kick, velocity=kick_onsets[1].velocity),
            midi.NoteOffEvent(tick=0, pitch=kick, velocity=0),

            midi.NoteOnEvent(tick=4, pitch=kick, velocity=kick_onsets[2].velocity),
            midi.NoteOffEvent(tick=0, pitch=kick, velocity=0),

            midi.NoteOnEvent(tick=3, pitch=snare, velocity=snare_onsets[1].velocity),
            midi.NoteOffEvent(tick=0, pitch=snare, velocity=0),

            midi.NoteOnEvent(tick=2, pitch=kick, velocity=kick_onsets[3].velocity),
            midi.NoteOffEvent(tick=0, pitch=kick, velocity=0)
        )  # type: tp.Tuple[midi.NoteEvent]

        self.assertEqual(len(actual_note_events), len(expected_note_events))

        for expected_event, actual_event in zip(expected_note_events, actual_note_events):
            self.assertEqual(expected_event.tick, actual_event.tick)
            if isinstance(expected_event, midi.NoteOnEvent):
                self.assertIsInstance(actual_event, midi.NoteOnEvent)
            else:
                assert isinstance(expected_event, midi.NoteOffEvent)
                self.assertIsInstance(actual_event, midi.NoteOffEvent)
            self.assertEqual(expected_event.velocity, actual_event.velocity)


# TODO Add tests for convert_time -> should always return an int when quantize is True

# TODO Add tests for MonophonicRhythmFactory

class TestOnsetStructuredArray(TestCase):

    def test_property_names(self):
        self.assertEqual(Onset.names, ("tick", "velocity"))

    def test_properties_are_unsigned_int(self):
        types = tuple(info[0] for info in Onset.fields.values())
        self.assertEqual(types, (np.uint32, np.uint32))

    # TODO Port these scaling tests to a function test for rescale_onset_series()

    # def test_scale_doesnt_affect_velocity(self):
    #     onset = Onset(123, 75)
    #     onset.scale(10, 20)
    #     self.assertEqual(onset.velocity, 75)
    #
    # def test_scale_tick_zero_stays_zero(self):
    #     onset = Onset(0, 75)
    #     onset.scale(10, 20)
    #     self.assertEqual(onset.tick, 0)
    #
    # def test_scale_tick_rounds_down(self):
    #     onset = Onset(2, 75).scale(10, 16)  # scale factor of 1.6
    #     self.assertEqual(onset.tick, 3)     # 2 * 1.6 = 3.2
    #
    # def test_scale_tick_rounds_up(self):
    #     onset = Onset(3, 75).scale(10, 16)  # scale factor of  1.6
    #     self.assertEqual(onset.tick, 5)     # 3 * 1.6 = 4.8


if __name__ == "__main__":
    main()

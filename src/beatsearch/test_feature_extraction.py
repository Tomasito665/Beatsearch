import typing as tp
from unittest import TestCase, main
from unittest.mock import MagicMock
from abc import ABCMeta, abstractmethod

# feature extractor base classes
from beatsearch.feature_extraction import FeatureExtractor, RhythmFeatureExtractor, \
    MonophonicRhythmFeatureExtractor, PolyphonicRhythmFeatureExtractor, OnsetPositionVector

# feature extractor implementations
from beatsearch.feature_extraction import BinaryOnsetVector, IOIVector, IOIHistogram, \
    BinarySchillingerChain, ChronotonicChain, IOIDifferenceVector, OnsetDensity

# misc
from beatsearch.rhythm import MonophonicRhythm, Unit
from beatsearch.test_rhythm import mock_onset


def get_rhythm_mock_with_23_rumba_clave_onsets(resolution):
    rhythm_mock = MagicMock(MonophonicRhythm)
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
    rhythm_mock.resolution = rhythm_mock.get_resolution.return_value
    rhythm_mock.get_onsets.return_value = mocked_onsets
    rhythm_mock.onsets = rhythm_mock.get_onsets.return_value
    rhythm_mock.get_duration_in_ticks.return_value = int(resolution * 4)
    rhythm_mock.duration_in_ticks = rhythm_mock.get_duration_in_ticks.return_value
    rhythm_mock.get_duration.return_value = int(resolution * 4)

    return rhythm_mock


class TestFeatureExtractor(TestCase):
    def test_not_instantiable(self):
        self.assertRaises(Exception, FeatureExtractor)


class TestRhythmFeatureExtractor(TestCase):
    def test_not_instantiable(self):
        self.assertRaises(Exception, RhythmFeatureExtractor)


class TestMonophonicRhythmFeatureExtractor(TestCase):
    def test_not_instantiable(self):
        self.assertRaises(Exception, MonophonicRhythmFeatureExtractor)


#######################################################
# Monophonic rhythm feature extractor implementations #
#######################################################


class TestMonophonicRhythmFeatureExtractorImplementationMixin(object, metaclass=ABCMeta):
    def test_instantiable(self):
        cls = self.get_impl_class()
        cls()  # shouldn't cause any problems

    def __init__(self, *args, **kw):
        # noinspection PyArgumentList
        super().__init__(*args, **kw)
        self.rhythm = get_rhythm_mock_with_23_rumba_clave_onsets(4)  # type: MonophonicRhythm
        self.feature_extractor = None  # type: tp.Union[MonophonicRhythmFeatureExtractor, None]

    # noinspection PyPep8Naming
    def setUp(self):
        cls = self.get_impl_class()
        self.feature_extractor = cls()
        self.feature_extractor.unit = "ticks"

    # noinspection PyUnresolvedReferences
    def test_unit_set_with_first_positional_constructor_argument(self):
        cls = self.get_impl_class()
        for unit in self.get_legal_units():
            with self.subTest(unit):
                obj = cls(unit)
                self.assertEqual(obj.unit, unit)

    # noinspection PyUnresolvedReferences
    def test_unit_set_with_named_constructor_argument(self):
        cls = self.get_impl_class()
        for unit in self.get_legal_units():
            with self.subTest(unit):
                obj = cls(unit=unit)
                self.assertEqual(obj.unit, unit)

    # noinspection PyUnresolvedReferences
    def test_unit_set_to_pre_processors_with_first_positional_constructor_argument(self):
        cls = self.get_impl_class()
        for unit in self.get_legal_units():
            obj = cls(unit=unit)
            for pre_processor in obj.pre_processors:
                with self.subTest("%s.%s" % (unit, pre_processor.__class__.__name__)):
                    self.assertEqual(pre_processor.unit, unit)

    # noinspection PyUnresolvedReferences
    def test_unit_set_to_preprocessors_with_named_constructor_argument(self):
        cls = self.get_impl_class()
        for unit in self.get_legal_units():
            obj = cls(unit=unit)
            for pre_processor in obj.pre_processors:
                with self.subTest("%s.%s" % (unit, pre_processor.__class__.__name__)):
                    self.assertEqual(pre_processor.unit, unit)

    # noinspection PyUnresolvedReferences
    def test_unit_property_sets_preprocessor_units(self):
        cls = self.get_impl_class()
        obj = cls()
        for unit in self.get_legal_units():
            obj.unit = unit
            for pre_processor in obj.pre_processors:
                with self.subTest("%s.%s" % (unit, pre_processor.__class__.__name__)):
                    self.assertEqual(pre_processor.unit, unit)

    @staticmethod
    def get_legal_units():
        return ["ticks"] + list(Unit.get_unit_values())

    @staticmethod
    @abstractmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        raise NotImplementedError


class TestBinaryOnsetVector(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return BinaryOnsetVector

    def test_process(self):
        expected_binary_ticks = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]
        actual_binary_ticks = self.feature_extractor.process(self.rhythm)
        self.assertEqual(actual_binary_ticks, expected_binary_ticks)


class TestIOIVector(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class():
        return IOIVector

    def test_process_in_pre_note_mode(self):
        extractor = self.feature_extractor  # type: IOIVector
        extractor.mode = IOIVector.Mode.PRE_NOTE

        expected_ioi_vector = [0, 3, 4, 3, 2]
        actual_ioi_vector = extractor.process(self.rhythm)
        self.assertEqual(actual_ioi_vector, expected_ioi_vector)

    def test_process_in_post_note_mode(self):
        extractor = self.feature_extractor  # type: IOIVector
        extractor.mode = IOIVector.Mode.POST_NOTE

        expected_ioi_vector = [3, 4, 3, 2, 4]
        actual_ioi_vector = extractor.process(self.rhythm)
        self.assertEqual(actual_ioi_vector, expected_ioi_vector)


class TestIOIHistogram(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return IOIHistogram

    def test_process(self):
        expected_histogram = (
            [1, 2, 2],  # occurrences
            [2, 3, 4]  # intervals
        )
        actual_histogram = self.feature_extractor.process(self.rhythm)
        self.assertEqual(actual_histogram, expected_histogram)


class TestBinarySchillingerChain(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return BinarySchillingerChain

    def test_process_with_values_one_zero(self):
        extractor = self.feature_extractor  # type: BinarySchillingerChain
        extractor.values = (1, 0)

        expected_chain = [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1]
        actual_chain = extractor.process(self.rhythm)
        self.assertEqual(actual_chain, expected_chain)

    def test_process_with_values_zero_one(self):
        extractor = self.feature_extractor  # type: BinarySchillingerChain
        extractor.values = (0, 1)

        expected_chain = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0]
        actual_chain = extractor.process(self.rhythm)
        self.assertEqual(actual_chain, expected_chain)


class TestChronotonicChain(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return ChronotonicChain

    def test_process(self):
        expected_chain = [3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 2, 2, 4, 4, 4, 4]
        actual_chain = self.feature_extractor.process(self.rhythm)
        self.assertEqual(actual_chain, expected_chain)


class TestIOIDifferenceVector(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return IOIDifferenceVector

    def test_process_in_non_cyclic_mode(self):
        extractor = self.feature_extractor  # type: IOIDifferenceVector
        extractor.cyclic = False

        expected_vector = [4 / 3, 3 / 4, 2 / 3, 4 / 2]
        actual_vector = extractor.process(self.rhythm)
        self.assertEqual(actual_vector, expected_vector)

    def test_process_in_cyclic_mode(self):
        extractor = self.feature_extractor  # type: IOIDifferenceVector
        extractor.cyclic = True

        expected_vector = [4 / 3, 3 / 4, 2 / 3, 4 / 2, 3 / 4]
        actual_vector = extractor.process(self.rhythm)
        self.assertEqual(actual_vector, expected_vector)


class TestOnsetPositionVector(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return OnsetPositionVector

    def test_process(self):
        expected_vector = [0, 3, 7, 10, 12]
        actual_vector = self.feature_extractor.process(self.rhythm)
        self.assertEqual(actual_vector, expected_vector)


# TODO Add test for SyncopationVector

class TestOnsetDensity(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):

    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return OnsetDensity

    def test_process(self):
        expected_onset_density = 0.3125  # rumba rhythm has 5 onsets and 16 available positions
        actual_onset_density = self.feature_extractor.process(self.rhythm)
        self.assertEqual(actual_onset_density, expected_onset_density)


class TestPolyphonicRhythmFeatureExtractor(TestCase):
    def test_not_instantiable(self):
        self.assertRaises(Exception, PolyphonicRhythmFeatureExtractor)


if __name__ == "__main__":
    main()

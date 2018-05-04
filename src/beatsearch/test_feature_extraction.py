import functools
import typing as tp
from fractions import Fraction
from unittest import TestCase, main
from unittest.mock import MagicMock, PropertyMock, patch
from collections import OrderedDict
from abc import ABCMeta, abstractmethod

# feature extractor base classes
from beatsearch.feature_extraction import FeatureExtractor, RhythmFeatureExtractor, \
    MonophonicRhythmFeatureExtractor, PolyphonicRhythmFeatureExtractor

# monophonic feature extractor implementations
from beatsearch.feature_extraction import (
    BinaryOnsetVector,
    NoteVector,
    BinarySchillingerChain,
    ChronotonicChain,
    OnsetDensity,
    OnsetPositionVector,
    SyncopationVector,
    SyncopatedOnsetRatio,
    MeanSyncopationStrength,
    IOIVector,
    IOIDifferenceVector,
    IOIHistogram
)

from beatsearch.feature_extraction import (
    MultiChannelMonophonicRhythmFeatureVector
)

# misc
from beatsearch.rhythm import Rhythm, Track, MonophonicRhythm, PolyphonicRhythm, Unit, TimeSignature
from beatsearch.test_rhythm import mock_onset


def set_rhythm_mock_properties(rhythm_mock, resolution, time_signature, duration_in_ticks):
    # resolution
    rhythm_mock.get_resolution.return_value = resolution
    rhythm_mock.resolution = resolution
    # time signature
    rhythm_mock.get_time_signature.return_value = time_signature
    rhythm_mock.time_signature = time_signature
    # duration (in ticks)
    rhythm_mock.get_duration_in_ticks.return_value = duration_in_ticks
    rhythm_mock.duration_in_ticks = duration_in_ticks
    rhythm_mock.get_duration.return_value = duration_in_ticks


def get_mono_rhythm_mock(rhythm_str: str, resolution: int, onset_char: str = "x") -> MonophonicRhythm:
    # returns rhythm with duration of one measure in 4/4

    rhythm_mock = MagicMock(MonophonicRhythm)
    onset_positions = tuple(resolution / 4.0 * i for i, c in enumerate(rhythm_str) if c == onset_char)
    mocked_onsets = tuple(mock_onset(tick, 100) for tick in onset_positions)

    rhythm_mock.get_onsets.return_value = mocked_onsets
    rhythm_mock.onsets = rhythm_mock.get_onsets.return_value
    set_rhythm_mock_properties(rhythm_mock, resolution, (4, 4), int(resolution) * 4)

    # noinspection PyTypeChecker
    return rhythm_mock


def get_mocked_track(name, onset_positions, resolution, time_signature, duration_in_ticks):
    track_mock = MagicMock(Track)
    mocked_onsets = tuple(mock_onset(tick, 100) for tick in onset_positions)

    track_mock.get_name.return_value = name
    track_mock.name = name
    track_mock.get_onsets.return_value = mocked_onsets
    track_mock.onsets = mocked_onsets

    set_rhythm_mock_properties(track_mock, resolution, time_signature, duration_in_ticks)
    return track_mock


def get_poly_rhythm_mock_with_songo_onsets(resolution):
    rhythm_mock = MagicMock(PolyphonicRhythm)
    assert resolution >= 4, "the songo rhythm is not representable with a resolution smaller than 4"
    ts = (4, 4)
    duration_in_ticks = 4 * resolution  # one measure 4/4

    kick_track_mock = get_mocked_track("kick", [
        int(resolution / 4.0 * 3),
        int(resolution / 4.0 * 6),
        int(resolution / 4.0 * 11),
        int(resolution / 4.0 * 14)
    ], resolution, ts, duration_in_ticks)

    snare_track_mock = get_mocked_track("snare", [
        int(resolution / 4.0 * 2),
        int(resolution / 4.0 * 5),
        int(resolution / 4.0 * 7),
        int(resolution / 4.0 * 9),
        int(resolution / 4.0 * 10),
        int(resolution / 4.0 * 13),
        int(resolution / 4.0 * 15)
    ], resolution, ts, duration_in_ticks)

    hihat_track_mock = get_mocked_track("hihat", [
        int(resolution / 4.0 * 0),
        int(resolution / 4.0 * 4),
        int(resolution / 4.0 * 8),
        int(resolution / 4.0 * 12)
    ], resolution, ts, duration_in_ticks)

    rhythm_mock.get_track_iterator.return_value = iter([kick_track_mock, snare_track_mock, hihat_track_mock])
    set_rhythm_mock_properties(rhythm_mock, resolution, ts, duration_in_ticks)
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


class TestRhythmFeatureExtractorImplementationMixin(object, metaclass=ABCMeta):
    def test_instantiable(self):
        cls = self.get_impl_class()
        cls()  # shouldn't cause any problems

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
    @abstractmethod
    def get_impl_class() -> tp.Type[RhythmFeatureExtractor]:
        raise NotImplementedError


#######################################################
# Monophonic rhythm feature extractor implementations #
#######################################################


class TestMonophonicRhythmFeatureExtractorImplementationMixin(TestRhythmFeatureExtractorImplementationMixin,
                                                              metaclass=ABCMeta):
    def __init__(self, *args, **kw):
        # noinspection PyArgumentList
        super().__init__(*args, **kw)
        self.rhythm = get_mono_rhythm_mock(self.get_rhythm_str(), 4)  # type: MonophonicRhythm
        self.feature_extractor = None  # type: tp.Union[MonophonicRhythmFeatureExtractor, None]

    # noinspection PyPep8Naming
    def setUp(self):
        cls = self.get_impl_class()
        self.feature_extractor = cls()
        self.feature_extractor.unit = 1/16

    @staticmethod
    def get_legal_units():
        # noinspection PyTypeChecker
        return [None] + list(Unit)

    @staticmethod
    @abstractmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        raise NotImplementedError

    @staticmethod
    def get_rhythm_str():
        return "x--x---x--x-x---"


class TestBinaryOnsetVector(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return BinaryOnsetVector

    def test_process(self):
        expected_binary_ticks = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]
        actual_binary_ticks = self.feature_extractor.process(self.rhythm)
        self.assertEqual(actual_binary_ticks, expected_binary_ticks)


class TestNoteVector(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):

    def setUp(self):
        super().setUp()

        # TODO mock time signature and its methods
        rhythm = self.rhythm
        rhythm.time_signature = TimeSignature(4, 4)
        rhythm.get_time_signature.return_value = rhythm.time_signature

    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return NoteVector

    @staticmethod
    def get_legal_units():
        # noinspection PyTypeChecker
        # Syncopation vector does not support tick-based computation
        return list(Unit)

    @staticmethod
    def create_note_vector(note_vec_str: str) -> tp.Sequence[tp.Tuple[str, int, int]]:
        note_vec_str = note_vec_str.split()
        get_e_type = lambda char: [NoteVector.NOTE, NoteVector.TIED_NOTE, NoteVector.REST]["NTR".index(char)]
        positions = functools.reduce(lambda out, x: out + [out[-1] + x], (int(s[1]) for s in note_vec_str), [0])
        return tuple((get_e_type(s[0]), int(s[1]), p) for s, p in zip(note_vec_str, positions))

    def test_defaults_to_tied_notes(self):
        extractor = self.feature_extractor  # type: NoteVector
        self.assertTrue(extractor.tied_notes)

    def test_defaults_to_cyclic(self):
        extractor = self.feature_extractor  # type: NoteVector
        self.assertTrue(extractor.cyclic)

    def test_process_with_tied_notes(self):
        extractor = self.feature_extractor  # type: NoteVector
        extractor.tied_notes = True

        expected_note_vector = self.create_note_vector("N2 T1 N1 T2 R1 N1 T2 N2 N4")
        actual_note_vector = extractor.process(self.rhythm)

        self.assertSequenceEqual(actual_note_vector, expected_note_vector)

    def test_process_without_tied_notes(self):
        extractor = self.feature_extractor  # type: NoteVector
        extractor.tied_notes = False

        expected_note_vector = self.create_note_vector("N2 R1 N1 R2 R1 N1 R2 N2 N4")
        actual_note_vector = extractor.process(self.rhythm)

        self.assertSequenceEqual(actual_note_vector, expected_note_vector)

    def test_process_cyclic_rhythm_counts_first_rest_as_tied_note(self):
        rhythm = get_mono_rhythm_mock("--x-x---x--x--x-", 4)  # rumba 23
        rhythm.get_time_signature.return_value = TimeSignature(4, 4)
        rhythm.time_signature = rhythm.get_time_signature.return_value

        extractor = self.feature_extractor  # type: NoteVector
        extractor.tied_notes = True
        extractor.cyclic = True

        self.assertEqual(self.create_note_vector("T2")[0], extractor.process(rhythm)[0])

    def test_process_non_cyclic_rhythm_doesnt_count_first_rest_as_tied_note(self):
        rhythm = get_mono_rhythm_mock("--x-x---x--x--x-", 4)  # rumba 23
        rhythm.get_time_signature.return_value = TimeSignature(4, 4)
        rhythm.time_signature = rhythm.get_time_signature.return_value

        extractor = self.feature_extractor  # type: NoteVector
        extractor.tied_notes = True
        extractor.cyclic = False

        self.assertEqual(self.create_note_vector("R2")[0], extractor.process(rhythm)[0])


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


class TestSyncopationVector(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return SyncopationVector

    @staticmethod
    def get_legal_units():
        # noinspection PyTypeChecker
        # Syncopation vector does not support tick-based computation
        return list(Unit)

    # TODO Add test_process


class TestSyncopatedOnsetRatio(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return SyncopatedOnsetRatio

    @staticmethod
    def get_legal_units():
        return TestSyncopationVector.get_legal_units()

    def test_default_ret_float(self):
        self.assertFalse(self.feature_extractor.ret_fraction, "should return a float by default")

    @patch.object(SyncopationVector, "process")
    def test_process_ret_fraction(self, mock_syncopation_vector_process):
        syncopations = "first", "second", "third"
        mock_syncopation_vector_process.return_value = syncopations

        extractor = self.feature_extractor  # type: SyncopatedOnsetRatio
        extractor.ret_fraction = True

        expected_ratio = Fraction(len(syncopations), 5)  # self.rhythm contains 5 onsets
        actual_ratio = extractor.process(self.rhythm)
        self.assertEqual(actual_ratio, expected_ratio)

    @patch.object(SyncopationVector, "process")
    def test_process_ret_float(self, mock_syncopation_vector_process):
        syncopations = "first", "second", "third"
        mock_syncopation_vector_process.return_value = syncopations

        extractor = self.feature_extractor  # type: SyncopatedOnsetRatio
        extractor.ret_fraction = False

        expected_ratio = 3 / float(5)
        actual_ratio = extractor.process(self.rhythm)
        self.assertAlmostEqual(actual_ratio, expected_ratio)


class TestMeanSyncopationStrength(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return MeanSyncopationStrength

    @staticmethod
    def get_legal_units():
        return TestSyncopationVector.get_legal_units()

    @patch.object(SyncopationVector, "process")
    def test_process(self, mock_syncopation_vector_process):
        syncopations = [0, 7], [1, 2], [2, 5]  # total sync strength = 14
        mock_syncopation_vector_process.return_value = syncopations
        self.rhythm.get_duration.return_value = 123

        expected_mean_sync_strength = 14 / 123
        actual_mean_sync_strength = self.feature_extractor.process(self.rhythm)
        self.assertEqual(actual_mean_sync_strength, expected_mean_sync_strength)


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


#######################################################
# Polyphonic rhythm feature extractor implementations #
#######################################################

class TestPolyphonicRhythmFeatureExtractorImplementationMixin(TestRhythmFeatureExtractorImplementationMixin,
                                                              metaclass=ABCMeta):
    def __init__(self, *args, **kw):
        # noinspection PyArgumentList
        super().__init__(*args, **kw)
        self.rhythm = get_poly_rhythm_mock_with_songo_onsets(4)  # type: PolyphonicRhythm
        self.feature_extractor = None  # type: tp.Union[PolyphonicRhythmFeatureExtractor, None]

    # noinspection PyPep8Naming
    def setUp(self):
        cls = self.get_impl_class()
        self.feature_extractor = cls()
        self.feature_extractor.unit = 1/16

    @staticmethod
    def get_legal_units():
        # noinspection PyTypeChecker
        return [None] + list(Unit)

    @staticmethod
    @abstractmethod
    def get_impl_class() -> tp.Type[PolyphonicRhythmFeatureExtractor]:
        raise NotImplementedError


class TestMultiChannelMonophonicRhythmFeatureVector(TestPolyphonicRhythmFeatureExtractorImplementationMixin,
                                                    TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[PolyphonicRhythmFeatureExtractor]:
        return MultiChannelMonophonicRhythmFeatureVector

    def test_process(self):
        poly_extractor = self.feature_extractor  # type: MultiChannelMonophonicRhythmFeatureVector

        mono_extractor_mock = MagicMock(MonophonicRhythmFeatureExtractor)
        mono_extractor_mock.process.side_effect = ["fake_feature_a", "fake_feature_b", "fake_feature_c"]

        expected_result_items = (
            ("kick", "fake_feature_a"),
            ("snare", "fake_feature_b"),
            ("hihat", "fake_feature_c")
        )

        poly_extractor.monophonic_extractor = mono_extractor_mock
        actual_result_items = tuple(poly_extractor.process(self.rhythm).items())
        self.assertEqual(actual_result_items, expected_result_items)


# TODO: Add tests for polyphonic syncopation vector
# TODO: Add tests for polyphonic syncopation vector (Witek)


if __name__ == "__main__":
    main()

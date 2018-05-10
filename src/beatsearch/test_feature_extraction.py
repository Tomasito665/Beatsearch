import functools
import typing as tp
from unittest import TestCase, main
from unittest.mock import MagicMock, patch
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
    MonophonicTensionVector,
    IOIVector,
    IOIDifferenceVector,
    IOIHistogram
)

from beatsearch.feature_extraction import (
    MultiTrackMonoFeature
)

# misc
from beatsearch.rhythm import Track, MonophonicRhythm, PolyphonicRhythm, Unit, TimeSignature
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

    # TODO Mock time signature
    set_rhythm_mock_properties(rhythm_mock, resolution, TimeSignature(4, 4), int(resolution) * 4)

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


def get_poly_rhythm_mock_with_songo_onsets(resolution: int = 4):
    rhythm_mock = MagicMock(spec=PolyphonicRhythm)
    assert resolution >= 4, "the songo rhythm is not representable with a resolution smaller than 4"
    ts = TimeSignature(4, 4)  # TODO Mock time signature
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

    tracks = kick_track_mock, snare_track_mock, hihat_track_mock
    rhythm_mock.get_track_iterator.return_value = iter(tracks)
    rhythm_mock.get_track_count.return_value = len(tracks)
    rhythm_mock.get_track_names.return_value = tuple(t.name for t in tracks)
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


# noinspection PyUnresolvedReferences
class TestRhythmFeatureExtractorImplementationMixin(object, metaclass=ABCMeta):
    def test_instantiable(self):
        cls = self.get_impl_class()
        cls()  # shouldn't cause any problems

    def test_unit_set_with_first_positional_constructor_argument(self):
        for unit in Unit:
            with self.subTest(unit):
                obj = self.get_impl_class()(unit)
                self.assertEqual(obj.unit, unit)

    def test_unit_set_with_named_constructor_argument(self):
        for unit in Unit:
            with self.subTest(unit):
                obj = self.get_impl_class()(unit=unit)
                self.assertEqual(obj.unit, unit)

    def test_unit_set_to_aux_extractors_with_first_positional_argument(self):
        for unit in Unit:
            obj = self.get_impl_class()(unit)
            for i, aux_extractor in enumerate(obj.get_auxiliary_extractors()):
                with self.subTest("aux_%i<%s>.%s" % (i, aux_extractor.__class__.__name__, unit)):
                    self.assertEqual(aux_extractor.unit, unit)

    def test_unit_set_to_aux_extractors_with_named_constructor_argument(self):
        for unit in Unit:
            obj = self.get_impl_class()(unit=unit)
            for i, aux_extractor in enumerate(obj.get_auxiliary_extractors()):
                with self.subTest("aux_%i<%s>.%s" % (i, aux_extractor.__class__.__name__, unit)):
                    self.assertEqual(aux_extractor.unit, unit)

    def test_auto_aux_registration_when_passed_master_to_constructor(self):
        master_mock = MagicMock(FeatureExtractor)
        obj = self.get_impl_class()(aux_to=master_mock)
        master_mock.register_auxiliary_extractor.assert_called_once_with(obj)

    def test_set_unit_to_none_if_tick_based_computation_is_supported(self):
        obj = self.get_impl_class()()
        if obj.supports_tick_based_computation():
            obj.unit = None
            self.assertIsNone(obj.unit)

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
        self.assertSequenceEqual(actual_binary_ticks, expected_binary_ticks)


class TestNoteVector(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return NoteVector

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

        extractor = self.feature_extractor  # type: NoteVector
        extractor.tied_notes = True
        extractor.cyclic = True

        self.assertEqual(self.create_note_vector("T2")[0], extractor.process(rhythm)[0])

    def test_process_non_cyclic_rhythm_doesnt_count_first_rest_as_tied_note(self):
        rhythm = get_mono_rhythm_mock("--x-x---x--x--x-", 4)  # rumba 23

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
        extractor.mode = IOIVector.PRE_NOTE

        expected_ioi_vector = [0, 3, 4, 3, 2]
        actual_ioi_vector = extractor.process(self.rhythm)
        self.assertSequenceEqual(actual_ioi_vector, expected_ioi_vector)

    def test_process_in_post_note_mode(self):
        extractor = self.feature_extractor  # type: IOIVector
        extractor.mode = IOIVector.POST_NOTE

        expected_ioi_vector = [3, 4, 3, 2, 4]
        actual_ioi_vector = extractor.process(self.rhythm)
        self.assertSequenceEqual(actual_ioi_vector, expected_ioi_vector)


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
        self.assertSequenceEqual(actual_chain, expected_chain)

    def test_process_with_values_zero_one(self):
        extractor = self.feature_extractor  # type: BinarySchillingerChain
        extractor.values = (0, 1)

        expected_chain = [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0]
        actual_chain = extractor.process(self.rhythm)
        self.assertSequenceEqual(actual_chain, expected_chain)


class TestChronotonicChain(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return ChronotonicChain

    def test_process(self):
        expected_chain = [3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 2, 2, 4, 4, 4, 4]
        actual_chain = self.feature_extractor.process(self.rhythm)
        self.assertSequenceEqual(actual_chain, expected_chain)


class TestIOIDifferenceVector(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return IOIDifferenceVector

    def test_process_in_non_cyclic_mode(self):
        extractor = self.feature_extractor  # type: IOIDifferenceVector
        extractor.cyclic = False

        expected_vector = [4 / 3, 3 / 4, 2 / 3, 4 / 2]
        actual_vector = extractor.process(self.rhythm)
        self.assertSequenceEqual(actual_vector, expected_vector)

    def test_process_in_cyclic_mode(self):
        extractor = self.feature_extractor  # type: IOIDifferenceVector
        extractor.cyclic = True

        expected_vector = [4 / 3, 3 / 4, 2 / 3, 4 / 2, 3 / 4]
        actual_vector = extractor.process(self.rhythm)
        self.assertSequenceEqual(actual_vector, expected_vector)


class TestOnsetPositionVector(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return OnsetPositionVector

    def test_process(self):
        expected_vector = [0, 3, 7, 10, 12]
        actual_vector = self.feature_extractor.process(self.rhythm)
        self.assertSequenceEqual(actual_vector, expected_vector)


class TestSyncopationVector(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return SyncopationVector

    # TODO Add test_process


class TestSyncopatedOnsetRatio(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return SyncopatedOnsetRatio

    def test_default_ret_float(self):
        self.assertFalse(self.feature_extractor.ret_fraction, "should return a float by default")

    @patch.object(SyncopationVector, "__process__")
    def test_process_ret_fraction(self, mock_syncopation_vector_process):
        syncopations = "first", "second", "third"
        mock_syncopation_vector_process.return_value = syncopations

        extractor = self.feature_extractor  # type: SyncopatedOnsetRatio
        extractor.ret_fraction = True

        expected_ratio = (len(syncopations), 5)  # self.rhythm contains 5 onsets
        actual_ratio = extractor.process(self.rhythm)
        self.assertEqual(actual_ratio, expected_ratio)

    @patch.object(SyncopationVector, "__process__")
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

    @patch.object(SyncopationVector, "__process__")
    def test_process(self, mock_syncopation_vector_process):
        syncopations = [0, 7], [1, 2], [2, 5]  # total sync strength = 14
        mock_syncopation_vector_process.return_value = syncopations
        self.rhythm.get_duration.return_value = 124

        expected_mean_sync_strength = 14 / 124
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


# TODO Add tests for MonophonicOnsetLikelihoodVector


class TestMonophonicTensionVector(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return MonophonicTensionVector

    @staticmethod
    def get_rhythm_str():
        return "x--x--x------x--"

    def test_defaults_to_cyclic(self):
        extractor = self.feature_extractor  # type: MonophonicTensionVector
        self.assertTrue(extractor.cyclic)

    def test_defaults_to_equal_upbeats_salience_prf(self):
        extractor = self.feature_extractor  # type: MonophonicTensionVector
        self.assertEqual(extractor.salience_profile_type, "equal_upbeats")

    SEMI_QUAVER_EQUAL_UPBEAT_SALIENCE_PRF = [0, -3, -2, -3, -1, -3, -2, -3, -1, -3, -2, -3, -1, -3, -2, -3]

    @patch.object(TimeSignature, "get_salience_profile", return_value=SEMI_QUAVER_EQUAL_UPBEAT_SALIENCE_PRF)
    def test_process(self, _):
        extractor = self.feature_extractor  # type: MonophonicTensionVector

        expected_tension_vector = tuple(int(c) for c in "0003332222221333")
        actual_tension_vector = extractor.process(self.rhythm)

        self.assertSequenceEqual(actual_tension_vector, expected_tension_vector)

    @patch.object(TimeSignature, "get_salience_profile", return_value=SEMI_QUAVER_EQUAL_UPBEAT_SALIENCE_PRF)
    def test_process_tension_during_first_tied_note_event_equals_last_with_cyclic_rhythm(self, _):
        rhythm = get_mono_rhythm_mock("----x-x-----x--x", 4)
        extractor = self.feature_extractor  # type: MonophonicTensionVector
        extractor.cyclic = True

        tension_vector = extractor.process(rhythm)
        self.assertSequenceEqual(tension_vector[0:4], [tension_vector[-1]] * 4)

    @patch.object(TimeSignature, "get_salience_profile", return_value=SEMI_QUAVER_EQUAL_UPBEAT_SALIENCE_PRF)
    def test_process_tension_during_first_tied_note_event_is_zero_with_non_cyclic_rhythm(self, _):
        rhythm = get_mono_rhythm_mock("----x-x-----x--x", 4)
        extractor = self.feature_extractor  # type: MonophonicTensionVector
        extractor.cyclic = False

        tension_vector = extractor.process(rhythm)
        self.assertSequenceEqual(tension_vector[0:4], [0] * 4)

# TODO Add tests for MonophonicTension
# TODO Add tests for MonophonicVariationVector


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
    @abstractmethod
    def get_impl_class() -> tp.Type[PolyphonicRhythmFeatureExtractor]:
        raise NotImplementedError


class TestMultiTrackMonoFeature(TestCase):
    def test_constructor_instantiates_mono_extractor_cls_forwarding_args_and_kwargs(self):
        mock_mono_extr_cls = MagicMock(spec=MonophonicRhythmFeatureExtractor.__class__)

        with patch("inspect.isclass", return_value=True):
            # noinspection PyTypeChecker
            MultiTrackMonoFeature(
                mock_mono_extr_cls, "positional_a", "positional_b",
                named_a="fake_a", named_b="fake_b"
            )

        mock_mono_extr_cls.assert_called_once_with(
            "positional_a", "positional_b",
            named_a="fake_a", named_b="fake_b"
        )

    def test_returns_mono_cls_and_mono_factors(self):
        mock_mono_extr_cls = MagicMock(spec=MonophonicRhythmFeatureExtractor.__class__)
        mock_mono_extr_obj = MagicMock(spec=MonophonicRhythmFeatureExtractor)
        mock_mono_extr_cls.return_value = mock_mono_extr_obj

        with patch("inspect.isclass", return_value=True):
            # noinspection PyTypeChecker
            mt_mono_feature_extr = MultiTrackMonoFeature(mock_mono_extr_cls)

        mono_factors = "fake-factor-1", "fake-factor-2", "fake-factor-3"
        expected_mt_factors = mock_mono_extr_obj.__class__, mono_factors
        mock_mono_extr_obj.__get_factors__.return_value = mono_factors
        actual_mt_factors = mt_mono_feature_extr.__get_factors__()
        self.assertSequenceEqual(actual_mt_factors, expected_mt_factors)

    def test_process_returns_mono_process_per_track(self):
        mock_mono_extr_cls = MagicMock(spec=MonophonicRhythmFeatureExtractor.__class__)
        mock_mono_extr_obj = MagicMock(spec=MonophonicRhythmFeatureExtractor)
        mock_mono_extr_cls.return_value = mock_mono_extr_obj

        with patch("inspect.isclass", return_value=True):
            # noinspection PyTypeChecker
            mt_mono_feature_extr = MultiTrackMonoFeature(mock_mono_extr_cls)

        mock_mono_extr_obj.__process__ = lambda fake_track, *_: "fake-%s-feature" % fake_track.name
        rhythm = get_poly_rhythm_mock_with_songo_onsets()
        expected_feature = tuple("fake-%s-feature" % name for name in rhythm.get_track_names())
        actual_feature = mt_mono_feature_extr.process(rhythm)

        self.assertSequenceEqual(actual_feature, expected_feature)

    def test_mono_extractor_registered_as_auxiliary_extractor(self):
        mock_mono_extr_cls = MagicMock(spec=MonophonicRhythmFeatureExtractor.__class__)
        mock_mono_extr_obj = MagicMock(spec=MonophonicRhythmFeatureExtractor)
        mock_mono_extr_cls.return_value = mock_mono_extr_obj

        with patch("inspect.isclass", return_value=True):
            # noinspection PyTypeChecker
            mt_mono_feature_extr = MultiTrackMonoFeature(mock_mono_extr_cls)

        expected_auxiliary_extractors = [mock_mono_extr_obj]
        actual_auxiliary_extractors = [*mt_mono_feature_extr.get_auxiliary_extractors(None)]
        self.assertSequenceEqual(actual_auxiliary_extractors, expected_auxiliary_extractors)


# TODO: Add tests for polyphonic syncopation vector
# TODO: Add tests for polyphonic syncopation vector (Witek)
# TODO: Add tests for polyphonic tension vector
# TODO: Add tests for polyphonic tension


if __name__ == "__main__":
    main()


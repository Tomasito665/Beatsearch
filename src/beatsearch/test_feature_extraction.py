import typing as tp
import numpy.testing as npt
from unittest import TestCase, main
from unittest.mock import MagicMock, patch, PropertyMock
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
    MonophonicOnsetLikelihoodVector,
    MonophonicVariabilityVector,
    MonophonicSyncopationVector,
    SyncopatedOnsetRatio,
    MeanMonophonicSyncopationStrength,
    MonophonicMetricalTensionVector,
    MonophonicMetricalTensionMagnitude,
    IOIVector,
    IOIDifferenceVector,
    IOIHistogram
)

from beatsearch.feature_extraction import (
    MultiTrackMonoFeature,
    PolyphonicMetricalTensionVector,
    PolyphonicMetricalTensionMagnitude,
    DistantPolyphonicSyncopationVector,
)

# misc
from beatsearch.rhythm import MonophonicRhythm, PolyphonicRhythm, Unit, TimeSignature
from beatsearch.test_rhythm import get_mono_rhythm_mock, get_poly_rhythm_mock_with_songo_onsets, \
    get_poly_rhythm_mock, get_mocked_onsets


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

    def test_defaults_to_tied_notes(self):
        extractor = self.feature_extractor  # type: NoteVector
        self.assertTrue(extractor.tied_notes)

    def test_defaults_to_cyclic(self):
        extractor = self.feature_extractor  # type: NoteVector
        self.assertTrue(extractor.cyclic)

    def test_process_with_tied_notes(self):
        extractor = self.feature_extractor  # type: NoteVector
        extractor.tied_notes = True

        expected_note_vector = create_note_vector_from_string("N2 T1 N1 T2 R1 N1 T2 N2 N4")
        actual_note_vector = extractor.process(self.rhythm)

        self.assertSequenceEqual(actual_note_vector, expected_note_vector)

    def test_process_without_tied_notes(self):
        extractor = self.feature_extractor  # type: NoteVector
        extractor.tied_notes = False

        expected_note_vector = create_note_vector_from_string("N2 R1 N1 R2 R1 N1 R2 N2 N4")
        actual_note_vector = extractor.process(self.rhythm)

        self.assertSequenceEqual(actual_note_vector, expected_note_vector)

    def test_process_cyclic_rhythm_counts_first_rest_as_tied_note(self):
        rhythm = get_mono_rhythm_mock("--x-x---x--x--x-", 4)  # rumba 23

        extractor = self.feature_extractor  # type: NoteVector
        extractor.tied_notes = True
        extractor.cyclic = True

        self.assertEqual(create_note_vector_from_string("T2")[0], extractor.process(rhythm)[0])

    def test_process_non_cyclic_rhythm_doesnt_count_first_rest_as_tied_note(self):
        rhythm = get_mono_rhythm_mock("--x-x---x--x--x-", 4)  # rumba 23

        extractor = self.feature_extractor  # type: NoteVector
        extractor.tied_notes = True
        extractor.cyclic = False

        self.assertEqual(create_note_vector_from_string("R2")[0], extractor.process(rhythm)[0])


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


def create_note_vector_from_string(note_vector_str: str, note_char: str = "N",
                                   tied_note_char: str = "T", rest_char: str = "R"):
    """Creates a note vector from the given string where each space-separated word represents a musical event. The
    events must be given as (at least) two-character words where the first character is one of the given note_char,
    tied-note char or rest char. The remaining characters must be interpretable as an integer and specify the duration
    of the event. The positions of the events are set automatically, cumulatively.
    """

    event_type_lookup_table = {
        note_char: NoteVector.NOTE,
        tied_note_char: NoteVector.TIED_NOTE,
        rest_char: NoteVector.REST,
    }

    note_vector = []
    step = 0

    for event_string in note_vector_str.split():
        assert len(event_string) >= 2
        e_type_char, *e_duration_str_chars = event_string
        e_duration_str = "".join(e_duration_str_chars)
        e_type = event_type_lookup_table[e_type_char]
        e_duration = int(e_duration_str)
        note_vector.append((e_type, e_duration, step))
        step += e_duration

    return tuple(note_vector)


class TestSyncopationVector(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return MonophonicSyncopationVector

    @staticmethod
    def get_rhythm_str():
        # Same rhythm as we manually computed the note vector for (see SEMI_QUAVER_NOTE_VECTOR)
        return "---x---x--x-x---"

    def test_defaults_to_equal_upbeats_salience_profile_type(self):
        extractor = self.feature_extractor  # type: MonophonicSyncopationVector
        self.assertEqual(extractor.salience_profile_type, "equal_upbeats")

    def test_defaults_to_cyclic(self):
        extractor = self.feature_extractor  # type: MonophonicSyncopationVector
        self.assertTrue(extractor.cyclic)

    SEMI_QUAVER_NOTE_VECTOR = create_note_vector_from_string("T2 R1 N1 T2 R1 N1 T2 N2 N4")
    SEMI_QUAVER_EQUAL_UPBEAT_SALIENCE_PRF = [0, -3, -2, -3, -1, -3, -2, -3, -1, -3, -2, -3, -1, -3, -2, -3]

    @patch.object(NoteVector, "__process__", return_value=SEMI_QUAVER_NOTE_VECTOR)
    @patch.object(TimeSignature, "get_salience_profile", return_value=SEMI_QUAVER_EQUAL_UPBEAT_SALIENCE_PRF)
    def test_process_cyclic(self, *_):
        extractor = self.feature_extractor  # type: MonophonicSyncopationVector
        extractor.cyclic = True
        expected_sync_vector = [(2, 3, 4), (2, 7, 8), (1, 12, 0)]
        actual_sync_vector = extractor.process(self.rhythm)
        self.assertSequenceEqual(actual_sync_vector, expected_sync_vector)

    @patch.object(NoteVector, "__process__", return_value=SEMI_QUAVER_NOTE_VECTOR)
    @patch.object(TimeSignature, "get_salience_profile", return_value=SEMI_QUAVER_EQUAL_UPBEAT_SALIENCE_PRF)
    def test_process_non_cyclic(self, *_):
        extractor = self.feature_extractor  # type: MonophonicSyncopationVector
        extractor.cyclic = False
        expected_sync_vector = [(2, 3, 4), (2, 7, 8)]
        actual_sync_vector = extractor.process(self.rhythm)
        self.assertSequenceEqual(actual_sync_vector, expected_sync_vector)


class TestSyncopatedOnsetRatio(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return SyncopatedOnsetRatio

    def test_default_ret_float(self):
        self.assertFalse(self.feature_extractor.ret_fraction, "should return a float by default")

    @patch.object(MonophonicSyncopationVector, "__process__")
    def test_process_ret_fraction(self, mock_syncopation_vector_process):
        syncopations = "first", "second", "third"
        mock_syncopation_vector_process.return_value = syncopations

        extractor = self.feature_extractor  # type: SyncopatedOnsetRatio
        extractor.ret_fraction = True

        expected_ratio = (len(syncopations), 5)  # self.rhythm contains 5 onsets
        actual_ratio = extractor.process(self.rhythm)
        self.assertEqual(actual_ratio, expected_ratio)

    @patch.object(MonophonicSyncopationVector, "__process__")
    def test_process_ret_float(self, mock_syncopation_vector_process):
        syncopations = "first", "second", "third"
        mock_syncopation_vector_process.return_value = syncopations

        extractor = self.feature_extractor  # type: SyncopatedOnsetRatio
        extractor.ret_fraction = False

        expected_ratio = 3 / float(5)
        actual_ratio = extractor.process(self.rhythm)
        self.assertAlmostEqual(actual_ratio, expected_ratio)


class TestMeanMonophonicSyncopationStrength(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return MeanMonophonicSyncopationStrength

    @patch.object(MonophonicSyncopationVector, "__process__")
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


class TestMonophonicOnsetLikelihoodVector(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return MonophonicOnsetLikelihoodVector

    @staticmethod
    def get_rhythm_str():
        return "x-x-x-x-x-xxx-xx"

    def test_defaults_to_cyclic_priors(self):
        extractor = self.feature_extractor  # type: MonophonicOnsetLikelihoodVector
        self.assertEqual(extractor.priors, "cyclic")

    def test_process_with_cyclic_priors(self):
        extractor = self.feature_extractor  # type: MonophonicOnsetLikelihoodVector
        extractor.priors = "cyclic"
        expected_onset_likelihood_vec = (1.0, 0.2, 0.8666667, 0.8, 0.8, 0.0, 0.8, 0.4666667,
                                         0.8, 0.0, 0.8, 0.0, 0.6666667, 0.1333333, 0.6, 0.2)
        actual_onset_likelihood_vec = extractor.process(self.rhythm)
        self.assertEqual(len(actual_onset_likelihood_vec), len(expected_onset_likelihood_vec))
        for actual_likelihood, expected_likelihood in zip(actual_onset_likelihood_vec, expected_onset_likelihood_vec):
            self.assertAlmostEqual(actual_likelihood, expected_likelihood)

    def test_process_with_optimistic_priors(self):
        extractor = self.feature_extractor  # type: MonophonicOnsetLikelihoodVector
        extractor.priors = "optimistic"
        expected_onset_likelihood_vec = (1.0, 0.0666667, 0.9333333, 0.0, 0.9333333, 0.0, 0.9333333, 0.0,
                                         0.9333333, 0.0, 0.9333333, 0.0, 0.8, 0.1333333, 0.7333333, 0.2)
        actual_onset_likelihood_vec = extractor.process(self.rhythm)
        self.assertEqual(len(actual_onset_likelihood_vec), len(expected_onset_likelihood_vec))
        for actual_likelihood, expected_likelihood in zip(actual_onset_likelihood_vec, expected_onset_likelihood_vec):
            self.assertAlmostEqual(actual_likelihood, expected_likelihood)

    def test_process_with_pessimistic_priors(self):
        extractor = self.feature_extractor  # type: MonophonicOnsetLikelihoodVector
        extractor.priors = "pessimistic"
        expected_onset_likelihood_vec = (0.0, 0.7333333, 0.0, 0.4, 0.1333333, 0.2, 0.1333333, 0.0666667,
                                         0.4, 0.0, 0.4, 0.0, 0.2666667, 0.1333333, 0.2, 0.2)
        actual_onset_likelihood_vec = extractor.process(self.rhythm)
        self.assertEqual(len(actual_onset_likelihood_vec), len(expected_onset_likelihood_vec))
        for actual_likelihood, expected_likelihood in zip(actual_onset_likelihood_vec, expected_onset_likelihood_vec):
            self.assertAlmostEqual(actual_likelihood, expected_likelihood)


class TestMonophonicVariabilityVector(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return MonophonicVariabilityVector

    @staticmethod
    def get_rhythm_str():
        return "x-x-x-x-x-xxx-xx"

    def test_defaults_to_cyclic_priors(self):
        extractor = self.feature_extractor  # type: MonophonicVariabilityVector
        self.assertEqual(extractor.priors, "cyclic")

    @patch.object(MonophonicOnsetLikelihoodVector, "__process__")
    @patch.object(BinaryOnsetVector, "__process__")
    def test_process(self, binary_onset_process_mock, onset_likelihood_process_mock):
        extractor = self.feature_extractor  # type: MonophonicVariabilityVector
        onset_likelihood_process_mock.return_value = (0.1, 0.3, 0.03, 0.99, 0.07, 0.00, 0.1, 1.0)
        binary_onset_process_mock.return_value = (0, 1, 1, 0, 0, 1, 0, 0)
        expected_variability_vec = (0.1, 0.7, 0.97, 0.99, 0.07, 1.00, 0.1, 1.0)
        actual_variability_vec = extractor.process(self.rhythm)
        self.assertEqual(actual_variability_vec, expected_variability_vec)


class TestMonophonicMetricalTensionVector(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return MonophonicMetricalTensionVector

    @staticmethod
    def get_rhythm_str():
        return "x--x--x------x--"

    def test_defaults_to_cyclic(self):
        extractor = self.feature_extractor  # type: MonophonicMetricalTensionVector
        self.assertTrue(extractor.cyclic)

    def test_defaults_to_equal_upbeats_salience_prf(self):
        extractor = self.feature_extractor  # type: MonophonicMetricalTensionVector
        self.assertEqual(extractor.salience_profile_type, "equal_upbeats")

    def test_defaults_to_not_normalized(self):
        extractor = self.feature_extractor  # type: MonophonicMetricalTensionVector
        self.assertFalse(extractor.normalize)

    SEMI_QUAVER_EQUAL_UPBEAT_SALIENCE_PRF = [0, -3, -2, -3, -1, -3, -2, -3, -1, -3, -2, -3, -1, -3, -2, -3]

    @patch.object(TimeSignature, "get_salience_profile", return_value=SEMI_QUAVER_EQUAL_UPBEAT_SALIENCE_PRF)
    def test_process(self, _):
        extractor = self.feature_extractor  # type: MonophonicMetricalTensionVector
        extractor.cyclic = True

        expected_tension_vector = tuple(int(c) for c in "0003332222221333")
        actual_tension_vector = extractor.process(self.rhythm)

        self.assertSequenceEqual(actual_tension_vector, expected_tension_vector)

    @patch.object(TimeSignature, "get_salience_profile", return_value=SEMI_QUAVER_EQUAL_UPBEAT_SALIENCE_PRF)
    def test_process_tension_during_first_tied_note_event_equals_last_with_cyclic_rhythm(self, _):
        rhythm = get_mono_rhythm_mock("----x-x-----x--x", 4)
        extractor = self.feature_extractor  # type: MonophonicMetricalTensionVector
        extractor.cyclic = True

        tension_vector = extractor.process(rhythm)
        self.assertSequenceEqual(tension_vector[0:4], [tension_vector[-1]] * 4)

    @patch.object(TimeSignature, "get_salience_profile", return_value=SEMI_QUAVER_EQUAL_UPBEAT_SALIENCE_PRF)
    def test_process_tension_during_first_tied_note_event_is_zero_with_non_cyclic_rhythm(self, _):
        rhythm = get_mono_rhythm_mock("----x-x-----x--x", 4)
        extractor = self.feature_extractor  # type: MonophonicMetricalTensionVector
        extractor.cyclic = False

        tension_vector = extractor.process(rhythm)
        self.assertSequenceEqual(tension_vector[0:4], [0] * 4)


class TestMonophonicMetricalTensionMagnitude(TestMonophonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[MonophonicRhythmFeatureExtractor]:
        return MonophonicMetricalTensionMagnitude

    @staticmethod
    def get_rhythm_str():
        return "x--x--x------x--"

    def test_defaults_to_cyclic(self):
        extractor = self.feature_extractor  # type: MonophonicMetricalTensionMagnitude
        self.assertTrue(extractor.cyclic)

    def test_defaults_to_equal_upbeats_salience_prf(self):
        extractor = self.feature_extractor  # type: MonophonicMetricalTensionMagnitude
        self.assertEqual(extractor.salience_profile_type, "equal_upbeats")

    @patch.object(MonophonicMetricalTensionVector, "__process__")
    def test_process_not_normalized(self, mock_metrical_tension_vec_process):
        extractor = self.feature_extractor  # type: MonophonicMetricalTensionVector
        extractor.normalize = False
        mock_metrical_tension_vec_process.return_value = (0/6, 0/6, 0/6, 6/6, 6/6, 1/6, 2/6, 4/6,
                                                          4/6, 4/6, 5/6, 1/6, 4/6, 6/6, 0/6, 6/6)
        expected_vector_magnitude = 2.576604138956718
        actual_vector_magnitude = extractor.process(self.rhythm)
        self.assertAlmostEqual(actual_vector_magnitude, expected_vector_magnitude)

    @patch.object(MonophonicMetricalTensionVector, "__process__")
    def test_process_normalized(self, mock_metrical_tension_vec_process):
        extractor = self.feature_extractor  # type: MonophonicMetricalTensionVector
        extractor.normalize = True
        mock_metrical_tension_vec_process.return_value = (0/6, 0/6, 0/6, 6/6, 6/6, 1/6, 2/6, 4/6,
                                                          4/6, 4/6, 5/6, 1/6, 4/6, 6/6, 0/6, 6/6)
        expected_vector_magnitude = 0.6441510347391795
        actual_vector_magnitude = extractor.process(self.rhythm)
        self.assertAlmostEqual(actual_vector_magnitude, expected_vector_magnitude)


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
        self.rhythm = get_poly_rhythm_mock_with_songo_onsets()  # type: PolyphonicRhythm
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

    def test_defaults_to_per_track(self):
        mock_mono_extr_cls = MagicMock(spec=MonophonicRhythmFeatureExtractor.__class__)
        with patch("inspect.isclass", return_value=True):
            # noinspection PyTypeChecker
            mt_mono_feature_extr = MultiTrackMonoFeature(mock_mono_extr_cls)
        self.assertEqual(mt_mono_feature_extr.multi_track_mode, MultiTrackMonoFeature.PER_TRACK)

    def test_returns_own_and_mono_factors(self):
        mock_mono_extr_cls = MagicMock(spec=MonophonicRhythmFeatureExtractor.__class__)
        mock_mono_extr_obj = MagicMock(spec=MonophonicRhythmFeatureExtractor)
        mock_mono_extr_cls.return_value = mock_mono_extr_obj
        fake_multi_track_mode = "fake-mt-mode"

        with patch("inspect.isclass", return_value=True):
            # noinspection PyTypeChecker
            mt_mono_feature_extr = MultiTrackMonoFeature(mock_mono_extr_cls)

        own_factors = mock_mono_extr_obj.__class__, fake_multi_track_mode
        mono_factors = "fake-factor-1", "fake-factor-2", "fake-factor-3"
        mock_mono_extr_obj.__get_factors__.return_value = mono_factors
        expected_mt_factors = own_factors, mono_factors

        with patch("beatsearch.feature_extraction.MultiTrackMonoFeature.multi_track_mode",
                   new_callable=PropertyMock) as mock_mt_mode_prop:
            mock_mt_mode_prop.return_value = fake_multi_track_mode
            actual_mt_factors = mt_mono_feature_extr.__get_factors__()

        self.assertSequenceEqual(actual_mt_factors, expected_mt_factors)

    def test_process_per_track(self):
        mock_mono_extr_cls = MagicMock(spec=MonophonicRhythmFeatureExtractor.__class__)
        mock_mono_extr_obj = MagicMock(spec=MonophonicRhythmFeatureExtractor)
        mock_mono_extr_cls.return_value = mock_mono_extr_obj

        with patch("inspect.isclass", return_value=True):
            # noinspection PyTypeChecker
            mt_mono_feature_extr = MultiTrackMonoFeature(mock_mono_extr_cls)
            mt_mono_feature_extr.multi_track_mode = MultiTrackMonoFeature.PER_TRACK

        mock_mono_extr_obj.__process__ = lambda fake_track, *_: "fake-%s-feature" % fake_track.name
        rhythm = get_poly_rhythm_mock_with_songo_onsets()
        expected_feature = tuple("fake-%s-feature" % name for name in rhythm.get_track_names())
        actual_feature = mt_mono_feature_extr.process(rhythm)

        self.assertSequenceEqual(actual_feature, expected_feature)

    def test_process_per_track_combination(self):
        mock_mono_extr_cls = MagicMock(spec=MonophonicRhythmFeatureExtractor.__class__)
        mock_mono_extr_obj = MagicMock(spec=MonophonicRhythmFeatureExtractor)
        mock_mono_extr_cls.return_value = mock_mono_extr_obj

        with patch("inspect.isclass", return_value=True):
            # noinspection PyTypeChecker
            mt_mono_feature_extr = MultiTrackMonoFeature(mock_mono_extr_cls)
            mt_mono_feature_extr.multi_track_mode = MultiTrackMonoFeature.PER_TRACK_COMBINATION

        # Let the mono extractor return the raw onsets
        mock_mono_extr_obj.__process__ = lambda fake_track_combi, *_: fake_track_combi.get_onsets()

        ###############################
        # Rhythm mock:                #
        #   kick:   x---xx--          #
        #   snare:  --x--xx-          #
        #                             #
        # Expected combinations:      #
        #   [kick]          x---xx--  #
        #   [snare]         --x--xx-  #
        #   [kick, snare]   x-x-xxx-  #
        ###############################

        kck_rhythm_str = "x---xx--"
        snr_rhythm_str = "--x--xx-"
        cmb_rhythm_str = "x-x-xxx-"

        resolution = 2
        velocity = 100

        mocked_rhythm, mocked_tracks = get_poly_rhythm_mock(
            [("kick", kck_rhythm_str, velocity), ("snare", snr_rhythm_str, velocity)], resolution, "x")
        mocked_kick_track, mocked_snare_track = mocked_tracks
        assert mocked_kick_track.name == "kick"
        assert mocked_snare_track.name == "snare"

        expected_feature = (
            ((0,), mocked_kick_track.get_onsets()),
            ((1,), mocked_snare_track.get_onsets()),
            ((0, 1), get_mocked_onsets(cmb_rhythm_str, velocity, "x"))
        )

        actual_feature = mt_mono_feature_extr.process(mocked_rhythm)
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


class TestPolyphonicMetricalTensionVector(TestPolyphonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[PolyphonicRhythmFeatureExtractor]:
        return PolyphonicMetricalTensionVector

    def test_defaults_to_cyclic(self):
        extractor = self.feature_extractor  # type: PolyphonicMetricalTensionVector
        self.assertTrue(extractor.cyclic)

    def test_defaults_to_equal_upbeats_salience_profile_type(self):
        extractor = self.feature_extractor  # type: PolyphonicMetricalTensionVector
        self.assertEqual("equal_upbeats", extractor.salience_profile_type)

    def test_defaults_to_not_include_combination_tracks(self):
        extractor = self.feature_extractor  # type: PolyphonicMetricalTensionVector
        self.assertFalse(extractor.include_combination_tracks)

    def test_defaults_to_not_normalized(self):
        extractor = self.feature_extractor  # type: PolyphonicMetricalTensionVector
        self.assertFalse(extractor.normalize)

    def test_cyclic_property_propagates_to_mono_extractor(self):
        extractor = self.feature_extractor  # type: PolyphonicMetricalTensionVector
        with patch("beatsearch.feature_extraction.MonophonicMetricalTensionVector.cyclic",
                   new_callable=PropertyMock) as prop_mock:
            extractor.cyclic = "fake-cyclic"
            prop_mock.assert_called_once_with("fake-cyclic")

    def test_salience_profile_type_property_propagates_to_mono_extractor(self):
        extractor = self.feature_extractor  # type: PolyphonicMetricalTensionVector
        with patch("beatsearch.feature_extraction.MonophonicMetricalTensionVector.salience_profile_type",
                   new_callable=PropertyMock) as prop_mock:
            extractor.salience_profile_type = "fake-salience-profile-type"
            prop_mock.assert_called_once_with("fake-salience-profile-type")

    def test_normalize_property_propagates_to_mono_extractor(self):
        extractor = self.feature_extractor  # type: PolyphonicMetricalTensionVector
        with patch("beatsearch.feature_extraction.MonophonicMetricalTensionVector.normalize",
                   new_callable=PropertyMock) as prop_mock:
            extractor.normalize = "fake-normalize"
            prop_mock.assert_called_once_with("fake-normalize")

    N_STEPS = 16

    KCK = "kick"
    SNR = "snare"
    HHT = "hht"

    KCK_RHYTHM_STR = ''          'x---x---x---x---'
    SNR_RHYTHM_STR = ''          '---x---x--x-x---'
    HHT_RHYTHM_STR = ''          '--x---x---x---x-'
    KCK_SNR_RHYTHM_STR = ''      'x--xx--xx-x-x---'
    KCK_HHT_RHYTHM_STR = ''      'x-x-x-x-x-x-x-x-'
    SNR_HHT_RHYTHM_STR = ''      '--xx--xx--x-x-x-'
    KCK_SNR_HHT_RHYTHM_STR = ''  'x-xxx-xxx-x-x-x-'

    KCK_TENSION_VEC = (0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    SNR_TENSION_VEC = (1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 2.0, 3.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0)
    HHT_TENSION_VEC = (2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)
    KCK_SNR_TENSION_VEC = (0.0, 0.0, 0.0, 3.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0)
    KCK_HHT_TENSION_VEC = (0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0)
    SNR_HHT_TENSION_VEC = (2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 2.0, 3.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0)
    KCK_SNR_HHT_TENSION_VEC = (0.0, 0.0, 2.0, 3.0, 1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0)
    SEMI_QUAVER_EQUAL_UPBEAT_SALIENCE_PRF = [0, -3, -2, -3, -1, -3, -2, -3, -1, -3, -2, -3, -1, -3, -2, -3]

    def _call_process(self, extractor: PolyphonicMetricalTensionVector):
        """Calls process on the given polyphonic metrical tension vector extractor, mocking:
            - :meth:`beatsearch.rhythm.MonophonicRhythm.create.from_monophonic_rhythms`
            - :meth:`beatsearch.feature_extraction.MonophonicMetricalTensionVector.__process__`

        The former one is mocked so that this test doesn't depend on a well-working rhythm-track merge function. The
        latter one is mocked so that this test doesn't depend on a well-working monophonic metrical tension vector
        extractor.
        """

        res = 4
        onset_char = "x"

        rhythm, [kck_track, snr_track, hht_track] = get_poly_rhythm_mock([
            (self.KCK, self.KCK_RHYTHM_STR, 100),
            (self.SNR, self.SNR_RHYTHM_STR, 100),
            (self.HHT, self.HHT_RHYTHM_STR, 100)
        ], res, onset_char)

        kck_snr_track = get_mono_rhythm_mock(self.KCK_SNR_RHYTHM_STR, res, onset_char)
        kck_hht_track = get_mono_rhythm_mock(self.KCK_HHT_RHYTHM_STR, res, onset_char)
        snr_hht_track = get_mono_rhythm_mock(self.SNR_HHT_RHYTHM_STR, res, onset_char)
        kck_snr_hht_track = get_mono_rhythm_mock(self.KCK_SNR_HHT_RHYTHM_STR, res, onset_char)

        tension_vectors_per_track = {
            kck_track: self.KCK_TENSION_VEC,
            snr_track: self.SNR_TENSION_VEC,
            hht_track: self.HHT_TENSION_VEC,
            kck_hht_track: self.KCK_HHT_TENSION_VEC,
            kck_snr_track: self.KCK_SNR_TENSION_VEC,
            snr_hht_track: self.SNR_HHT_TENSION_VEC,
            kck_snr_hht_track: self.KCK_SNR_HHT_TENSION_VEC
        }

        # Create fake track-merge function that returns our own manually merged tracks
        def fake_create_mono_rhythm_from_other_mono_rhythms_func(*other_mono_rhythms):
            other_mono_rhythms = tuple(other_mono_rhythms)  # type: tp.Tuple[MonophonicRhythm, ...]
            n_other_mono_rhythms = len(other_mono_rhythms)
            if n_other_mono_rhythms == 1:
                return other_mono_rhythms[0]
            if n_other_mono_rhythms == 3:
                return kck_snr_hht_track
            assert n_other_mono_rhythms == 2
            # noinspection PyTypeChecker
            return {
                (kck_track, snr_track): kck_snr_track,
                (kck_track, hht_track): kck_hht_track,
                (snr_track, hht_track): snr_hht_track
            }[other_mono_rhythms]

        # Create fake monophonic metrical tension process function that returns our own manually pre-computed
        # monophonic metrical tension vectors, given one of the tracks returned by our merge mock function above
        def fake_mono_tension_process_func(mono_rhythm: MonophonicRhythm, _):  # unused _ is aux_fts
            onsets = mono_rhythm.get_onsets()
            try:
                track = next(t for t in tension_vectors_per_track.keys() if onsets == t.onsets)
            except StopIteration:
                assert False, "Unexpected mono rhythm: %s (with onsets: %s)" % (mono_rhythm, str(onsets))
            return tension_vectors_per_track[track]

        # Actually call process patched with our mock-functions
        with patch.object(MonophonicMetricalTensionVector, "__process__") as mock_mono_tension_process, \
                patch.object(MonophonicRhythm.create, "from_monophonic_rhythms") as mock_create_merged_rhythm:
            mock_create_merged_rhythm.side_effect = fake_create_mono_rhythm_from_other_mono_rhythms_func
            mock_mono_tension_process.side_effect = fake_mono_tension_process_func
            process_ret = extractor.process(rhythm)

        return process_ret

    def test_process_not_normalized_without_combination_tracks(self):
        weights = {self.KCK: 2, self.SNR: 1, self.HHT: 0.5}

        extractor = self.feature_extractor  # type: PolyphonicMetricalTensionVector
        extractor.include_combination_tracks = False
        extractor.set_instrument_weights(weights)
        extractor.normalize = False
        extractor.cyclic = True

        expected_tension_vec = tuple(
            (weights[self.KCK] * self.KCK_TENSION_VEC[i] +
             weights[self.SNR] * self.SNR_TENSION_VEC[i] +
             weights[self.HHT] * self.HHT_TENSION_VEC[i]) for i in range(self.N_STEPS))

        actual_tension_vec = self._call_process(extractor)
        npt.assert_almost_equal(actual_tension_vec, expected_tension_vec)

    def test_process_normalized_without_combination_tracks(self):
        weights = {self.KCK: 2, self.SNR: 1, self.HHT: 0.5}
        weight_normalizer = sum(weights.values())

        extractor = self.feature_extractor  # type: PolyphonicMetricalTensionVector
        extractor.include_combination_tracks = False
        extractor.set_instrument_weights(weights)
        extractor.normalize = True
        extractor.cyclic = True

        expected_tension_vec = tuple(
            ((weights[self.KCK] * self.KCK_TENSION_VEC[i] / weight_normalizer) +
             (weights[self.SNR] * self.SNR_TENSION_VEC[i] / weight_normalizer) +
             (weights[self.HHT] * self.HHT_TENSION_VEC[i]) / weight_normalizer) for i in range(self.N_STEPS))

        actual_tension_vec = self._call_process(extractor)
        npt.assert_almost_equal(actual_tension_vec, expected_tension_vec)

    def test_process_not_normalized_with_combination_tracks(self):
        weights = {self.KCK: 2, self.SNR: 1, self.HHT: 0.5}

        extractor = self.feature_extractor  # type: PolyphonicMetricalTensionVector
        extractor.include_combination_tracks = True
        extractor.set_instrument_weights(weights)
        extractor.normalize = False
        extractor.cyclic = True

        expected_tension_vec = tuple(
            ((weights[self.KCK] * self.KCK_TENSION_VEC[i]) +
             (weights[self.SNR] * self.SNR_TENSION_VEC[i]) +
             (weights[self.HHT] * self.HHT_TENSION_VEC[i]) +
             (weights[self.KCK] * weights[self.SNR] * self.KCK_SNR_TENSION_VEC[i]) +
             (weights[self.KCK] * weights[self.HHT] * self.KCK_HHT_TENSION_VEC[i]) +
             (weights[self.SNR] * weights[self.HHT] * self.SNR_HHT_TENSION_VEC[i]) +
             (weights[self.KCK] * weights[self.SNR] * weights[self.HHT] * self.KCK_SNR_HHT_TENSION_VEC[i]))
            for i in range(self.N_STEPS))

        actual_tension_vec = self._call_process(extractor)
        npt.assert_almost_equal(actual_tension_vec, expected_tension_vec)

    def test_process_normalized_with_combination_tracks(self):
        weights = {self.KCK: 2, self.SNR: 1, self.HHT: 0.5}

        combination_weights = [
            weights[self.KCK],
            weights[self.SNR],
            weights[self.HHT],
            weights[self.KCK] * weights[self.SNR],
            weights[self.KCK] * weights[self.HHT],
            weights[self.SNR] * weights[self.HHT],
            weights[self.KCK] * weights[self.SNR] * weights[self.HHT]
        ]

        # Weight normalizer is now the sum of all combination weights
        w_nrm = sum(combination_weights)

        extractor = self.feature_extractor  # type: PolyphonicMetricalTensionVector
        extractor.include_combination_tracks = True
        extractor.set_instrument_weights(weights)
        extractor.normalize = True
        extractor.cyclic = True

        expected_tension_vec = tuple(
            ((weights[self.KCK] * self.KCK_TENSION_VEC[i]) / w_nrm +
             (weights[self.SNR] * self.SNR_TENSION_VEC[i]) / w_nrm +
             (weights[self.HHT] * self.HHT_TENSION_VEC[i]) / w_nrm +
             (weights[self.KCK] * weights[self.SNR] * self.KCK_SNR_TENSION_VEC[i]) / w_nrm +
             (weights[self.KCK] * weights[self.HHT] * self.KCK_HHT_TENSION_VEC[i]) / w_nrm +
             (weights[self.SNR] * weights[self.HHT] * self.SNR_HHT_TENSION_VEC[i]) / w_nrm +
             (weights[self.KCK] * weights[self.SNR] * weights[self.HHT] * self.KCK_SNR_HHT_TENSION_VEC[i]) / w_nrm)
            for i in range(self.N_STEPS))

        actual_tension_vec = self._call_process(extractor)
        npt.assert_almost_equal(actual_tension_vec, expected_tension_vec)


class TestPolyphonicMetricalTensionMagnitude(TestPolyphonicRhythmFeatureExtractorImplementationMixin, TestCase):
    @staticmethod
    def get_impl_class() -> tp.Type[PolyphonicRhythmFeatureExtractor]:
        return PolyphonicMetricalTensionMagnitude

    def test_defaults_to_cyclic(self):
        extractor = self.feature_extractor  # type: PolyphonicMetricalTensionMagnitude
        self.assertTrue(extractor.cyclic)

    def test_defaults_to_equal_upbeats_salience_profile_type(self):
        extractor = self.feature_extractor  # type: PolyphonicMetricalTensionMagnitude
        self.assertEqual("equal_upbeats", extractor.salience_profile_type)

    def test_defaults_to_not_include_combination_tracks(self):
        extractor = self.feature_extractor  # type: PolyphonicMetricalTensionMagnitude
        self.assertFalse(extractor.include_combination_tracks)

    def test_defaults_to_normalized(self):
        extractor = self.feature_extractor  # type: PolyphonicMetricalTensionMagnitude
        self.assertTrue(extractor.normalize)

    def test_cyclic_property_propagates_to_vector_extractor(self):
        extractor = self.feature_extractor  # type: PolyphonicMetricalTensionMagnitude
        with patch("beatsearch.feature_extraction.PolyphonicMetricalTensionVector.cyclic",
                   new_callable=PropertyMock) as prop_mock:
            extractor.cyclic = "fake-cyclic"
            prop_mock.assert_called_once_with("fake-cyclic")

    def test_salience_profile_type_property_propagates_to_vector_extractor(self):
        extractor = self.feature_extractor  # type: PolyphonicMetricalTensionMagnitude
        with patch("beatsearch.feature_extraction.PolyphonicMetricalTensionVector.salience_profile_type",
                   new_callable=PropertyMock) as prop_mock:
            extractor.salience_profile_type = "fake-salience-profile-type"
            prop_mock.assert_called_once_with("fake-salience-profile-type")

    def test_normalize_property_propagates_to_vector_extractor(self):
        extractor = self.feature_extractor  # type: PolyphonicMetricalTensionMagnitude
        with patch("beatsearch.feature_extraction.PolyphonicMetricalTensionVector.normalize",
                   new_callable=PropertyMock) as prop_mock:
            extractor.normalize = "fake-normalize"
            prop_mock.assert_called_once_with("fake-normalize")

    @patch.object(PolyphonicMetricalTensionVector, "__process__")
    def test_process_not_normalized(self, mock_metrical_tension_vec_process):
        extractor = self.feature_extractor  # type: PolyphonicMetricalTensionMagnitude
        extractor.normalize = False
        mock_metrical_tension_vec_process.return_value = (0/6, 0/6, 0/6, 6/6, 6/6, 1/6, 2/6, 4/6,
                                                          4/6, 4/6, 5/6, 1/6, 4/6, 6/6, 0/6, 6/6)
        expected_vector_magnitude = 2.576604138956718
        actual_vector_magnitude = extractor.process(self.rhythm)
        self.assertAlmostEqual(actual_vector_magnitude, expected_vector_magnitude)

    @patch.object(PolyphonicMetricalTensionVector, "__process__")
    def test_process_normalized(self, mock_metrical_tension_vec_process):
        extractor = self.feature_extractor  # type: PolyphonicMetricalTensionMagnitude
        extractor.normalize = True
        mock_metrical_tension_vec_process.return_value = (0/6, 0/6, 0/6, 6/6, 6/6, 1/6, 2/6, 4/6,
                                                          4/6, 4/6, 5/6, 1/6, 4/6, 6/6, 0/6, 6/6)
        expected_vector_magnitude = 0.6441510347391795
        actual_vector_magnitude = extractor.process(self.rhythm)
        self.assertAlmostEqual(actual_vector_magnitude, expected_vector_magnitude)


# TODO: Add tests for DistantPolyphonicSyncopationVector
# TODO: Add tests for PolyphonicSyncopationVector
# TODO: Add tests for MeanPolyphonicSyncopationStrength
# TODO: Add tests for MeanDistantPolyphonicSyncopationStrength


if __name__ == "__main__":
    main()

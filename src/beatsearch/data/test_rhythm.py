import unittest
import copy
import math
from unittest.mock import MagicMock

from beatsearch.data.rhythm import (
    Rhythm,
    TimeSignature,
    HammingDistanceMeasure,
    TrackDistanceMeasure,
    EuclideanIntervalVectorDistanceMeasure,
    IntervalDifferenceVectorDistanceMeasure,
    SwapDistanceMeasure,
    ChronotonicDistanceMeasure,
)

# noinspection PyUnresolvedReferences
import midi


class TestRhythm(unittest.TestCase):
    # broken c major chord
    onset_data = {
        60: ((0, 90), (1402, 127)),   # c
        64: ((375, 95), (924, 120)),  # e
        67: ((624, 100), )            # g
    }
    time_signature = TimeSignature(6, 8)
    rhythm_name = "The_Star-Spangled_Banner_beat"
    ppq = 240
    bpm = 180
    duration = 2160  # 3 3/4 measures with a resolution of 240 ppq
    midi_file_path = "/home/user/obama/The_Star-Spangled_Banner_beat.mid"
    to_midi_note_duration = 32

    def setUp(self):
        self.rhythm = Rhythm(
            name=TestRhythm.rhythm_name,
            bpm=TestRhythm.bpm,
            time_signature=TestRhythm.time_signature,
            data=copy.deepcopy(TestRhythm.onset_data),
            data_ppq=TestRhythm.ppq,
            duration=TestRhythm.duration,
            midi_file_path=TestRhythm.midi_file_path,
            ceil_duration_to_measure=False
        )
        self.to_midi_result = self.rhythm.to_midi(TestRhythm.to_midi_note_duration)

    def tearDown(self):
        self.rhythm = None

    def test_name_property_equals_name_given_to_constructor(self):
        self.assertEqual(self.rhythm.name, TestRhythm.rhythm_name)

    def test_name_property_is_writeable(self):
        self.rhythm.name = "new_name"
        self.assertEqual(self.rhythm.name, "new_name")

    def test_bpm_property_equals_bpm_given_to_constructor(self):
        self.assertEqual(self.rhythm.bpm, TestRhythm.bpm)

    def test_bpm_property_is_writeable(self):
        expected_bpm = 150
        assert expected_bpm != self.bpm
        self.rhythm.bpm = expected_bpm
        self.assertEqual(self.rhythm.bpm, expected_bpm)

    def test_time_signature_property_equals_ts_given_to_constructor(self):
        self.assertEqual(self.rhythm.time_signature, TestRhythm.time_signature)

    def test_measure_duration_is_correct(self):
        self.assertEqual(self.rhythm.get_measure_duration(), 720)

    def test_beat_duration_is_correct(self):
        self.assertEqual(self.rhythm.get_beat_duration(), 120)

    def test_get_resolution_returns_resolution_given_to_constructor(self):
        self.assertEqual(self.rhythm.get_resolution(), TestRhythm.ppq)

    def test_get_duration_returns_duration_given_to_constructor(self):
        self.assertEqual(self.rhythm.get_duration(), TestRhythm.duration)

    def test_one_track_is_created_per_onset_pitch(self):
        data = {
            1: ((0, 0), ),
            6: ((0, 0), ),
            4: ((0, 0), ),
            9: ((0, 0), )
        }

        rhythm = Rhythm("", 120, TimeSignature(4, 4), data, 240, 1)
        for pitch in list(data.keys()):
            track = rhythm.get_track(pitch)
            self.assertIsInstance(track, Rhythm.Track)

    def test_track_onsets_are_correct(self):
        actual_onset_data = {
            60: self.rhythm.get_track(60).onsets,
            64: self.rhythm.get_track(64).onsets,
            67: self.rhythm.get_track(67).onsets
        }

        self.assertEqual(actual_onset_data, self.onset_data)

    def test_duration_rescales_correctly_when_resolution_upscale(self):
        assert self.ppq == 240
        assert self.duration == 2160

        self.rhythm.set_resolution(960)
        self.assertEqual(self.rhythm.get_duration(), 8640)

    def test_duration_rescales_correctly_when_resolution_downscale(self):
        assert self.ppq == 240
        assert self.duration == 2160

        self.rhythm.set_resolution(22)
        self.assertEqual(self.rhythm.get_duration(), 198)

    def test_track_onsets_rescale_correctly_when_resolution_upscale(self):
        assert self.ppq == 240  # expected onsets computed assuming an original resolution of 240 ppq
        self.rhythm.set_resolution(960)  # rescale time resolution to 960 ppq

        # expected onset data when rescaling from 240 ppq to 960 ppq
        expected_onset_data = {
            60: ((0, 90), (5608, 127)),     # c
            64: ((1500, 95), (3696, 120)),  # e
            67: ((2496, 100),)              # g
        }

        actual_onset_data = {
            60: self.rhythm.get_track(60).onsets,
            64: self.rhythm.get_track(64).onsets,
            67: self.rhythm.get_track(67).onsets
        }

        self.assertEqual(actual_onset_data, expected_onset_data)

    def test_track_onsets_rescale_correctly_when_resolution_downscale(self):
        assert self.ppq == 240  # expected onsets computed assuming an original resolution of 240 ppq
        self.rhythm.set_resolution(22)  # rescale time resolution to 22 ppq

        # expected onset data when rescaling from 240 ppq to 22 ppq
        expected_onset_data = {
            60: ((0, 90), (129, 127)),   # c
            64: ((34, 95), (85, 120)),   # e
            67: ((57, 100), )            # g
        }

        actual_onset_data = {
            60: self.rhythm.get_track(60).onsets,
            64: self.rhythm.get_track(64).onsets,
            67: self.rhythm.get_track(67).onsets
        }

        self.assertEqual(actual_onset_data, expected_onset_data)

    def test_midi_has_only_one_track(self):
        pattern = self.rhythm.to_midi()
        self.assertEqual(len(pattern), 1)

    def test_midi_has_correct_meta_data(self):
        track = self.rhythm.to_midi()[0]
        self.assertIsInstance(track[0], midi.TrackNameEvent)
        self.assertIsInstance(track[1], midi.TimeSignatureEvent)
        self.assertIsInstance(track[2], midi.SetTempoEvent)
        self.assertIsInstance(track[-1], midi.EndOfTrackEvent)

    def test_track_pre_note_inter_onset_intervals(self):
        onset_data = {60: ((0, 127), (3, 127), (7, 127), (10, 127), (12, 127))}
        expected_intervals = [0, 3, 4, 3, 2]
        rhythm = Rhythm("", 120, TimeSignature(4, 4), onset_data, 4, 16)
        actual_intervals = rhythm.get_track(60).get_pre_note_inter_onset_intervals()
        self.assertEqual(actual_intervals, expected_intervals)

    def test_track_post_note_inter_onset_intervals(self):
        onset_data = {60: ((0, 127), (3, 127), (7, 127), (10, 127), (12, 127))}
        expected_intervals = [3, 4, 3, 2, 4]
        rhythm = Rhythm("", 120, TimeSignature(4, 4), onset_data, 4, 16)
        actual_intervals = rhythm.get_track(60).get_post_note_inter_onset_intervals()
        self.assertEqual(actual_intervals, expected_intervals)

    def test_non_cyclic_track_interval_difference_vector_is_correct(self):
        onset_data = {60: ((0, 127), (3, 127), (7, 127), (10, 127), (12, 127))}
        rhythm = Rhythm("", 120, TimeSignature(4, 4), onset_data, 4, 16)
        expected_interval_difference_vector = [4. / 3., 3. / 4., 2. / 3., 4. / 2.]
        actual_interval_difference_vector = rhythm.get_track(60).get_interval_difference_vector(cyclic=False)
        self.assertEqual(actual_interval_difference_vector, expected_interval_difference_vector)

    def test_cyclic_track_interval_difference_vector_is_correct(self):
        onset_data = {60: ((0, 127), (3, 127), (7, 127), (10, 127), (12, 127))}
        rhythm = Rhythm("", 120, TimeSignature(4, 4), onset_data, 4, 16)
        expected_interval_difference_vector = [4. / 3., 3. / 4., 2. / 3., 4. / 2., 3. / 4.]
        actual_interval_difference_vector = rhythm.get_track(60).get_interval_difference_vector(cyclic=True)
        self.assertEqual(actual_interval_difference_vector, expected_interval_difference_vector)

    def test_track_to_binary_without_resolution_change(self):
        onset_data = {60: ((0, 127), (3, 127), (7, 127), (10, 127), (12, 127))}
        rhythm = Rhythm("", 120, TimeSignature(4, 4), onset_data, 4, 16)
        expected_binary = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]
        actual_binary = rhythm.get_track(60).get_binary("ticks")
        self.assertEqual(actual_binary, expected_binary)

    def test_track_binary_with_down_scale_resolution_and_not_quantized_input_data(self):
        onset_data = {60: ((7, 127), (176, 127), (421, 127), (611, 127), (713, 127))}
        rhythm = Rhythm("", 120, TimeSignature(4, 4), onset_data, 240, 960)
        expected_binary = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]
        actual_binary = rhythm.get_track(60).get_binary("sixteenths")
        self.assertEqual(actual_binary, expected_binary)

    def test_track_binary_schillinger_chain_is_correct(self):
        onset_data = {60: ((0, 127), (3, 127), (7, 127), (10, 127), (12, 127))}
        rhythm = Rhythm("", 120, TimeSignature(4, 4), onset_data, 4, 16)
        expected_schillinger_chain = [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1]
        actual_schillinger_chain = rhythm.get_track(60).get_binary_schillinger_chain("ticks")
        self.assertEqual(actual_schillinger_chain, expected_schillinger_chain)

    def test_track_binary_schillinger_chain_only_consists_of_given_binary_values(self):
        values = ("real_madrid", "fc_barcelona")
        schillinger_chain = self.rhythm.get_track(60).get_binary_schillinger_chain(values=values)
        self.assertTrue(all(val == values[0] or val == values[1] for val in schillinger_chain))

    def test_track_chronotonic_chain_is_correct(self):
        onset_data = {60: ((0, 127), (3, 127), (7, 127), (10, 127), (12, 127))}
        rhythm = Rhythm("", 120, TimeSignature(4, 4), onset_data, 4, 16)
        expected_chronotonic_chain = [3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 2, 2, 4, 4, 4, 4]
        actual_chronotonic_chain = rhythm.get_track(60).get_chronotonic_chain()
        self.assertEqual(actual_chronotonic_chain, expected_chronotonic_chain)

    def test_track_onset_times_are_correct(self):
        onset_data = {60: ((0, 127), (3, 127), (7, 127), (10, 127), (12, 127))}
        rhythm = Rhythm("", 120, TimeSignature(4, 4), onset_data, 4, 16)
        expected_onset_times = [0, 3, 7, 10, 12]
        actual_onset_times = rhythm.get_track(60).get_onset_times("ticks")
        self.assertEqual(actual_onset_times, expected_onset_times)

    def test_track_interval_histogram_is_correct(self):
        onset_data = {60: ((0, 127), (3, 127), (7, 127), (10, 127), (12, 127))}
        rhythm = Rhythm("", 120, TimeSignature(4, 4), onset_data, 4, 16)
        expected_histogram = (
            [1, 2, 2],
            [2, 3, 4]
        )
        actual_histogram = rhythm.get_track(60).get_interval_histogram("ticks")
        self.assertEqual(actual_histogram[0], expected_histogram[0])
        self.assertEqual(actual_histogram[1], expected_histogram[1])

    def test_track_get_resolution_returns_same_as_rhythm_get_resolution(self):
        self.assertEqual(self.rhythm.get_track(60).get_resolution(), self.rhythm.get_resolution())


class TestTrackMeasures(unittest.TestCase):
    def setUp(self):
        self.tr_a = Rhythm.Track(((0, 127), (3, 127), (7, 127), (10, 127), (12, 127)))
        self.tr_b = Rhythm.Track(((1, 127), (3, 127), (6, 127), (10, 127), (14, 127)))
        self.tr_aa = Rhythm.Track(((0, 127), (3, 127), (7, 127), (10, 127), (12, 127),
                                   (16, 127), (19, 127), (23, 127), (26, 127), (28, 127)))

        fake_rhythm = type("Rhythm", (object, ), dict(get_duration=lambda *_: None))
        fake_rhythm_dur_16, fake_rhythm_dur_32 = fake_rhythm(), fake_rhythm()
        fake_rhythm_dur_32.get_duration = MagicMock(return_value=32)
        fake_rhythm_dur_16.get_duration = MagicMock(return_value=16)

        for t in [self.tr_a, self.tr_b]:
            t.rhythm = fake_rhythm_dur_16
        self.tr_aa.rhythm = fake_rhythm_dur_32

        for t in [self.tr_a, self.tr_b, self.tr_aa]:
            t.get_resolution = MagicMock(return_value=4)

        self.len_policies = ["exact", "multiple", "fill"]

    def test_track_measure_length_policy_property_given_to_constructor(self):
        t = TrackDistanceMeasure("quarters", length_policy="exact")
        self.assertEqual(t.length_policy, "exact")

    ####################
    # HAMMING DISTANCE #
    ####################

    def test_hamming_distance_is_correct(self):
        d_expected = 6
        for lp in self.len_policies:
            m = HammingDistanceMeasure("ticks", lp)
            self.assertEqual(m.get_distance(self.tr_a, self.tr_b), d_expected)

    def test_auto_hamming_distance_is_zero(self):
        for lp in self.len_policies:
            m = HammingDistanceMeasure("ticks", lp)
            self.assertEqual(m.get_distance(self.tr_a, self.tr_a), 0)

    def test_hamming_distance_single_to_double_is_zero(self):
        for lp in self.len_policies[1:]:  # skip "exact"
            m = HammingDistanceMeasure("ticks", lp)
            self.assertEqual(m.get_distance(self.tr_a, self.tr_aa), 0)

    #############################
    # EUCLIDEAN VECTOR DISTANCE #
    #############################

    def test_euclidean_vector_distance_is_correct(self):
        d_expected = math.sqrt(11)
        for lp in self.len_policies:
            m = EuclideanIntervalVectorDistanceMeasure("ticks", lp, quantize=True)
            self.assertEqual(m.get_distance(self.tr_a, self.tr_b), d_expected)

    def test_auto_euclidean_vector_distance_is_zero(self):
        for lp in self.len_policies:
            m = EuclideanIntervalVectorDistanceMeasure("ticks", lp)
            self.assertEqual(m.get_distance(self.tr_a, self.tr_a), 0)

    def test_euclidean_vector_distance_single_to_double_is_zero(self):
        for lp in self.len_policies[1:]:  # skip "exact"
            m = EuclideanIntervalVectorDistanceMeasure("ticks", lp)
            self.assertEqual(m.get_distance(self.tr_a, self.tr_aa), 0)

    #######################################
    # INTERVAL DIFFERENCE VECTOR DISTANCE #
    #######################################

    def test_interval_difference_vector_distance_is_correct(self):
        d_expected = 4.736111111111111
        for lp in self.len_policies:
            m = IntervalDifferenceVectorDistanceMeasure("ticks", lp, quantize=True, cyclic=True)
            self.assertAlmostEqual(m.get_distance(self.tr_a, self.tr_b), d_expected)

    def test_auto_interval_difference_vector_distance_is_zero(self):
        for lp in self.len_policies:
            m = IntervalDifferenceVectorDistanceMeasure("ticks", lp)
            self.assertEqual(m.get_distance(self.tr_a, self.tr_a), 0)

    def test_interval_difference_vector_distance_single_to_double_is_zero(self):
        for lp in self.len_policies[1:]:  # skip "exact"
            m = IntervalDifferenceVectorDistanceMeasure("ticks", lp)
            self.assertEqual(m.get_distance(self.tr_a, self.tr_aa), 0)

    #################
    # SWAP DISTANCE #
    #################

    def test_swap_distance_is_correct(self):
        d_expected = 4
        for lp in self.len_policies:
            m = SwapDistanceMeasure("ticks", length_policy=lp, quantize=True)
            self.assertEqual(m.get_distance(self.tr_a, self.tr_b), d_expected)

    def test_auto_swap_distance_is_zero(self):
        for lp in self.len_policies:
            m = SwapDistanceMeasure("ticks", lp)
            self.assertEqual(m.get_distance(self.tr_a, self.tr_a), 0)

    def test_swap_distance_single_to_double_is_zero(self):
        for lp in self.len_policies[1:]:
            m = SwapDistanceMeasure("ticks", lp)
            self.assertEqual(m.get_distance(self.tr_a, self.tr_aa), 0)

    ########################
    # CHRONOTONIC DISTANCE #
    ########################

    def test_chronotonic_distance_is_correct(self):
        d_expected = 19
        for lp in self.len_policies:
            m = ChronotonicDistanceMeasure("ticks", lp)
            self.assertEqual(m.get_distance(self.tr_a, self.tr_b), d_expected)

    def test_auto_chronotonic_distance_is_zero(self):
        for lp in self.len_policies:
            m = ChronotonicDistanceMeasure("ticks", lp)
            self.assertEqual(m.get_distance(self.tr_a, self.tr_a), 0)

    def test_chronotonic_distance_single_to_double_is_zero(self):
        for lp in self.len_policies[1:]:
            m = ChronotonicDistanceMeasure("ticks", lp)
            self.assertEqual(m.get_distance(self.tr_a, self.tr_aa), 0)


if __name__ == "__main__":
    unittest.main()
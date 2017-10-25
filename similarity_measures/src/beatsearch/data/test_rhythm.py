import unittest
import copy
import midi
from rhythm import Rhythm, TimeSignature


class TestRhythm(unittest.TestCase):
    # broken c major chord
    onset_data = {
        60: ((0, 90), (1402, 127)),   # c
        64: ((375, 95), (924, 120)),  # e
        67: ((624, 100), )            # g
    }
    time_signature = TimeSignature(3, 4)
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

    def test_bpm_property_equals_bpm_given_to_constructor(self):
        self.assertEqual(self.rhythm.bpm, TestRhythm.bpm)

    def test_time_signature_property_equals_ts_given_to_constructor(self):
        self.assertEqual(self.rhythm.time_signature, TestRhythm.time_signature)

    def test_ppq_property_equals_ppq_given_to_constructor(self):
        self.assertEqual(self.rhythm.get_resolution(), TestRhythm.ppq)

    def test_duration_property_equals_duration_given_to_constructor(self):
        self.assertEqual(self.rhythm.get_duration(), TestRhythm.duration)

    def test_one_track_is_created_per_onset_pitch(self):
        data = {
            1: ((0, 0), ),
            6: ((0, 0), ),
            4: ((0, 0), ),
            9: ((0, 0), )
        }

        rhythm = Rhythm("", 120, TimeSignature(4, 4), data, 240, 1)
        for pitch in data.keys():
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

    def test_track_to_binary_without_resolution_change(self):
        onset_data = {60: ((0, 127), (3, 127), (7, 127), (10, 127), (12, 127))}
        rhythm = Rhythm("", 120, TimeSignature(4, 4), onset_data, 4, 16)
        expected_binary = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]
        actual_binary = rhythm.get_track(60).get_binary('ticks')
        self.assertEqual(actual_binary, expected_binary)

    def test_track_binary_with_down_scale_resolution_and_not_quantized_input_data(self):
        onset_data = {60: ((7, 127), (176, 127), (421, 127), (611, 127), (713, 127))}
        rhythm = Rhythm("", 120, TimeSignature(4, 4), onset_data, 240, 960)
        expected_binary = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]
        actual_binary = rhythm.get_track(60).get_binary('sixteenths')
        self.assertEqual(actual_binary, expected_binary)

    def test_binary_schillinger_chain_is_correct(self):
        onset_data = {60: ((0, 127), (3, 127), (7, 127), (10, 127), (12, 127))}
        rhythm = Rhythm("", 120, TimeSignature(4, 4), onset_data, 4, 16)
        expected_schillinger_chain = [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1]
        actual_schillinger_chain = rhythm.get_track(60).get_binary_schillinger_chain('ticks')
        self.assertEqual(actual_schillinger_chain, expected_schillinger_chain)

    def test_chronotonic_chain_is_correct(self):
        onset_data = {60: ((0, 127), (3, 127), (7, 127), (10, 127), (12, 127))}
        rhythm = Rhythm("", 120, TimeSignature(4, 4), onset_data, 4, 16)
        expected_chronotonic_chain = [3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 2, 2, 4, 4, 4, 4]
        actual_chronotonic_chain = rhythm.get_track(60).get_chronotonic_chain()
        self.assertEqual(actual_chronotonic_chain, expected_chronotonic_chain)

if __name__ == '__main__':
    unittest.main()

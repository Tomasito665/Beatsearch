import os
import itertools
import midi


def scale_tick(tick, ppq_from, ppq_to):
    n_quarter = int(tick // ppq_from)  # beat number

    tick_beat = n_quarter * ppq_from  # ticks at start of the quarter
    tick_rest = tick - tick_beat  # ticks into the beat

    scaled_beat = n_quarter * ppq_to
    scaled_rest = round(tick_rest / float(ppq_from) * ppq_to)

    return scaled_beat + int(scaled_rest)


class TimeSignature(object):
    """
    This class represents a musical time signature, consisting of a numerator and a denominator.
    """

    def __init__(self, numerator, denominator):
        self._numerator = numerator
        self._denominator = denominator

    @property
    def numerator(self):
        return self._numerator

    @property
    def denominator(self):
        return self._denominator

    def to_midi_event(self):
        return midi.TimeSignatureEvent(
            numerator=self.numerator,
            denominator=self.denominator,
            metronome=24,
            thirtyseconds=8
        )

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.numerator == other.numerator and self.denominator == other.denominator

    @staticmethod
    def from_midi_event(midi_event):
        if not isinstance(midi_event, midi.TimeSignatureEvent):
            raise ValueError("Expected midi.TimeSignatureEvent")
        numerator = midi_event.get_numerator()
        denominator = midi_event.get_denominator()
        return TimeSignature(numerator, denominator)


class Rhythm(object):
    """
    Represents a rhythm, containing onset times and velocities per instrument. Holds meta data about the rhythm like
    name, bpm and time signature.
    """

    def __init__(self, name, bpm, time_signature, data, data_ppq, duration, rescale_to_ppq=None, midi_file_path=""):
        """
        Constructs a new rhythm.

        :param name:               The name of the rhythm
        :param bpm:                Beats per minute; tempo of the rhythm
        :param time_signature:     Time signature of the rhythm
        :param data:               The onset data; dictionary holding onsets by MIDI key where each onset is an
                                    (absolute tick, velocity) tuple
        :param data_ppq:           Resolution of the onset data and the length in PPQ (pulses per quarter note)
        :param duration:           The duration of the rhythm in ticks
        :param rescale_to_ppq:     Resolution to rescale the onset data (and the length) to in pulses per quarter note or None
                                    for no rescaling
        :param midi_file_path:     The path to the midi file or an empty string if this rhythm is not
                                    attached to a MIDI file
        """

        if not isinstance(time_signature, TimeSignature):
            raise ValueError("Time signature should be a Rhythm.TimeSignature instance")

        self._name = str(name)
        self._bpm = float(bpm)
        self._time_signature = time_signature
        self._tracks = {}
        self._duration = -1
        self._ppq = data_ppq
        self._midi_file_path = str(midi_file_path) if midi_file_path else None

        # add tracks
        for pitch, onsets in data.iteritems():
            self._tracks[pitch] = Rhythm.Track(onsets, self)

        # set duration by property setter to raise exception on invalid duration
        self.duration = duration

        if rescale_to_ppq is not None:
            self.ppq = rescale_to_ppq

    class Track(object):
        """
        Contains the actual onset data for one instrument.
        """

        def __init__(self, data, rhythm=None):
            """
            Creates a new rhythm track.

            :param data: An iterable returning an (absolute tick, velocity) tuple for each iteration.
            :param rhythm: The rhythm object which this track is belongs to.
            """

            self._data = tuple(data)
            self._rhythm = rhythm

        @property
        def rhythm(self):
            """
            The rhythm object whch this track belongs to.
            """

            return self._rhythm

        @rhythm.setter
        def rhythm(self, rhythm):
            self._rhythm = rhythm

        @property
        def onsets(self):
            """
            The onsets of this rhythm track where each onset is an (absolute tick, velocity) tuple pair. This is a
            read-only property.
            """

            return self._data

        def get_binary(self, resolution=None):
            """
            Returns the binary representation of the note onsets of this track where each step where a note onset
            happens is denoted with a 1; otherwise with a 0. The given resolution is the resolution in PPQ (pulses per
            quarter note) of the binary vector.

            :param resolution: the resolution of the binary grid (defaults to self.ppq)
            :return: the binary representation of the note onsets of the given pitch
            """

            rhythm = self.rhythm

            if resolution is None:
                resolution = rhythm.ppq

            binary_string = [0] * scale_tick(rhythm.duration, rhythm.ppq, resolution)

            for onset in self._data:
                tick = onset[0]
                scaled_tick = scale_tick(tick, rhythm.ppq, resolution)
                binary_string[scaled_tick] = 1

            return binary_string

        def get_pre_note_inter_onset_intervals(self):
            """
            Returns the time difference between the current note and the previous note for all notes of this track. The
            first note will return the time difference with the start of the rhythm.

            For example, given the Rumba Clave rhythm:
              X--X---X--X-X---
              0  3   4  3 2

            :return: pre note inter-onset interval vector
            """

            current_tick = 0
            intervals = []

            for onset in self._data:
                onset_tick = onset[0]
                intervals.append(onset_tick - current_tick)
                current_tick = onset_tick

            return intervals

        def get_post_note_inter_onset_intervals(self):
            """
            Returns the time difference between the current note and the next note for all notes in this rhythm track.
            The last note will return the time difference with the end (duration) of the rhythm.

            For example, given the Rumba Clave rhythm:
              X--X---X--X-X---
              3  4   3  2 4

            :return: post note inter-onset interval vector
            """

            last_onset_tick = -1
            intervals = []

            for onset in self._data:
                onset_tick = onset[0]
                if last_onset_tick < 0:
                    last_onset_tick = onset_tick
                    continue
                intervals.append(onset_tick - last_onset_tick)
                last_onset_tick = onset_tick

            intervals.append(self.rhythm.duration - last_onset_tick)
            return intervals

        def __set_resolution__(self, new_resolution):  # used by Rhythm.ppq setter
            old_resolution = self.rhythm.ppq

            def scale_onset(onset):
                tick, velocity = onset
                scaled_tick = scale_tick(tick, old_resolution, new_resolution)
                return scaled_tick, velocity

            rescaled_onsets = map(scale_onset, self._data)
            self._data = tuple(rescaled_onsets)

    @property
    def name(self):
        """
        The name of the rhythm.
        """

        return self._name

    @name.setter
    def name(self, name):
        self._name = str(name)

    @property
    def bpm(self):
        """
        The tempo of this rhythm in beats per minute. This is a read-only property.
        """

        return self._bpm

    @property
    def time_signature(self):
        """
        The time signature of this rhythm. This is a read-only property.
        """

        return self._time_signature

    @property
    def ppq(self):
        """
        The time resolution in pulses per quarter note.
        """

        return self._ppq

    @ppq.setter
    def ppq(self, new_ppq):
        if new_ppq <= 0:
            raise ValueError("Expected a number greater than zero but got %s" % str(new_ppq))

        new_ppq = int(new_ppq)
        old_ppq = self._ppq

        if old_ppq == new_ppq:
            return

        for track in self._tracks.values():
            track.__set_resolution__(new_ppq)

        self._duration = scale_tick(self._duration, old_ppq, new_ppq)

    @property
    def duration(self):
        """
        The duration of the rhythm in ticks (see ppq).
        """

        return self._duration

    @duration.setter
    def duration(self, duration):
        last_onset = 0

        for onsets in map(lambda track: track.onsets, self._tracks.values()):
            last_onset = max(onsets[-1][0], last_onset)

        if duration < last_onset:
            raise ValueError("Expected a duration of at least %i but got %i" % (last_onset, duration))

        self._duration = duration

    def get_track(self, pitch):
        """
        Returns the rhythm track of the instrument on the given pitch or None if there is no track for the given pitch.

        :param pitch: pitch number
        :return: rhythm track
        """

        try:
            return self._tracks[pitch]
        except KeyError:
            return None

    def to_midi(self, note_duration=32):
        """
        Convert this rhythm to a MIDI.

        :param note_duration: note duration in ticks
        :return: MIDI pattern
        """

        midi_track = midi.Track(tick_relative=False)  # create track and add metadata events

        midi_track.append(midi.TrackNameEvent(text=self._name))  # track name
        midi_track.append(self._time_signature.to_midi_event())  # time signature
        midi_track.append(midi.SetTempoEvent(bpm=self._bpm))     # tempo

        # add note events
        for pitch, track in self._tracks.iteritems():
            onsets = track.onsets
            for onset in onsets:
                note_abs_tick = onset[0]
                note_on = midi.NoteOnEvent(tick=note_abs_tick, pitch=pitch)
                note_off = midi.NoteOffEvent(tick=note_abs_tick + note_duration, pitch=pitch)
                midi_track.extend([note_on, note_off])

        # sort the events in chronological order and convert to relative delta-time
        midi_track = midi.Track(sorted(midi_track, key=lambda event: event.tick), tick_relative=False)
        midi_track.make_ticks_rel()

        # add end of track event
        midi_track.append(midi.EndOfTrackEvent())

        # save the midi file
        return midi.Pattern([midi_track])

    @staticmethod
    def create_from_midi(pattern, path=None, resolution=240):
        """
        Constructs a new rhythm from a MIDI file.

        :param pattern: midi pattern
        :param path: midi source file or None
        :param resolution: resolution to the rescale the rhythm to in pulses per quarter note
        :return: Rhythm object
        """

        n_tracks_containing_note_events = sum(any(isinstance(e, midi.NoteEvent) for e in track) for track in pattern)

        if n_tracks_containing_note_events > 1:
            raise ValueError("This MIDI pattern has multiple tracks with note events (%i)",
                             n_tracks_containing_note_events)

        pattern.make_ticks_abs()
        track = list(itertools.chain(*pattern))  # merge all tracks into one
        track = midi.Track(sorted(track, key=lambda event: event.tick))  # sort in chronological order

        args = {
            'name': os.path.splitext(os.path.basename(path))[0],
            'bpm': 120,
            'time_signature': None,
            'data': {},
            'duration': 0,
            'data_ppq': pattern.resolution,
            'rescale_to_ppq': resolution,
            'midi_file_path': path
        }

        ts_midi_event = None  # time signature MIDI event

        for msg in track:
            if isinstance(msg, midi.NoteOnEvent):
                midi_key = msg.get_pitch()
                if midi_key not in args['data']:
                    args['data'][midi_key] = []
                onset = (msg.tick, msg.get_velocity())
                args['data'][midi_key].append(onset)
            elif isinstance(msg, midi.TimeSignatureEvent):
                if ts_midi_event is None:
                    ts_midi_event = msg
                elif ts_midi_event != msg:
                    raise ValueError("Time signature changes are not supported")
            elif isinstance(msg, midi.SetTempoEvent):
                args['bpm'] = msg.get_bpm()
            elif isinstance(msg, midi.EndOfTrackEvent):
                args['duration'] = msg.tick

        if ts_midi_event is None:
            raise ValueError("No time signature found in '%s'" % path)

        try:
            args['time_signature'] = TimeSignature.from_midi_event(ts_midi_event)
        except ValueError:
            raise ValueError("No time signature MIDI event")

        return Rhythm(**args)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_tracks']
        state['__track_onsets__'] = {}
        for pitch, track in self._tracks.iteritems():
            state['__track_onsets__'][pitch] = track.onsets
        return state

    def __setstate__(self, state):
        state['_tracks'] = {}
        for pitch, onsets in state['__track_onsets__'].iteritems():
            state['_tracks'][pitch] = Rhythm.Track(onsets, self)
        del state['__track_onsets__']
        self.__dict__.update(state)

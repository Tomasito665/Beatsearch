import inspect
import os
import itertools
from functools import wraps
import midi
import math


def musical_unit_enum(**kwargs):
    props = dict(map(lambda key: (key, kwargs[key][0]), kwargs))
    props['__quarter_unit_scale_factors'] = dict(kwargs.values())

    # noinspection PyDecorator
    @staticmethod
    def exists(unit):
        return unit in props.values()

    props['exists'] = exists
    return type('Enum', (), props)


Unit = musical_unit_enum(
    FULL=('fulls', 0.25),
    HALF=('halves', 0.5),
    QUARTER=('quarters', 1.0),
    EIGHTH=('eighths', 2.0),
    SIXTEENTH=('sixteenths', 4.0)
)


def convert_time(time, unit_from, unit_to, quantize=False):
    """
    Converts the given time to a given unit. The given units can be either a resolution (in Pulses
    Per Quarter Note) or one of these musical time units: 'fulls', 'halves', 'quarters', 'eighths or,
    'sixteenths'.

    :param time: time value
    :param unit_from: the original unit of the given time value.
    :param unit_to: the unit to convert the time value to.
    :param quantize: whether or not to round to the nearest target unit. If quantize is true, this
                     function will return an integer. If false, this function will return a float.
    :return: converted time
    """

    if unit_from == unit_to:
        return time

    # noinspection PyUnresolvedReferences
    quarter_unit_scale_factors = Unit.__quarter_unit_scale_factors

    try:
        time_in_quarters = time / float(unit_from)
    except ValueError:
        try:
            time_in_quarters = time / quarter_unit_scale_factors[unit_from]
        except KeyError:
            raise ValueError("Unknown time unit: %s" % unit_from)

    try:
        converted_time = time_in_quarters * float(unit_to)
    except ValueError:
        try:
            converted_time = time_in_quarters * quarter_unit_scale_factors[unit_to]
        except KeyError:
            print quarter_unit_scale_factors
            raise ValueError("Unknown target time unit: %s" % unit_to)

    return int(round(converted_time)) if quantize else converted_time


def concretize_unit(f_get_res=lambda *args, **kwargs: args[0].get_resolution()):
    """
    This function returns a decorator which converts the "unit" argument of a function to a resolution in PPQ.

    :param f_get_res: callable that should return the resolution. The result of this function will be replaced for
                      "unit" arguments with a value of 'ticks'. This callable is passed the *args and **kwargs of the
                      function being decorated.
    :return: a decorator function
    """

    def concretize_unit_decorator(func):
        arg_spec = inspect.getargspec(func)

        try:
            unit_arg_index = arg_spec.args.index("unit")
            n_args_before_default_args = len(arg_spec.args) - len(arg_spec.defaults)
            default_unit = arg_spec.defaults[unit_arg_index - n_args_before_default_args]
        except ValueError:
            unit_arg_index = -1
            if not arg_spec.keywords:
                raise ValueError("Function %s doesn't have a 'unit' argument, "
                                 "neither does it accept **kwargs" % func.__name__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Retrieve the unit or get the default unit if not given
            try:
                unit = kwargs["unit"]
                unit_in_kwargs = True
            except KeyError:
                try:
                    unit = args[unit_arg_index]
                except IndexError:
                    try:
                        unit = default_unit
                    except NameError:
                        raise ValueError("Required named 'unit' argument not found")
                unit_in_kwargs = False

            # Concretize the unit to the rhythm's resolution for tick unit
            if unit == 'ticks':
                unit = f_get_res(*args, **kwargs)

            # Replace unit by concretized unit
            if unit_in_kwargs:
                kwargs["unit"] = unit
            else:
                try:
                    args = list(args)
                    args[unit_arg_index] = unit
                except IndexError:
                    kwargs["unit"] = unit

            return func(*args, **kwargs)
        return wrapper
    return concretize_unit_decorator


class TimeSignature(object):
    """
    This class represents a musical time signature, consisting of a numerator and a denominator.
    """

    def __init__(self, numerator, denominator):
        self._numerator = numerator
        self._denominator = denominator

        if denominator == 4:
            self._beat_unit = Unit.QUARTER
        elif denominator == 8:
            self._beat_unit = Unit.EIGHTH
        else:
            raise ValueError("Unknown denominator: %s" % str(denominator))

    @property
    def numerator(self):
        return self._numerator

    @property
    def denominator(self):
        return self._denominator

    def get_beat_unit(self):
        return self._beat_unit

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

    def __init__(self, name, bpm, time_signature, data, data_ppq, duration=None,
                 ceil_duration_to_measure=True, rescale_to_ppq=None, midi_file_path=""):
        """
        Constructs a new rhythm.

        :param name:               The name of the rhythm
        :param bpm:                Beats per minute; tempo of the rhythm
        :param time_signature:     Time signature of the rhythm
        :param data:               The onset data; dictionary holding onsets by MIDI key where each onset is an
                                   (absolute tick, velocity) tuple
        :param data_ppq:           Resolution of the onset data and the length in PPQ (pulses per quarter note)
        :param duration:           The duration of the rhythm in ticks. When given None, the duration will be set to
                                   first downbeat after the last note in the rhythm.
        :param rescale_to_ppq:     Resolution to rescale the onset data (and the length) to in pulses per quarter note or
                                   None for no rescaling
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

        # default the duration to the timestamp of the last note
        if duration is None:
            duration = self._get_last_onset_tick()

        if ceil_duration_to_measure:
            measure_duration = self.get_measure_duration('ticks')
            n_measures = duration / measure_duration
            # round up the duration to the nearest musical measure
            duration = int(math.ceil(n_measures * measure_duration))

        self.set_duration(duration)

        if rescale_to_ppq is not None:
            self.set_resolution(rescale_to_ppq)

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
            The rhythm object which this track belongs to.
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

        @concretize_unit()
        def get_binary(self, unit='eighths'):
            """
            Returns the binary representation of the note onsets of this track where each step where a note onset
            happens is denoted with a 1; otherwise with a 0. The given resolution is the resolution in PPQ (pulses per
            quarter note) of the binary vector.

            :param unit:
            :return: the binary representation of the note onsets of the given pitch
            """

            resolution = self.get_resolution()
            duration = self.rhythm.get_duration(unit)
            binary_string = [0] * int(math.ceil(duration))

            for onset in self._data:
                tick = onset[0]
                pulse = convert_time(tick, resolution, unit, quantize=True)
                try:
                    binary_string[pulse] = 1
                except IndexError:
                    pass  # when quantization pushes note after end of rhythm

            return binary_string

        @concretize_unit()
        def get_pre_note_inter_onset_intervals(self, unit='ticks'):
            """
            Returns the time difference between the current note and the previous note for all notes of this track in
            eighths. The first note will return the time difference with the start of the rhythm.

            For example, given the Rumba Clave rhythm:
              X--X---X--X-X---
              0  3   4  3 2

            :return: pre note inter-onset interval vector
            """

            resolution = self.get_resolution()
            current_tick = 0
            intervals = []

            for onset in self._data:
                onset_tick = onset[0]
                delta_tick = onset_tick - current_tick
                intervals.append(convert_time(delta_tick, resolution, unit))
                current_tick = onset_tick

            return intervals

        @concretize_unit()
        def get_post_note_inter_onset_intervals(self, unit='ticks', quantize=False):
            """
            Returns the time difference between the current note and the next note for all notes in this rhythm track
            in eighths. The last note will return the time difference with the end (duration) of the rhythm.

            For example, given the Rumba Clave rhythm:
              X--X---X--X-X---
              3  4   3  2 4

            :return: post note inter-onset interval vector
            """

            intervals = []
            onset_ticks = [onset[0] for onset in self._data] + [self.rhythm.get_duration()]
            last_onset_tick = -1

            for onset_tick in onset_ticks:
                if last_onset_tick < 0:
                    last_onset_tick = onset_tick
                    continue

                delta_in_ticks = onset_tick - last_onset_tick
                delta_in_units = convert_time(delta_in_ticks, self.get_resolution(), unit, quantize=quantize)

                intervals.append(delta_in_units)
                last_onset_tick = onset_tick

            return intervals

        def get_binary_schillinger_chain(self, unit='ticks', values=(0, 1)):
            """
            Returns the Schillinger notation of this rhythm track where each onset is a change of a "binary note".

            For example, given the Rumba Clave rhythm and with values (0, 1):
              X--X---X--X-X---
              0001111000110000

            :param unit: the unit to quantize on ('ticks' is no quantization)
            :param values: binary vector to be used in the schillinger chain. E.g. when given ('a', 'b'), the returned
                           schillinger chain will consist of 'a' and 'b'.
            :return: Schillinger rhythm vector as a list
            """

            chain, i = self.get_binary(unit), 0
            value_i = 0
            while i < len(chain):
                if chain[i] == 1:
                    value_i = 1 - value_i
                chain[i] = values[value_i]
                i += 1
            return chain

        def get_chronotonic_chain(self, unit='ticks'):
            """
            Returns the chronotonic chain representation of this rhythm track.

            For example, given the Rumba Clave rhythm:
              X--X---X--X-X---
              3334444333224444

            :param unit: unit
            :return: the chronotonic chain as a list
            """

            chain, i, delta = self.get_binary(unit), 0, 0
            while i < len(chain):
                if chain[i] == 1:
                    j = i + 1
                    while j < len(chain) and chain[j] == 0:
                        j += 1
                    delta = j - i
                chain[i] = delta
                i += 1
                pass
            return chain

        @concretize_unit()
        def get_onset_times(self, unit='ticks', quantize=False):
            """
            Returns the onset times in the given unit.

            :param unit: the unit of the onset times
            :param quantize: whether or not the onset times must be quantized to the given unit
            :return: a list with the onset times of this rhythm track
            """

            return map(lambda onset: convert_time(onset[0], self.get_resolution(), unit, quantize), self.onsets)

        def get_resolution(self):
            """
            Returns the resolution of this rhythm in PPQ (pulses per quarter note).

            :return: the resolution in PPQ
            """

            return self.rhythm.get_resolution()

        # used internally by rhythm.set_resolution to rescale
        # the onsets to a new resolution
        def __set_resolution__(self, new_resolution):
            old_resolution = self.get_resolution()

            def scale_onset(onset):
                tick, velocity = onset
                scaled_tick = convert_time(tick, old_resolution, new_resolution, quantize=True)
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

    def get_resolution(self):
        """
        Returns the resolution of this rhythm in PPQ (pulses per quarter note).

        :return: resolution in PPQ
        """

        return self._ppq

    def set_resolution(self, ppq):
        """
        Rescales this rhythm to the given resolution in PPQ (pulses per quarter note).

        :param ppq: resolution to rescale this rhythm to in PPQ
        """

        if ppq <= 0:
            raise ValueError("Expected a number greater than zero but got %s" % str(ppq))

        ppq = int(ppq)
        old_ppq = self._ppq

        if old_ppq == ppq:
            return

        for track in self._tracks.values():
            track.__set_resolution__(ppq)

        self._duration = convert_time(self._duration, old_ppq, ppq, quantize=True)
        self._ppq = ppq

    @concretize_unit()
    def get_duration(self, unit='ticks'):
        """
        Returns the duration of the rhythm in the given time unit.

        :param unit: the unit of the returned duration
        :return: the duration in the given time unit
        """

        return convert_time(self._duration, self._ppq, unit)

    @concretize_unit()
    def set_duration(self, duration, unit='ticks'):
        """
        Sets the duration of the rhythm in the given time unit.

        :param duration: new duration of the rhythm
        :param unit: unit of the given duration
        """

        duration_in_ticks = convert_time(duration, unit, self._ppq)
        last_onset_tick = self._get_last_onset_tick()

        if duration_in_ticks < last_onset_tick:
            last_onset_in_given_unit = self._get_last_onset_tick(unit)
            raise ValueError("Expected a duration of at least %s but got %s" % (last_onset_in_given_unit, duration))

        self._duration = duration_in_ticks

    @concretize_unit()
    def get_beat_duration(self, unit='ticks'):
        """
        Returns the duration of one musical beat, based on the time signature.

        :param unit: unit of the returned duration
        :return: the duration of one beat in the given unit
        """

        beat_unit = self.time_signature.get_beat_unit()
        return convert_time(1.0, beat_unit, unit)

    @concretize_unit()
    def get_measure_duration(self, unit='ticks'):
        """
        Returns the duration of one musical measure, based on the time signature.

        :param unit: unit of the returned duration
        :return: the duration of one measure in the given unit
        """

        numerator = self.time_signature.numerator
        beat_unit = self.time_signature.get_beat_unit()
        return convert_time(numerator, beat_unit, unit)

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

    def track_iter(self):
        """
        Returns an iterator over the tracks, returning both the pitch and the track on each iteration.

        :return: track iterator
        """

        return self._tracks.iteritems()

    def track_count(self):
        """
        Returns the number of tracks in this rhythm

        :return: number of tracks
        """

        return len(self._tracks)

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
    def create_from_midi(pattern, path=None, resolution=240, end_on_eot_event=False):
        """
        Constructs a new rhythm from a MIDI file.

        :param pattern: midi pattern
        :param path: midi source file or None
        :param resolution: resolution to the rescale the rhythm to in pulses per quarter note
        :param end_on_eot_event: when true, the duration of the rhythms is set according to the last EOT (End Of Track)
                                 MIDI event, otherwise it is set based on the last note.
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
                args['duration'] = max(msg.tick, args['duration'])

        if ts_midi_event is None:
            raise ValueError("No time signature found in '%s'" % path)

        try:
            args['time_signature'] = TimeSignature.from_midi_event(ts_midi_event)
        except ValueError:
            raise ValueError("No time signature MIDI event")

        if not end_on_eot_event:
            args.pop('duration', 0)

        return Rhythm(**args)

    @concretize_unit()
    def _get_last_onset_tick(self, unit='ticks'):
        last_onset_tick = 0
        for onsets in map(lambda track: track.onsets, self._tracks.values()):
            last_onset_tick = max(onsets[-1][0], last_onset_tick)
        return convert_time(last_onset_tick, self._ppq, unit)

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

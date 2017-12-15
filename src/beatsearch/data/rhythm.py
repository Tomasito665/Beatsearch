import inspect
import os
import itertools
from typing import Union, Iterable, Tuple, Any
from collections import OrderedDict
from functools import wraps
import midi
import math
from beatsearch.config import __USE_NUMPY__
from beatsearch.utils import friendly_named_class, inject_numpy


class Unit(object):
    _units = OrderedDict()
    _unit_names = []

    def __init__(self, name, value, scale_factor_from_quarter_note):
        self._name = str(name)
        self._value = value
        self._scale_from_quarter = float(scale_factor_from_quarter_note)
        Unit._units[value] = self

    class UnknownTimeUnit(Exception):
        pass

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value

    @property
    def scale_factor_from_quarter_note(self):
        return self._scale_from_quarter

    @staticmethod
    def exists(unit_value):
        return unit_value in Unit._units

    @staticmethod
    def get(unit):  # type: (Union[str, Unit]) -> Unit
        """
        Returns the Unit object for the given unit value, e.g. "quarters" or "eighths". If the given unit is already
        a Unit object, this method will return that.

        :return: Unit object for given unit value or Unit if given Unit as argument
        """

        try:
            unit_value = unit.value
        except AttributeError:
            unit_value = unit

        try:
            return Unit._units[unit_value]
        except KeyError:
            raise Unit.UnknownTimeUnit(unit_value)

    @staticmethod
    def get_unit_names():
        """
        Returns a tuple containing the names of the units.

        :return: tuple with unit names
        """

        if len(Unit._unit_names) != len(Unit._units):
            Unit._unit_names = tuple(unit.name for unit in Unit._units.values())
        return Unit._unit_names


Unit.FULL = Unit("Full", "fulls", 0.25)
Unit.HALF = Unit("Half", "halves", 0.5)
Unit.QUARTER = Unit("Quarter", "quarters", 1.0)
Unit.EIGHTH = Unit("Eighth", "eighths", 2.0)
Unit.SIXTEENTH = Unit("Sixteenth", "sixteenths", 4.0)
Unit.THIRTY_SECOND = Unit("Thirty-second", "thirty-seconds", 8.0)


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
        return int(round(time)) if quantize else time

    try:
        # if unit_from is a time resolution
        time_in_quarters = time / float(unit_from)  # if unit_from is a time resolution
    except (ValueError, TypeError):
        # if unit_from is musical unit
        unit_from = Unit.get(unit_from)
        time_in_quarters = time / unit_from.scale_factor_from_quarter_note

    try:
        # if unit_to is a time resolution
        converted_time = time_in_quarters * float(unit_to)
    except (ValueError, TypeError):
        # if unit_to is a musical unit
        unit_to = Unit.get(unit_to)
        converted_time = time_in_quarters * unit_to.scale_factor_from_quarter_note

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


def check_iterables_meet_length_policy(iterables, len_policy):
    """
    Checks whether the given iterables meet a certain duration policy. Duration policies are:
        'exact' - met when all iterables have the exact same length and are not empty
        'multiple' - met when the length of the largest iterable is a multiple of all the other iterable lengths
        'fill' - met when any of the chains is empty

    :param iterables: iterables to check the lengths of
    :param len_policy: one of {'exact', 'multiple' or 'fill'}
    :return length of the largest iterable
    """

    if not iterables:
        return

    l = [len(c) for c in iterables]  # lengths

    if len_policy == 'exact':
        if not all(x == l[0] for x in l):
            raise ValueError("When length policy is set to 'exact', iterables should have the same lengths")
    elif len_policy == 'multiple':
        if not all(x % l[0] == 0 or l[0] % x == 0 for x in l):
            raise ValueError("When length policy is set to 'multiple', the length of the largest "
                             "iterable should be a multiple of all the other iterable lengths")
    elif len_policy != 'fill':
        raise ValueError("Unknown length policy: '%s'" % len_policy)

    return max(l)


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

    def to_midi_event(self, metronome=24, thirty_seconds=8):
        return midi.TimeSignatureEvent(
            numerator=self.numerator,
            denominator=self.denominator,
            metronome=metronome,
            thirtyseconds=thirty_seconds
        )

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.numerator == other.numerator and self.denominator == other.denominator

    def __str__(self):
        return "%i/%i" % (self.numerator, self.denominator)

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
                 ceil_duration_to_measure=True, rescale_to_ppq=None, midi_file_path="", midi_metronome=None):
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
        :param rescale_to_ppq:     Resolution to rescale the onset data (and the length) to in pulses per quarter note
                                   or None for no rescaling
        :param midi_file_path:     The path to the midi file or an empty string if this rhythm is not
                                   attached to a MIDI file
        :param midi_metronome:     Midi TimeSignature event metronome
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
        self._midi_metronome = midi_metronome

        # add tracks
        for pitch, onsets in data.iteritems():
            self._tracks[pitch] = Rhythm.Track(onsets, self)

        # default the duration to the timestamp of the last note
        if duration is None:
            duration = self.get_last_onset_time('ticks')

        if ceil_duration_to_measure:
            measure_duration = self.get_measure_duration('ticks')
            n_measures = int(math.ceil(duration / measure_duration))
            duration = n_measures * measure_duration

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

        # TODO Add cyclic option to include the offset in the last onset's interval
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

        @inject_numpy
        def get_interval_histogram(self, unit='eighths', numpy=None):
            """
            Returns the number of occurrences of all the inter-onset intervals from the smallest to the biggest
            interval. The inter onset intervals are retrieved with get_post_note_inter_onset_intervals().

            For example, given the Rumba Clave rhythm, with inter-onset vector [3, 4, 3, 2, 4]:
                (
                    [1, 2, 2],  # occurrences
                    [2, 3, 4]   # bins (interval durations)
                )


            :return: an (occurrences, bins) tuple
            """

            intervals = self.get_post_note_inter_onset_intervals(unit, quantize=True)
            histogram = numpy.histogram(intervals, range(min(intervals), max(intervals) + 2))
            occurrences = histogram[0].tolist()
            bins = histogram[1].tolist()[:-1]
            return occurrences, bins

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
            return chain

        def get_interval_difference_vector(self, cyclic=True, unit='ticks', quantize=False):
            """
            Returns the interval difference vector (aka difference of rhythm vector). For each
            note, this is the difference between the current onset interval and the next onset
            interval. So, if N is the number of onsets, the returned vector will have a length
            of N - 1. This is different with cyclic rhythms, where the last onset's interval is
            compared with the first onset's interval. In this case, the length will be N.

            For example, given the post-note inter-onset interval vector for the Rumba clave:
              [3, 4, 3, 2, 4]

            The interval difference vector would be:
               With cyclic set to False: [4/3, 3/4, 2/3, 4/2]
               With cyclic set to True:  [4/3, 3/4, 2/3, 4/2, 3/4]

            :param cyclic: whether or not to treat this rhythm as a cyclic rhythm or not
            :param unit: time unit
            :param quantize: whether or not the inter onset interval vector should be quantized
            :return: interval difference vector
            """

            vector = self.get_post_note_inter_onset_intervals(unit, quantize)
            if cyclic:
                vector.append(vector[0])
            i = 0
            while i < len(vector) - 1:
                try:
                    vector[i] = vector[i + 1] / float(vector[i])
                except ZeroDivisionError:
                    vector[i] = float('inf')
                i += 1
            vector.pop()
            return vector

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
        The tempo of this rhythm in beats per minute.
        """

        return self._bpm

    @bpm.setter
    def bpm(self, bpm):
        bpm = float(bpm)
        if bpm <= 0:
            raise ValueError("Expected a BPM greater than zero but got %s" % bpm)
        self._bpm = bpm

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
        last_onset_tick = self.get_last_onset_time()

        if duration_in_ticks < last_onset_tick:
            last_onset_in_given_unit = self.get_last_onset_time(unit)
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

    @inject_numpy
    def get_interval_histogram(self, unit='ticks', numpy=None):
        """
        Returns the interval histogram of all the tracks combined.

        :return: combined interval histogram of all the tracks in this rhythm
        """

        intervals = []

        for _, track in self.track_iter():
            track_intervals = track.get_post_note_inter_onset_intervals(unit, quantize=unit)
            intervals.extend(track_intervals)

        histogram = numpy.histogram(intervals, range(min(intervals), max(intervals) + 2))
        occurrences = histogram[0].tolist()
        bins = histogram[1].tolist()[:-1]
        return occurrences, bins

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

    def to_midi(self, note_duration=0):  # type: (int) -> midi.Pattern
        """
        Convert this rhythm to a MIDI.

        :param note_duration: note duration in ticks
        :return: MIDI pattern
        """

        midi_track = midi.Track(tick_relative=False)  # create track and add metadata events

        midi_track.append(midi.TrackNameEvent(text=self._name))  # track name
        midi_metronome = 24 if self._midi_metronome is None else self._midi_metronome
        midi_track.append(self._time_signature.to_midi_event(midi_metronome))  # time signature
        midi_track.append(midi.SetTempoEvent(bpm=self._bpm))  # tempo

        # add note events
        for pitch, track in self._tracks.iteritems():
            pitch = int(pitch)
            onsets = track.onsets
            for onset in onsets:
                note_abs_tick = onset[0]
                velocity = onset[1]
                # channel 9 for drums
                note_on = midi.NoteOnEvent(tick=note_abs_tick, pitch=pitch, velocity=velocity, channel=9)
                note_off = midi.NoteOffEvent(tick=note_abs_tick + note_duration, pitch=pitch, channel=9)
                midi_track.extend([note_on, note_off])

        # sort the events in chronological order and convert to relative delta-time
        midi_track = midi.Track(sorted(midi_track, key=lambda event: event.tick), tick_relative=False)
        midi_track.make_ticks_rel()

        # add end of track event
        midi_track.append(midi.EndOfTrackEvent())

        # save the midi file
        return midi.Pattern(
            [midi_track],
            format=0,
            resolution=self.get_resolution()
        )

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
            'midi_metronome': None,
            'data': {},
            'duration': 0,
            'data_ppq': pattern.resolution,
            'rescale_to_ppq': resolution,
            'midi_file_path': path
        }

        ts_midi_event = None  # time signature MIDI event

        for msg in track:
            if isinstance(msg, midi.NoteOnEvent):
                midi_key = str(msg.get_pitch())
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
            args['midi_metronome'] = ts_midi_event.get_metronome()
        except ValueError:
            raise ValueError("No time signature MIDI event")

        if not end_on_eot_event:
            args.pop('duration', 0)

        return Rhythm(**args)

    @concretize_unit()
    def get_last_onset_time(self, unit='ticks'):
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


class DistanceMeasure(object):
    def get_distance(self, obj_a, obj_b):
        raise NotImplementedError


class TrackDistanceMeasure(DistanceMeasure):
    """
    Distance measure for rhythm tracks (a single instrument of a rhythm).
    """

    LENGTH_POLICIES = ['exact', 'multiple', 'fill']

    class UnknownLengthPolicy(Exception):
        def __init__(self, given_length_policy):
            super(TrackDistanceMeasure.UnknownLengthPolicy, self).__init__(
                "Given %s, please choose between: %s" % (
                    given_length_policy, TrackDistanceMeasure.LENGTH_POLICIES))

    def __init__(self, unit, length_policy):
        """
        Creates a new track distance measure.

        :param unit: see internal_unit property
        :param length_policy: see documentation on length_policy property
        """

        self._len_policy = ''
        self._internal_unit = 0
        self._output_unit = 0

        self.length_policy = length_policy
        self.internal_unit = unit
        self.output_unit = unit

    @property
    def length_policy(self):
        """
        The length policy determines how permissive the track similarity measure is with variable sized track vector
        (N = onset count) or chain (N = duration) representations. Given two track representations X and Y, the length
        policy should be one of:

            'exact': len(X) must equal len(Y)
            'multiple': len(X) must be a multiple of len(Y) or vice-versa
            'fill': len(X) and len(Y) must not be empty

        Implementations of TrackSimilarityMeasure.get_distance will throw a ValueError if the representations of the
        given track do not meet the requirements according to the length policy.
        """

        return self._len_policy

    @length_policy.setter
    def length_policy(self, length_policy):
        valid_policies = TrackDistanceMeasure.LENGTH_POLICIES
        if length_policy not in valid_policies:
            raise TrackDistanceMeasure.UnknownLengthPolicy(length_policy)
        self._len_policy = length_policy

    @property
    def internal_unit(self):
        """
        Unit used internally for distance computation. This unit is given to __get_iterable__
        """

        return self._internal_unit

    @internal_unit.setter
    def internal_unit(self, internal_unit):
        if internal_unit != 'ticks':
            convert_time(1, 1, internal_unit)
        self._internal_unit = internal_unit

    @property
    def output_unit(self):
        """
        The output unit is the unit of the distance returned by get_distance. E.g. when the output_unit is 'ticks', the
        distance returned by get_distance will be in 'ticks'.
        """

        return self._output_unit

    @output_unit.setter
    def output_unit(self, output_unit):
        if output_unit != 'ticks':
            # validates the given unit
            convert_time(1, 1, output_unit)
        self._output_unit = output_unit

    def get_distance(self, track_a, track_b):
        """
        Returns the distance between the given tracks.

        :param track_a: track to compare to track b
        :param track_b: track to compare to track a
        :return: distance between the given tracks
        """

        internal_unit_in_ticks = self._internal_unit == 'ticks'
        output_unit_in_ticks = self._output_unit == 'ticks'
        res_a, res_b = track_a.get_resolution(), track_b.get_resolution()
        if (internal_unit_in_ticks or output_unit_in_ticks) and res_a != res_b:
            raise ValueError("%s unit set to 'ticks', but given tracks have "
                             "different resolutions (%i != %i)" %
                             ("Internal" if internal_unit_in_ticks else "Output", res_a, res_b))
        internal_unit = res_a if internal_unit_in_ticks else self._internal_unit
        output_unit = res_a if output_unit_in_ticks else self._output_unit

        tracks = [track_a, track_b]
        iterables = [self.__get_iterable__(t, internal_unit) for t in tracks]
        cookies = [self.__get_cookie__(t, internal_unit) for t in tracks]
        max_len = self._check_if_iterables_meet_len_policy(*iterables)
        distance = self.__compute_distance__(max_len, *(iterables + cookies))
        return convert_time(distance, internal_unit, output_unit, quantize=False)

    def __get_iterable__(self, track, unit):
        """
        Should prepare and return the track representation on which the similarity measure is based. The returned
        vectors will be length policy checked.

        :param track: the track
        :param unit: the representation should be in the given unit
        :return: desired track representation to use in __compute_distance__
        """

        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def __get_cookie__(self, track, unit):
        """
        The result of this method will be passed to __compute_distance__, both for track a and track b. By default,
        the cookie is the track itself.

        :param track: the track
        :param unit: the unit given to __get_iterable__
        :return: cookie to use in __compute_distance__
        """

        return track

    def __compute_distance__(self, max_len, iterable_a, iterable_b, cookie_a, cookie_b):
        """
        The result of this method is returned by get_distance. If that method is given two tracks a and b, this method
        is given both the iterables of a and b and the cookies of a and b, returned by respectively __get_iterable__
        and __get_cookie__.

        :param max_len: max(len(iterable_a), len(iterable_b))
        :param iterable_a: the result of __get_iterable__, given track a
        :param iterable_b: the result of __get_iterable__, given track b
        :param cookie_a: the result of __get_cookie__, given track a
        :param cookie_b: the result of __get_cookie__, given track b
        :return: the distance between track a and b
        """

        raise NotImplementedError

    def _check_if_iterables_meet_len_policy(self, iterable_a, iterable_b):
        if not iterable_a or not iterable_b:
            return

        l = [len(iterable_a), len(iterable_b)]

        if self.length_policy == "exact":
            if l[0] != l[1]:
                raise ValueError("When length policy is set to \"exact\", both iterables "
                                 "should have the same number of elements")

        elif self.length_policy == "multiple":
            if not all(x % l[0] == 0 or l[0] % x == 0 for x in l):
                raise ValueError("When length policy is set to \"multiple\", the length of the largest "
                                 "iterable should be a multiple of all the other iterable lengths")

        elif self.length_policy != "fill":
            raise ValueError("Unknown length policy: \"%s\"" % self.length_policy)

        return max(l)

    __measures__ = {}  # track distance implementations by __friendly_name__

    @classmethod
    def get_measures(cls, friendly_name=True):
        """
        Returns an ordered dictionary containing implementations of TrackDistanceMeasure by name.

        :param friendly_name: when True, the name will be retrieved with __friendly_name__ instead of __name__
        :return: an ordered dictionary containing all subclasses of TrackDistanceMeasure by name
        """

        if len(TrackDistanceMeasure.__measures__) != TrackDistanceMeasure.__subclasses__():
            measures = OrderedDict()
            for tdm in cls.__subclasses__():
                name = tdm.__name__
                if friendly_name:
                    try:
                        name = tdm.__friendly_name__
                    except AttributeError:
                        pass
                measures[name] = tdm
            cls.__measures__ = measures

        return TrackDistanceMeasure.__measures__

    @classmethod
    def get_measure_names(cls):
        measures = cls.get_measures()
        return measures.keys()

    @classmethod
    def get_measure_by_name(cls, measure_name):
        measures = cls.get_measures()
        try:
            return measures[measure_name]
        except KeyError:
            raise ValueError("No measure with name: '%s'" % measure_name)


@friendly_named_class("Hamming distance")
class HammingDistanceMeasure(TrackDistanceMeasure):
    """
    The hamming distance is based on the binary chains of the rhythms. The hamming distance is the sum of indexes
    where the binary rhythm chains do not match. The hamming distance is always an integer.
    """

    def __init__(self, unit='eighths', length_policy='multiple'):
        super(HammingDistanceMeasure, self).__init__(unit, length_policy)

    def __get_iterable__(self, track, unit):
        return track.get_binary(unit)

    def __compute_distance__(self, n, cx, cy, *cookies):  # cx = (binary) chain x
        hamming_distance, i = 0, 0
        while i < n:
            x = cx[i % len(cx)]
            y = cy[i % len(cy)]
            hamming_distance += x != y
            i += 1
        return hamming_distance


@friendly_named_class("Euclidean interval vector distance")
class EuclideanIntervalVectorDistanceMeasure(TrackDistanceMeasure):
    """
    The euclidean interval vector distance is the euclidean distance between the inter-onset vectors of the rhythms.
    """

    def __init__(self, unit='ticks', length_policy='exact', quantize=False):
        super(EuclideanIntervalVectorDistanceMeasure, self).__init__(unit, length_policy)
        self.quantize = quantize

    def __get_iterable__(self, track, unit):
        return track.get_post_note_inter_onset_intervals(unit, self.quantize)

    def __compute_distance__(self, n, vx, vy, *cookies):
        sum_squared_dt, i = 0, 0
        while i < n:
            dt = vx[i % len(vx)] - vy[i % len(vy)]
            sum_squared_dt += dt * dt
            i += 1
        return math.sqrt(sum_squared_dt)


@friendly_named_class("Interval difference vector distance")
class IntervalDifferenceVectorDistanceMeasure(TrackDistanceMeasure):
    """
    The interval difference vector distance is based on the interval difference vectors of the rhythms.
    """

    def __init__(self, unit='ticks', length_policy='fill', quantize=False, cyclic=True):
        super(IntervalDifferenceVectorDistanceMeasure, self).__init__(unit, length_policy)
        self.quantize = quantize
        self.cyclic = cyclic

    def __get_iterable__(self, track, unit):
        return track.get_interval_difference_vector(self.cyclic, unit, self.quantize)

    def __compute_distance__(self, n, vx, vy, *cookies):
        summed_fractions, i = 0, 0
        while i < n:
            x = float(vx[i % len(vx)])
            y = float(vy[i % len(vy)])
            numerator, denominator = (x, y) if x > y else (y, x)
            try:
                summed_fractions += numerator / denominator
            except ZeroDivisionError:
                return float('inf')
            i += 1
        return summed_fractions - n


@friendly_named_class("Swap distance")
class SwapDistanceMeasure(TrackDistanceMeasure):
    """
    The swap distance is the minimal number of swap operations required to transform one track to another track. A swap
    is an interchange of a one and a zero that are adjacent to each other in the binary representations of the rhythms.

    Although the concept of the swap distance is based on the rhythm's binary chain, this implementation uses the
    absolute onset times of the onsets. This makes it possible to work with floating point swap operations (0.x swap
    operation). Enable this by setting quantize to True in the constructor.
    """

    def __init__(self, unit='eighths', length_policy='multiple', quantize=False):
        super(SwapDistanceMeasure, self).__init__(unit, length_policy)
        self.quantize = quantize

    def __get_iterable__(self, track, unit):
        return track.get_onset_times(unit, self.quantize)

    def __get_cookie__(self, track, unit):
        return int(math.ceil(track.rhythm.get_duration(unit)))

    def __compute_distance__(self, n, vx, vy, dur_x, dur_y):
        swap_distance, i, x_offset, y_offset = 0, 0, 0, 0
        while i < n:
            x_offset = i // len(vx) * dur_x
            y_offset = i // len(vy) * dur_y
            x = vx[i % len(vx)] + x_offset
            y = vy[i % len(vy)] + y_offset
            swap_distance += abs(x - y)
            i += 1
        return swap_distance


@friendly_named_class("Chronotonic distance")
class ChronotonicDistanceMeasure(TrackDistanceMeasure):
    """
    The chronotonic distance is the area difference (aka measure K) of the rhythm's chronotonic chains.
    """

    def __init__(self, unit='eighths', length_policy='multiple'):
        super(ChronotonicDistanceMeasure, self).__init__(unit, length_policy)

    def __get_iterable__(self, track, unit):
        return track.get_chronotonic_chain(unit)

    def __compute_distance__(self, n, cx, cy, *args):
        chronotonic_distance, i = 0, 0
        while i < n:
            x = cx[i % len(cx)]
            y = cy[i % len(cy)]
            # assuming that each pulse is a unit
            chronotonic_distance += abs(x - y)
            i += 1
        return chronotonic_distance


TRACK_WILDCARDS = ["*", "a*", "b*"]  # NOTE: Don't change the wildcard order or the code will break


def rhythm_pair_track_iterator(rhythm_a, rhythm_b, tracks):
    # type: (Rhythm, Rhythm, Union[str, Iterable[Any]]) -> Tuple[Any, Tuple[Rhythm.Track, Rhythm.Track]]

    """
    Returns an iterator over the tracks of the given rhythms. Each iteration yields a track name and a pair of tracks
    with that track name; rhythm_a.get_track(name) and rhythm_b.get_track(name). This is yielded as a tuple:
        (track_name, (track_a, track_b))

    The tracks to iterate over can be specified with the 'tracks' argument:

        [iterable] - Iterate over tracks with the names in the given iterable
        '*'        - Iterate over all track names. If the rhythms contain tracks with the same name, this will just
                     result in one iteration. E.g. if both rhythms contain a track named 'foo', this will result in one
                     iteration yielding ('foo', (track_a, track_b))).
        'a*'      - Iterate over track names in rhythm a
        'b*'      - Iterate over track names in rhythm b

    :param rhythm_a: the first rhythm
    :param rhythm_b: the second rhythm
    :param tracks: either an iterable containing the track names, or one of the wildcards ['*', 'a*' or 'b*']
    :return: an iterator over the tracks of the given rhythms
    """

    try:
        wildcard_index = TRACK_WILDCARDS.index(tracks)
    except ValueError:
        wildcard_index = -1
        try:
            tracks = iter(tracks)
        except TypeError:
            raise ValueError("Excepted an iterable or "
                             "one of %s but got %s" % (TRACK_WILDCARDS, tracks))

    it_a, it_b = rhythm_a.track_iter(), rhythm_b.track_iter()

    if wildcard_index == -1:  # if given specific tracks
        for track_name in tracks:
            track_a = rhythm_a.get_track(track_name)
            track_b = rhythm_b.get_track(track_name)
            yield (track_name, (track_a, track_b))

    elif wildcard_index == 0:  # if '*' wildcard
        names = set()
        for name, track_a in it_a:
            names.add(name)
            track_b = rhythm_b.get_track(name)
            names.add(name)
            yield (name, (track_a, track_b))
        for name, track_b in it_b:
            if name in names:
                continue
            track_a = rhythm_a.get_track(name)
            yield (name, (track_a, track_b))

    elif wildcard_index == 1:  # if wildcard 'a*'
        for name, track_a in it_a:
            track_b = rhythm_b.get_track(name)
            yield (name, (track_a, track_b))

    elif wildcard_index == 2:  # if wildcard 'b*'
        for name, track_b in it_b:
            track_a = rhythm_a.get_track(name)
            yield (name, (track_a, track_b))

    else:
        assert False


class RhythmDistanceMeasure(DistanceMeasure):
    def __init__(self, track_distance_measure=HammingDistanceMeasure, tracks='a*', normalize=True):
        # type: (Union[TrackDistanceMeasure, type], Union[str, Iterable[Any]]) -> None
        self._tracks = []
        self.tracks = tracks
        self._track_distance_measure = None
        self.track_distance_measure = track_distance_measure
        self.normalize = normalize

    @property
    def track_distance_measure(self):
        """
        The distance measure used to compute the distance between the rhythm tracks; an instance of TrackDistanceMeasure.
        """

        return self._track_distance_measure

    @track_distance_measure.setter
    def track_distance_measure(self, track_distance_measure):
        """
        Setter for the track distance measure. This should be either a TrackDistanceMeasure subclass or instance. When
        given a class, the measure will be initialized with no arguments.
        """

        if inspect.isclass(track_distance_measure) and \
                issubclass(track_distance_measure, TrackDistanceMeasure):
            track_distance_measure = track_distance_measure()
        elif not isinstance(track_distance_measure, TrackDistanceMeasure):
            raise ValueError("Expected a TrackDistanceMeasure subclass or "
                             "instance, but got '%s'" % track_distance_measure)
        self._track_distance_measure = track_distance_measure

    @property
    def tracks(self):
        """
        The tracks to iterate over when computing the distance. See rhythm_pair_track_iterator.
        """

        return self._tracks

    @tracks.setter
    def tracks(self, tracks):
        self._tracks = tracks

    def get_distance(self, rhythm_a, rhythm_b):
        """
        Returns the average track distance from the tracks of rhythm a and b. When the track distance can't be computed,
        duration of the longest rhythm is used as a distance. This can either happen when the a certain track doesn't
        exist in the two rhythms (e.g. 'snare' track in rhythm_a but not in rhythm_b) or when the track distance measure
        raises an exception when computing the distance.

        See RhythmSimilarityMeasure.tracks to specify which tracks should measured.

        :param rhythm_a: the first rhythm
        :param rhythm_b: the second rhythm
        :return: the average track distance of the two rhythms
        """

        measure = self.track_distance_measure
        unit = measure.output_unit
        duration = max(rhythm_a.get_duration(unit), rhythm_b.get_duration(unit))
        n_tracks, total_distance = 0, 0

        for name, tracks in rhythm_pair_track_iterator(rhythm_a, rhythm_b, self.tracks):
            distance = duration
            if None not in tracks:
                try:
                    distance = measure.get_distance(tracks[0], tracks[1])
                except ValueError:
                    pass
            total_distance += distance
            n_tracks += 1

        average_distance = float(total_distance) / n_tracks
        return average_distance / duration if self.normalize else average_distance


def create_rumba_rhythm(resolution=240, track=38, bpm=120):
    """
    Utility function that creates a one-bar rumba rhythm.

    :param resolution: rhythm resolution
    :param track: which track to place the rumba pattern on
    :param bpm: rhythm tempo in beats per minute
    :return: rumba rhythm
    """

    onset_data = {track: ((0, 127), (3, 127), (7, 127), (10, 127), (12, 127))}
    rhythm = Rhythm("<Dummy: Rumba>", bpm, TimeSignature(4, 4), onset_data, 4, 16)
    rhythm.set_resolution(resolution)
    return rhythm

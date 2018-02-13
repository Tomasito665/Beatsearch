import os
import socket
import threading
from collections import namedtuple

try:
    # noinspection PyUnresolvedReferences
    import beyond.Reaper
except ModuleNotFoundError:
    raise ModuleNotFoundError("beyond.Reaper not found. Download it here: "
                              "https://forum.cockos.com/showthread.php?t=129696")
import typing as tp
from tkinter import *
from tempfile import TemporaryFile
# noinspection PyUnresolvedReferences
from beatsearch_dirs import BS_ROOT, BS_LIB
sys.path.append(BS_LIB)
from beatsearch.app.control import BSController, BSRhythmPlayer, BSRhythmLoopLoader
from beatsearch.rhythm import MidiRhythm, PolyphonicRhythmImpl, TimeSignature, Unit
from beatsearch.app.view import BSApp
from beatsearch.utils import get_default_beatsearch_rhythms_fpath
# noinspection PyUnresolvedReferences
import midi
# noinspection PyUnresolvedReferences
ReaperApi = Reaper


def main(input_track_name, output_track_name):
    player = ReaperRhythmPlayer()
    rhythm_loader = ReaperRhythmLoopLoader()

    def find_reaper_thread_target():
        try:
            with ReaperApi as api:
                print("Connected to Reaper")

                try:
                    bs_input_track = ReaperUtils.find_track_by_name(api, input_track_name)
                except ValueError:
                    bs_input_track = ReaperUtils.insert_track(api, 0, input_track_name)

                try:
                    bs_output_track = ReaperUtils.find_track_by_name(api, output_track_name)
                except ValueError:
                    bs_output_track = ReaperUtils.insert_track(api, 0, output_track_name)

            rhythm_loader.set_input_track(bs_input_track)
            player.set_output_track(bs_output_track)

        except socket.timeout:
            if not app.is_closed:
                app.destroy()

    print("Trying to connect to Reaper...")
    threading.Thread(target=find_reaper_thread_target).start()

    print("Initializing controller...")
    controller = BSController(rhythm_player=player)
    controller.register_rhythm_loader(rhythm_loader)

    print("Initializing view...")
    app = BSApp(controller, baseName="beatsearch-reaper-plugin")
    app.mainloop()


class ReaperRhythmPlayer(BSRhythmPlayer):
    """Implementation of BSRhythmPlayer that uses Reaper as MIDI output for the rhythms"""

    def __init__(self, output_track: str = None):
        super(ReaperRhythmPlayer, self).__init__()
        self._tmp_files = set()              # type: tp.Set[str]
        self._start_pos = -1                 # type: float
        self._end_pos = -1                   # type: float
        self._end_pos_first_rhythm = -1      # type: float
        self._output_track = output_track    # type: str
        self._current_rhythms = []           # type: tp.List[PolyphonicRhythmImpl]
        self._is_playing = False             # type: bool

        if output_track is not None:
            self.set_output_track(output_track)

    @property
    def output_track(self) -> str:
        return self._output_track

    @output_track.setter
    def output_track(self, output_track: str) -> None:
        self.set_output_track(output_track)

    def set_output_track(self, output_track: str, reaper_api: ReaperApi = None):
        if reaper_api is None:
            with ReaperApi as api:
                self.set_output_track(output_track, api)
            return

        reaper_api.OnStopButton()
        reaper_api.GetSetRepeat(1)  # enable repeat
        ReaperUtils.clear_track(reaper_api, self._output_track)
        self._output_track = output_track

    def playback_rhythms(self, rhythms: tp.Iterable[MidiRhythm]) -> None:
        with ReaperApi as api:
            api.OnPauseButton()
            self.populate_track(api, rhythms)
            ReaperUtils.set_time_selection(api, self._start_pos, self._end_pos)
            ReaperUtils.enable_metronome(api, False)
            api.RPR_CSurf_OnMuteChange(self.output_track, False)  # unmute output track
            api.RPR_CSurf_OnSoloChange(self.output_track, True)  # solo output track
            api.OnPlayButton()
        self._is_playing = True

    def stop_playback(self) -> None:
        with ReaperApi as api:
            api.OnStopButton()

            api.RPR_CSurf_OnMuteChange(self.output_track, True)  # mute output track
            api.RPR_CSurf_OnSoloChange(self.output_track, False)  # un-solo output track

            if self._current_rhythms:
                assert self._start_pos > 0
                assert self._end_pos_first_rhythm > 0
                ReaperUtils.set_time_selection(api, self._start_pos, self._end_pos_first_rhythm)

            ReaperUtils.enable_metronome(api, True)
        self._is_playing = False

    def is_playing(self) -> bool:
        return self._is_playing

    def set_repeat(self, enabled: bool) -> None:
        print("ReaperRhythmPlayer.set_repeat() not available. Repeat is always enabled.", file=sys.stderr)

    def get_repeat(self) -> bool:
        return True

    def clean_temp_files(self) -> None:
        for fpath in self._tmp_files:
            os.remove(fpath)

    def populate_track(self, api, rhythms: tp.Iterable[MidiRhythm]) -> None:
        if rhythms == self._current_rhythms:
            return

        ReaperUtils.clear_track(api, self._output_track)
        ReaperUtils.delete_all_time_signature_markers(api)
        measure_duration = ReaperUtils.get_measure_duration(api)

        # Reaper freaks out when we try to time select starting from 0 so
        # we place the rhythms at the start of the second measure
        api.SetEditCurPos(measure_duration, False, False)
        self._start_pos = api.GetCursorPosition()
        self._end_pos_first_rhythm = -1

        for rhythm in rhythms:
            ReaperUtils.insert_time_signature_marker(api, rhythm.time_signature)
            temp_file = self.render_rhythm_and_insert_to_track(self._output_track, rhythm, api)
            self._tmp_files.add(temp_file)

            if self._end_pos_first_rhythm < 0:
                self._end_pos_first_rhythm = api.GetCursorPosition()

        self._end_pos = api.GetCursorPosition()
        self._current_rhythms = rhythms

        time_signature_proj_start = ReaperUtils.get_time_signature_at_time(api, 0)
        ReaperUtils.insert_time_signature_marker(api, time_signature_proj_start)

    @staticmethod
    def render_rhythm_and_insert_to_track(output_track, rhythm: MidiRhythm, reaper_api):
        ReaperUtils.set_selected_tracks(reaper_api, output_track)
        with TemporaryFile(prefix="beatsearch-reaper", suffix=rhythm.name + ".mid", delete=False) as f:
            pattern = rhythm.as_midi_pattern()
            midi.write_midifile(f, pattern)
            fpath = f.name
        reaper_api.InsertMedia(fpath, 0)
        return fpath


class ReaperRhythmLoopLoader(BSRhythmLoopLoader):
    SOURCE_NAME = "Reaper MIDI media item"

    def __init__(self, input_track=None):
        super().__init__()
        self._input_track = input_track

    def set_input_track(self, track_id):
        self._input_track = track_id

    def is_available(self):
        return self._input_track is not None

    # noinspection PyUnboundLocalVariable
    def __load__(self, **kwargs):
        track = self._input_track

        with ReaperApi as api:
            track_name = ReaperUtils.get_track_name(api, track)

            try:
                first_selected_item = next(ReaperUtils.selected_items_on_track_iter(api, track))
            except StopIteration:
                first_selected_item = None

            if first_selected_item is not None:
                media_take = api.GetMediaItemTake(first_selected_item, 0)
                media_take_name = api.GetSetMediaItemTakeInfo_String(media_take, "P_NAME", 0, False)[3]
                position = api.RPR_GetMediaItemInfo_Value(first_selected_item, "D_POSITION")
                time_signature = ReaperUtils.get_time_signature_at_time(api, position)
                bpm = ReaperUtils.get_tempo_at_time(api, position)

                midi_resolution = ReaperUtils.get_midi_resolution(api, media_take)
                midi_notes = tuple(ReaperUtils.midi_note_iter(api, media_take))

        # Note: raise outside ReaperApi context, otherwise it will be caught by beyond.Reaper
        if first_selected_item is None:
            raise self.LoadingError("Please select a media item on track \"%s\"" % track_name)

        midi_track = midi.Track(tick_relative=False)  # create track and add metadata events

        # add meta data
        midi_track.append(midi.TrackNameEvent(text=media_take_name))
        midi_track.append(midi.SetTempoEvent(bpm=bpm))
        midi_track.append(time_signature.to_midi_event())

        # add note on/off events
        for note in midi_notes:
            if note.muted:
                continue
            midi_track.extend(note.to_midi_event_pair())

        # sort the events in chronological order and convert to relative delta-time
        midi_track = midi.Track(sorted(midi_track, key=lambda event: event.tick), tick_relative=False)
        midi_track.make_ticks_rel()

        # add end of track event
        midi_track.append(midi.EndOfTrackEvent())

        # create midi pattern
        midi_pattern = midi.Pattern(
            [midi_track],
            format=0,
            resolution=midi_resolution
        )

        return MidiRhythm(midi_pattern=midi_pattern, name=media_take_name)

    @classmethod
    def get_source_name(cls):
        return cls.SOURCE_NAME


class ReaperUtils:

    @staticmethod
    def track_iter(reaper_api: ReaperApi):
        """
        Returns an iterator over the tracks in the current REAPER project, yielding a track id on each iteration.

        :return: iterator over tracks in current REAPER project
        """

        n_tracks, i = reaper_api.GetNumTracks(), 0
        while i < n_tracks:
            yield reaper_api.GetTrack(0, i)
            i += 1

    @staticmethod
    def media_item_iter(reaper_api: ReaperApi, track_id):
        """
        Returns an iterator over the media items in the given track, yielding a media item id on each iteration.

        :param reaper_api: reaper api handle
        :param track_id: the track whose media items to iterate over
        :return: iterator over media items in given track
        """

        n_media_items, i = reaper_api.GetTrackNumMediaItems(track_id), 0
        print("Iterating over track %s, which has %i media items" % (track_id, n_media_items))
        while i < n_media_items:
            yield reaper_api.GetTrackMediaItem(track_id, i)
            i += 1

    @staticmethod
    def clear_track(reaper_api: ReaperApi, media_track_id):
        """
        Removes all media items from the given media track.
        """

        while reaper_api.GetTrackNumMediaItems(media_track_id) > 0:
            media_item = reaper_api.GetTrackMediaItem(media_track_id, 0)
            reaper_api.DeleteTrackMediaItem(media_track_id, media_item)

    @staticmethod
    def get_time_selection(reaper_api: ReaperApi):
        """
        Returns a tuple with two elements; the start and the end of the current time selection. The times returned by
        this function are in seconds.
        """

        return reaper_api.GetSet_LoopTimeRange(0, 0, 0, 0, 0)[2:4]

    @staticmethod
    def reset_time_selection(reaper_api: ReaperApi):
        """
        Sets both the start and the end of the time selection to the start of the project.
        """

        reaper_api.SetEditCurPos(0, False, False)
        for i in range(2):
            reaper_api.MoveEditCursor(0, True)

    @classmethod
    def set_time_selection(cls, reaper_api: ReaperApi, t_start, t_end, move_cursor_to_start=True):
        """
        Sets the time selection to the given start and end position in seconds.
        """

        cls.reset_time_selection(reaper_api)
        reaper_api.SetEditCurPos(t_start, False, False)
        reaper_api.MoveEditCursor(t_end - t_start, True)
        if move_cursor_to_start:
            reaper_api.SetEditCurPos(t_start, False, False)

    @classmethod
    def find_track_by_name(cls, reaper_api: ReaperApi, track_name):
        """
        Returns the first track with in the current REAPER project with the given track name. Returns None if no class
        found with given name. Note that this is a linear search (running time of O(N)).

        :param reaper_api: reaper api handle
        :param track_name: track name as a string
        :return: the track id of the first track with the given name
        """

        for track_id in cls.track_iter(reaper_api):
            current_track_name = cls.get_track_name(reaper_api, track_id)
            if current_track_name == track_name:
                return track_id
        raise ValueError("No track named '%s'" % track_name)

    @staticmethod
    def get_track_name(reaper_api: ReaperApi, track_id, max_name_length=64):
        """
        Returns the name of the given track.

        :param reaper_api: reaper api
        :param track_id: track id
        :param max_name_length: string buffer size for track name
        :return: track name of given track
        """

        return reaper_api.GetTrackName(track_id, 0, max_name_length)[2]

    @classmethod
    def set_selected_tracks(cls, reaper_api: ReaperApi, *tracks_to_select):
        """
        Sets the track selection to the given tracks. This will unselect all tracks that are not specified in the
        arguments to this function.
        """

        for track_id in cls.track_iter(reaper_api):
            reaper_api.SetTrackSelected(track_id, track_id in tracks_to_select)

    @staticmethod
    def insert_track(reaper_api: ReaperApi, position=0, name=""):
        """
        Inserts a new track to the current REAPER project.

        :param reaper_api: reaper api
        :param position: position to insert the track to (0 = beginning)
        :param name: track name
        :return: the id of the newly created track
        """

        n_tracks = reaper_api.GetNumTracks()
        position = max(0, min(n_tracks, position))
        reaper_api.InsertTrackAtIndex(position, False)
        track_id = reaper_api.GetTrack(0, position)
        if name:
            reaper_api.GetSetMediaTrackInfo_String(track_id, "P_NAME", name, 1)
        return track_id

    class MIDINote(namedtuple("MIDINote", [
        "selected", "muted", "startppqpos",
        "endppqpos", "channel", "pitch", "velocity"
    ])):
        def to_midi_event_pair(self):
            """
            Converts this Reaper MIDI note to a midi.NoteOn/midi.NoteOff event pair. Note that timing is absolute.

            :return: tuple containing the note on and note off midi events for this note
            """

            channel = self.channel
            pitch = self.pitch
            t_start, t_end = self.startppqpos, self.endppqpos

            note_on = midi.NoteOnEvent(tick=t_start, pitch=pitch, velocity=self.velocity, channel=channel)
            note_off = midi.NoteOffEvent(tick=t_end, pitch=pitch, channel=channel)

            return note_on, note_off

    @classmethod
    def midi_note_iter(cls, reaper_api: ReaperApi, media_take):
        """
        Returns an iterator over the MIDI notes of the given MIDI media take. The iterator yields a MIDINote.

        :param reaper_api: reaper api
        :param media_take: MIDI media take
        :return: iterator over MIDI notes in given MIDI media take yielding a MIDINote namedtuple
        """

        if not reaper_api.TakeIsMIDI(media_take):
            raise ValueError("Given media take \"%s\" is not a MIDI take" % media_take)

        n_notes = reaper_api.RPR_MIDI_CountEvts(media_take, 0, 0, 0)[2]

        for i in range(n_notes):
            result, _take, _note_ix, *note_info = reaper_api.RPR_MIDI_GetNote(media_take, i, 0, 0, 0, 0, 0, 0, 0)
            if not result:
                raise Exception("Something went wrong while retrieving the "
                                "%ith MIDI note from MIDI take \"%s\"" % (i, media_take))
            yield cls.MIDINote(*note_info)

    @classmethod
    def midi_event_iter(cls, reaper_api: ReaperApi, media_take):
        """
        Returns an iterator over all MIDI events in the given MIDI media take. The iterator yields a string.

        :param reaper_api:
        :param media_take:
        :return:
        """

        if True:
            raise Exception("Sorry, this method is currently broken. See to-do in loop")

        if not reaper_api.TakeIsMIDI(media_take):
            raise ValueError("Given media take \"%s\" is not a MIDI take" % media_take)

        n_events = reaper_api.RPR_MIDI_CountEvts(media_take, 0, 0, 0)[0]

        for i in range(n_events):
            # TODO fix utf-8 decode error
            event_info = reaper_api.RPR_MIDI_GetEvt(media_take, i, 0, 0, 0, "", 256)
            print(event_info)

    @staticmethod
    def get_midi_resolution(reaper_api: ReaperApi, media_take):
        """
        Returns the MIDI resolution in ticks per quarter note (PPQN) for the given MIDI media take.

        :param reaper_api: reaper api
        :param media_take: MIDI media take
        :return: the MIDI resolution in PPQN of the given MIDI media take
        """

        if not reaper_api.TakeIsMIDI(media_take):
            raise ValueError("Given media take \"%s\" is not a MIDI take" % media_take)

        t_start_take = reaper_api.RPR_MIDI_GetProjTimeFromPPQPos(media_take, 0)
        res = reaper_api.RPR_MIDI_GetPPQPosFromProjQN(media_take, 1.0)
        res_offset = reaper_api.RPR_MIDI_GetPPQPosFromProjTime(media_take, t_start_take * 2)

        return res + res_offset

    @classmethod
    def selected_items_on_track_iter(cls, reaper_api: ReaperApi, track_id):
        """
        Returns an iterator over the selected media items on the given track.

        :param reaper_api: reaper api
        :param track_id: track id
        :return: an iterator over the selected media items on the given track
        """

        for media_item in cls.items_on_track_iter(reaper_api, track_id):
            if reaper_api.IsMediaItemSelected(media_item):
                yield media_item

    @staticmethod
    def items_on_track_iter(reaper_api: ReaperApi, track_id):
        """
        Returns an iterator over the media items on the given track.

        :param reaper_api: reaper api
        :param track_id: track id
        :return: an iterator over the media items on the given track
        """

        n_items = reaper_api.CountTrackMediaItems(track_id)

        for item_ix in range(n_items):
            yield reaper_api.GetTrackMediaItem(track_id, item_ix)

    @staticmethod
    def get_time_signature(reaper_api: ReaperApi, time=0, project=-1):
        """
        Returns the time signature of a reaper project at the given time.

        :param reaper_api: reaper api
        :param time: time
        :param project: reaper project id (-1 for current project)
        :return: a TimeSignature object representing the time signature of a Reaper project and the given time
        """

        numerator, denominator = reaper_api.RPR_TimeMap_GetTimeSigAtTime(project, time, 0, 0, 0)[2:4]
        return TimeSignature(numerator, denominator)

    @classmethod
    def get_measure_duration(cls, reaper_api: ReaperApi, time=0, project=-1):
        """
        Returns the duration of a measure of a Reaper project at a given time.

        :param reaper_api: reaper api
        :param time: time
        :param project: reaper project id (-1 for current project)
        :return: the duration of one measure at the given time for the given Reaper project
        """

        time_signature = cls.get_time_signature(reaper_api, time, project)
        beat_unit = time_signature.get_beat_unit()

        # noinspection PyUnresolvedReferences
        if beat_unit == Unit.QUARTER:
            n_quarter_notes_per_measure = time_signature.numerator
        else:
            # noinspection PyUnresolvedReferences
            assert beat_unit == Unit.EIGHTH
            n_quarter_notes_per_measure = time_signature.numerator / 2.0

        return reaper_api.RPR_TimeMap_QNToTime(n_quarter_notes_per_measure)

    RPR_COMMAND_ENABLE_METRONOME = 41745
    """Reaper action id for the metronome enable command"""

    RPR_COMMAND_DISABLE_METRONOME = 41746
    """Reaper action id for the metronome disable command"""

    @classmethod
    def enable_metronome(cls, reaper_api: ReaperApi, enabled=True):
        """
        Enables or disables the Reaper metronome.

        :param reaper_api: reaper api
        :param enabled: true for enabling the metronome, false for disabling
        :return: None
        """

        action_id = cls.RPR_COMMAND_ENABLE_METRONOME if enabled else cls.RPR_COMMAND_DISABLE_METRONOME
        reaper_api.RPR_Main_OnCommand(action_id, 0)

    @classmethod
    def delete_all_time_signature_markers(cls, reaper_api: ReaperApi):
        """
        Removes all timesignature/tempo markers.

        :param reaper_api: reaper api
        :return: None
        """

        def get_marker_count():
            return reaper_api.RPR_CountTempoTimeSigMarkers(-1)

        n_markers = get_marker_count()

        while n_markers > 0:
            reaper_api.RPR_DeleteTempoTimeSigMarker(-1, int(n_markers > 1))
            n_markers = get_marker_count()

    @classmethod
    def insert_time_signature_marker(cls, reaper_api: ReaperApi, time_signature, position=-1):
        """
        Adds a time signature marker on the given position.

        :param reaper_api: reaper api
        :param position: position in seconds or -1 for current position
        :param time_signature: time signature
        :return: None
        """

        bpm = cls.get_project_tempo(reaper_api)
        numerator = time_signature.numerator
        denominator = time_signature.denominator
        if position < 0:
            position = reaper_api.GetCursorPosition()
        reaper_api.RPR_SetTempoTimeSigMarker(-1, -1, position, -1, -1, bpm, numerator, denominator, False)

    @classmethod
    def get_project_tempo(cls, reaper_api: ReaperApi):
        """
        Returns the project tempo in bpm.

        :param reaper_api: reaper api
        :return: project tempo in bpm
        """

        return reaper_api.RPR_GetProjectTimeSignature2(0, 0, 0)[1]

    @classmethod
    def get_time_signature_at_time(cls, reaper_api: ReaperApi, position=-1) -> TimeSignature:
        """
        Returns the time signature at the start of the current project.

        :param reaper_api: reaper api
        :param position: position in seconds, -1 for current position
        :return: time signature at the start of the current project as a TimeSignature object
        """

        if position < 0:
            position = reaper_api.GetCursorPosition()

        numerator, denominator = reaper_api.RPR_TimeMap_GetTimeSigAtTime(-1, position, 0, 0, 0)[2:4]
        return TimeSignature(numerator, denominator)

    @classmethod
    def get_tempo_at_time(cls, reaper_api: ReaperApi, position=-1):
        """
        Returns the tempo in bpm at the given position.

        :param reaper_api: reaper api
        :param position: position in seconds, -1 for current position
        :return: tempo in bpm at the given position
        """

        if position < 0:
            position = reaper_api.GetCursorPosition()

        return reaper_api.RPR_TimeMap_GetTimeSigAtTime(-1, position, 0, 0, 0)[4]


if __name__ == "__main__":
    main("BS_Input", "BS_Output")

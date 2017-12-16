import os
import sys
import midi
import threading
from io import StringIO
from tempfile import TemporaryFile
from beatsearch_dirs import BS_LIB, BS_ROOT
sys.path.append(BS_LIB)
# noinspection PyUnresolvedReferences
from reaper_python import (
    RPR_ClearConsole,
    RPR_GetPlayState,
    RPR_ShowConsoleMsg,
    RPR_GetPlayPosition,
    RPR_OnPlayButton,
    RPR_InsertMedia,
    RPR_GetSetRepeat,
    RPR_GetTrackNumMediaItems,
    RPR_GetTrackMediaItem,
    RPR_OnStopButton,
    RPR_InsertTrackAtIndex,
    RPR_GetTrackName,
    RPR_MoveEditCursor,
    RPR_GetSet_LoopTimeRange,
    RPR_GetCursorPosition,
    RPR_GetNumTracks,
    RPR_SetTrackSelected,
    RPR_GetSetMediaTrackInfo_String,
    RPR_SetEditCurPos,
    RPR_GetTrack,
    RPR_DeleteTrackMediaItem,
)

# Disables Numpy imports (Reaper crashes when trying to import numpy)
from beatsearch.config import enable_numpy
enable_numpy(False)

from beatsearch.app.control import BSController, BSRhythmPlayer
from beatsearch.app.view import BSApp


def main():
    try:
        bs_input_track = ReaperUtils.find_track_by_name("BS_Output")
    except ValueError:
        bs_input_track = ReaperUtils.insert_track(0, "BS_Output")
    player = ReaperRhythmPlayer(bs_input_track)
    player.set_repeat(True)

    print("Initializing controller...")
    controller = BSController(rhythm_player=player)
    controller.set_corpus(os.path.join(BS_ROOT, "data", "rhythms.pkl"))

    print("Initializing view...")
    app = BSApp(controller, baseName="beatsearch-reaper-plugin")
    app.mainloop()


class ReaperConsoleOut(object, StringIO):
    def __init__(self, prefix="", **kwargs):
        super(ReaperConsoleOut, self).__init__(**kwargs)
        self._prefix = str(prefix)
        self.has_prefix = len(self._prefix) > 0
        self._prev_write = "\n"

    def write(self, s):
        if self.has_prefix:
            show_prefix = self._prev_write.endswith("\n")
            s = "%s %s" % (self._prefix, s) if show_prefix else s
        RPR_ShowConsoleMsg(s)
        self._prev_write = s


class ReaperRhythmPlayer(BSRhythmPlayer):
    """Implementation of BSRhythmPlayer that uses Reaper as MIDI output for the rhythms"""

    CHECK_ENDED_FREQUENCY = 15

    def __init__(self, output_track):
        super(ReaperRhythmPlayer, self).__init__()
        self._timer = None
        self._check_ended_interval = 1.0 / self.CHECK_ENDED_FREQUENCY
        self._tmp_files = set()
        self._start_pos = -1
        self._end_pos = -1
        self._prev_pos = -1
        self._output_track = output_track
        self._current_rhythms = []
        ReaperUtils.clear_track(self._output_track)

    def playback_rhythms(self, rhythms):
        self.populate_track(rhythms)

        ReaperUtils.set_time_selection(self._start_pos, self._end_pos)
        RPR_OnPlayButton()

        self._timer = threading.Timer(self._check_ended_interval, self._check_if_playback_ended)
        self._timer.start()

    def stop_playback(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        RPR_OnStopButton()

    def is_playing(self):
        return self._timer is not None

    def set_repeat(self, enabled):
        RPR_GetSetRepeat(int(bool(enabled)))

    def get_repeat(self):
        return bool(RPR_GetSetRepeat(-1))

    def clean_temp_files(self):
        for fpath in self._tmp_files:
            os.remove(fpath)

    def populate_track(self, rhythms):
        if rhythms == self._current_rhythms:
            print("Rhythms are the same... not re-populating track")
            return

        ReaperUtils.clear_track(self._output_track)
        RPR_SetEditCurPos(1, False, False)
        self._start_pos = RPR_GetCursorPosition()

        for rhythm in rhythms:
            temp_file = self.render_rhythm_and_insert_to_track(self._output_track, rhythm)
            self._tmp_files.add(temp_file)

        self._end_pos = RPR_GetCursorPosition()
        self._current_rhythms = rhythms

    @staticmethod
    def render_rhythm_and_insert_to_track(reaper_track, rhythm):
        ReaperUtils.set_selected_tracks(reaper_track)
        with TemporaryFile(prefix="beatsearch-reaper", suffix=rhythm.name + ".mid", delete=False) as f:
            pattern = rhythm.to_midi()
            midi.write_midifile(f, pattern)
            fpath = f.name
        RPR_InsertMedia(fpath, 0)
        return fpath

    def _on_playback_ended(self):
        self._prev_pos = -1
        self._timer = None
        RPR_OnStopButton()
        self.on_playback_ended()

    def _check_if_playback_ended(self):
        pos = RPR_GetPlayPosition()
        if pos >= self._end_pos or RPR_GetPlayState() == 0:
            return self._on_playback_ended()
        self._prev_pos = pos
        self._timer = threading.Timer(self._check_ended_interval, self._check_if_playback_ended)
        self._timer.start()
    #
    # def __on_new_loop_mode__(self, new_loop_mode):
    #     if new_loop_mode is None:
    #         RPR_GetSetRepeat(0)
    #     elif new_loop_mode == self.LOOP_MODE_SEQUENCE:
    #         RPR_GetSetRepeat(1)
    #     elif new_loop_mode == self.LOOP_MODE_SHUFFLE:
    #         raise NotImplementedError("Shuffle mode not implemented yet")


class ReaperUtils(object):
    def __init__(self):
        raise Exception("Please use the static functions")

    @staticmethod
    def track_iter():
        """
        Returns an iterator over the tracks in the current REAPER project, yielding a track id on each iteration.

        :return: iterator over tracks in current REAPER project
        """

        n_tracks, i = RPR_GetNumTracks(), 0
        while i < n_tracks:
            yield RPR_GetTrack(0, i)
            i += 1

    @staticmethod
    def media_item_iter(track_id):
        """
        Returns an iterator over the media items in the given track, yielding a media item id on each iteration.

        :param track_id: the track whose media items to iterate over
        :return: iterator over media items in given track
        """

        n_media_items, i = RPR_GetTrackNumMediaItems(track_id), 0
        print("Iterating over track %s, which has %i media items" % (track_id, n_media_items))
        while i < n_media_items:
            yield RPR_GetTrackMediaItem(track_id, i)
            i += 1

    @staticmethod
    def clear_track(media_track_id):
        """
        Removes all media items from the given media track.
        """

        while RPR_GetTrackNumMediaItems(media_track_id) > 0:
            media_item = RPR_GetTrackMediaItem(media_track_id, 0)
            RPR_DeleteTrackMediaItem(media_track_id, media_item)

    @staticmethod
    def get_time_selection():
        """
        Returns a tuple with two elements; the start and the end of the current time selection. The times returned by
        this function are in seconds.
        """

        return RPR_GetSet_LoopTimeRange(0, 0, 0, 0, 0)[2:4]

    @staticmethod
    def reset_time_selection():
        """
        Sets both the start and the end of the time selection to the start of the project.
        """

        RPR_SetEditCurPos(0, False, False)
        for i in range(2):
            RPR_MoveEditCursor(0, True)

    @staticmethod
    def set_time_selection(t_start, t_end, move_cursor_to_start=True):
        """
        Sets the time selection to the given start and end position in seconds.
        """

        ReaperUtils.reset_time_selection()
        RPR_SetEditCurPos(t_start, False, False)
        RPR_MoveEditCursor(t_end - t_start, True)
        if move_cursor_to_start:
            RPR_SetEditCurPos(t_start, False, False)

    @staticmethod
    def find_track_by_name(track_name):
        """
        Returns the first track with in the current REAPER project with the given track name. Returns None if no class
        found with given name. Note that this is a linear search (running time of O(N)).

        :param track_name: track name as a string
        :return: the track id of the first track with the given name
        """

        for track_id in ReaperUtils.track_iter():
            current_track_name = RPR_GetTrackName(track_id, 0, 64)[2]
            if current_track_name == track_name:
                return track_id
        raise ValueError("No track named '%s'" % track_name)

    @staticmethod
    def set_selected_tracks(*tracks_to_select):
        """
        Sets the track selection to the given tracks. This will unselect all tracks that are not specified in the
        arguments to this function.
        """

        for track_id in ReaperUtils.track_iter():
            RPR_SetTrackSelected(track_id, track_id in tracks_to_select)

    @staticmethod
    def insert_track(position=0, name=""):
        """
        Inserts a new track to the current REAPER project.

        :param position: position to insert the track to (0 = beginning)
        :param name: track name
        :return: the id of the newly created track
        """

        n_tracks = RPR_GetNumTracks()
        position = max(0, min(n_tracks, position))
        RPR_InsertTrackAtIndex(position, False)
        track_id = RPR_GetTrack(0, position)
        if name:
            RPR_GetSetMediaTrackInfo_String(track_id, "P_NAME", name, 1)
        return track_id


if __name__ == "__main__":
    RPR_ClearConsole()

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = ReaperConsoleOut("[BeatSearch]")
    sys.stderr = ReaperConsoleOut("[BeatSearch Err]")

    try:
        main()
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

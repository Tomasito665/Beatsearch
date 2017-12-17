import os
import socket
import threading

try:
    # noinspection PyUnresolvedReferences
    import beyond.Reaper
except ModuleNotFoundError:
    raise ModuleNotFoundError("beyond.Reaper not found. Download it here: "
                              "https://forum.cockos.com/showthread.php?t=129696")
from tkinter import *
from tempfile import TemporaryFile
# noinspection PyUnresolvedReference
from beatsearch_dirs import BS_ROOT, BS_LIB
sys.path.append(BS_LIB)
from beatsearch.app.control import BSController, BSRhythmPlayer
from beatsearch.app.view import BSApp
# noinspection PyUnresolvedReferences
import midi
# noinspection PyUnresolvedReferences
ReaperApi = Reaper


def main():
    player = ReaperRhythmPlayer()

    def find_reaper_thread_target():
        try:
            with ReaperApi as api:
                print("Connected to Reaper")
                try:
                    bs_input_track = ReaperUtils.find_track_by_name(api, "BS_Output")
                except ValueError:
                    bs_input_track = ReaperUtils.insert_track(api, 0, "BS_Output")
            player.set_output_track(bs_input_track)
        except socket.timeout:
            app.destroy()

    print("Trying to connect to Reaper...")
    threading.Thread(target=find_reaper_thread_target).start()

    print("Initializing controller...")
    controller = BSController(rhythm_player=player)
    controller.set_corpus(os.path.join(BS_ROOT, "data", "rhythms.pkl"))

    print("Initializing view...")
    app = BSApp(controller, baseName="beatsearch-reaper-plugin")
    app.mainloop()


class ReaperRhythmPlayer(BSRhythmPlayer):
    """Implementation of BSRhythmPlayer that uses Reaper as MIDI output for the rhythms"""

    def __init__(self, output_track=None):
        super(ReaperRhythmPlayer, self).__init__()
        self._tmp_files = set()
        self._start_pos = -1
        self._end_pos = -1
        self._prev_pos = -1
        self._output_track = output_track
        self._current_rhythms = []
        self._is_playing = False

        if output_track is not None:
            self.set_output_track(output_track)

    @property
    def output_track(self):
        return self._output_track

    @output_track.setter
    def output_track(self, output_track):
        self.set_output_track(output_track)

    def set_output_track(self, output_track, reaper_api=None):
        if reaper_api is None:
            with ReaperApi as api:
                self.set_output_track(output_track, api)
            return

        reaper_api.OnStopButton()
        reaper_api.GetSetRepeat(1)  # enable repeat
        ReaperUtils.clear_track(reaper_api, self._output_track)
        self._output_track = output_track

    def playback_rhythms(self, rhythms):
        with ReaperApi as api:
            api.OnPauseButton()
            self.populate_track(api, rhythms)
            ReaperUtils.set_time_selection(api, self._start_pos, self._end_pos)
            api.OnPlayButton()
        self._is_playing = True

    def stop_playback(self):
        with ReaperApi as api:
            api.OnStopButton()
        self._is_playing = False

    def is_playing(self):
        return self._is_playing

    def set_repeat(self, enabled):
        print("ReaperRhythmPlayer.set_repeat() not available. Repeat is always enabled.", file=sys.stderr)

    def get_repeat(self):
        return True

    def clean_temp_files(self):
        for fpath in self._tmp_files:
            os.remove(fpath)

    def populate_track(self, api, rhythms):
        if rhythms == self._current_rhythms:
            return

        ReaperUtils.clear_track(api, self._output_track)
        api.SetEditCurPos(1, False, False)
        self._start_pos = api.GetCursorPosition()

        for rhythm in rhythms:
            temp_file = self.render_rhythm_and_insert_to_track(self._output_track, rhythm, api)
            self._tmp_files.add(temp_file)

        self._end_pos = api.GetCursorPosition()
        self._current_rhythms = rhythms

    @staticmethod
    def render_rhythm_and_insert_to_track(output_track, rhythm, reaper_api):
        ReaperUtils.set_selected_tracks(reaper_api, output_track)
        with TemporaryFile(prefix="beatsearch-reaper", suffix=rhythm.name + ".mid", delete=False) as f:
            pattern = rhythm.to_midi()
            midi.write_midifile(f, pattern)
            fpath = f.name
        reaper_api.InsertMedia(fpath, 0)
        return fpath


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
            current_track_name = reaper_api.GetTrackName(track_id, 0, 64)[2]
            if current_track_name == track_name:
                return track_id
        raise ValueError("No track named '%s'" % track_name)

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


if __name__ == "__main__":
    main()

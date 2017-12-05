from collections import OrderedDict, namedtuple
from functools import wraps
from inspect import isclass
import typing as tp
import threading
import numpy as np
from sortedcollections import OrderedSet
from beatsearch.data.rhythm import (
    Rhythm,
    create_rumba_rhythm,
    RhythmDistanceMeasure,
    TrackDistanceMeasure,
    HammingDistanceMeasure
)
from beatsearch.data.rhythmcorpus import RhythmCorpus
from beatsearch.utils import no_callback, type_check_and_instantiate_if_necessary


class BSRhythmPlayer(object):
    def __init__(self):
        self._on_playback_ended_callback = no_callback

    def playback_rhythms(self, rhythms):  # type: ([Rhythm]) -> None
        raise NotImplementedError

    def stop_playback(self):  # type: () -> None
        raise NotImplementedError

    def is_playing(self):  # type: () -> bool
        raise NotImplementedError

    @property
    def on_playback_ended(self):  # type: () -> tp.Callable
        return self._on_playback_ended_callback

    @on_playback_ended.setter
    def on_playback_ended(self, callback):  # type: (tp.Callable) -> None
        self._on_playback_ended_callback = callback


class BSFakeRhythmPlayer(BSRhythmPlayer):

    def __init__(self, playback_duration=2.0, rhythm=create_rumba_rhythm(track=36)):
        super(BSFakeRhythmPlayer, self).__init__()
        self._playback_duration = playback_duration
        self._timer = None
        self._rhythm = rhythm

    def playback_rhythms(self, rhythms):
        @wraps(self.on_playback_ended)
        def on_playback_ended():
            self._timer = None
            self.on_playback_ended()
        self._timer = threading.Timer(self._playback_duration, on_playback_ended)
        self._timer.start()

    def stop_playback(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def is_playing(self):
        return self._timer is not None


class BSController(object):
    _rhythm_info_types = None

    # events
    RHYTHM_SELECTION = "<Rhythm-Selection>"
    RHYTHM_PLAYBACK_START = "<RhythmPlayback-Start>"
    RHYTHM_PLAYBACK_STOP = "<RhythmPlayback-Stop>"
    CORPUS_LOADED = "<Corpus-Loaded>"
    DISTANCES_TO_TARGET_UPDATED = "<TargetDistances-Updated>"
    TARGET_RHYTHM_SET = "<TargetRhythm-Set>"

    # description of data returned by get_rhythm_info
    RHYTHM_INFO_STRUCTURE = OrderedDict((
        ("I", int),
        ("Distance to target", float),
        ("Name", str),
        ("BPM", float),
        ("Timesignature", str),
        ("Track count", int)
    ))

    def __init__(self, distance_measure=HammingDistanceMeasure, rhythm_player=None):
        # type: (tp.Type[TrackDistanceMeasure], tp.Union[BSRhythmPlayer, tp.Type[BSRhythmPlayer], None]) -> None

        self._corpus = None
        self._distances_to_target = np.empty(0)
        self._distances_went_stale = False
        self._rhythm_measure = RhythmDistanceMeasure()
        self._rhythm_selection = OrderedSet()
        self._target_rhythm = None
        self._target_rhythm_prev_update = None
        self._lock = threading.Lock()
        self._rhythm_player = None
        self.set_distance_measure(distance_measure)
        self.set_rhythm_player(rhythm_player)
        self._callbacks = dict((event, OrderedSet()) for event in [
            BSController.RHYTHM_SELECTION,
            BSController.CORPUS_LOADED,
            BSController.DISTANCES_TO_TARGET_UPDATED,
            BSController.RHYTHM_PLAYBACK_START,
            BSController.RHYTHM_PLAYBACK_STOP,
            BSController.TARGET_RHYTHM_SET
        ])

    def load_corpus(self, in_file):
        """
        Loads a rhythm corpus, given either the path or the file handle itself to the rhythms pickle file.

        :param in_file: either file path or file handle of rhythms pickle file
        """

        self._corpus = RhythmCorpus.load(in_file)
        # order is important, this depends on _corpus
        self.reset_distances()
        self._fire_event(BSController.CORPUS_LOADED)

    def unload_corpus(self):
        """
        Unloads the rhythm corpus.
        """

        self._corpus = None
        # order is important, this depends on _corpus being None
        self.reset_distances()

    def has_corpus_loaded(self):
        """
        Returns whether or not a corpus has been loaded.

        :return: True if corpus has been loaded; False otherwise
        """

        return self._corpus is not None

    @property
    def target_rhythm(self):
        return self._target_rhythm

    @target_rhythm.setter
    def target_rhythm(self, target_rhythm):
        if target_rhythm == self._target_rhythm:
            return
        if target_rhythm is None:
            self._target_rhythm = None
            return
        if not isinstance(target_rhythm, Rhythm):
            raise TypeError("Expected a Rhythm but got \"%s\"" % target_rhythm)
        self._target_rhythm = target_rhythm
        self._distances_went_stale = True
        self._fire_event(self.TARGET_RHYTHM_SET)

    def set_rhythm_player(self, rhythm_player):
        # type: (tp.Union[BSRhythmPlayer, tp.Type[BSRhythmPlayer], None]) -> None

        # stop and reset previous player
        if self._rhythm_player is not None:
            self._rhythm_player.stop_playback()
            self._fire_event(self.RHYTHM_PLAYBACK_STOP)
            self._rhythm_player.on_playback_ended = no_callback

        if rhythm_player is None:
            self._rhythm_player = None
            return

        rhythm_player = type_check_and_instantiate_if_necessary(rhythm_player, BSRhythmPlayer, allow_none=True)
        rhythm_player.on_playback_ended = lambda *args: self._fire_event(self.RHYTHM_PLAYBACK_STOP)
        self._rhythm_player = rhythm_player

    def set_distance_measure(self, track_distance_measure):
        if isinstance(track_distance_measure, str):
            try:
                track_distance_measure = TrackDistanceMeasure.get_measure_by_name(track_distance_measure)
            except KeyError:
                raise ValueError("Unknown distance measure: '%s'" % str(track_distance_measure))
        self._rhythm_measure.track_distance_measure = track_distance_measure
        self._distances_went_stale = True

    def set_tracks_to_compare(self, tracks):
        self._rhythm_measure.tracks = tracks
        self._distances_went_stale = True

    def get_rhythm_count(self):
        try:
            return len(self._corpus)
        except TypeError:
            return 0

    def reset_distances(self):  # TODO should this method be public?
        """
        Resets the internal array containing the distances to the target rhythm. This will fill the array with np.inf.
        If the corpus length has changed, this will also re-allocate the array.
        """

        n_rhythms = self.get_rhythm_count()
        with self._lock:
            if len(self._distances_to_target) == n_rhythms:
                self._distances_to_target.fill(np.inf)
            else:
                self._distances_to_target = np.full(n_rhythms, np.inf)

    def update_distances(self):
        self._precondition_check_corpus_loaded()
        self._precondition_check_target_rhythm_set()

        measure = self._rhythm_measure
        target_rhythm = self._target_rhythm

        if target_rhythm == self._target_rhythm_prev_update:
            self._distances_went_stale = True

        # nothing to update
        if not self._distances_went_stale:
            return

        if target_rhythm is None:
            self.reset_distances()  # set distance to NaN
            self._target_rhythm_prev_update = None
            return

        with self._lock:
            for i, scanned_rhythm in enumerate(self._corpus):
                distance = measure.get_distance(target_rhythm, scanned_rhythm)
                self._distances_to_target[i] = distance

        self._distances_went_stale = False
        self._target_rhythm_prev_update = target_rhythm
        self._fire_event(self.DISTANCES_TO_TARGET_UPDATED)

    def get_rhythm_info(self):
        """
        Returns an iterator yielding rhythm info on each iteration. The info is returned as a tuple containing the
        rhythm index, the distance to the target rhythm, its name, its BPM, its time signature and its track count. To
        get an array with the names of the attributes use BSController.get_rhythm_info_types_by_names.

        Warning: This method locks this class. The lock will be release only when iteration is done.

        :return: iteration yielding rhythm info
        """

        with self._lock:
            for i, rhythm in enumerate(self._corpus):
                d_to_target = self._distances_to_target[i]
                yield (
                    i,
                    "NaN" if d_to_target == np.inf else d_to_target,
                    rhythm.name,
                    rhythm.bpm,
                    str(rhythm.time_signature),
                    rhythm.track_count()
                )

    def get_corpus_name(self):
        return self._corpus.name

    def get_rhythm_by_index(self, rhythm_ix):
        self._precondition_check_rhythm_index(rhythm_ix)
        return self._corpus[rhythm_ix]

    @staticmethod
    def get_rhythm_info_names():
        structure = BSController.RHYTHM_INFO_STRUCTURE
        return structure.keys()

    @staticmethod
    def get_rhythm_info_types():
        structure = BSController.RHYTHM_INFO_STRUCTURE
        return structure.values()

    def rhythm_selection_set(self, selected_rhythms):
        self._rhythm_selection.clear()
        for rhythm_ix in selected_rhythms:
            self._precondition_check_rhythm_index(rhythm_ix)
            self._rhythm_selection.add(rhythm_ix)
        self._fire_event(BSController.RHYTHM_SELECTION)

    def rhythm_player_set(self):
        return self._rhythm_player is not None

    def rhythm_selection_clear(self):
        self.rhythm_selection_set([])

    def get_rhythm_selection(self):
        return tuple(self._rhythm_selection)

    def are_rhythms_playing_back(self):
        return self._rhythm_player.is_playing()

    def is_target_rhythm_set(self):
        return self._target_rhythm is not None

    def playback_selected_rhythms(self):
        self._precondition_check_corpus_loaded()
        self._precondition_check_rhythm_player_set()
        selected_rhythm_indices = self.get_rhythm_selection()
        rhythms = [self.get_rhythm_by_index(i) for i in selected_rhythm_indices]
        self._rhythm_player.playback_rhythms(rhythms)
        self._fire_event(BSController.RHYTHM_PLAYBACK_START)

    def stop_rhythm_playback(self):
        self._precondition_check_rhythm_player_set()
        self._rhythm_player.stop_playback()
        self._fire_event(BSController.RHYTHM_PLAYBACK_STOP)

    def bind(self, event_name, callback):
        if not callable(callback):
            raise TypeError("Expected a callable")
        self._precondition_check_event_name(event_name)
        event_callbacks = self._callbacks[event_name]
        if callback in event_callbacks:
            event_callbacks.remove(callback)
        self._callbacks[event_name].add(callback)

    def unbind(self, event_name, callback):
        self._precondition_check_event_name(event_name)
        event_callbacks = self._callbacks[event_name]
        if callback not in event_callbacks:
            return False
        self._callbacks[event_name].remove(callback)
        return True

    def _fire_event(self, event_name, *args, **kwargs):
        self._precondition_check_event_name(event_name)
        for clb in self._callbacks[event_name]:
            clb(*args, **kwargs)

    def _on_rhythm_playback_ended(self):
        self._fire_event(BSController.RHYTHM_PLAYBACK_STOP)

    def _precondition_check_rhythm_index(self, rhythm_ix):
        n_rhythms = self.get_rhythm_count()
        if not (0 <= rhythm_ix < n_rhythms):
            raise IndexError("Expected rhythm index in range [0, %i]" % (n_rhythms - 1))

    def _precondition_check_rhythm_player_set(self):
        if not self.rhythm_player_set():
            raise Exception("Rhythm player not set")

    def _precondition_check_event_name(self, event_name):
        if event_name not in self._callbacks:
            raise ValueError("Unknown event \"%s\"" % event_name)

    def _precondition_check_corpus_loaded(self):
        if not self.has_corpus_loaded():
            raise Exception("Corpus not loaded")

    def _precondition_check_target_rhythm_set(self):
        if not self.is_target_rhythm_set():
            raise Exception("Target rhythm not set")

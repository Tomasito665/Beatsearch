from collections import OrderedDict
from functools import wraps
import typing as tp
import threading
from sortedcollections import OrderedSet
from beatsearch.rhythm import (
    RhythmLoop,
    PolyphonicRhythmImpl,
    create_rumba_rhythm
)
from beatsearch.metrics import (
    MonophonicRhythmDistanceMeasure,
    HammingDistanceMeasure,
    SummedMonophonicRhythmDistance,
    Quantizable
)
from beatsearch.config import __USE_NUMPY__
from beatsearch.rhythmcorpus import RhythmCorpus
from beatsearch.utils import no_callback, type_check_and_instantiate_if_necessary
if __USE_NUMPY__:
    import numpy


class BSRhythmPlayer(object):
    def __init__(self):
        self._on_playback_ended_callback = no_callback

    def playback_rhythms(self, rhythms: tp.Iterable[PolyphonicRhythmImpl]) -> None:
        raise NotImplementedError

    def stop_playback(self):  # type: () -> None
        raise NotImplementedError

    def is_playing(self) -> bool:
        raise NotImplementedError

    def set_repeat(self, enabled: bool) -> None:
        raise NotImplementedError

    def get_repeat(self) -> bool:
        raise NotImplementedError

    @property
    def on_playback_ended(self) -> tp.Union[tp.Callable, None]:
        return self._on_playback_ended_callback

    @on_playback_ended.setter
    def on_playback_ended(self, callback: tp.Union[tp.Callable, None]) -> None:
        self._on_playback_ended_callback = callback


class BSFakeRhythmPlayer(BSRhythmPlayer):

    def __init__(self, playback_duration: float = 2.0, rhythm: PolyphonicRhythmImpl = create_rumba_rhythm()):
        super(BSFakeRhythmPlayer, self).__init__()
        self._playback_duration = playback_duration  # type: float
        self._timer = None                           # type: tp.Union[threading.Timer, None]
        self._rhythm = rhythm                        # type: PolyphonicRhythmImpl
        self._repeat = False                         # type: bool

    def playback_rhythms(self, rhythms: tp.Iterable[PolyphonicRhythmImpl]) -> None:
        @wraps(self.on_playback_ended)
        def on_playback_ended():
            if self.get_repeat():
                self.playback_rhythms(rhythms)
            else:
                self._timer = None
                self.on_playback_ended()
        self._timer = threading.Timer(self._playback_duration, on_playback_ended)
        self._timer.start()

    def stop_playback(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def is_playing(self) -> bool:
        return self._timer is not None

    def set_repeat(self, enabled: bool) -> None:
        self._repeat = enabled

    def get_repeat(self) -> bool:
        return self._repeat


class BSRhythmLoopLoader(object):
    def __init__(self):
        self._on_loading_error = no_callback

    class LoadingError(Exception):
        pass

    def load(self, **kwargs):  # type: (tp.Any) -> tp.Union[None, RhythmLoop]
        if not self.is_available():
            return None
        try:
            return self.__load__(**kwargs)
        except self.LoadingError as e:
            if self.on_loading_error is no_callback:
                raise e
            self.on_loading_error(e)
            return None

    def is_available(self):  # type: () -> bool
        raise NotImplementedError

    def __load__(self, **kwargs):  # type: (tp.Any) -> RhythmLoop
        raise NotImplementedError

    @classmethod
    def get_source_name(cls):
        raise NotImplementedError

    @property
    def on_loading_error(self):
        return self._on_loading_error

    @on_loading_error.setter
    def on_loading_error(self, callback):
        if not callable(callback):
            raise TypeError("Expected callable but got \"%s\"" % callback)
        self._on_loading_error = callback


class BSSelectedRhythmLoopLoader(BSRhythmLoopLoader):
    SOURCE_NAME = "selected rhythm"

    def __init__(self, controller):  # type: (BSController) -> None
        super().__init__()
        self.controller = controller

    @classmethod
    def get_source_name(cls):
        return cls.SOURCE_NAME

    def is_available(self):
        controller = self.controller
        rhythm_selection = controller.get_rhythm_selection()
        return len(rhythm_selection) >= 1

    def __load__(self, **kwargs):
        controller = self.controller
        rhythm_selection = controller.get_rhythm_selection()
        rhythm_ix = rhythm_selection[0]
        rhythm = controller.get_rhythm_by_index(rhythm_ix)
        return rhythm


class BSController(object):

    # actions
    RHYTHM_SELECTION = "<Rhythm-Selection>"
    RHYTHM_PLAYBACK_START = "<RhythmPlayback-Start>"
    RHYTHM_PLAYBACK_STOP = "<RhythmPlayback-Stop>"
    CORPUS_LOADED = "<Corpus-Loaded>"
    DISTANCES_TO_TARGET_UPDATED = "<TargetDistances-Updated>"
    TARGET_RHYTHM_SET = "<TargetRhythm-Set>"
    DISTANCE_MEASURE_SET = "<DistanceMeasure-Set>"
    RHYTHM_LOADER_REGISTERED = "<RhythmLoader-Registered>"

    # description of data returned by get_rhythm_data
    RHYTHM_DATA_TYPES = OrderedDict((
        ("I", int),
        ("Distance to target", float),
        ("Name", str),
        ("BPM", float),
        ("Timesignature", str),
        ("Track count", int),
        ("Measure count", int)
    ))

    @staticmethod
    def get_rhythm_data_attr_names():
        structure = BSController.RHYTHM_DATA_TYPES
        return tuple(structure.keys())  # TODO Avoid re-allocation

    @staticmethod
    def get_rhythm_data_attr_types():
        structure = BSController.RHYTHM_DATA_TYPES
        return tuple(structure.values())  # TODO Avoid re-allocation

    def __init__(
            self, distance_measure: MonophonicRhythmDistanceMeasure = HammingDistanceMeasure,
            rhythm_player: tp.Union[BSRhythmPlayer, tp.Type[BSRhythmPlayer], None] = None
    ):
        self._corpus = None
        if __USE_NUMPY__:
            self._distances_to_target = numpy.empty(0)
        else:
            self._distances_to_target = []
        self._distances_to_target_rhythm_are_stale = False
        self._rhythm_measure = SummedMonophonicRhythmDistance()  # type: SummedMonophonicRhythmDistance
        self._rhythm_selection = OrderedSet()
        self._target_rhythm = None
        self._target_rhythm_prev_update = None
        self._lock = threading.Lock()
        self._rhythm_player = None
        self._callbacks = dict((action, OrderedSet()) for action in [
            BSController.RHYTHM_SELECTION,
            BSController.CORPUS_LOADED,
            BSController.DISTANCES_TO_TARGET_UPDATED,
            BSController.RHYTHM_PLAYBACK_START,
            BSController.RHYTHM_PLAYBACK_STOP,
            BSController.TARGET_RHYTHM_SET,
            BSController.DISTANCE_MEASURE_SET,
            BSController.RHYTHM_LOADER_REGISTERED
        ])
        self._rhythm_loaders = OrderedDict()  # rhythm loaders by loader source name
        self.set_rhythm_player(rhythm_player)
        self.set_distance_measure(distance_measure)
        # automatically register a loader for the currently selected rhythm
        self.register_rhythm_loader(BSSelectedRhythmLoopLoader(self))

    def set_corpus(self, corpus):
        """
        Sets the rhythm corpus. If the given corpus is a filename or a file handle, this method will try to load a new
        corpus and set that.

        :param corpus: a rhythm corpus or a file(name) of a rhythm corpus file or None
        """

        if not isinstance(corpus, RhythmCorpus) and corpus is not None:
            corpus = RhythmCorpus.load(corpus)

        self._corpus = corpus

        with self._lock:
            self._reset_distances_to_target_rhythm()

        self.clear_rhythm_selection()
        self._dispatch(self.CORPUS_LOADED)

    def is_corpus_set(self):
        """
        Returns whether or not a corpus has been set.

        :return: True if a corpus has been set; False otherwise
        """

        return self._corpus is not None

    def get_corpus_name(self):
        """
        Returns the name of the current corpus or an empty string if no corpus is set.

        :return: name of current corpus or an empty string
        """

        try:
            return self._corpus.name
        except AttributeError:
            return ""

    def get_corpus_fname(self):
        """
        Returns the filename (including its extension) of the current corpus or an empty string if no corpus is set.

        :return: filename of the current corpus or an empty string
        """

        try:
            return self._corpus.fname
        except AttributeError:
            return ""

    def get_rhythm_count(self):
        """
        Returns the number of rhythms in the current rhythm corpus.

        :return: number of rhythms in current corpus or 0 if no corpus set
        """

        try:
            return len(self._corpus)
        except TypeError:
            return 0

    def get_rhythm_by_index(self, rhythm_ix):
        """
        Returns a rhythm by its index in the current corpus.

        :param rhythm_ix: rhythm index in current corpus
        :return: the Rhythm object on the given index in the corpus
        """

        self._precondition_check_corpus_set()
        self._precondition_check_rhythm_index(rhythm_ix)
        return self._corpus[rhythm_ix]

    def set_target_rhythm(self, target_rhythm):  # type: (tp.Union[RhythmLoop, None]) -> None
        """
        Sets the target rhythm.

        :param target_rhythm: target rhythm or None
        """

        if target_rhythm == self._target_rhythm or target_rhythm is None:
            self._target_rhythm = target_rhythm
            return

        if not isinstance(target_rhythm, RhythmLoop):
            raise TypeError("Expected a RhythmLoop but got \"%s\"" % str(target_rhythm))

        self._target_rhythm = target_rhythm
        self._distances_to_target_rhythm_are_stale = True
        self._dispatch(self.TARGET_RHYTHM_SET)

    def get_target_rhythm(self):
        """
        Returns the current target rhythm or None if there is no target rhythm set.

        :return: target rhythm
        """

        return self._target_rhythm

    def is_target_rhythm_set(self):
        """
        Returns whether or not a target rhythm has been set.

        :return: True if target rhythm has been set; False otherwise
        """

        return self._target_rhythm is not None

    def get_rhythm_data(self):
        """
        Returns an iterator yielding rhythm information on each iteration, such as the distance to the target rhythm,
        the name of the rhythm or the tempo. This data is yielded as a tuple. The type and names of the attributes in
        these tuples can be found in the ordered dictionary BSController.RHYTHM_DATA_TYPES, which gives name and type
        information for the elements in the tuple.

        Warning: This method locks this class. The lock will be released only when iteration is done or when the
        generator some time when the generator is destroyed.

        :return: iteration yielding rhythm info
        """

        with self._lock:
            for i, rhythm in enumerate(self._corpus):
                d_to_target = self._distances_to_target[i]
                yield (
                    i,
                    "NaN" if d_to_target == float("inf") else d_to_target,
                    rhythm.name,
                    rhythm.bpm,
                    str(rhythm.time_signature),
                    rhythm.get_track_count(),
                    int(rhythm.get_duration_in_measures())
                )

    def set_rhythm_player(self, rhythm_player):  # type: (tp.Union[BSRhythmPlayer, None]) -> None
        """
        Sets the rhythm player.

        :param rhythm_player: rhythm player or None
        :return:
        """

        if self.is_rhythm_player_set():
            self.stop_rhythm_playback()
            self._rhythm_player.on_playback_ended = no_callback

        if rhythm_player is None:
            self._rhythm_player = None
            return

        rhythm_player = type_check_and_instantiate_if_necessary(rhythm_player, BSRhythmPlayer, allow_none=True)
        rhythm_player.on_playback_ended = lambda *args: self._dispatch(self.RHYTHM_PLAYBACK_STOP)
        self._rhythm_player = rhythm_player

    def is_rhythm_player_set(self):  # type: () -> bool
        """
        Returns whether or not a rhythm player has been set.

        :return: True if a rhythm player has been set; False otherwise.
        """

        return self._rhythm_player is not None

    def is_rhythm_player_playing(self):
        """
        Returns whether or not the rhythm player is currently playing. Returns False if no rhythm player has been set.

        :return: True if rhythm player is playing; False otherwise
        """

        try:
            return self._rhythm_player.is_playing()
        except AttributeError:
            return False

    def playback_selected_rhythms(self):
        """
        Starts playing the currently selected rhythms. The rhythm player will be used for rhythm playback.

        :return: None
        """

        self._precondition_check_corpus_set()
        self._precondition_check_rhythm_player_set()
        selected_rhythm_indices = self.get_rhythm_selection()
        rhythms = [self.get_rhythm_by_index(i) for i in selected_rhythm_indices]

        if rhythms:
            self._rhythm_player.playback_rhythms(rhythms)
            self._dispatch(BSController.RHYTHM_PLAYBACK_START)

    def stop_rhythm_playback(self):
        """
        Stops rhythm playback.

        :return: None
        """

        self._precondition_check_rhythm_player_set()
        self._rhythm_player.stop_playback()
        self._dispatch(BSController.RHYTHM_PLAYBACK_STOP)

    def set_distance_measure(self, track_distance_measure):
        """
        Sets the measure use to get the distance between individual rhythm tracks.

        :param track_distance_measure: the distance measure, one of:
            MonophonicRhythmDistanceMeasure - the track distance measure itself
            Type[MonophonicRhythmDistanceMeasure] - the track distance class, must be a subclass of
                                                    MonophonicRhythmDistanceMeasure
            str - the name of the MonophonicRhythmDistanceMeasure class (cls.__friendly_name__)
        """
        # type: (tp.Union[str, MonophonicRhythmDistanceMeasure, tp.Type[MonophonicRhythmDistanceMeasure]]) -> None

        if isinstance(track_distance_measure, str):
            try:
                track_distance_measure = MonophonicRhythmDistanceMeasure.get_measure_by_name(track_distance_measure)
            except KeyError:
                raise ValueError("Unknown distance measure: \"%s\"" % str(track_distance_measure))

        self._rhythm_measure.monophonic_measure = track_distance_measure
        self._distances_to_target_rhythm_are_stale = True
        self._dispatch(self.DISTANCE_MEASURE_SET)

    def is_current_distance_measure_quantizable(self):
        return isinstance(self._rhythm_measure.monophonic_measure, Quantizable)

    def set_measure_quantization_unit(self, unit):
        """
        Sets the quantization for the distance measures.

        :param unit: quantization unit
        :return: None
        """

        if not self.is_current_distance_measure_quantizable():
            return False

        track_distance_measure = self._rhythm_measure.monophonic_measure
        track_distance_measure.internal_unit = unit
        track_distance_measure.quantize_enabled = True
        self._distances_to_target_rhythm_are_stale = True
        return True

    def set_tracks_to_compare(self, tracks):
        """
        Sets the tracks to compare.

        :param tracks: tracks to compare as a list or one of the wildcards ['*', 'a*', 'b*']. See
                       rhythm_pair_track_iterator for further info.
        """

        self._rhythm_measure.tracks = tracks
        self._distances_to_target_rhythm_are_stale = True

    def calculate_distances_to_target_rhythm(self):
        """
        Calculates and updates the distances between the rhythms in the corpus and the current target rhythm. This
        function call will be ignored if the distances are already up to date. The next call to get_rhythm_data will
        yield the updated distances.
        """

        self._precondition_check_corpus_set()
        self._precondition_check_target_rhythm_set()

        measure = self._rhythm_measure
        target_rhythm = self._target_rhythm

        if target_rhythm == self._target_rhythm_prev_update:
            self._distances_to_target_rhythm_are_stale = True

        # nothing to update
        if not self._distances_to_target_rhythm_are_stale:
            return

        if target_rhythm is None:
            self._reset_distances_to_target_rhythm()  # set distance to NaN
            self._target_rhythm_prev_update = None
            return

        with self._lock:
            for i, scanned_rhythm in enumerate(self._corpus):
                distance = measure.get_distance(target_rhythm, scanned_rhythm)
                self._distances_to_target[i] = distance

        self._distances_to_target_rhythm_are_stale = False
        self._target_rhythm_prev_update = target_rhythm
        self._dispatch(self.DISTANCES_TO_TARGET_UPDATED)

    def set_rhythm_selection(self, selected_rhythms):
        """
        Sets the rhythm selection.

        :param selected_rhythms: the indices in the current corpus of the rhythms to select
        """

        if self.is_rhythm_player_set() and self.is_rhythm_player_playing():
            self.stop_rhythm_playback()

        self._rhythm_selection.clear()
        for rhythm_ix in selected_rhythms:
            self._precondition_check_rhythm_index(rhythm_ix)
            self._rhythm_selection.add(rhythm_ix)
        self._dispatch(BSController.RHYTHM_SELECTION)

    def clear_rhythm_selection(self):
        """
        Clears the rhythm selection.
        """

        self.set_rhythm_selection([])

    def get_rhythm_selection(self):
        """
        Returns the corpus indices of the currently selected rhythms.

        :return: tuple containing the corpus indices of the currently selected rhythms
        """

        return tuple(self._rhythm_selection)

    def register_rhythm_loader(self, loader: BSRhythmLoopLoader):
        """
        Register a new rhythm loader.

        :param loader: rhythm loader
        :return: None
        """

        loader_class = loader.__class__
        if loader_class in self._rhythm_loaders:
            raise ValueError("Already registered a rhythm loader for loader type: \"%s\"" % loader_class)
        self._rhythm_loaders[loader.__class__] = loader
        self._dispatch(self.RHYTHM_LOADER_REGISTERED, loader)

    def get_rhythm_loader_source_names(self):
        """
        Returns a tuple with the rhythm source names of the currently registered rhythm loaders. See the
        BSRhythmLoopLoader.source_name property.

        :return: tuple containing the source names of the currently registered rhythm loaders
        """

        return tuple(loader.get_source_name() for loader in self._rhythm_loaders.values())

    def get_rhythm_loader_count(self):
        """
        Returns the number of currently registered rhythm loaders.

        :return: number of currently registered rhythm loaders
        """

        return len(self._rhythm_loaders)

    def get_rhythm_loader_iterator(self) -> (tp.Iterator[tp.Tuple[tp.Type[BSRhythmLoopLoader], BSRhythmLoopLoader]]):
        """
        Returns an iterator over the currently registered rhythm loaders.

        :return: iterator over the currently registered rhythm loaders
        """

        return iter(self._rhythm_loaders.items())

    def get_rhythm_loader(self, loader_type: tp.Type[BSRhythmLoopLoader]):  # type: (str) -> BSRhythmLoopLoader
        """
        Returns the rhythm loader, given the loader type.

        :param loader_type: source name
        :return: loader
        """

        try:
            loader = self._rhythm_loaders[loader_type]
        except KeyError:
            raise ValueError("No rhythm loader registered of type \"%s\"" % loader_type)

        return loader

    def bind(self, action, callback):
        """
        Adds a callback to the given action. Callback order is preserved. When the given callback is already bound to
        the given action, the callback will be moved to the end of the callback chain.

        :param action: action
        :param callback: callable that will be called when the given action occurs
        """

        if not callable(callback):
            raise TypeError("Expected a callable")
        self._precondition_check_action(action)
        action_callbacks = self._callbacks[action]
        if callback in action_callbacks:
            action_callbacks.remove(callback)
        self._callbacks[action].add(callback)

    def unbind(self, action, callback):
        """
        Removes a callback from the given action.

        :param action: action
        :param callback: callable
        :return: True if the action was removed successfully or False if the callback was never added to
                 the given action
        """

        self._precondition_check_action(action)
        action_callbacks = self._callbacks[action]
        if callback not in action_callbacks:
            return False
        self._callbacks[action].remove(callback)
        return True

    def _reset_distances_to_target_rhythm(self):  # Note: the caller should acquire the lock
        n_rhythms = self.get_rhythm_count()
        if len(self._distances_to_target) == n_rhythms:
            if __USE_NUMPY__:
                # noinspection PyUnresolvedReferences
                self._distances_to_target.fill(numpy.inf)
            else:
                for i in range(n_rhythms):
                    self._distances_to_target[i] = 0
        else:
            if __USE_NUMPY__:
                self._distances_to_target = numpy.full(n_rhythms, numpy.inf)
            else:
                self._distances_to_target = [float("inf")] * n_rhythms

    def _dispatch(self, action, *args, **kwargs):
        self._precondition_check_action(action)
        for clb in self._callbacks[action]:
            clb(*args, **kwargs)

    def _on_rhythm_playback_ended(self):
        self._dispatch(BSController.RHYTHM_PLAYBACK_STOP)

    def _precondition_check_rhythm_index(self, rhythm_ix):
        n_rhythms = self.get_rhythm_count()
        if not (0 <= rhythm_ix < n_rhythms):
            raise IndexError("Expected rhythm index in range [0, %i]" % (n_rhythms - 1))

    def _precondition_check_rhythm_player_set(self):
        if not self.is_rhythm_player_set():
            raise Exception("Rhythm player not set")

    def _precondition_check_action(self, action):
        if action not in self._callbacks:
            raise ValueError("Unknown action \"%s\"" % action)

    def _precondition_check_corpus_set(self):
        if not self.is_corpus_set():
            raise Exception("Corpus not loaded")

    def _precondition_check_target_rhythm_set(self):
        if not self.is_target_rhythm_set():
            raise Exception("Target rhythm not set")

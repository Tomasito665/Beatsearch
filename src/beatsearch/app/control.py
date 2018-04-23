import os
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import wraps
import typing as tp
import numpy as np
import threading
from sortedcollections import OrderedSet
from beatsearch.rhythm import (
    RhythmLoop,
    MidiRhythm,
    MidiRhythmCorpus,
    PolyphonicRhythm,
    create_rumba_rhythm,
    MidiDrumMappingReducer
)
from beatsearch.metrics import (
    MonophonicRhythmDistanceMeasure,
    HammingDistanceMeasure,
    SummedMonophonicRhythmDistance
)
from beatsearch.app.config import BSConfig
from beatsearch.utils import no_callback, type_check_and_instantiate_if_necessary, Quantizable


def get_rhythm_corpus(config: BSConfig) -> MidiRhythmCorpus:
    """Utility function to load the MIDI rhythm corpus, given a BeatSearch config object

    This function will try to load the MIDI rhythm corpus from cache. If the cache file is missing, out of date or
    corrupt, it will re-load the corpus from the MIDI root directory.

    :param config: config object
    :return: midi rhythm corpus object
    """

    midi_dir = config.midi_root_directory.get()
    cache_fpath = config.get_midi_root_directory_cache_fpath(midi_dir)
    resolution = config.rhythm_resolution.get()
    mapping_reducer = config.mapping_reducer.get()

    # try to load from cache
    if cache_fpath:
        cached_corpus = MidiRhythmCorpus()

        try:
            cached_corpus.load_from_cache_file(cache_fpath)
        except MidiRhythmCorpus.BadCacheFormatError:
            pass

        if cached_corpus.has_loaded() and \
                cached_corpus.is_up_to_date(midi_dir) and \
                cached_corpus.rhythm_resolution == resolution:

            cached_corpus.midi_mapping_reducer = mapping_reducer
            return cached_corpus

        # forget cache file if it is out of date or corrupt
        config.forget_midi_root_directory_cache_file(midi_dir, remove_cache_file=True)

    # if loading from cache didn't succeed, load from midi directory
    corpus = MidiRhythmCorpus(
        path=midi_dir,
        rhythm_resolution=resolution,
        midi_mapping_reducer=mapping_reducer
    )

    # create new cache file
    with config.add_midi_root_directory_cache_file(midi_dir) as cache_file:
        corpus.save_to_cache_file(cache_file)

    return corpus


class BSRhythmPlayer(object):
    def __init__(self):
        self._on_playback_ended_callback = no_callback

    def playback_rhythms(self, rhythms: tp.Iterable[PolyphonicRhythm]) -> None:
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

    def __init__(self, playback_duration: float = 2.0, rhythm: PolyphonicRhythm = create_rumba_rhythm()):
        super(BSFakeRhythmPlayer, self).__init__()
        self._playback_duration = playback_duration  # type: float
        self._timer = None                           # type: tp.Union[threading.Timer, None]
        self._rhythm = rhythm                        # type: PolyphonicRhythm
        self._repeat = False                         # type: bool

    def playback_rhythms(self, rhythms: tp.Iterable[PolyphonicRhythm]) -> None:
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


class BSMidiRhythmLoader(object, metaclass=ABCMeta):
    """MIDI rhythm loader base class"""

    def __init__(self):
        self._on_loading_error = no_callback  # type: tp.Callable[[BSMidiRhythmLoader.LoadingError], tp.Any]
        self.midi_mapping_reducer = None      # type: tp.Union[tp.Type[MidiDrumMappingReducer]], None]
        self.rhythm_resolution = 0            # type: int

    class LoadingError(Exception):
        """Exception raised in  __load__"""
        pass

    def load(self) -> tp.Union[MidiRhythm, None]:
        """Loads and returns the midi rhythm

        :return: MidiRhythm object or None
        :raises LoadingError: if no on_loading_error callback is set
        """

        if not self.is_available():
            return None

        resolution = self.rhythm_resolution
        mapping_reducer = self.midi_mapping_reducer

        try:
            rhythm = self.__load__(resolution, mapping_reducer)
        except self.LoadingError as e:
            if self.on_loading_error is no_callback:
                raise e
            self.on_loading_error(e)
            return None

        assert rhythm.resolution == resolution, "__load__ ignored the requested resolution of %i" % resolution
        assert rhythm.midi_mapping_reducer is mapping_reducer, \
            "__load__ ignored the requested mapping reducer \"%s\"" % str(mapping_reducer)

        return rhythm

    @abstractmethod
    def is_available(self) -> bool:
        """Returns whether this loader is available

        :return: True if this loader is available and ready to use; False otherwise
        """

        raise NotImplementedError

    @abstractmethod
    def __load__(self, rhythm_resolution: int,
                 mapping_reducer: tp.Optional[tp.Type[MidiDrumMappingReducer]]) -> MidiRhythm:
        """Loads and returns a midi rhythm

        :param rhythm_resolution: resolution of the rhythm to load
        :param mapping_reducer: MIDI drum mapping reducer of the rhythm to load
        :return: rhythm with given resolution and mapping reducer
        :raises LoadingError: if something went wrong while loading
        """

        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_source_name(cls) -> str:
        """Returns the name of this source, which will be used in the view as a label"""
        raise NotImplementedError

    @property
    def on_loading_error(self) -> tp.Union[tp.Callable[[LoadingError], tp.Any]]:
        """Callback receiving a LoadingError parameter

        This callback will be called from load() if a LoadingError was raised in that method. If this property is not
        set, the LoadingError will be raised (otherwise this method will be called and the exception won't be raisen).
        """

        return self._on_loading_error

    @on_loading_error.setter
    def on_loading_error(self, callback: tp.Union[tp.Callable[[LoadingError], tp.Any]]):
        if not callable(callback):
            raise TypeError("Expected callable but got \"%s\"" % callback)
        self._on_loading_error = callback

    def bind_to_config(self, config: BSConfig) -> None:
        config.mapping_reducer.add_binding(lambda reducer: setattr(self, "midi_mapping_reducer", reducer))
        config.rhythm_resolution.add_binding(lambda res: setattr(self, "rhythm_resolution", res))


class BSSelectedMidiRhythmLoader(BSMidiRhythmLoader):
    SOURCE_NAME = "selected rhythm"

    def __init__(self, controller):  # type: (BSController) -> None
        super().__init__()
        self.controller = controller

    @classmethod
    def get_source_name(cls) -> str:
        return cls.SOURCE_NAME

    def is_available(self) -> bool:
        controller = self.controller
        rhythm_selection = controller.get_rhythm_selection()
        return len(rhythm_selection) >= 1

    def __load__(self, rhythm_resolution: int,
                 mapping_reducer: tp.Optional[tp.Type[MidiDrumMappingReducer]]) -> MidiRhythm:
        controller = self.controller
        rhythm_selection = controller.get_rhythm_selection()
        rhythm_ix = rhythm_selection[0]
        rhythm = controller.get_rhythm_by_index(rhythm_ix)
        assert rhythm.resolution == rhythm_resolution
        assert rhythm.midi_mapping_reducer is mapping_reducer
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
        self._config = BSConfig()
        self._corpus = None  # type: MidiRhythmCorpus
        self._corpus_resolution = -1
        self._distances_to_target = np.empty(0)
        self._distances_to_target_rhythm_are_stale = False
        self._rhythm_measure = SummedMonophonicRhythmDistance()  # type: SummedMonophonicRhythmDistance
        self._rhythm_selection = OrderedSet()
        self._target_rhythm = None                               # type: tp.Union[MidiRhythm, None]
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
        self.register_rhythm_loader(BSSelectedMidiRhythmLoader(self))
        # setup config change handlers
        self._setup_config()
        # if a midi root directory is set, load the corpus
        if self.get_config().midi_root_directory.get():
            self.load_corpus()

    def load_corpus(self):
        """Loads the rhythm corpus based on the settings in the current configuration

        Loads the rhythm corpus based on the current state of the BSConfig object. This method may be called multiple
        times throughout the lifespan of the app, e.g. after changing the rhythm resolution or the midi root directory
        in the BSConfig object.

        :return: None
        """

        prev_corpus = self._corpus
        prev_corpus_id = prev_corpus.id if prev_corpus else None
        corpus = get_rhythm_corpus(self._config)

        with self._lock:
            self._corpus = corpus
            if corpus.id != prev_corpus_id:
                self._reset_distances_to_target_rhythm()

        if corpus.id != prev_corpus_id:
            self.clear_rhythm_selection()

        self._dispatch(self.CORPUS_LOADED)

    def is_corpus_loaded(self):
        """Returns whether the rhythm corpus has loaded

        :return: True if the rhythm corpus has loaded; False otherwise
        """

        return self._corpus is not None

    def get_corpus_id(self):
        """Returns the id of the current rhythm corpus or None if no corpus

        :return: id of current rhythm corpus
        """

        return self._corpus.id if self.is_corpus_loaded() else None

    def get_corpus_rootdir(self):
        """Returns the root directory of the rhythm corpus

        Returns the path to the root directory of the current rhythm corpus. Returns an empty string if no corpus has
        been set.

        :return: root directory of current rhythm corpus or empty string
        """

        return self._config.midi_root_directory.get()

    def get_corpus_rootdir_name(self):
        """Returns the name of the current rhythm corpus' root directory

        Returns the directory name of the current rhythm corpus' root directory. Returns an empty string if no corpus
        has been set.

        :return: root directory name of current rhythm corpus or empty string
        """

        rootdir = self.get_corpus_rootdir()

        try:
            return os.path.split(rootdir)[-1]
        except IndexError:
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

        if not self._corpus:
            return

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

    def get_config(self) -> BSConfig:
        return self._config

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
        track_distance_measure.unit = unit
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

    def register_rhythm_loader(self, loader: BSMidiRhythmLoader) -> None:
        """
        Register a new rhythm loader.

        :param loader: rhythm loader
        :return: None
        """

        loader_class = loader.__class__
        if loader_class in self._rhythm_loaders:
            raise ValueError("Already registered a rhythm loader for loader type: \"%s\"" % loader_class)

        config = self._config
        loader.bind_to_config(config)

        self._rhythm_loaders[loader.__class__] = loader
        self._dispatch(self.RHYTHM_LOADER_REGISTERED, loader)

    def get_rhythm_loader_source_names(self):
        """
        Returns a tuple with the rhythm source names of the currently registered rhythm loaders. See the
        BSMidiRhythmLoader.source_name property.

        :return: tuple containing the source names of the currently registered rhythm loaders
        """

        return tuple(loader.get_source_name() for loader in self._rhythm_loaders.values())

    def get_rhythm_loader_count(self):
        """
        Returns the number of currently registered rhythm loaders.

        :return: number of currently registered rhythm loaders
        """

        return len(self._rhythm_loaders)

    def get_rhythm_loader_iterator(self) -> (tp.Iterator[tp.Tuple[tp.Type[BSMidiRhythmLoader], BSMidiRhythmLoader]]):
        """
        Returns an iterator over the currently registered rhythm loaders.

        :return: iterator over the currently registered rhythm loaders
        """

        return iter(self._rhythm_loaders.items())

    def get_rhythm_loader(self, loader_type: tp.Type[BSMidiRhythmLoader]):  # type: (str) -> BSMidiRhythmLoader
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
            self._distances_to_target.fill(np.inf)
        else:
            self._distances_to_target = np.full(n_rhythms, np.inf)

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
        if not self.is_corpus_loaded():
            raise Exception("Corpus not loaded")

    def _precondition_check_target_rhythm_set(self):
        if not self.is_target_rhythm_set():
            raise Exception("Target rhythm not set")

    def _setup_config(self):
        config = self._config
        config.mapping_reducer.add_binding(self._on_mapping_reducer_changed)
        config.rhythm_resolution.add_binding(self._on_rhythm_resolution_changed)

    def _on_mapping_reducer_changed(self, reducer: tp.Type[MidiDrumMappingReducer]):
        target_rhythm = self._target_rhythm
        if not target_rhythm:
            return
        target_rhythm.set_midi_drum_mapping_reducer(reducer)

    def _on_rhythm_resolution_changed(self, resolution: int):
        target_rhythm = self._target_rhythm
        if not target_rhythm:
            return
        target_rhythm.set_resolution(resolution)

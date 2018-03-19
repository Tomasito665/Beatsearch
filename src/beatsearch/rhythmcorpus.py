import os
import sys
import uuid
import pickle
import typing as tp
from collections import namedtuple
from beatsearch.rhythm import MidiRhythm, MidiDrumMappingReducer, get_drum_mapping_reducer_implementation
from beatsearch.utils import get_midi_files_in_directory
from beatsearch.config import BSConfig


RhythmFileInfo = namedtuple("RhythmFileInfo", ["path", "modified_time"])


class RhythmCorpus(object):
    RHYTHM_IX = 0
    FILE_DATA_IX = 1

    class CorpusStateError(Exception):
        pass

    def __init__(
            self,
            root_dir: tp.Optional[str] = None,
            rhythm_resolution: tp.Optional[int] = None,
            midi_mapping_reducer: tp.Optional[MidiDrumMappingReducer] = None
    ):
        self._root_dir = None              # type: tp.Union[str, None]
        self._rhythm_resolution = None     # type: tp.Union[int, None]
        self._midi_mapping_reducer = None  # type: tp.Union[tp.Type[MidiDrumMappingReducer], None]
        self._rhythm_data = None           # type: tp.Union[tp.Tuple[tp.Tuple[MidiRhythm, RhythmFileInfo], ...]]
        self._id = uuid.uuid4()            # type: uuid.UUID

        # calling setters (validation is handled there)
        self.root_directory = root_dir
        self.rhythm_resolution = rhythm_resolution
        self.midi_mapping_reducer = midi_mapping_reducer

    def load(self, config: tp.Optional[BSConfig] = None):
        """Load the corpus

        Loads the corpus. When a config object is given, this method will try to load from cache first. Also, it will
        write the cache filename to the config object.

        If this method is called when the root_directory and/or rhythm_resolution attributes is not set, the caller
        should provide a config object. The root_directory and rhythm_resolution attributes will then automatically be
        set according to the configuration file.

        :param config: BSConfig object or None
        :return: None
        """

        if not config:
            assert self._root_dir, "root_directory should be set if no BSConfig object is given"
            assert self._rhythm_resolution, "rhythm_resolution should be set if no BSConfig object is given"
            assert self._midi_mapping_reducer, "midi_mapping_reducer should be set if no BSConfig object is given"

        midi_dir = self._root_dir or config.midi_root_directory.get()
        cache_fpath = config.get_midi_root_directory_cache_fpath(midi_dir) if config else ""
        resolution = self._rhythm_resolution or config.rhythm_resolution.get()
        mapping_reducer = self._midi_mapping_reducer or config.mapping_reducer.get()

        self.root_directory = midi_dir
        self.rhythm_resolution = resolution
        self.midi_mapping_reducer = mapping_reducer

        # try to load from cache
        if cache_fpath:
            if self._load_from_cache(cache_fpath):
                return
            else:
                # remove the cache for this midi root directory if we weren't able to load it (and
                # therefore it's probably erroneous or out of date)
                config.forget_midi_root_directory_cache_file(midi_dir, remove_cache_file=True)

        # load directly from the directory if couldn't load from cache
        self._load_from_directory()

        if not config:
            return  # we can't create a new cache file without a config object

        pickle_data = {
            'res': resolution,
            'mapping_reducer': getattr(mapping_reducer, "__name__", "None"),
            'data': self._rhythm_data
        }

        # create new cache for new directory
        with config.add_midi_root_directory_cache_file(midi_dir) as cache_file:
            pickle.dump(pickle_data, cache_file)

    def is_loaded(self):
        """Returns whether this corpus has already loaded

        Returns whether this rhythm corpus has already been loaded. This will return true after a successful call to
        load().

        :return: True if this corpus has already loaded: False otherwise
        """

        return self._rhythm_data is not None

    @property
    def rhythm_resolution(self):
        """The resolution in PPQN

        Tick resolution in PPQN (pulses-per-quarter-note) of the rhythms within this corpus. This property will become
        a read-only property after the corpus has loaded.

        :return: resolution in PPQN of the rhythms in this corpus
        """

        return self._rhythm_resolution

    @rhythm_resolution.setter
    def rhythm_resolution(self, resolution: tp.Union[int, None]):
        if self.is_loaded():
            raise self.CorpusStateError()

        if resolution is None:
            self._rhythm_resolution = None
            return

        resolution = int(resolution)
        if resolution <= 0:
            raise ValueError("resolution should be greater than zero")

        self._rhythm_resolution = resolution

    @property
    def midi_mapping_reducer(self) -> tp.Union[tp.Type[MidiDrumMappingReducer], None]:
        return self._midi_mapping_reducer

    @midi_mapping_reducer.setter
    def midi_mapping_reducer(self, midi_mapping_reducer: tp.Type[MidiDrumMappingReducer]):
        if self.is_loaded():
            raise self.CorpusStateError
        self._midi_mapping_reducer = midi_mapping_reducer

    @property
    def root_directory(self):
        """MIDI root directory

        Root directory containing the MIDI files that where used to create the rhythms within this corpus. This property
        will become a read-only property after the corpus has loaded.

        :return: MIDI root directory
        """

        return self._root_dir

    @root_directory.setter
    def root_directory(self, root_dir: tp.Union[str, None]):
        if self.is_loaded():
            raise self.CorpusStateError()

        if not root_dir:
            self._root_dir = None
            return

        if not os.path.isdir(root_dir):
            raise ValueError("no such directory: %s" % root_dir)

        self._root_dir = root_dir

    @property
    def id(self):
        """The id of this rhythm corpus

        The UUID id of this rhythm corpus. This is a read-only property.
        """

        return self._id

    def __getitem__(self, i):
        """Returns the i-th rhythm"""
        return self._rhythm_data[i][self.RHYTHM_IX]

    def __len__(self):
        """Returns the number of rhythms within this corpus"""
        return len(self._rhythm_data)

    def __iter__(self):
        """Returns an iterator over the rhythms within this corpus"""
        return iter(data[self.RHYTHM_IX] for data in self._rhythm_data)

    # loads the rhythm corpus from the root directory
    def _load_from_directory(self):
        midi_dir = self._root_dir
        resolution = self._rhythm_resolution
        mapping_reducer = self._midi_mapping_reducer

        def get_rhythm_data():
            for f_path in get_midi_files_in_directory(midi_dir):
                f_path = f_path.replace("\\", "/")  # TODO handle this properly with a utility function "normalize_path"
                try:
                    rhythm = MidiRhythm(f_path, midi_mapping_reducer_cls=mapping_reducer)
                    rhythm.set_resolution(resolution)
                    print("%s: OK" % f_path)
                except (TypeError, ValueError) as e:
                    print("%s: ERROR, %s" % (f_path, str(e)))
                    continue
                m_time = os.path.getmtime(f_path)
                file_info = RhythmFileInfo(path=f_path, modified_time=m_time)
                yield rhythm, file_info

        self._rhythm_data = tuple(get_rhythm_data())

    # loads the rhythm corpus from a cache file
    def _load_from_cache(self, cache_fpath: str):
        if not cache_fpath or not os.path.isfile(cache_fpath):
            return False

        with open(cache_fpath, "rb") as pickle_file:
            try:
                unpickled_data = pickle.load(pickle_file)
            except (ValueError, TypeError) as e:
                print("Error loading midi root directory cache: %s" % e, file=sys.stderr)
                return False

        try:
            rhythm_resolution = unpickled_data['res']
            mapping_reducer_name = unpickled_data['mapping_reducer']
            rhythm_data = unpickled_data['data']
        except KeyError:
            print("Midi root directory cache has bad format: %s" % cache_fpath)
            return False

        if mapping_reducer_name == "None":
            mapping_reducer = None
        else:
            mapping_reducer = get_drum_mapping_reducer_implementation(mapping_reducer_name)

        if not self._is_rhythm_data_up_to_date(rhythm_data, rhythm_resolution, mapping_reducer):
            print("Midi root directory cache file not up to date: %s" % cache_fpath)
            return False

        assert rhythm_resolution == self._rhythm_resolution
        self._rhythm_data = rhythm_data
        return True

    # returns whether the rhythm file data is up to date with the "real word" files
    def _is_rhythm_data_up_to_date(
            self, rhythm_data: tp.Tuple[tp.Tuple[MidiRhythm, RhythmFileInfo], ...],
            rhythm_resolution: int,
            mapping_reducer: tp.Union[MidiDrumMappingReducer, None]
    ):
        if rhythm_resolution != self._rhythm_resolution:
            return False

        if mapping_reducer != self._midi_mapping_reducer:
            return False

        for file_info in (data[self.FILE_DATA_IX] for data in rhythm_data):
            fpath = file_info.path

            # rhythm data is not up to date if either the file doesn't exist anymore or the file has been modified
            if not os.path.isfile(fpath) or os.path.getmtime(fpath) != file_info.modified_time:
                return False

        n_cached_midi_files = len(rhythm_data)
        n_actual_midi_files = sum(1 for _ in get_midi_files_in_directory(self._root_dir))

        # won't be equal if new MIDI files have been added
        return n_cached_midi_files == n_actual_midi_files


__all__ = ['RhythmCorpus']

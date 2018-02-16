import os
import uuid
import pickle
import typing as tp
from collections import namedtuple
from beatsearch.rhythm import MidiRhythm
from beatsearch.utils import get_midi_files_in_directory
from beatsearch.config import BSConfig


RhythmFileInfo = namedtuple("RhythmFileInfo", ["path", "modified_time"])


class RhythmCorpus(object):
    RHYTHM_IX = 0
    FILE_DATA_IX = 1

    def __init__(self, root_dir: str, rhythm_resolution: int):
        if not os.path.isdir(root_dir):
            raise IOError("no such directory: \"%s\"" % root_dir)

        if rhythm_resolution <= 0:
            raise ValueError("expected rhythm resolution greater than zero but got %i" % rhythm_resolution)

        self._root_dir = root_dir                     # type: str
        self._rhythm_resolution = rhythm_resolution   # type: int
        self._rhythm_data = tuple()                   # type: tp.Tuple[tp.Tuple[MidiRhythm, RhythmFileInfo], ...]
        self._id = uuid.uuid4()                       # type: uuid.UUID

    def load(self, config: BSConfig = None):
        midi_dir = self._root_dir
        cache_fpath = config.get_midi_root_directory_cache_fpath(midi_dir) if config else ""
        resolution = self._rhythm_resolution

        # try to load from cache
        if cache_fpath:
            if self._load_from_cache(cache_fpath):
                return True
            else:
                # remove the cache for this midi root directory if we weren't able to load it (and
                # therefore it's probably erroneous or out of date)
                config.forget_midi_root_directory_cache_file(midi_dir, remove_cache_file=True)

        if not self._load_from_directory():
            raise Exception("error occurred loading midi rhythms from: \"%s\"" % midi_dir)

        if not config:
            # we can't create a new cache file without a config object
            return True

        pickle_data = {
            'res': resolution,
            'data': self._rhythm_data
        }

        # create new cache for new directory
        with config.add_midi_root_directory_cache_file(midi_dir) as cache_file:
            pickle.dump(pickle_data, cache_file)

        return True

    @property
    def rhythm_resolution(self):
        """The resolution in PPQN

        Tick resolution in PPQN (pulses-per-quarter-note) of the rhythms within this corpus. This is a read-only
        property.

        :return: resolution in PPQN of the rhythms in this corpus
        """

        return self._rhythm_resolution

    @property
    def root_directory(self):
        """MIDI root directory

        Root directory containing the MIDI files that where used to create the rhythms within this corpus. This is a
        read-only property.

        :return: MIDI root directory
        """

        return self._root_dir

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

        def get_rhythm_data():
            for f_path in get_midi_files_in_directory(midi_dir):
                f_path = f_path.replace("\\", "/")  # TODO handle this properly with a utility function "normalize_path"
                try:
                    rhythm = MidiRhythm(f_path)
                    rhythm.set_resolution(resolution)
                    print("%s: OK" % f_path)
                except (TypeError, ValueError) as e:
                    print("%s: ERROR, %s" % (f_path, str(e)))
                    continue
                m_time = os.path.getmtime(f_path)
                file_info = RhythmFileInfo(path=f_path, modified_time=m_time)
                yield rhythm, file_info

        self._rhythm_data = tuple(get_rhythm_data())
        return True

    # loads the rhythm corpus from a cache file
    def _load_from_cache(self, cache_fpath: str):
        if not cache_fpath or not os.path.isfile(cache_fpath):
            return False

        with open(cache_fpath, "rb") as pickle_file:
            unpickled_data = pickle.load(pickle_file)

        try:
            rhythm_resolution = unpickled_data['res']
            rhythm_data = unpickled_data['data']
        except KeyError:
            print("Midi root directory cache has bad format: %s" % cache_fpath)
            return False

        if not self._is_rhythm_data_up_to_date(rhythm_data, rhythm_resolution):
            print("Midi root directory cache file not up to date: %s" % cache_fpath)
            return False

        assert rhythm_resolution == self._rhythm_resolution
        self._rhythm_data = rhythm_data
        return True

    # returns whether the rhythm file data is up to date with the "real word" files
    def _is_rhythm_data_up_to_date(
            self, rhythm_data: tp.Tuple[tp.Tuple[MidiRhythm, RhythmFileInfo], ...],
            rhythm_resolution: int
    ):
        if rhythm_resolution != self._rhythm_resolution:
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

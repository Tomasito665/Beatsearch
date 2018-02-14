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
    DEFAULT_RESOLUTION = 960

    def __init__(self, root_dir, config: BSConfig, ignore_pickle_file=False):
        root_dir = str(root_dir)
        resolution = config.get_rhythm_resolution() or self.DEFAULT_RESOLUTION

        if not os.path.isdir(root_dir):
            raise IOError("no such directory: %s" % root_dir)

        print("Loading rhythms from: %s" % root_dir)
        self._root_dir = root_dir             # type: str
        self._config = config                 # type: BSConfig
        self._rhythm_resolution = resolution  # type: int
        self._rhythm_data = tuple()           # type: tp.Tuple[tp.Tuple[MidiRhythm, RhythmFileInfo], ...]
        self._uuid = uuid.uuid4()             # type: uuid.UUID

        if not self.__load_from_pickle_file() and not ignore_pickle_file:
            print("Parsing MIDI files: %s" % root_dir)
            self.__load_from_root_dir()
            pickle_file = self.__write_pickle()
            config.set_rhythm_corpus_pickle_name(self._root_dir, pickle_file)
            config.save()

    @property
    def root_directory(self):
        return self._root_dir

    @property
    def rhythm_resolution(self):
        return self._rhythm_resolution

    @rhythm_resolution.setter
    def rhythm_resolution(self, resolution):
        resolution = int(resolution)
        if resolution <= 0:
            raise ValueError("rhythm resolution should be greater than zero")
        old_res = self._rhythm_resolution
        if old_res == resolution:
            return
        print("Rescaling corpus from %i PPQN to %i PPQN" % (self._rhythm_resolution, resolution))
        for rhythm in self:
            rhythm.set_resolution(resolution)

    @property
    def id(self):
        return self._uuid

    def __load_from_pickle_file(self):
        config = self._config
        root_dir = self._root_dir
        pickle_data_dir = self.get_rhythm_pickle_directory()
        pickle_name = config.get_rhythm_corpus_pickle_name(root_dir)
        pickle_fname = "%s.pkl" % pickle_name if pickle_name else ""
        pickle_fpath = os.path.join(pickle_data_dir, pickle_fname)

        def remove_pickle_file():
            config.set_rhythm_corpus_pickle_name(root_dir, None)
            os.remove(pickle_fpath)

        if not pickle_name:
            print("No pickle file found")
            return False

        print("Loading rhythms from cached pickle file: %s" % pickle_fpath)

        with open(pickle_fpath, "rb") as pickle_file:
            pickle_data = pickle.load(pickle_file)

        try:
            root_dir = pickle_data['root_dir']
            rhythm_resolution = pickle_data['res']
            rhythm_data = pickle_data['rhythm_data']
        except KeyError:
            print("Pickle file content error")
            remove_pickle_file()
            return False

        if not self.__is_rhythm_data_up_to_date(rhythm_data) or \
                rhythm_resolution != self._rhythm_resolution:
            print("Pickle file not up to date")
            remove_pickle_file()
            return False

        self._root_dir = root_dir
        self._rhythm_resolution = rhythm_resolution
        self._rhythm_data = rhythm_data

        return True

    def __is_rhythm_data_up_to_date(self, rhythm_data: tp.Tuple[tp.Tuple[MidiRhythm, RhythmFileInfo], ...]):
        for file_info in (data[self.FILE_DATA_IX] for data in rhythm_data):
            fpath = file_info.path

            if not os.path.isfile(fpath) or os.path.getmtime(fpath) != file_info.modified_time:
                # rhythm data is not up to date if either the file doesn't exist anymore or the
                # file has been modified
                return False

        n_pickle_midi_files = len(rhythm_data)
        n_actual_midi_files = sum(1 for _ in get_midi_files_in_directory(self._root_dir))
        return n_pickle_midi_files == n_actual_midi_files  # won't be equal if new MIDI files have been added

    def __load_from_root_dir(self):
        root_dir = self._root_dir
        resolution = self._rhythm_resolution

        def get_rhythm_data():
            for f_path in get_midi_files_in_directory(root_dir):
                try:
                    rhythm = MidiRhythm(f_path)
                    rhythm.set_resolution(resolution)
                except (TypeError, ValueError) as e:
                    print("%s: [Error] %s" % (f_path, e))
                    continue

                m_time = os.path.getmtime(f_path)
                file_info = RhythmFileInfo(path=f_path, modified_time=m_time)

                yield rhythm, file_info

        self._rhythm_data = tuple(get_rhythm_data())

    def __write_pickle(self):
        pickle_file_dir = self.get_rhythm_pickle_directory(True)

        def generate_pickle_fpath():
            f_name = "rhythms_%s.pkl" % str(uuid.uuid4())
            return os.path.join(pickle_file_dir, f_name)

        # loop in case that there already exists a file with the generated path
        f_path = generate_pickle_fpath()
        while os.path.isfile(f_path):
            f_path = generate_pickle_fpath()

        with open(f_path, "wb") as f:
            pickle_data = {
                'root_dir': self._root_dir,
                'res': self._rhythm_resolution,
                'rhythm_data': self._rhythm_data
            }

            pickle.dump(pickle_data, f)

        basename = os.path.basename(f_path)
        return os.path.splitext(basename)[0]

    def get_rhythm_pickle_directory(self, mkdir=True):
        ini_file_dir = self._config.root_dir
        pickle_file_dir = os.path.join(ini_file_dir, "data")
        if mkdir and not os.path.isdir(pickle_file_dir):
            os.mkdir(pickle_file_dir)
        return pickle_file_dir

    def __getitem__(self, item):
        return self._rhythm_data[item][self.RHYTHM_IX]

    def __len__(self):
        return len(self._rhythm_data)

    def __iter__(self):
        return iter(data[self.RHYTHM_IX] for data in self._rhythm_data)

import os
import configparser


class BSConfig(object):

    # Config file sections
    DEFAULT = "DEFAULT"
    SETTINGS = "SETTINGS"
    SERIALIZED_RHYTHM_CORPORA = "SERIALIZED_RHYTHM_CORPORA"

    def __init__(self, ini_file):
        # explicitly set only "=" as delimiter (we don't want : as a delimiter, because we'll have
        # paths as keys, starting with "C:\..." and then the key will be just "C"
        self._parser = configparser.ConfigParser(delimiters="=")
        self._ini_file = str(ini_file)
        ini_file_dir = os.path.dirname(ini_file)

        if not os.path.isdir(ini_file_dir):
            raise IOError("can't read/write config file in non-existent directory: %s" % ini_file_dir)

        if os.path.isfile(ini_file):
            with open(ini_file, "r", encoding="utf8") as f:
                self._parser.read_file(f)

    def set_rhythm_resolution(self, resolution):
        """Sets the rhythm resolution or 0 to remove from config file

        :param resolution: a positive integer or 0 to remove the resolution from the config file
        :return: None
        """

        if self.SETTINGS not in self._parser:
            self._parser[self.SETTINGS] = {}
        self._parser[self.SETTINGS]['rhythm_resolution'] = str(resolution)
        if resolution == 0:
            del self._parser[self.SETTINGS]['rhythm_resolution']

    def get_rhythm_resolution(self):
        """Returns the rhythm resolution or 0 if unspecified

        :return: rhythm resolution (0 for unspecified)
        """

        try:
            return self._parser[self.SETTINGS].getint('rhythm_resolution')
        except (KeyError, ValueError):
            return 0

    def set_rhythm_root_directory(self, rhythm_root_dir):
        """Sets the rhythm root directory

        :param rhythm_root_dir: root directory containing the MIDI files
        :return: None
        """

        if self.SETTINGS not in self._parser:
            self._parser[self.SETTINGS] = {}
        self._parser[self.SETTINGS]['rhythm_root_dir'] = str(rhythm_root_dir)
        if not rhythm_root_dir:
            del self._parser[self.SETTINGS]['rhythm_root_dir']

    def get_rhythm_root_directory(self):
        """Returns the rhythm root directory

        Returns the directory containing the MIDI files or an empty string if no root directory is set.

        :return: root directory containing the MIDI files or an empty string
        """

        try:
            return self._parser[self.SETTINGS]['rhythm_root_dir']
        except KeyError:
            return ""

    def set_rhythm_corpus_pickle_name(self, corpus_root_dir, pickle_name):
        """Sets the name of the pickle file containing the rhythms within the given root directory

        Sets the name (without extension) of the pickle file containing the cached rhythms for the given rhythm root
        directory.

        :param corpus_root_dir: root directory of which the rhythms are cached in the pickle file
        :param pickle_name: name of the pickle file (without .pkl) containing the cached rhythms within the given
                            directory
        :return: None
        """

        corpus_root_dir = os.path.normpath(corpus_root_dir)
        corpus_root_dir = corpus_root_dir.replace("\\", "/")

        if self.SERIALIZED_RHYTHM_CORPORA not in self._parser:
            self._parser[self.SERIALIZED_RHYTHM_CORPORA] = {}

        self._parser[self.SERIALIZED_RHYTHM_CORPORA][corpus_root_dir] = str(pickle_name)

        if not pickle_name:
            del self._parser[self.SERIALIZED_RHYTHM_CORPORA][corpus_root_dir]

    def get_rhythm_corpus_pickle_name(self, corpus_root_dir):
        """Returns the name of the pickle file containing the rhythms within the given root directory

        Returns the name of the pickle file (without extension) containing the rhythms within the given root directory
        or an empty string if there's no cache of the given directory.

        :param corpus_root_dir: root directory of the rhythms
        :return: name of the pickle file (without .pkl) containing the cached rhythms within the given directory
        """

        corpus_root_dir = os.path.normpath(corpus_root_dir)
        corpus_root_dir = corpus_root_dir.replace("\\", "/")  # always forward slashes, also on Windows

        try:
            return self._parser[self.SERIALIZED_RHYTHM_CORPORA][corpus_root_dir]
        except KeyError:
            return ""

    def save(self):
        with open(self._ini_file, "w", encoding="utf-8") as configfile:
            self._parser.write(configfile)

    @property
    def root_dir(self):
        return os.path.dirname(self._ini_file)

import os
import configparser


class BeatsearchConfig(object):
    DEFAULT = "DEFAULT"
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

    def set_rhythm_corpus_pickle_file(self, corpus_root_dir, pickle_file):
        corpus_root_dir = str(corpus_root_dir)
        pickle_file = str(pickle_file)

        if self.SERIALIZED_RHYTHM_CORPORA not in self._parser:
            self._parser[self.SERIALIZED_RHYTHM_CORPORA] = {}

        if pickle_file:
            pickle_file = os.path.normpath(pickle_file)
            self._parser[self.SERIALIZED_RHYTHM_CORPORA][corpus_root_dir] = pickle_file
        else:
            del self._parser[self.SERIALIZED_RHYTHM_CORPORA][corpus_root_dir]

    def get_rhythm_corpus_pickle_file(self, corpus_root_dir):
        corpus_root_dir = os.path.normpath(corpus_root_dir)

        try:
            return self._parser[self.SERIALIZED_RHYTHM_CORPORA][corpus_root_dir]
        except KeyError:
            return ""

    def save(self):
        with open(self._ini_file, "w", encoding="utf-8") as configfile:
            self._parser.write(configfile)

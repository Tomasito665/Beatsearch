import os
import pickle


class RhythmCorpus(list):
    def __init__(self, rhythms=(), name="", fpath=""):
        super(RhythmCorpus, self).__init__(rhythms)
        self._name = name
        self._fpath = fpath

    @staticmethod
    def load(in_file):
        """
        Loads a rhythm corpus based on the given binary file.

        :param in_file: either the path of the *.pkl file or the file itself
        :return: rhythm corpus
        """

        try:
            rhythms = pickle.load(in_file)
            fpath = in_file.name
        except TypeError:
            fpath = in_file
            with open(in_file, 'rb') as f:
                rhythms = pickle.load(f)
        name = os.path.splitext(os.path.basename(fpath))[0]
        return RhythmCorpus(rhythms, name, fpath)

    @property
    def name(self):
        return self._name

    @property
    def fname(self):
        fpath = self._fpath
        return os.path.basename(fpath)

    def append(self, _):  # TODO We might extend tuple instead of list and get rid of these methods
        raise Exception("Can't append rhythms to rhythm corpus")

    def extend(self, _):
        raise Exception("Can't extend rhythms to rhythm corpus")

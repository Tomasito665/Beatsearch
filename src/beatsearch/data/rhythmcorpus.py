import os
# noinspection PyPep8Naming
import cPickle as pickle


class RhythmCorpus(list):
    def __init__(self, rhythms=(), name=""):
        super(RhythmCorpus, self).__init__(rhythms)
        self._name = name

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
        return RhythmCorpus(rhythms, name)

    @property
    def name(self):
        return self._name

    def append(self, _):  # TODO We might extend tuple instead of list and get rid of these methods
        raise Exception("Can't append rhythms to rhythm corpus")

    def extend(self, _):
        raise Exception("Can't extend rhythms to rhythm corpus")

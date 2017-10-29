# noinspection PyPep8Naming
import cPickle as pickle


class RhythmCorpus(list):
    def __init__(self, rhythms=()):
        super(RhythmCorpus, self).__init__(rhythms)

    @staticmethod
    def load(in_file):
        """
        Loads a rhythm corpus based on the given binary file.

        :param in_file: either the path of the *.pkl file or the file itself
        :return: rhythm corpus
        """

        try:
            rhythms = pickle.load(in_file)
        except TypeError:
            with open(in_file, 'rb') as f:
                rhythms = pickle.load(f)
        return RhythmCorpus(rhythms)

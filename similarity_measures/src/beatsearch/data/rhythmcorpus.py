# noinspection PyPep8Naming
import cPickle as pickle


class RhythmCorpus(list):
    def __init__(self, rhythms=()):
        super(RhythmCorpus, self).__init__(rhythms)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            rhythms = pickle.load(f)
        return RhythmCorpus(rhythms)

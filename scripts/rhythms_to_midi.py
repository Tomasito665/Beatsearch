import argparse
import os
import sys
import midi
from time import time
from create_pickle import log_replace
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "src"))
from beatsearch.rhythmcorpus import RhythmCorpus
from beatsearch.utils import print_progress_bar


def get_args():
    parser = argparse.ArgumentParser(description="Exports rhythms to MIDI files")
    parser.add_argument("--corpus", type=argparse.FileType('r'),
                        help="The *.pkl file containing the rhythms to export to MIDI",
                        default="./data/rhythms.pkl")
    parser.add_argument("--dir", default="./output/midi",
                        help="Directory to save the MIDI files to")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    log_replace("Loading rhythms from: %s" % args.corpus.name)
    corpus = RhythmCorpus.load(args.corpus)
    log_replace("Loaded rhythms from '%s' containing %i rhythms\n" % (args.corpus.name, len(corpus)))

    if not os.path.isdir(args.dir):
        os.makedirs(args.dir)

    rhythm_i = 1
    n_rhythms = len(corpus)
    t_start = time()

    for rhythm in corpus:
        print_progress_bar(rhythm_i, n_rhythms,
                           "Exporting rhythms to MIDI...", "[%i/%i]" %
                           (rhythm_i, n_rhythms), starting_time=t_start, fill="O")

        pattern = rhythm.to_midi()
        path = os.path.join(args.dir, "%s.mid" % rhythm.name)
        midi.write_midifile(path, pattern)
        rhythm_i += 1

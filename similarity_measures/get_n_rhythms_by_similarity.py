import os
import sys
import argparse
import midi
import numpy as np
from argparse import RawTextHelpFormatter
from create_pickle import log_replace
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "src"))
from beatsearch.data.rhythm import Rhythm
from beatsearch.data.rhythmcorpus import RhythmCorpus


def get_args():
    parser = argparse.ArgumentParser(
        description="Prints the rhythm names that return the highest "
                    "similarity score with the given input rhythm.",
        formatter_class=RawTextHelpFormatter)
    parser.add_argument("target_rhythm", type=argparse.FileType('r'),
                        help="The MIDI file of the target rhythm")
    parser.add_argument("--corpus", type=argparse.FileType('r'),
                        help="The *.pkl file containing the rhythms",
                        default="./data/rhythms.pkl")
    parser.add_argument("--track", type=int, default=36,
                        help="The rhythm track to compare")
    parser.add_argument("-n", type=int, default=10, help="Number of rhythms to return")
    parser.add_argument("--method", type=int, default=0, help="""Which similarity measure to use:
    0) Hamming distance
    1) Euclidean inter-onset interval vector distance
    2) Interval difference vector distance
""")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # parse the target midi file
    target_rhythm_midi = midi.read_midifile(args.target_rhythm)
    target_rhythm = Rhythm.create_from_midi(target_rhythm_midi, args.target_rhythm.name)
    target_track = target_rhythm.get_track(args.track)

    if target_track is None:
        print "Target rhythm has no track named '%s'" % args.track

    # load corpus
    log_replace("Loading rhythms from: %s" % args.corpus.name)
    corpus = RhythmCorpus.load(args.corpus)
    log_replace("Loaded rhythms from '%s' containing %i rhythms\n" % (args.corpus.name, len(corpus)))

    similarity_funcs = (
        Rhythm.Track.get_hamming_distance_to,
        Rhythm.Track.get_euclidean_inter_onset_vector_distance_to,
        Rhythm.Track.get_interval_difference_vector_distance_to
    )

    try:
        get_similarity = similarity_funcs[args.method].im_func
    except IndexError:
        print "Given unknown similarity method: '%i'" % args.method
        sys.exit(-1)

    distances = []

    for rhythm in corpus:
        track = rhythm.get_track(args.track)
        if track is None:
            distances.append(float("inf"))
            continue
        try:
            distance = get_similarity(target_track, track, 'eighths')
        except ValueError:
            distances.append(float("inf"))
            continue
        distances.append(distance)

    sorted_indexes = np.argsort(distances)
    print "\nThe %i most similar rhythms to '%s', when measured with '%s' on track '%s' are:" \
          % (args.n, target_rhythm.name, get_similarity.__name__, args.track)

    for i in range(args.n):
        index = sorted_indexes[i]
        rhythm = corpus[index]
        distance = distances[index]
        print "    %i) (d = %.2f) %s" % (i + 1, distance, rhythm.name)

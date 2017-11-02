import os
import sys
import argparse

import midi
import numpy as np
from argparse import RawTextHelpFormatter
from create_pickle import log_replace
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "src"))
from beatsearch.data.rhythmcorpus import RhythmCorpus
from beatsearch.data.rhythm import Rhythm, get_track_distance_measures

SIM_MEASURES = get_track_distance_measures()
SIM_MEASURE_NAMES = SIM_MEASURES.keys()
SIM_MEASURE_CLASSES = SIM_MEASURES.values()


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
    info_sim_measures = ''.join(["\n   %i) %s" % (j + 1, name) for j, name in enumerate(SIM_MEASURE_NAMES)])
    parser.add_argument("--measure", type=int, default=0, help="Which similarity measure to use:%s" % info_sim_measures)
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

    try:
        distance_measure_index = args.measure - 1
        if distance_measure_index < 0:
            raise IndexError
        distance_measure = SIM_MEASURE_CLASSES[args.measure - 1]()
    except IndexError:
        print "Given unknown similarity method: %i (choose between: %s)" % \
              (args.measure, range(1, len(SIM_MEASURE_CLASSES) + 1))
        sys.exit(-1)

    distances = []

    for rhythm in corpus:
        track = rhythm.get_track(args.track)
        if track is None:
            distances.append(float("inf"))
            continue
        try:
            distance = distance_measure.get_distance(target_track, track)
        except ValueError:
            distances.append(float("inf"))
            continue
        distances.append(distance)

    sorted_indexes = np.argsort(distances)
    print "\nThe %i most similar rhythms to '%s', when measured with '%s' on track '%s' are:" \
          % (args.n, target_rhythm.name, distance_measure.__class__.__name__, args.track)

    for i in range(args.n):
        index = sorted_indexes[i]
        rhythm = corpus[index]
        distance = distances[index]
        formatted_d = "%.2f" % distance if type(distance) == float else str(distance)
        print "    %i) (d = %s) %s" % (i + 1, formatted_d, rhythm.name)

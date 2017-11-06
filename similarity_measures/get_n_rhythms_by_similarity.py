# coding=utf-8
import os
import sys
import argparse
import textwrap
import midi
import numpy as np
from create_pickle import log_replace
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "src"))
from beatsearch.data.rhythmcorpus import RhythmCorpus
from beatsearch.utils import err_print
from beatsearch.data.rhythm import Rhythm, RhythmDistanceMeasure, get_track_distance_measures

SIM_MEASURES = get_track_distance_measures()
SIM_MEASURE_NAMES = SIM_MEASURES.keys()
SIM_MEASURE_CLASSES = SIM_MEASURES.values()


def get_args():
    info_sim_measures = "".join(["\n%s%i) %s" % (" " * 16, j + 1, name) for j, name in enumerate(SIM_MEASURE_NAMES)])

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""                                                                                                                                
                       ____             _   ____                      _     
                      | __ )  ___  __ _| |_/ ___|  ___  __ _ _ __ ___| |__  
                      |  _ \ / _ \/ _` | __\___ \ / _ \/ _` | '__/ __| '_ \ 
                      | |_) |  __/ (_| | |_ ___) |  __/ (_| | | | (__| | | |
                      |____/ \___|\__,_|\__|____/ \___|\__,_|_|  \___|_| |_|                                                                                                                                                                                                                                          
        
            This script returns a list with rhythms that return the lowest distance 
            to the given target rhythm. The script will search through rhythms in the 
            given rhythm corpus. This corpus should be a *.pkl file and it can be set 
            with the --corpus flag.
            
            The rhythms are compared as a whole by default. However, if you just want 
            to compare certain tracks, you can specify that with the -t flag. For 
            example, when run with "-t 36 38", only the kick and the snare tracks will 
            be compared. You can also provide these wildcards instead of explicitly 
            setting the track names:
                *   Compares all tracks. E.g. if the target rhythm has a kick but no 
                    snare and the compared rhythm has a snare but no kick, both tracks 
                    will be compared. In this particular case, this will yield the 
                    maximum distance, because there is nothing to compare.
                a*  Compares all tracks of the target rhythm with the equally named 
                    tracks in the compared rhythm.
                b*  Compares all tracks of the compared rhythm to the equally named 
                    tracks in the target rhythm.
            
            The track distance measure can be set with the -t flag, which 
            can be set to: %s
            
        """ % info_sim_measures)
    )

    parser.add_argument(
        "target_rhythm",
        metavar="TARGET_RHYTHM",
        type=argparse.FileType("r"),
        help="specifies the path to the target rhythm MIDI file"
    )

    parser.add_argument(
        "--corpus",
        type=argparse.FileType("r"),
        metavar="CORPUS",
        dest="corpus",
        help="specifies the path to the *.pkl corpus file",
        default="./data/rhythms.pkl"
    )

    parser.add_argument(
        "-t",
        dest='tracks',
        type=str,
        default=['a*'],
        nargs='*',
        metavar='T',
        help="specifies the tracks to compare"
    )

    parser.add_argument(
        "-n",
        type=int,
        default=10,
        help="specifies the number of rhythms to return"
    )

    parser.add_argument(
        "--measure",
        type=int,
        dest="measure",
        default=1,
        help="specifies which track similarity measure to use"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    tracks = args.tracks[0] if len(args.tracks) == 1 else args.tracks

    # parse the target midi file
    target_rhythm_midi = midi.read_midifile(args.target_rhythm)
    target_rhythm = Rhythm.create_from_midi(target_rhythm_midi, args.target_rhythm.name)

    # load corpus
    log_replace("Loading rhythms from: %s" % args.corpus.name)
    corpus = RhythmCorpus.load(args.corpus)
    log_replace("Loaded rhythms from '%s' containing %i rhythms\n" % (args.corpus.name, len(corpus)))

    try:
        distance_measure_index = args.measure - 1
        if distance_measure_index < 0:
            raise IndexError
        track_distance_measure = SIM_MEASURE_CLASSES[args.measure - 1]()
    except IndexError:
        err_print("Given unknown similarity method: %i (choose between: %s)" % (
            args.measure, range(1, len(SIM_MEASURE_CLASSES) + 1)))
        sys.exit(-1)

    measure = RhythmDistanceMeasure(track_distance_measure, tracks)
    distances = [measure.get_distance(target_rhythm, other) for other in corpus]

    sorted_indexes = np.argsort(distances)
    print "\nThe %i most similar rhythms to '%s', when measured with '%s' on tracks '%s' are:" \
          % (args.n, target_rhythm.name, track_distance_measure.__class__.__name__, tracks)

    for i in range(args.n):
        index = sorted_indexes[i]
        rhythm = corpus[index]
        distance = distances[index]
        formatted_d = "%.4f" % distance if type(distance) == float else str(distance)
        print "    %i) (d = %s) %s" % (i + 1, formatted_d, rhythm.name)

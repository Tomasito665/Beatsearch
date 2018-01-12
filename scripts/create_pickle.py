import os
import pickle
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "src"))
from beatsearch.data.rhythm import MidiRhythm


def create_ascii_spinner(spinner_chars, n_iters_per_char=32):
    i = 0
    while True:
        yield spinner_chars[i // n_iters_per_char]
        i = (i + 1) % (len(spinner_chars) * n_iters_per_char)


def log_replace(msg):
    sys.stdout.write("\r%s" % msg)
    sys.stdout.flush()


if __name__ == '__main__':
    try:
        root_dir = sys.argv[1]
    except IndexError:
        raise Exception("No root directory given")

    try:
        output = sys.argv[2]
    except IndexError:
        output = "./data/rhythms.pkl"

    rhythms = []
    n_midi_files = 0

    spinner = create_ascii_spinner("...ooOO@*         ")

    for directory, subdirectories, files in os.walk(root_dir):
        for f_name in files:
            if os.path.splitext(f_name)[1] != ".mid":
                continue

            n_midi_files += 1
            path = os.path.join(root_dir, directory, f_name)
            log_replace("Parsing %ith rhythm... %s" % (len(rhythms) + 1, next(spinner)))

            try:
                rhythm = MidiRhythm(path)
                rhythms.append(rhythm)
            except (TypeError, ValueError) as e:
                print(e, path)
                continue

    log_replace("Successfully parsed %i/%i rhythms" % (len(rhythms), n_midi_files))

    log_replace("\nDumping rhythms to: %s" % output)
    pickle.dump(rhythms, open(output, "wb"))
    log_replace("Dumped rhythms to: %s" % output)

import subprocess
from collections import OrderedDict
from time import time
from beatsearch.rhythm import MonophonicRhythm, MonophonicRhythmRepresentationsMixin

N_ITER_PER_REPRESENTATION_PER_RHYTHM = 10000


def get_git_revision_short_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").rstrip()


if __name__ == "__main__":
    rhythms = OrderedDict()

    # noinspection PyUnresolvedReferences
    rhythms['rumba'] = MonophonicRhythm.create.from_string("--x-x---x--x---x")
    # noinspection PyUnresolvedReferences
    rhythms['clave32'] = MonophonicRhythm.create.from_string("--x-x---x--x--x-")
    # noinspection PyUnresolvedReferences
    rhythms['clave23'] = MonophonicRhythm.create.from_string("x--x---x--x-x---")

    representation_functions = (
        MonophonicRhythmRepresentationsMixin.get_binary,
        MonophonicRhythmRepresentationsMixin.get_pre_note_inter_onset_intervals,
        MonophonicRhythmRepresentationsMixin.get_post_note_inter_onset_intervals,
        MonophonicRhythmRepresentationsMixin.get_interval_histogram,
        MonophonicRhythmRepresentationsMixin.get_binary_schillinger_chain,
        MonophonicRhythmRepresentationsMixin.get_chronotonic_chain,
        MonophonicRhythmRepresentationsMixin.get_interval_difference_vector,
        MonophonicRhythmRepresentationsMixin.get_onset_times
    )

    representation_function_kwargs = dict(unit="eighths")
    n_iters_per_representation = N_ITER_PER_REPRESENTATION_PER_RHYTHM * len(rhythms)

    print("Monophonic Rhythm Representations Benchmark")
    print("  Beatsearch git commit hash: %s" % get_git_revision_short_hash())
    print("  Number of iterations per representation: %i" % n_iters_per_representation)
    print("")

    t_start_global = time()

    for get_representation in representation_functions:
        t_total_representation = 0

        for rhythm_name, rhythm in rhythms.items():
            t_start_rhythm = time()

            for i in range(N_ITER_PER_REPRESENTATION_PER_RHYTHM):
                _ = get_representation(rhythm, **representation_function_kwargs)

            t_end_rhythm = time()
            t_total_rhythm = t_end_rhythm - t_start_rhythm
            t_average_rhythm = t_total_rhythm / N_ITER_PER_REPRESENTATION_PER_RHYTHM
            t_total_representation += t_total_rhythm

        t_average_representation = t_total_representation / n_iters_per_representation
        print("> %s: %f" % (get_representation.__name__, t_average_representation))

    t_end_global = time()
    t_total_global = t_end_global - t_start_global

    print("\n")
    print("Done. Total time: %.3g seconds" % t_total_global)

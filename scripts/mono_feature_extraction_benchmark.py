import subprocess
from collections import OrderedDict
from time import time
from beatsearch.rhythm import MonophonicRhythm, Unit
from beatsearch.feature_extraction import MonophonicRhythmFeatureExtractor

N_ITER_PER_REPRESENTATION_PER_RHYTHM = 10000


def get_git_revision_short_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").rstrip()


if __name__ == "__main__":
    rhythms = OrderedDict()

    # noinspection PyUnresolvedReferences
    rhythms['rumba'] = MonophonicRhythm.create.from_string("--x-x---x--x---x", (4, 4))
    # noinspection PyUnresolvedReferences
    rhythms['clave32'] = MonophonicRhythm.create.from_string("--x-x---x--x--x-", (4, 4))
    # noinspection PyUnresolvedReferences
    rhythms['clave23'] = MonophonicRhythm.create.from_string("x--x---x--x-x---", (4, 4))

    mono_feature_extractors = tuple(cls(unit=Unit.EIGHTH) for cls in MonophonicRhythmFeatureExtractor.__subclasses__())
    n_iters_per_representation = N_ITER_PER_REPRESENTATION_PER_RHYTHM * len(rhythms)

    print("Monophonic Rhythm Representations Benchmark")
    print("  Beatsearch git commit hash: %s" % get_git_revision_short_hash())
    print("  Number of iterations per representation: %i" % n_iters_per_representation)
    print("")

    t_start_global = time()

    for feature_extractor in mono_feature_extractors:
        t_total_feature_extractor = 0

        for rhythm_name, rhythm in rhythms.items():
            t_start_rhythm = time()

            for i in range(N_ITER_PER_REPRESENTATION_PER_RHYTHM):
                _ = feature_extractor.process(rhythm)

            t_end_rhythm = time()
            t_total_rhythm = t_end_rhythm - t_start_rhythm
            t_average_rhythm = t_total_rhythm / N_ITER_PER_REPRESENTATION_PER_RHYTHM
            t_total_feature_extractor += t_total_rhythm

        t_average_feature_extractor = t_total_feature_extractor / n_iters_per_representation
        print("> %s: %f" % (feature_extractor.__class__.__name__, t_average_feature_extractor))

    t_end_global = time()
    t_total_global = t_end_global - t_start_global

    print("\n")
    print("Done. Total time: %.3g seconds" % t_total_global)

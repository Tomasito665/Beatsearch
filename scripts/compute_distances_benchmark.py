import subprocess
from time import time
from collections import OrderedDict
from beatsearch.app.control import BSController
from beatsearch.metrics import MonophonicRhythmDistanceMeasure
from beatsearch.utils import get_default_beatsearch_rhythms_fpath
measures = MonophonicRhythmDistanceMeasure.get_measures()

ITERATIONS_PER_MEASURE = 15
CORPUS = get_default_beatsearch_rhythms_fpath()


def get_git_revision_short_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").rstrip()


if __name__ == "__main__":
    controller = BSController()
    short_hash = get_git_revision_short_hash()

    print("BeatSearch, commit: %s" % short_hash)
    print("Rhythm corpus: %s (%i rhythms)\n" % (CORPUS, controller.get_rhythm_count()))

    target = controller.get_rhythm_by_index(0)
    controller.set_target_rhythm(target)
    average_compute_times_by_measure = OrderedDict()

    t_start_global = time()

    for name, m in measures.items():
        print("> %s" % name)
        controller.set_distance_measure(m)
        d_compute_times = []

        for i in range(ITERATIONS_PER_MEASURE):
            t_start_d_calculation = time()
            controller.calculate_distances_to_target_rhythm()
            t_end_d_calculation = time()
            t_d_calculation = t_end_d_calculation - t_start_d_calculation
            print("    #%i: %.5f seconds" % (i + 1, t_d_calculation))
            d_compute_times.append(t_d_calculation)

        total_d_calculation_time = sum(d_compute_times)
        average_d_calculation_time = total_d_calculation_time / ITERATIONS_PER_MEASURE
        average_compute_times_by_measure[m] = average_d_calculation_time
        print("    Average: %.5f seconds\n" % average_d_calculation_time)

    measure_class_name_lengths = tuple(len(m.__name__) for m in measures.values())
    longest_class_name_length = max(measure_class_name_lengths)

    print("Average computation times:")
    for m, average_time in average_compute_times_by_measure.items():
        m_name = ("%s:" % m.__name__).ljust(longest_class_name_length + 1)
        print("  - %s %.5f seconds" % (m_name, average_time))

    t_end_global = time()
    print("\nTotal computing time: %.5f seconds" % (t_end_global - t_start_global))

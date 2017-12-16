import os
import sys
import midi
from time import time
from matplotlib import pyplot as plt
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "src"))
from beatsearch.data.rhythmcorpus import RhythmCorpus
from beatsearch.data.rhythm import Unit
from beatsearch.graphics.plot import RhythmPlotter
from beatsearch.utils import print_progress_bar


if __name__ == '__main__':
    try:
        corpus_binary = sys.argv[1]
    except IndexError:
        corpus_binary = "./data/rhythms.pkl"

    try:
        target_dir = sys.argv[2]
    except IndexError:
        target_dir = "./output/notations"
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)

    corpus = RhythmCorpus.load(corpus_binary)
    n_rhythms = len(corpus)

    def p(_func_name, _name, **_args): return _func_name, _name, _args

    single_unit_plots = (
        p('tedas', "tedas_without_quantize", quantize=False),
        p('polygon', "polygon_without_quantize", quantize=False),
        p('spectral', "spectral_without_quantize", quantize=False)
    )

    multi_unit_plots = (
        p('tedas', "tedas_quantized", quantize=True),
        p('polygon', "polygon_quantized", quantize=True),
        p('spectral', "spectral_quantized", quantize=True),
        p('chronotonic', "chronotonic"),
        p('schillinger', "schillinger"),
        p('inter_onset_interval_histogram', "ioi_histogram")
    )

    single_unit = Unit.THIRTYSECOND

    multi_units = {
        '04': Unit.QUARTER,
        '08': Unit.EIGHTH,
        '16': Unit.SIXTEENTH,
        '32': Unit.THIRTYSECOND
    }

    plotter = RhythmPlotter()
    n_steps = \
        n_rhythms * len(single_unit_plots) + \
        n_rhythms * len(multi_units) * len(multi_unit_plots) + \
        n_rhythms  # One MIDI export per rhythm

    starting_time = time()
    step_i, rhythm_i = 0, 0

    for rhythm in corpus:
        rhythm_dir = os.path.join(target_dir, rhythm.name)

        if not os.path.isdir(rhythm_dir):
            os.makedirs(rhythm_dir)

        plotter.unit = single_unit

        def print_progress():
            print_progress_bar(step_i, n_steps, "Rendering rhythm plots",
                               "[%i/%i]" % (rhythm_i, n_rhythms), 2,
                               starting_time=starting_time, fill='O')

        midi_file_path = os.path.join(rhythm_dir, "%s.mid" % rhythm.name)
        midi_pattern = rhythm.to_midi()
        midi.write_midifile(midi_file_path, midi_pattern)

        # noinspection PyShadowingNames
        def plot_and_save(plot_func_name, name, plot_args):
            print_progress()
            figure = getattr(plotter, plot_func_name)(rhythm, **plot_args)
            path = os.path.join(rhythm_dir, name)
            plt.savefig(path)
            plt.close(figure)

        for plot_func_name, name, plot_args in single_unit_plots:
            plot_and_save(plot_func_name, name, plot_args)
            step_i += 1

        for plot_func_name, name, plot_args in multi_unit_plots:
            for unit_name, unit in multi_units.items():
                plotter.unit = unit
                plot_and_save(plot_func_name, "%s_%s" % (name, unit_name), plot_args)
                step_i += 1

        rhythm_i += 1

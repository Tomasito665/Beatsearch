import os
import sys
from create_pickle import create_ascii_spinner, log_replace
from matplotlib import pyplot as plt
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "src"))
from beatsearch.data.rhythmcorpus import RhythmCorpus
from beatsearch.data.rhythm import Unit
from beatsearch.graphics.plot import RhythmPlotter


def plot_and_save(figure, name, output):
    plt.savefig(os.path.join(output, name))
    plt.close(figure)

if __name__ == '__main__':
    try:
        target_dir = sys.argv[1]
    except IndexError:
        raise Exception("No target directory given")

    try:
        corpus_binary = sys.argv[2]
    except IndexError:
        corpus_binary = "./data/rhythms.pkl"

    corpus = RhythmCorpus.load(corpus_binary)
    spinner = create_ascii_spinner("...ooOO@*         ")
    plot = RhythmPlotter('eighths')

    unit_names = {
        Unit.QUARTER: '04',
        Unit.EIGHTH: '08',
        Unit.SIXTEENTH: '16',
        Unit.THIRTYSECOND: '32'
    }

    i = 0
    for rh in corpus:
        d = os.path.join(target_dir, rh.name)

        if not os.path.isdir(d):
            os.makedirs(d)

        log_replace("Rendering notations %ith of %i rhythms... %s" % (i + 1, len(corpus), spinner.next()))

        plot.unit = Unit.THIRTYSECOND

        plot_and_save(plot.tedas(rh, quantize=False), 'tedas_without_quantize', d)
        plot_and_save(plot.polygon(rh, quantize=False), 'polygon_without_quantize', d)
        plot_and_save(plot.spectral(rh, quantize=False), 'spectral_without_quantize', d)

        for unit in [Unit.QUARTER, Unit.EIGHTH, Unit.SIXTEENTH, Unit.THIRTYSECOND]:
            plot.unit = unit
            unit_name = unit_names[unit]
            plot_and_save(plot.tedas(rh, quantize=True), 'tedas_quantized_%s' % unit_name, d)
            plot_and_save(plot.polygon(rh, quantize=True), 'polygon_quantized_%s' % unit_name, d)
            plot_and_save(plot.spectral(rh, quantize=True), 'spectral_quantized_%s' % unit_name, d)
            plot_and_save(plot.chronotonic(rh), 'chronotonic_%s' % unit_name, d)
            plot_and_save(plot.schillinger(rh), 'schillinger_%s' % unit_name, d)
            plot_and_save(plot.inter_onset_interval_histogram(rh), 'inter-onset_interval_histo_%s' % unit_name, d)

        i += 1

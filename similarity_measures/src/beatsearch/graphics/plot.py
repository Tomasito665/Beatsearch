import math
import matplotlib.pyplot as plt
from beatsearch.data.rhythm import convert_time
from beatsearch.data.rhythm import Unit
from itertools import cycle

# make room for the labels
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


# plot method decorator
def plot(plot_type, subplot_layout='combined'):
    plot_colors = ['#b71c1c', '#4A148C', '#1A237E', '#01579B', '#004D40', '#33691E', '#F57F17', '#E65100', '#3E2723']

    def add_axes(figure, plotter, f_setup, *args, **kwargs):
        axes = figure.add_subplot(*args, **kwargs)
        try:
            setup_result = f_setup(plotter, axes)
        except TypeError:
            setup_result = None
        return axes, setup_result

    def get_axes_combined(figure, plotter, f_setup, *_):
        axes, setup_result = add_axes(figure, plotter, f_setup, 111)
        while True:
            yield (axes, setup_result)

    def get_axes_v_stack(figure, plotter, f_setup, n_tracks):
        track_i = 0
        while track_i < n_tracks:
            yield add_axes(figure, plotter, f_setup, n_tracks, 1, track_i + 1)
            track_i += 1

    def get_axes_h_stack(figure, plotter, f_setup, n_tracks):
        track_i = 0
        while track_i < n_tracks:
            yield add_axes(figure, plotter, f_setup, 1, n_tracks, track_i + 1)
            track_i += 1

    get_axes = {
        'combined': get_axes_combined,
        'v_stack': get_axes_v_stack,
        'h_stack': get_axes_h_stack
    }

    class PlotDescriptor(object):

        def __init__(self, f_plot=None, f_setup=None):
            self.f_plot = f_plot
            self.f_setup = f_setup

        def __get__(self, plotter, _=None):
            def wrapper(rhythm, *args, **kwargs):
                title = "%s - %s" % (plot_type, rhythm.name)
                figure = plt.figure(title)
                concrete_unit = to_concrete_unit(plotter.unit, rhythm)

                color_pool = cycle(plot_colors)
                n_tracks = rhythm.track_count()
                track_i = 0

                try:
                    axes_it = get_axes[subplot_layout](figure, plotter, self.f_setup, n_tracks)
                except KeyError:
                    raise ValueError("Unknown subplot layout: %s" % subplot_layout)

                # keep track of track names and plot handles for legend
                track_names = []
                plot_handles = []

                for track_name, track in rhythm.track_iter():
                    axes, setup_result = axes_it.next()
                    handle = self.f_plot(
                        self=plotter,
                        track=track,
                        axes=axes,
                        track_i=track_i,
                        color=color_pool.next(),
                        concrete_unit=concrete_unit,
                        setup_result=setup_result,
                        n_pulses=convert_time(rhythm.duration, rhythm.ppq, concrete_unit),
                        *args, **kwargs
                    )[0]
                    axes.set_xlabel(plotter.unit)
                    track_names.append(track_name)
                    plot_handles.append(handle)
                    track_i += 1

                figure.legend(plot_handles, track_names, loc="center right")
                plt.draw()
                return figure

            wrapper.__name__ = self.f_plot.__name__
            return wrapper

        def plot(self, f_plot):
            return type(self)(f_plot, self.f_setup)

        def setup(self, f_setup):
            return type(self)(self.f_plot, f_setup)

    return PlotDescriptor


class RhythmPlotter(object):
    def __init__(self, unit='ticks', background_color='#38302E', foreground_color='#6F6866', accent_color='#DB5461'):
        self._color = {
            'background': background_color,
            'foreground': foreground_color,
            'accent': accent_color
        }
        self._unit = None
        self.unit = unit

    @property
    def unit(self):
        return self._unit

    @unit.setter
    def unit(self, unit):
        if unit != 'ticks' and not Unit.exists(unit):
            raise ValueError("Unknown unit: %s" % str(unit))
        self._unit = unit

    @plot('Schillinger Rhythm Notation')
    def schillinger(self, track, **kwargs):
        axes = kwargs['axes']
        axes.yaxis.set_ticklabels([])
        axes.yaxis.set_visible(False)

        # each schillinger chain is drawn on a different vertical position
        lo_y = kwargs['track_i'] + kwargs['track_i'] * 0.25
        hi_y = lo_y + 1.0

        # compute schillinger chain and plot it
        schillinger_chain = track.get_binary_schillinger_chain(kwargs['concrete_unit'], (lo_y, hi_y))
        return axes.plot(schillinger_chain, drawstyle='steps-pre', color=kwargs['color'], linewidth=2.5)

    @plot("Chronotonic notation")
    def chronotonic(self, track, **kwargs):
        chronotonic_chain = track.get_chronotonic_chain(kwargs['concrete_unit'])
        return kwargs['axes'].plot(chronotonic_chain, '--.', color=kwargs['color'])

    @plot("Polygon notation", subplot_layout='combined')
    def polygon(self, track, quantize=False, pulse_circle_ratio=0.05, pulse_circle_color='black', **kwargs):
        axes = kwargs['axes']

        # disable labels
        axes.xaxis.set_visible(False)
        axes.yaxis.set_visible(False)
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticklabels([])

        # get main circle size
        main_center = kwargs['setup_result'].center
        main_radius = kwargs['setup_result'].radius

        if self.unit != 'ticks':
            pulse_circle_styles = {
                'radius': main_radius * pulse_circle_ratio,
                'linewidth': kwargs['setup_result'].get_linewidth() * 0.75,
                'edgecolor': pulse_circle_color,
                'facecolor': 'white'
            }

            for i in range(int(kwargs['n_pulses'])):
                relative_t = float(i) / kwargs['n_pulses']
                pulse_circle_pos = get_coordinates_on_circle(main_center, main_radius, relative_t)
                pulse_circle = plt.Circle(pulse_circle_pos, **pulse_circle_styles)
                axes.add_artist(pulse_circle)

        # retrieve onset times
        onset_times = track.get_onset_times(kwargs['concrete_unit'], quantize=quantize)

        # coordinates of line end points
        coordinates_x = []
        coordinates_y = []

        for t in onset_times:
            relative_t = float(t) / kwargs['n_pulses']
            x, y, = get_coordinates_on_circle(main_center, main_radius, relative_t)
            coordinates_x.append(x)
            coordinates_y.append(y)

        # add first coordinate at the end to close the shape
        coordinates_x.append(coordinates_x[0])
        coordinates_y.append(coordinates_y[0])

        # plot the lines on the circle
        return axes.plot(coordinates_x, coordinates_y, '-o', color=kwargs['color'])

    @polygon.setup
    def polygon(self, axes):
        axes.axis('equal')
        axes.axis([0, 1, 0, 1])

        # add base rhythm circle
        position = 0.5, 0.5
        radius = 0.3

        circle = plt.Circle(position, radius, facecolor='white', edgecolor=(0, 0, 0, 0), linewidth=4.0)
        axes.add_artist(circle)
        return circle

    @plot("Spectral notation", subplot_layout='v_stack')
    def spectral(self, track, quantize=False, **kwargs):
        axes = kwargs['axes']
        axes.xaxis.set_visible(False)
        axes.xaxis.set_ticklabels([])

        # compute inter onset intervals and draw bars
        concrete_unit = kwargs['concrete_unit']
        inter_onsets = track.get_post_note_inter_onset_intervals(concrete_unit, quantize=quantize)
        return axes.bar(range(len(inter_onsets)), inter_onsets, width=0.95, color=kwargs['color'])

    @plot("TEDAS Notation", subplot_layout='v_stack')
    def tedas(self, track, quantize=False, **kwargs):
        axes = kwargs['axes']
        axes.xaxis.set_visible(False)
        axes.xaxis.set_ticklabels([])

        concrete_unit = kwargs['concrete_unit']
        inter_onsets = track.get_post_note_inter_onset_intervals(concrete_unit, quantize=quantize)
        onset_times = track.get_onset_times(concrete_unit, quantize=quantize)

        styles = {
            'edgecolor': kwargs['color'],
            'facecolor': (0, 0, 0, 0),
            'linewidth': 2.0
        }

        return axes.bar(onset_times, inter_onsets, width=inter_onsets, align='edge', **styles)


def get_coordinates_on_circle(circle_position, circle_radius, x):
    x *= 2.0
    p_x = circle_radius * math.sin(x * math.pi) + circle_position[0]
    p_y = circle_radius * math.cos(x * math.pi) + circle_position[1]
    return p_x, p_y


def to_concrete_unit(unit, rhythm):
    if unit == 'ticks':
        return rhythm.ppq
    return unit

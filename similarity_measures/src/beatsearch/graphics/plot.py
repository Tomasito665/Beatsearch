import math
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from beatsearch.data.rhythm import concretize_unit
from beatsearch.data.rhythm import Unit
from itertools import cycle

# make room for the labels
from matplotlib import rcParams

from beatsearch.utils import merge_dicts

rcParams.update({'figure.autolayout': True})


# plot method decorator
def plot(plot_type, subplot_layout='combined', share_axis=None, **decorator_kwargs):
    plot_colors = ['#b71c1c', '#4A148C', '#1A237E', '#01579B', '#004D40', '#33691E', '#F57F17', '#E65100', '#3E2723']

    share_axis_arg_options = {
        None: (False, False),
        'both': (True, True),
        'x': (True, False),
        'y': (False, True)
    }

    try:
        share_x, share_y = share_axis_arg_options[share_axis]  # parse share_axis argument
    except KeyError:
        raise ValueError("Unknown axis '%s', choose between %s" % (share_axis, share_axis_arg_options.keys()))

    # helper function to add a subplot and setup the
    # axes with the given setup function
    def add_axes(figure, f_setup, f_setup_kwargs=None, *args, **kwargs):
        axes = figure.add_subplot(*args, **kwargs)
        if f_setup_kwargs is None:
            f_setup_kwargs = {}
        if callable(f_setup):
            setup_result = f_setup(axes=axes, **f_setup_kwargs)
        else:
            setup_result = None
        return axes, setup_result

    def get_axes_combined_iterator(figure, f_setup, f_setup_kwargs=None, *_, **__):
        axes, setup_result = add_axes(figure, f_setup, f_setup_kwargs, 111)
        while True:
            yield (axes, setup_result)

    def create_get_stacked_axes_iterator(direction='horizontal'):
        def get_stacked_axes_iterator(figure, f_setup, f_setup_kwargs=None, *_, **kwargs):
            n_tracks = kwargs['n_tracks']
            track_i = 0
            prev_subplot = None

            while track_i < n_tracks:

                # NOTE: This will break as soon as the argument list (or order) of add_axes changes
                add_axes_args = [figure, f_setup, f_setup_kwargs]

                if direction == 'horizontal':
                    add_axes_args.extend([1, n_tracks, track_i + 1])
                elif direction == 'vertical':
                    add_axes_args.extend([n_tracks, 1, track_i + 1])
                else:
                    raise ValueError("Unknown direction: %s" % direction)

                add_axes_kwargs = {
                    'sharex': prev_subplot if share_x else None,
                    'sharey': prev_subplot if share_y else None
                }

                add_axes_result = add_axes(*add_axes_args, **add_axes_kwargs)
                yield add_axes_result

                prev_subplot = add_axes_result[0]
                track_i += 1

        return get_stacked_axes_iterator

    get_axes = {
        'combined': get_axes_combined_iterator,
        'v_stack': create_get_stacked_axes_iterator(direction='vertical'),
        'h_stack': create_get_stacked_axes_iterator(direction='horizontal')
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

                setup_kwargs = merge_dicts(decorator_kwargs, {
                    'self': plotter,
                    'concrete_unit': concrete_unit,
                    'n_pulses': int(math.ceil(rhythm.get_duration(concrete_unit)))
                })

                try:
                    axes_it = get_axes[subplot_layout](figure, self.f_setup,
                                                       f_setup_kwargs=setup_kwargs, n_tracks=n_tracks)
                except KeyError:
                    raise ValueError("Unknown subplot layout: %s" % subplot_layout)

                # keep track of track names and plot handles for legend
                track_names = []
                plot_handles = []

                for track_name, track in rhythm.track_iter():
                    axes, setup_result = axes_it.next()  # get axes for next subplot

                    handle = self.f_plot(
                        track=track,
                        axes=axes,
                        track_i=track_i,
                        color=color_pool.next(),
                        setup_result=setup_result,
                        *args, **merge_dicts(kwargs, setup_kwargs)
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

        # plot musical grid
        self._plot_rhythm_grid(axes, track, kwargs['concrete_unit'])

        # compute schillinger chain and plot it
        schillinger_chain = track.get_binary_schillinger_chain(kwargs['concrete_unit'], (lo_y, hi_y))
        return axes.plot(schillinger_chain, drawstyle='steps-pre', color=kwargs['color'], linewidth=2.5)

    @plot("Chronotonic notation")
    def chronotonic(self, track, **kwargs):
        axes = kwargs['axes']
        chronotonic_chain = track.get_chronotonic_chain(kwargs['concrete_unit'])
        self._plot_rhythm_grid(axes, track, kwargs['concrete_unit'])
        return axes.plot(chronotonic_chain, '--.', color=kwargs['color'])

    @plot("Polygon notation", subplot_layout='combined', pulse_circle_ratio=0.05, pulse_circle_color='black', max_pulse_circle_count=16)
    def polygon(self, track, quantize=False, **kwargs):
        axes = kwargs['axes']
        n_pulses = kwargs['n_pulses']

        # disable labels
        axes.xaxis.set_visible(False)
        axes.yaxis.set_visible(False)
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticklabels([])

        # get main circle size and position
        main_center = kwargs['setup_result'].center
        main_radius = kwargs['setup_result'].radius

        # retrieve onset times
        onset_times = track.get_onset_times(kwargs['concrete_unit'], quantize=quantize)

        # coordinates of line end points
        coordinates_x = []
        coordinates_y = []

        for t in onset_times:
            relative_t = float(t) / n_pulses
            x, y, = get_coordinates_on_circle(main_center, main_radius, relative_t)
            coordinates_x.append(x)
            coordinates_y.append(y)

        # add first coordinate at the end to close the shape
        coordinates_x.append(coordinates_x[0])
        coordinates_y.append(coordinates_y[0])

        # plot the lines on the circle
        return axes.plot(coordinates_x, coordinates_y, '-o', color=kwargs['color'])

    @polygon.setup
    def polygon(self, axes, **kwargs):
        # avoid stretching the aspect ratio
        axes.axis('equal')
        axes.axis([0, 1, 0, 1])

        # add base rhythm circle
        main_radius = 0.3
        main_center = 0.5, 0.5
        circle = plt.Circle(main_center, main_radius, color=(0, 0, 0, 0))  # circle is transparent for now
        axes.add_artist(circle)

        # if the unit is in ticks, don't bother drawing the pulses (since there would
        # be way to many pulses to draw, that would be a mess)
        if self.unit == 'ticks':
            return circle

        max_pulse_circle_count = kwargs['max_pulse_circle_count']
        n_pulses = kwargs['n_pulses']
        draw_pulse_circles = (n_pulses <= max_pulse_circle_count)

        pulse_circle_styles = {
            'radius': main_radius * 0.05,
            'linewidth': 2.5,
            'facecolor': 'white',
            'edgecolor': 'black'
        }

        for i in range(n_pulses):
            relative_t = float(i) / n_pulses
            pos_on_circle = get_coordinates_on_circle(main_center, main_radius, relative_t)

            if draw_pulse_circles:
                pulse_circle = plt.Circle(pos_on_circle, **pulse_circle_styles)
                axes.add_artist(pulse_circle)
            else:
                axes.plot(
                    [main_center[0], pos_on_circle[0]],
                    [main_center[1], pos_on_circle[1]],
                    color='gray',
                    linewidth=0.2,
                    alpha=0.35
                )

        return circle

    @plot("Spectral notation", subplot_layout='v_stack', share_axis='x')
    def spectral(self, track, quantize=False, **kwargs):
        axes = kwargs['axes']
        axes.xaxis.set_visible(False)
        axes.xaxis.set_ticklabels([])

        # compute inter onset intervals and draw bars
        concrete_unit = kwargs['concrete_unit']
        inter_onsets = track.get_post_note_inter_onset_intervals(concrete_unit, quantize=quantize)
        return axes.bar(range(len(inter_onsets)), inter_onsets, width=0.95, color=kwargs['color'])

    @plot("TEDAS Notation", subplot_layout='v_stack', share_axis='x')
    def tedas(self, track, quantize=False, **kwargs):
        axes = kwargs['axes']
        concrete_unit = kwargs['concrete_unit']
        inter_onsets = track.get_post_note_inter_onset_intervals(concrete_unit, quantize=quantize)
        onset_times = track.get_onset_times(concrete_unit, quantize=quantize)

        styles = {
            'edgecolor': kwargs['color'],
            'facecolor': colors.to_rgba(kwargs['color'], 0.18),
            'linewidth': 2.0
        }

        self._plot_rhythm_grid(axes, track, kwargs['concrete_unit'], )
        return axes.bar(onset_times, inter_onsets, width=inter_onsets, align='edge', **styles)

    @staticmethod
    @concretize_unit(lambda ax, rhythm, *args, **kw: rhythm)
    def _plot_rhythm_grid(axes, track, unit, axis='x'):
        rhythm = track.rhythm

        duration = rhythm.get_duration(unit)
        measure_duration = rhythm.get_measure_duration(unit)
        beat_duration = rhythm.get_beat_duration(unit)

        measure_grid_ticks = np.arange(0, duration + 1, measure_duration)
        beat_grid_ticks = np.arange(0, duration + 1, beat_duration)

        if len(axis) > 2:
            raise ValueError("Illegal axis: %s" % axis)

        axes.set_xticks(measure_grid_ticks)
        axes.set_xticks(beat_grid_ticks, minor=True)
        axes.set_yticks(measure_grid_ticks)
        axes.set_yticks(beat_grid_ticks, minor=True)

        axes.set_axisbelow(True)
        axes.grid(which='minor', alpha=0.2, axis=axis)
        axes.grid(which='major', alpha=0.5, axis=axis)


def get_coordinates_on_circle(circle_position, circle_radius, x):
    x *= 2.0
    p_x = circle_radius * math.sin(x * math.pi) + circle_position[0]
    p_y = circle_radius * math.cos(x * math.pi) + circle_position[1]
    return p_x, p_y


# TODO replace whatever calls this function with @concretize_unit decorator
def to_concrete_unit(unit, rhythm):
    if unit == 'ticks':
        return rhythm.get_resolution()
    return unit

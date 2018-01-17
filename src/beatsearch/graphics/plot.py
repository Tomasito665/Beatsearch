import math
from functools import wraps

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import typing as tp
from beatsearch.rhythm import concretize_unit, Unit, RhythmLoop, Rhythm, Track
from beatsearch.utils import merge_dicts
from itertools import cycle

# make room for the labels
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


# noinspection PyPep8Naming
class plot(object):
    """
    Decorator for RhythmLoopPlotter plotting methods.
    """

    # http://colorbrewer2.org/#type=diverging&scheme=Spectral&n=6
    colors = ['#d53e4f', '#fc8d59', '#fee08b', '#e6f598', '#99d594', '#3288bd']

    def __init__(self, title, subplot_layout='combined', share_axis=None, *_, **kwargs):
        self._title = title
        self._subplot_layout = subplot_layout
        self._share_x, self._share_y = plot.parse_axis_arg_str(share_axis)
        self._named_decorator_args = kwargs
        self._f_fill_subplot = None
        self._f_setup_subplot = None
        self._rhythm_plotter = None
        self.draw = None  # type: tp.Union[tp.Callable, None]

    class Descriptor(object):
        """
        This class provides the 'plot' decorator of a handle to the RhythmLoopPlotter instance and it enables sets the
        function pointer to the subplot setup function for methods decorated with @foo.setup. This enables the following
        syntax:

            @plot()
            def polygon(axes):
                ...

            @polygon.setup(axes):
                ...
        """

        def __init__(self, obj_decorator):
            self.obj_decorator = obj_decorator

        def setup(self, f_setup_subplot):
            self.obj_decorator._f_setup_subplot = f_setup_subplot
            return self

        def __get__(self, obj, obj_type):
            if obj_type != RhythmLoopPlotter:
                raise ValueError("Expected a RhythmLoopPlotter object but got a '%s'" % str(obj))
            self.obj_decorator._rhythm_plotter = obj
            return self.obj_decorator.draw

    def __call__(self, f_fill_subplot, *args, **kwargs):
        self._f_fill_subplot = f_fill_subplot

        @wraps(self._draw)
        def draw_f_wrapper(*args_, **kwargs_):
            self._draw(*args_, **kwargs_)

        # noinspection PyUnresolvedReferences
        draw_f_wrapper.__doc__ = self._draw.__doc__.format(self._title)
        self.draw = draw_f_wrapper

        return plot.Descriptor(obj_decorator=self)

    def _draw(
            self,
            rhythm_loop: RhythmLoop,
            figure: tp.Union[plt.Figure, None] = None,
            figure_kwargs: tp.Dict[str, tp.Any] = None,
            legend_kwargs: tp.Dict[str, tp.Any] = None,
            *args, **kwargs
    ):
        """
        Plots the {0} of the given drum loop on a matplotlib figure and returns the figure object.

        :param rhythm_loop: the rhythm loop to plot
        :param figure: figure to draw the loop
        :param figure_kwargs: keyword arguments for the creation of the figure. This argument is ignored if a custom
                              figure has been provided
        :param legend_kwargs: keyword arguments for the call to Figure.legend
        :return: matplotlib figure object
        """

        track_iterator = rhythm_loop.get_track_iterator()

        # the figure to add the subplot(s) to
        figure = figure or plt.figure("%s - %s" % (self._title, rhythm_loop.name), **(figure_kwargs or {}))

        rhythm_plotter = self._rhythm_plotter
        concrete_unit = to_concrete_unit(rhythm_plotter.unit, rhythm_loop)

        # the setup method will also receive the args given to the decorator
        setup_args = merge_dicts(self._named_decorator_args, {
            'self': rhythm_plotter,
            'rhythm': rhythm_loop,
            'concrete_unit': concrete_unit,
            'n_pulses': int(math.ceil(rhythm_loop.get_duration(concrete_unit))),
            'n_tracks': rhythm_loop.get_track_count()
        })

        # this iterator will return both an axes object and a setup result object at each iteration
        subplots = self._get_subplot_iterator(figure, setup_args)

        # this iterator will cycle through the plot colors
        color_pool = cycle(plot.colors)

        # decorated function
        fill_subplot = self._f_fill_subplot

        # keep track of track names and plot handles for legend
        track_names = []
        plot_handles = []

        track_i = 0
        for track in track_iterator:
            axes, axes_setup_result = next(subplots)

            handle = fill_subplot(
                track=track,
                axes=axes,
                track_i=track_i,
                color=next(color_pool),
                setup_result=axes_setup_result,
                *args, **merge_dicts(kwargs, setup_args)  # TODO merge_dicts not necessary anymore now with Python 3
            )[0]

            axes.set_xlabel(rhythm_plotter.unit)
            track_names.append(track.name)
            plot_handles.append(handle)
            track_i += 1

        figure.legend(plot_handles, track_names, loc="center right", **(legend_kwargs or {}))
        plt.draw()  # draw axes on figure

        return figure

    def _get_subplot_iterator(self, figure, setup_args):
        layout = self._subplot_layout

        if layout == 'combined':
            subplot_iterator_creator = self._combined_subplot_iterator
        elif layout == 'v_stack':
            subplot_iterator_creator = self._create_stacked_subplot_iterator(direction='vertical')
        elif layout == 'h_stack':
            subplot_iterator_creator = self._create_stacked_subplot_iterator(direction='horizontal')
        else:
            raise ValueError("Unknown subplot layout: '%s'" % layout)

        return subplot_iterator_creator(figure, setup_args)

    def _add_and_setup_subplot(self, figure, setup_args, *args, **kwargs):
        axes = figure.add_subplot(*args, **kwargs)
        f_setup = self._f_setup_subplot
        if callable(f_setup):
            setup_result = f_setup(axes=axes, **setup_args)
        else:
            setup_result = None
        return axes, setup_result

    def _combined_subplot_iterator(self, figure, setup_args):
        res = self._add_and_setup_subplot(figure, setup_args, 111)
        while True:
            yield res

    def _create_stacked_subplot_iterator(self, direction='horizontal'):
        subplot_args_by_direction = {
            'horizontal': lambda n_tracks, track_i: [1, n_tracks, track_i + 1],
            'vertical': lambda n_tracks, track_i: [n_tracks, 1, track_i + 1]
        }

        if direction not in subplot_args_by_direction:
            raise ValueError("Unknown direction: '%s'" % direction)

        def get_stacked_axes_iterator(figure, setup_args):
            n_tracks = setup_args['n_tracks']
            track_i = 0
            prev_subplot = None

            while track_i < n_tracks:
                args = [figure, setup_args] + subplot_args_by_direction[direction](n_tracks, track_i)
                kwargs = {
                    'sharex': prev_subplot if self._share_x else None,
                    'sharey': prev_subplot if self._share_y else None
                }
                res = self._add_and_setup_subplot(*args, **kwargs)
                yield res
                prev_subplot = res[0]
                track_i += 1

        return get_stacked_axes_iterator

    @staticmethod
    def parse_axis_arg_str(axis_arg_str):
        axis_arg_options = {
            None: (False, False),
            'both': (False, False),
            'x': (True, False),
            'y': (False, True)
        }

        try:
            return axis_arg_options[axis_arg_str]
        except KeyError:
            raise ValueError("Unknown axis '%s', choose between %s" % (axis_arg_str, list(axis_arg_options.keys())))


class RhythmLoopPlotter(object):
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

    @plot("Schillinger Rhythm Notation")
    def schillinger(self, track: Track, **kwargs):
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
    def chronotonic(self, track: Track, **kwargs):
        axes = kwargs['axes']
        chronotonic_chain = track.get_chronotonic_chain(kwargs['concrete_unit'])
        self._plot_rhythm_grid(axes, track, kwargs['concrete_unit'])
        return axes.plot(chronotonic_chain, '--.', color=kwargs['color'])

    @plot("Polygon notation", subplot_layout='combined', max_pulse_circle_count=16)
    def polygon(self, track: Track, quantize=False, **kwargs):
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
    def polygon(self, **kwargs):
        # avoid stretching the aspect ratio
        axes = kwargs['axes']
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
        unit = kwargs['concrete_unit']
        n_pulses = kwargs['n_pulses']
        measure_duration = kwargs['rhythm'].get_measure_duration(unit)
        draw_pulse_circles = (n_pulses <= max_pulse_circle_count)

        pulse_circle_styles = {
            'radius': main_radius * 0.05,
            'linewidth': 2.5,
            'facecolor': 'white',
            'edgecolor': 'black'
        }

        for i in range(n_pulses):
            is_down_beat = i % measure_duration == 0
            relative_t = float(i) / n_pulses
            pos_on_circle = get_coordinates_on_circle(main_center, main_radius, relative_t)

            if draw_pulse_circles:
                pulse_circle = plt.Circle(pos_on_circle, **pulse_circle_styles)
                axes.add_artist(pulse_circle)

            axes.plot(
                [main_center[0], pos_on_circle[0]],
                [main_center[1], pos_on_circle[1]],
                color='gray',
                linewidth=1 if is_down_beat else 0.3,
                alpha=0.9 if is_down_beat else 0.5
            )

        return circle

    @plot("Spectral notation", subplot_layout='v_stack', share_axis='x')
    def spectral(self, track: Track, quantize=False, **kwargs):
        axes = kwargs['axes']
        axes.xaxis.set_visible(False)
        axes.xaxis.set_ticklabels([])

        # compute inter onset intervals and draw bars
        concrete_unit = kwargs['concrete_unit']
        inter_onsets = track.get_post_note_inter_onset_intervals(concrete_unit, quantize=quantize)
        return axes.bar(list(range(len(inter_onsets))), inter_onsets, width=0.95, color=kwargs['color'])

    @plot("TEDAS Notation", subplot_layout='v_stack', share_axis='x')
    def tedas(self, track: Track, quantize=False, **kwargs):
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

    @plot("Inter-onset interval histogram")
    def inter_onset_interval_histogram(self, track: Track, **kwargs):
        axes = kwargs['axes']
        occurrences, interval_durations = track.get_interval_histogram(kwargs['concrete_unit'])
        return axes.bar(interval_durations, occurrences, color=kwargs['color'])

    @staticmethod
    @concretize_unit(lambda ax, rhythm, *args, **kw: rhythm)
    def _plot_rhythm_grid(axes, rhythm: Rhythm, unit, axis='x'):
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

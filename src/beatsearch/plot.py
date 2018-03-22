import math
import enum
import numpy as np
import typing as tp
import matplotlib.artist
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import to_rgba
from matplotlib.patches import Wedge
from abc import ABCMeta, abstractmethod
from itertools import cycle, repeat
from beatsearch.rhythm import Unit, UnitType, parse_unit_argument, RhythmLoop, Rhythm, Track
from beatsearch.feature_extraction import IOIVector, BinarySchillingerChain, \
    RhythmFeatureExtractor, ChronotonicChain, OnsetPositionVector, IOIHistogram
from beatsearch.utils import Quantizable

# make room for the labels
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


class SnapsToGridPolicy(enum.Enum):
    ALWAYS = enum.auto()
    NEVER = enum.auto()
    ADJUSTABLE = enum.auto()
    NOT_APPLICABLE = enum.auto()


def get_coordinates_on_circle(circle_position, circle_radius, x):
    x *= 2.0
    p_x = circle_radius * math.sin(x * math.pi) + circle_position[0]
    p_y = circle_radius * math.cos(x * math.pi) + circle_position[1]
    return p_x, p_y


def plot_rhythm_grid(axes: plt.Axes, rhythm: Rhythm, unit: Unit, axis='x'):  # TODO fix axis option
    duration = rhythm.get_duration(unit, ceil=True)
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


class Orientation(enum.Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class SubplotLayout(object, metaclass=ABCMeta):
    def __init__(self, share_axis=None):
        self.share_x, self.share_y = self.parse_axis_arg_str(share_axis)

    @abstractmethod
    def inflate(self, figure: plt.Figure, n_subplots: int, subplot_kwargs: tp.Dict[str, tp.Any]) \
            -> tp.Iterable[plt.Axes]:
        """
        Should create and add subplots to the given figure and return the subplot axes. Implementations of this method
        should pay attention to the share_x and share_y attributes and link the axes of the subplots accordingly.

        :param figure: figure to add the subplots to
        :param n_subplots: subplot count
        :param subplot_kwargs: named arguments to pass to the figure.add_subplot method
        :return: iterable of axes
        """

        raise NotImplementedError

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


class CombinedSubplotLayout(SubplotLayout):
    def inflate(self, figure: plt.Figure, n_subplots: int, subplot_kwargs: tp.Dict[str, tp.Any]) \
            -> tp.Iterable[plt.Axes]:
        axes = figure.add_subplot(111, **subplot_kwargs)
        return repeat(axes, n_subplots)


class StackedSubplotLayout(SubplotLayout):
    __subplot_args_by_orientation = {
        Orientation.HORIZONTAL: lambda n, i: [1, n, i + 1],
        Orientation.VERTICAL: lambda n, i: [n, 1, i + 1]
    }

    def __init__(self, orientation: Orientation = Orientation.VERTICAL, share_axis: tp.Optional[str] = "both"):
        super().__init__(share_axis)

        self.check_orientation(orientation)
        self._orientation = orientation  # type: Orientation

    def inflate(self, figure: plt.Figure, n_subplots: int, subplot_kwargs: tp.Dict[str, tp.Any]) \
            -> tp.Iterable[plt.Axes]:

        orientation = self._orientation
        get_subplot_positional_args = self.__subplot_args_by_orientation[orientation]
        prev_subplot = None

        for subplot_ix in range(n_subplots):
            subplot_positional_args = get_subplot_positional_args(n_subplots, subplot_ix)

            # noinspection SpellCheckingInspection
            subplot = figure.add_subplot(*subplot_positional_args, **{
                **subplot_kwargs,
                'sharex': prev_subplot if self.share_x else None,
                'sharey': prev_subplot if self.share_y else None
            })

            yield subplot
            prev_subplot = subplot

    @classmethod
    def check_orientation(cls, orientation: Orientation):
        if orientation not in cls.__subplot_args_by_orientation:
            raise ValueError("Unknown orientation: %s" % str(orientation))


class RhythmLoopPlotter(object, metaclass=ABCMeta):
    # http://colorbrewer2.org/#type=diverging&scheme=Spectral&n=6
    COLORS = ['#d53e4f', '#fc8d59', '#fee08b', '#e6f598', '#99d594', '#3288bd']

    def __init__(
            self, unit: UnitType,
            subplot_layout: SubplotLayout,
            feature_extractors: tp.Optional[tp.Dict[str, RhythmFeatureExtractor]] = None,
            snap_to_grid_policy: SnapsToGridPolicy = SnapsToGridPolicy.NOT_APPLICABLE,
            snaps_to_grid: bool = None
    ):
        self._subplot_layout = subplot_layout            # type: SubplotLayout
        self._feature_extractors = feature_extractors or dict()  # type: tp.Dict[str, RhythmFeatureExtractor]
        self._unit = None                                # type: Unit
        self._snaps_to_grid = None                       # type: bool
        self._snap_to_grid_policy = snap_to_grid_policy  # type: SnapsToGridPolicy

        if snaps_to_grid is None:
            if snap_to_grid_policy in [SnapsToGridPolicy.NEVER, SnapsToGridPolicy.ADJUSTABLE]:
                snaps_to_grid = False
            elif snap_to_grid_policy == SnapsToGridPolicy.ALWAYS:
                snaps_to_grid = True

        self.snaps_to_grid = snaps_to_grid  # call setter
        self.set_unit(unit)

    @parse_unit_argument
    def set_unit(self, unit: UnitType) -> None:
        # bind unit of feature extractors to rhythm loop plotter unit
        for feature_extractor in self._feature_extractors.values():
            feature_extractor.set_unit(unit)
        self._unit = unit

    def get_unit(self) -> Unit:
        return self._unit

    @property
    def unit(self) -> Unit:
        return self.get_unit()

    @unit.setter
    def unit(self, unit: UnitType):
        self.set_unit(unit)

    @property
    def snap_to_grid_policy(self):
        return self._snap_to_grid_policy

    @property
    def snaps_to_grid(self) -> bool:
        return self._snaps_to_grid

    @snaps_to_grid.setter
    def snaps_to_grid(self, snaps_to_grid):
        policy = self._snap_to_grid_policy

        if policy == SnapsToGridPolicy.NOT_APPLICABLE:
            if snaps_to_grid is not None:
                raise RuntimeError("Snaps to grid is not applicable for %s" % self.__class__.__name__)
            return

        if policy != SnapsToGridPolicy.ADJUSTABLE:
            if policy == SnapsToGridPolicy.ALWAYS:
                assert snaps_to_grid
            elif policy == SnapsToGridPolicy.NEVER:
                assert not snaps_to_grid

        snaps_to_grid = bool(snaps_to_grid)
        self._snaps_to_grid = snaps_to_grid

        # update quantizable feature extractors
        for quantizable_extractor in (ext for ext in self._feature_extractors.values() if isinstance(ext, Quantizable)):
            quantizable_extractor.set_quantize_enabled(snaps_to_grid)

    def draw(
            self,
            rhythm_loop: RhythmLoop,
            figure: tp.Optional[plt.Figure] = None,
            figure_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
            legend_kwargs: tp.Dict[str, tp.Any] = None,
            subplot_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ) -> plt.Figure:
        """
        Plots the the given drum loop on a matplotlib figure and returns the figure object.

        :param rhythm_loop: the rhythm loop to plot
        :param figure: figure to draw the loop
        :param figure_kwargs: keyword arguments for the creation of the figure. This argument is ignored if a custom
                              figure has been provided
        :param subplot_kwargs: keyword arguments for the call to Figure.add_subplot
        :param legend_kwargs: keyword arguments for the call to Figure.legend
        :return: matplotlib figure object
        """

        unit = self.get_unit()
        n_tracks = rhythm_loop.get_track_count()

        # the figure to add the subplot(s) to
        plot_type_name = self.get_plot_type_name()
        figure = figure or plt.figure("%s - %s" % (plot_type_name, rhythm_loop.name), **(figure_kwargs or {}))

        # add subplots to figure
        subplots = iter(self._subplot_layout.inflate(figure, n_tracks, subplot_kwargs or dict()))
        color_pool = cycle(self.COLORS)
        prev_subplot, subplot_setup_ret = None, None

        # named arguments given both to __setup_subplot__ and __draw_track__
        common_kwargs = {
            'n_pulses': rhythm_loop.get_duration(unit, ceil=True),
            'n_tracks': rhythm_loop.get_track_count()
        }

        # keep track of track names and plot handles for legend
        track_names = []
        subplots_handles = []

        for ix, track in enumerate(rhythm_loop.get_track_iterator()):
            subplot = next(subplots)

            # don't setup subplot if we already did
            if subplot != prev_subplot:
                subplot_setup_ret = self.__setup_subplot__(rhythm_loop, subplot, **common_kwargs)

            # draw the track on the subplot
            draw_ret = self.__draw_track__(
                track, subplot, track_ix=ix, color=next(color_pool),
                setup_ret=subplot_setup_ret, **common_kwargs
            )

            if isinstance(draw_ret, matplotlib.artist.Artist):
                handle = draw_ret
            elif isinstance(draw_ret[0], matplotlib.artist.Artist):
                handle = draw_ret[0]
            else:
                raise TypeError("__draw_track__ must return either a single artist or a non-empty list of artists")

            track_names.append(track.name)
            subplots_handles.append(handle)
            prev_subplot = subplot

        legend_kwargs = {
            'loc': "center right",
            **(legend_kwargs or {}),
            **self.get_legend_kwargs()
        }

        figure.legend(subplots_handles, track_names, **legend_kwargs)
        return figure

    def get_feature_extractor(self, feature_extractor_name):
        return self._feature_extractors[feature_extractor_name]

    # noinspection PyMethodMayBeStatic
    def get_legend_kwargs(self):
        """Override this method to add extra named arguments to the figure.legend() function call in draw()"""

        return {}

    @classmethod
    @abstractmethod
    def get_plot_type_name(cls):
        raise NotImplementedError

    @abstractmethod
    def __setup_subplot__(self, rhythm_loop: RhythmLoop, axes: plt.Axes, **kw):
        raise NotImplementedError

    @abstractmethod
    def __draw_track__(self, rhythm_track: Track, axes: plt.Axes, **kw):
        raise NotImplementedError


class SchillingerRhythmNotation(RhythmLoopPlotter):
    PLOT_TYPE_NAME = "Schillinger rhythm notation"

    def __init__(self, unit: UnitType = Unit.EIGHTH):
        super().__init__(
            unit=unit,
            subplot_layout=CombinedSubplotLayout(),
            feature_extractors={'schillinger': BinarySchillingerChain()},
            snap_to_grid_policy=SnapsToGridPolicy.ALWAYS
        )

    def __setup_subplot__(self, rhythm_loop: RhythmLoop, axes: plt.Axes, **kw):
        axes.yaxis.set_ticklabels([])
        axes.yaxis.set_visible(False)
        plot_rhythm_grid(axes, rhythm_loop, self.get_unit())  # plot musical grid

    def __draw_track__(self, rhythm_track: Track, axes: plt.Axes, **kw):
        # each schillinger chain is drawn on a different vertical pos
        lo_y = kw['track_ix'] + kw['track_ix'] * 0.25
        hi_y = lo_y + 1.0

        schillinger_extractor = self.get_feature_extractor("schillinger")  # type: BinarySchillingerChain
        schillinger_extractor.values = (lo_y, hi_y)

        # compute schillinger chain and plot it
        schillinger_chain = schillinger_extractor.process(rhythm_track)
        return axes.plot(schillinger_chain, drawstyle="steps-pre", color=kw['color'], linewidth=2.5)

    @classmethod
    def get_plot_type_name(cls):
        return cls.PLOT_TYPE_NAME


class ChronotonicNotation(RhythmLoopPlotter):
    PLOT_TYPE_NAME = "Chronotonic notation"

    def __init__(self, unit: UnitType = Unit.EIGHTH):
        super().__init__(
            unit=unit,
            subplot_layout=CombinedSubplotLayout(),
            feature_extractors={'chronotonic': ChronotonicChain()},
            snap_to_grid_policy=SnapsToGridPolicy.ALWAYS
        )

    @classmethod
    def get_plot_type_name(cls):
        return cls.PLOT_TYPE_NAME

    def __setup_subplot__(self, rhythm_loop: RhythmLoop, axes: plt.Axes, **kw):
        plot_rhythm_grid(axes, rhythm_loop, self.get_unit())

    def __draw_track__(self, rhythm_track: Track, axes: plt.Axes, **kw):
        chronotonic_chain = self.get_feature_extractor("chronotonic").process(rhythm_track)
        return axes.plot(chronotonic_chain, "--.", color=kw['color'])


class PolygonNotation(RhythmLoopPlotter):
    PLOT_TYPE_NAME = "Polygon notation"

    def __init__(self, unit: UnitType = Unit.EIGHTH):
        super().__init__(
            unit=unit,
            subplot_layout=CombinedSubplotLayout(),
            feature_extractors={'onset_positions': OnsetPositionVector()},
            snap_to_grid_policy=SnapsToGridPolicy.ADJUSTABLE
        )

    @classmethod
    def get_plot_type_name(cls):
        return cls.PLOT_TYPE_NAME

    def __setup_subplot__(self, rhythm_loop: RhythmLoop, axes: plt.Axes, **kw):
        # avoid stretching the aspect ratio
        axes.axis('equal')
        # noinspection PyTypeChecker
        axes.axis([0, 1, 0, 1])

        # add base rhythm circle
        main_radius = 0.3
        main_center = 0.5, 0.5

        # draws a wedge from the given start pulse to the given end pulse
        def draw_wedge(pulse_start, pulse_end, center=main_center, radius=main_radius, **kw_):
            theta_1, theta_2 = (((90 - (pulse / n_pulses * 360)) % 360) for pulse in (pulse_end, pulse_start))
            axes.add_artist(Wedge(center, radius, theta_1, theta_2, **kw_))

        unit = self.get_unit()
        n_pulses = kw['n_pulses']
        n_pulses_per_measure = int(rhythm_loop.get_measure_duration(unit))

        try:
            n_measures = int(n_pulses / n_pulses_per_measure)
        except ZeroDivisionError:
            n_measures = 0

        # measure wedges
        for i_measure in range(0, n_measures, 2):
            from_pulse = i_measure * n_pulses_per_measure
            to_pulse = (i_measure + 1) * n_pulses_per_measure
            draw_wedge(from_pulse, to_pulse, radius=1.0, fc=to_rgba("gray", 0.25))

        # main circle
        circle = plt.Circle(main_center, main_radius, fc="white")
        axes.add_artist(circle)

        # draw the pulse wedges
        for i_pulse in range(0, n_pulses, 2):
            draw_wedge(i_pulse, i_pulse + 1, fc=to_rgba("gray", 0.25))

        return circle

    def __draw_track__(self, rhythm_track: Track, axes: plt.Axes, **kw):
        n_pulses = kw['n_pulses']

        # disable labels
        axes.xaxis.set_visible(False)
        axes.yaxis.set_visible(False)
        axes.xaxis.set_ticklabels([])
        axes.yaxis.set_ticklabels([])

        # get main circle size and position
        main_center = kw['setup_ret'].center
        main_radius = kw['setup_ret'].radius

        # retrieve onset times
        onset_times = self.get_feature_extractor("onset_positions").process(rhythm_track)

        # coordinates of line end points
        coordinates_x = []
        coordinates_y = []

        if n_pulses > 0:
            for t in onset_times:
                relative_t = float(t) / n_pulses
                x, y, = get_coordinates_on_circle(main_center, main_radius, relative_t)
                coordinates_x.append(x)
                coordinates_y.append(y)

            # add first coordinate at the end to close the shape
            coordinates_x.append(coordinates_x[0])
            coordinates_y.append(coordinates_y[0])

        # plot the lines on the circle
        return axes.plot(coordinates_x, coordinates_y, '-o', color=kw['color'])


class SpectralNotation(RhythmLoopPlotter):
    PLOT_TYPE_NAME = "Spectral notation"

    def __init__(self, unit: UnitType = Unit.EIGHTH):
        super().__init__(
            unit=unit,
            subplot_layout=StackedSubplotLayout(Orientation.VERTICAL),
            feature_extractors={'ioi_vector': IOIVector()},
            snap_to_grid_policy=SnapsToGridPolicy.ADJUSTABLE
        )

    @classmethod
    def get_plot_type_name(cls):
        return cls.PLOT_TYPE_NAME

    def __setup_subplot__(self, rhythm_loop: RhythmLoop, axes: plt.Axes, **kw):
        axes.xaxis.set_visible(False)
        axes.xaxis.set_ticklabels([])

    def __draw_track__(self, rhythm_track: Track, axes: plt.Axes, **kw):
        # compute inter onset intervals and draw bars
        ioi_vector = self.get_feature_extractor("ioi_vector").process(rhythm_track)
        return axes.bar(list(range(len(ioi_vector))), ioi_vector, width=0.95, color=kw['color'])


class TEDASNotation(RhythmLoopPlotter):
    PLOT_TYPE_NAME = "TEDAS Notation"

    def __init__(self, unit: UnitType = Unit.EIGHTH):
        super().__init__(
            unit=unit,
            subplot_layout=StackedSubplotLayout(Orientation.VERTICAL),
            feature_extractors={
                'onset_positions': OnsetPositionVector(quantize_enabled=True),
                'ioi_vector': IOIVector(quantize_enabled=True)
            },
            snap_to_grid_policy=SnapsToGridPolicy.ADJUSTABLE
        )

    @classmethod
    def get_plot_type_name(cls):
        return cls.PLOT_TYPE_NAME

    def __setup_subplot__(self, rhythm_loop: RhythmLoop, axes: plt.Axes, **kw):
        plot_rhythm_grid(axes, rhythm_loop, self.get_unit())

    def __draw_track__(self, rhythm_track: Track, axes: plt.Axes, **kw):
        ioi_vector = self.get_feature_extractor("ioi_vector").process(rhythm_track)
        onset_positions = self.get_feature_extractor("onset_positions").process(rhythm_track)

        # noinspection SpellCheckingInspection
        styles = {
            'edgecolor': kw['color'],
            'facecolor': colors.to_rgba(kw['color'], 0.18),
            'linewidth': 2.0
        }

        return axes.bar(onset_positions, ioi_vector, width=ioi_vector, align="edge", **styles)


class IOIHistogramPlot(RhythmLoopPlotter):
    PLOT_TYPE_NAME = "Inter-onset interval histogram"

    def __init__(self, unit: UnitType = Unit.EIGHTH):
        super().__init__(
            unit=unit,
            subplot_layout=CombinedSubplotLayout(),
            feature_extractors={'ioi_histogram': IOIHistogram()},
            snap_to_grid_policy=SnapsToGridPolicy.NOT_APPLICABLE
        )

    @classmethod
    def get_plot_type_name(cls):
        return cls.PLOT_TYPE_NAME

    def __setup_subplot__(self, rhythm_loop: RhythmLoop, axes: plt.Axes, **kw):
        return None

    def __draw_track__(self, rhythm_track: Track, axes: plt.Axes, **kw):
        occurrences, interval_durations = self.get_feature_extractor("ioi_histogram").process(rhythm_track)
        return axes.bar(interval_durations, occurrences, color=kw['color'])


class BoxNotation(RhythmLoopPlotter):
    PLOT_TYPE_NAME = "Box notation"

    def __init__(self, unit: UnitType = Unit.EIGHTH):
        super().__init__(
            unit=unit,
            subplot_layout=CombinedSubplotLayout(),
            feature_extractors={'onset_positions': OnsetPositionVector()},
            snap_to_grid_policy=SnapsToGridPolicy.ALWAYS
        )

        self.line_width = 1
        self.line_color = "black"
        self.rel_padx = 0.25, 0.05
        self.rel_pady = 0.15, 0.15

    @classmethod
    def get_plot_type_name(cls):
        return cls.PLOT_TYPE_NAME

    def __setup_subplot__(self, rhythm_loop: RhythmLoop, axes: plt.Axes, **kw):
        # axes.axis("equal")  # avoid stretching the aspect ratio

        for axis in [axes.xaxis, axes.yaxis]:
            axis.set_ticklabels([])
            axis.set_visible(False)

        line_color = self.line_color
        line_width = self.line_width

        main_width = kw['n_pulses'] + 1
        main_height = kw['n_tracks']

        padx = tuple(rel_pad * main_width for rel_pad in self.rel_padx)
        pady = tuple(rel_pad * main_height for rel_pad in self.rel_pady)

        # reversed Y axis so that 0, 0 is in the upper left corner
        axes.set_ylim(main_height + pady[1], 0 - pady[0])
        axes.set_xlim(0 - padx[0], main_width + padx[1])

        # main rectangle (containing everything but the track names)
        axes.add_artist(plt.Rectangle(
            [0, 0], main_width, main_height, fill=False,
            edgecolor=line_color, linewidth=line_width
        ))

        # add horizontal lines
        for track_ix in range(1, kw['n_tracks']):
            axes.add_artist(plt.Line2D(
                [0, main_width], [track_ix, track_ix],
                color=line_color, linewidth=line_width
            ))

        # add vertical lines
        for pulse in range(kw['n_pulses'] + 1):
            axes.add_artist(plt.Line2D(
                [pulse, pulse], [0, main_height],
                color=line_color, linewidth=line_width
            ))

        # add track names
        for track_ix, track in enumerate(rhythm_loop.get_track_iterator()):
            axes.text(
                0, track_ix + 0.5, "%s " % track.get_name(),  # one ' ' character spacing
                verticalalignment="center", horizontalalignment="right"
            )

    def __draw_track__(self, rhythm_track: Track, axes: plt.Axes, **kw):
        onset_positions = self.get_feature_extractor("onset_positions").process(rhythm_track)
        track_ix = kw['track_ix']

        for onset in onset_positions:
            axes.add_artist(plt.Rectangle(
                [onset, track_ix],
                width=1, height=1, facecolor=kw['color'],
                edgecolor="black", linewidth=0.75
            ))

        return plt.Rectangle([0, 0], 1, 1)

    def get_legend_kwargs(self):
        # effectively hide the legend (move so far out that it's not visible anymore)
        # we don't need the legend as we already draw the track names in the plot itself
        return {'loc': "lower right", 'bbox_to_anchor': (0, 0)}


def get_rhythm_loop_plotter_classes():
    """Returns RhythmLoopPlotter subclasses

    :return: RhythmLoopPlotter subclasses as a list
    """

    return RhythmLoopPlotter.__subclasses__()


__all__ = [
    # Rhythm plotters
    'RhythmLoopPlotter',

    'SchillingerRhythmNotation', 'ChronotonicNotation', 'PolygonNotation',
    'SpectralNotation', 'TEDASNotation', 'IOIHistogram', 'BoxNotation',
    'get_rhythm_loop_plotter_classes',

    # Subplot layouts
    'SubplotLayout', 'CombinedSubplotLayout', 'StackedSubplotLayout', 'Orientation',

    # Misc
    'SnapsToGridPolicy', 'plot_rhythm_grid'
]

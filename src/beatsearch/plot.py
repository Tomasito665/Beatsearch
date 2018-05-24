import math
import enum
import anytree
import numpy as np
import typing as tp
import matplotlib.artist
import matplotlib.pyplot as plt
from fractions import Fraction
from matplotlib.colors import to_rgba
from matplotlib.patches import Wedge
from abc import ABCMeta, abstractmethod
from itertools import cycle, repeat
from beatsearch.rhythm import Unit, UnitType, parse_unit_argument, RhythmLoop, Rhythm, Track, MeterTreeNode
from beatsearch.feature_extraction import IOIVector, BinarySchillingerChain, \
    RhythmFeatureExtractor, ChronotonicChain, OnsetPositionVector, IOIHistogram, DistantPolyphonicSyncopationVector, \
    PolyphonicSyncopationVector, MonophonicMetricalTensionVector, PolyphonicMetricalTensionVector, \
    MonophonicVariabilityVector, MultiTrackMonoFeature
from beatsearch.utils import QuantizableMixin, generate_abbreviations, Rectangle2D, Point2D, \
    find_all_concrete_subclasses

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


def plot_rhythm_grid(axes: plt.Axes, rhythm: Rhythm, unit: Unit, axis="x"):
    duration = rhythm.get_duration(unit, ceil=True)
    measure_duration = rhythm.get_measure_duration(unit)
    beat_duration = rhythm.get_beat_duration(unit)

    measure_grid_ticks = np.arange(0, duration + 1, measure_duration)
    beat_grid_ticks = np.arange(0, duration + 1, beat_duration)

    if len(axis) > 2:
        raise ValueError("Illegal axis: %s" % axis)

    if axis in ("x", "both"):
        axes.set_xticks(measure_grid_ticks)
        axes.set_xticks(beat_grid_ticks, minor=True)
    elif axis in ("y", "both"):
        axes.set_yticks(measure_grid_ticks)
        axes.set_yticks(beat_grid_ticks, minor=True)

    axes.set_axisbelow(True)
    axes.grid(which='minor', alpha=0.2, axis=axis)
    axes.grid(which='major', alpha=0.5, axis=axis)


@parse_unit_argument
def plot_box_notation_grid(
        axes: plt.Axes,
        rhythm: RhythmLoop,
        unit: UnitType,
        position: Point2D = Point2D(0, 0),
        textbox_width: int = 2,
        line_width: int = 1,
        line_color: str = "black",
        beat_colors=(to_rgba("black", 0.21), to_rgba("black", 0.09))
):
    # hide axis
    for axis in [axes.xaxis, axes.yaxis]:
        axis.set_ticklabels([])
        axis.set_visible(False)

    common_height = rhythm.get_track_count()

    text_box = Rectangle2D(
        width=textbox_width, height=common_height,
        x=position.x, y=position.y
    )

    grid_box = Rectangle2D(
        width=rhythm.get_duration(unit, ceil=True),
        height=common_height,
        x=text_box.x + text_box.width, y=text_box.y
    )

    container = Rectangle2D(
        width=(text_box.width + grid_box.width), height=grid_box.height,
        x=text_box.x, y=text_box.y
    )

    # draw beat rectangles
    if beat_colors:
        timesig = rhythm.get_time_signature()
        beat_unit = timesig.get_beat_unit()
        n_steps_per_beat = beat_unit.convert(1, unit, quantize=True)
        n_beats = rhythm.get_duration(beat_unit, ceil=True)

        for beat in range(n_beats):
            beat_in_measure = beat % timesig.numerator
            step = beat * n_steps_per_beat
            axes.add_artist(plt.Rectangle(
                [grid_box.x + step, grid_box.y],
                width=n_steps_per_beat, height=grid_box.height,
                facecolor=beat_colors[beat_in_measure % 2],
                fill=True
            ))

    # draw main box
    axes.add_artist(plt.Rectangle(
        container.position, container.width, container.height,
        fill=False, edgecolor=line_color, linewidth=line_width
    ))

    # add horizontal lines
    for track_ix in range(1, grid_box.height):
        line_y = track_ix + position.x
        axes.add_artist(plt.Line2D(
            container.x_bounds, [line_y, line_y],
            color=line_color, linewidth=line_width
        ))

    # add vertical lines
    for step in range(grid_box.width):
        step_x = grid_box.x + step
        axes.add_artist(plt.Line2D(
            [step_x, step_x], [grid_box.y_bounds],
            color=line_color, linewidth=line_width,
            solid_capstyle="butt"
        ))

    track_names = tuple(t.name for t in rhythm.get_track_iterator())
    track_ids = generate_abbreviations(track_names, max_abbreviation_len=3)
    track_y_positions = dict()
    text_x = text_box.x + (text_box.width / 2)

    # draw track name ids
    if text_box.width > 0:
        for track_ix, [track_name, track_id] in enumerate(zip(track_names, track_ids)):
            axes.text(
                text_x, text_box.y + track_ix + 0.5, track_id,
                verticalalignment="center", horizontalalignment="center"
            )
            track_y_positions[track_name] = track_ix

    return {
        'grid_box': grid_box,
        'text_box': text_box,
        'container': container,
        'track_y_data': track_y_positions
    }


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


# http://colorbrewer2.org/#type=diverging&scheme=Spectral&n=6
DEFAULT_TRACK_COLORS = ['#d53e4f', '#fc8d59', '#fee08b', '#e6f598', '#99d594', '#3288bd']


class RhythmLoopPlotter(object, metaclass=ABCMeta):
    def __init__(
            self, unit: UnitType,
            subplot_layout: SubplotLayout,
            feature_extractors: tp.Optional[tp.Dict[str, RhythmFeatureExtractor]] = None,
            snap_to_grid_policy: SnapsToGridPolicy = SnapsToGridPolicy.NOT_APPLICABLE,
            snaps_to_grid: bool = None,
            draws_own_legend: bool = False,
            track_colors: tp.Iterable[str] = DEFAULT_TRACK_COLORS
    ):
        self._subplot_layout = subplot_layout            # type: SubplotLayout
        self._feature_extractors = feature_extractors or dict()  # type: tp.Dict[str, RhythmFeatureExtractor]
        self._unit = None                                # type: Unit
        self._snaps_to_grid = None                       # type: bool
        self._draws_own_legend = draws_own_legend        # type: bool
        self._snap_to_grid_policy = snap_to_grid_policy  # type: SnapsToGridPolicy
        self._track_colors = None                        # type: tp.Tuple[str, ...]

        if snaps_to_grid is None:
            if snap_to_grid_policy in [SnapsToGridPolicy.NEVER, SnapsToGridPolicy.ADJUSTABLE]:
                snaps_to_grid = False
            elif snap_to_grid_policy == SnapsToGridPolicy.ALWAYS:
                snaps_to_grid = True

        # call setters
        self.snaps_to_grid = snaps_to_grid
        self.track_colors = track_colors
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
        for quantizable_extractor in (ext for ext in self._feature_extractors.values()
                                      if isinstance(ext, QuantizableMixin)):
            quantizable_extractor.quantize = snaps_to_grid

    @property
    def track_colors(self):
        return self._track_colors

    @track_colors.setter
    def track_colors(self, track_colors: tp.Iterable[str]):
        track_colors = tuple(track_colors)
        for color in track_colors:
            try:
                to_rgba(color)
            except ValueError:
                raise ValueError("Invalid track color: '%s'" % str(color))
        self._track_colors = track_colors

    def draw(
            self,
            rhythm_loop: RhythmLoop,
            figure: tp.Optional[plt.Figure] = None,
            legend: bool = True,
            figure_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
            legend_kwargs: tp.Dict[str, tp.Any] = None,
            subplot_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
            show: bool = False
    ) -> plt.Figure:
        """
        Plots the the given drum loop on a matplotlib figure and returns the figure object.

        :param rhythm_loop: the rhythm loop to plot
        :param figure: figure to draw the loop
        :param legend: set to False to hide the legend
        :param figure_kwargs: keyword arguments for the creation of the figure. This argument is ignored if a custom
                              figure has been provided
        :param subplot_kwargs: keyword arguments for the call to Figure.add_subplot
        :param legend_kwargs: keyword arguments for the call to Figure.legend
        :param show: when set to True, matplotlib.pyplot.show() will be called automatically from this method
        :return: matplotlib figure object
        """

        unit = self.get_unit()
        n_tracks = rhythm_loop.get_track_count()

        # the figure to add the subplot(s) to
        plot_type_name = self.get_plot_type_name()
        figure = figure or plt.figure("%s - %s" % (plot_type_name, rhythm_loop.name), **(figure_kwargs or {}))

        # add subplots to figure
        subplots = iter(self._subplot_layout.inflate(figure, n_tracks, subplot_kwargs or dict()))
        color_pool = cycle(self.track_colors)
        prev_subplot, subplot_setup_ret = None, None

        # named arguments given both to __setup_subplot__ and __draw_track__
        common_kwargs = {
            'n_pulses': rhythm_loop.get_duration(unit, ceil=True),
            'n_tracks': rhythm_loop.get_track_count()
        }

        # keep track of track names and artist handles for legend
        track_names = []
        artist_handles = []

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
            artist_handles.append(handle)
            prev_subplot = subplot

        legend_kwargs = {
            'loc': "center right",
            **(legend_kwargs or {}),
            **self.get_legend_kwargs()
        }

        if legend and not self._draws_own_legend:
            figure.legend(artist_handles, track_names, **legend_kwargs)

        if show:
            plt.show()

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

    def __init__(self, unit: UnitType = Unit.EIGHTH, *_,
                 track_colors: tp.Iterable[str] = DEFAULT_TRACK_COLORS):
        super().__init__(
            unit=unit,
            subplot_layout=CombinedSubplotLayout(),
            feature_extractors={'schillinger': BinarySchillingerChain()},
            snap_to_grid_policy=SnapsToGridPolicy.ALWAYS, track_colors=track_colors
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

    def __init__(self, unit: UnitType = Unit.EIGHTH, *_, track_colors: tp.Iterable[str] = DEFAULT_TRACK_COLORS):
        super().__init__(
            unit=unit,
            subplot_layout=CombinedSubplotLayout(),
            feature_extractors={'chronotonic': ChronotonicChain(), 'mt_ioi': MultiTrackMonoFeature(IOIVector)},
            snap_to_grid_policy=SnapsToGridPolicy.ALWAYS,
            track_colors=track_colors
        )

    @classmethod
    def get_plot_type_name(cls):
        return cls.PLOT_TYPE_NAME

    def __setup_subplot__(self, rhythm_loop: RhythmLoop, axes: plt.Axes, **kw):
        plot_rhythm_grid(axes, rhythm_loop, self.get_unit())
        mt_ioi_vec = self.get_feature_extractor("mt_ioi").process(rhythm_loop)
        max_duration = max(ioi for ioi_vec in mt_ioi_vec for ioi in ioi_vec)
        axes.set_ylabel("chronotonic value")
        axes.set_xlabel("atomic beat")
        axes.set_yticks([*range(1, max_duration + 1)])
        axes.grid(which="major", alpha=0.2, axis="y")
        axes.set_ylim(0, max_duration + 1)

    def __draw_track__(self, rhythm_track: Track, axes: plt.Axes, **kw):
        chronotonic_chain = self.get_feature_extractor("chronotonic").process(rhythm_track)
        return axes.plot(chronotonic_chain, "-o", color=kw['color'])


class PolygonNotation(RhythmLoopPlotter):
    PLOT_TYPE_NAME = "Polygon notation"

    def __init__(self, unit: UnitType = Unit.EIGHTH, show_step_wedges: bool = True,
                 show_step_numbers: bool = False, *_, track_colors: tp.Iterable[str] = DEFAULT_TRACK_COLORS):
        super().__init__(
            unit=unit,
            subplot_layout=CombinedSubplotLayout(),
            feature_extractors={'onset_positions': OnsetPositionVector()},
            snap_to_grid_policy=SnapsToGridPolicy.ADJUSTABLE,
            track_colors=track_colors
        )

        self._show_step_wedges = None
        self._show_step_numbers = None

        self.show_step_wedges = show_step_wedges
        self.show_step_numbers = show_step_numbers

    @classmethod
    def get_plot_type_name(cls):
        return cls.PLOT_TYPE_NAME

    @property
    def show_step_wedges(self) -> bool:
        """Set to False to hide step markings"""
        return self._show_step_wedges

    @show_step_wedges.setter
    def show_step_wedges(self, show_step_wedges: bool):
        self._show_step_wedges = bool(show_step_wedges)

    @property
    def show_step_numbers(self):
        """Set to False to hide step numbering"""
        return self._show_step_numbers

    @show_step_numbers.setter
    def show_step_numbers(self, show_step_labels: bool):
        self._show_step_numbers = bool(show_step_labels)

    def __setup_subplot__(self, rhythm_loop: RhythmLoop, axes: plt.Axes, **kw):
        # avoid stretching the aspect ratio
        axes.axis('equal')
        # noinspection PyTypeChecker
        axes.axis([0, 1, 0, 1])

        # add base rhythm circle
        main_radius = 0.27 if self.show_step_numbers else 0.3
        main_center = 0.5, 0.5
        background_colors = "white", "#dfdfdf"

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
            draw_wedge(from_pulse, to_pulse, radius=1.0, fc=background_colors[1])

        # main circle
        circle = plt.Circle(main_center, main_radius, fc="white")
        axes.add_artist(circle)

        # draw the pulse wedges
        if self.show_step_wedges:
            for i_pulse in range(0, n_pulses, 2):
                draw_wedge(i_pulse, i_pulse + 1, fc=background_colors[1])

        # draw pulse numbers
        if self.show_step_numbers:
            for i_pulse in range(n_pulses):
                i_measure = int(i_pulse / n_pulses_per_measure)
                axes.text(
                    *get_coordinates_on_circle(main_center, main_radius * 1.18, float(i_pulse) / n_pulses),
                    i_pulse + 1, horizontalalignment="center", verticalalignment="center", fontsize="smaller",
                    bbox={'boxstyle': "circle", 'fc': background_colors[i_measure % 2], 'ec': "none", 'pad': 0.5}
                )

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

    def __init__(self, unit: UnitType = Unit.EIGHTH, *_, track_colors: tp.Iterable[str] = DEFAULT_TRACK_COLORS):
        super().__init__(
            unit=unit,
            subplot_layout=StackedSubplotLayout(Orientation.VERTICAL),
            feature_extractors={'ioi_vector': IOIVector()},
            snap_to_grid_policy=SnapsToGridPolicy.ADJUSTABLE,
            track_colors=track_colors
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

    def __init__(self, unit: UnitType = Unit.EIGHTH, *_, track_colors: tp.Iterable[str] = DEFAULT_TRACK_COLORS):
        super().__init__(
            unit=unit,
            subplot_layout=StackedSubplotLayout(Orientation.VERTICAL),
            feature_extractors={
                'onset_positions': OnsetPositionVector(quantize=True),
                'ioi_vector': IOIVector(quantize=True)
            },
            snap_to_grid_policy=SnapsToGridPolicy.ADJUSTABLE,
            track_colors=track_colors
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
            'facecolor': to_rgba(kw['color'], 0.18),
            'linewidth': 2.0
        }

        return axes.bar(onset_positions, ioi_vector, width=ioi_vector, align="edge", **styles)


class IOIHistogramPlot(RhythmLoopPlotter):
    PLOT_TYPE_NAME = "Inter-onset interval histogram"

    def __init__(self, unit: UnitType = Unit.EIGHTH, *_, track_colors: tp.Iterable[str] = DEFAULT_TRACK_COLORS):
        super().__init__(
            unit=unit,
            subplot_layout=CombinedSubplotLayout(),
            feature_extractors={'ioi_histogram': IOIHistogram()},
            snap_to_grid_policy=SnapsToGridPolicy.NOT_APPLICABLE,
            track_colors=track_colors
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

    DEFAULT_BACKGROUND = (to_rgba("black", 0.21), to_rgba("black", 0.09))
    SQUARE, CIRCLE = "square", "circle"

    def __init__(
            self, unit: UnitType = Unit.EIGHTH, onset_symbol: str = CIRCLE, track_ids: bool = True,
            background: tp.Union[tp.Any, tp.Tuple[tp.Any, tp.Any]] = DEFAULT_BACKGROUND, line_color: tp.Any = "black",
            line_width: int = 1, extra_feature_extractors: tp.Optional[tp.Dict[str, RhythmFeatureExtractor]] = None,
            *_, track_colors: tp.Iterable[str] = DEFAULT_TRACK_COLORS
    ):
        super().__init__(
            unit=unit,
            subplot_layout=CombinedSubplotLayout(),
            feature_extractors={
                'onset_positions': OnsetPositionVector(),
                **(extra_feature_extractors or dict())
            },
            snap_to_grid_policy=SnapsToGridPolicy.ALWAYS,
            draws_own_legend=True,
            track_colors=track_colors
        )

        self._onset_symbol = None       # type: str
        self._show_track_labels = None  # type: bool
        self._relative_padding = None   # type: float
        self._background = None         # type: tp.Tuple[tp.Any, tp.Any]
        self._line_color = None         # type: tp.Any
        self._line_width = None         # type: int

        self.rel_pad_x = 0.15
        self.onset_symbol = onset_symbol
        self.track_ids = track_ids
        self.background = background
        self.line_color = line_color
        self.line_width = line_width

    @classmethod
    def get_plot_type_name(cls):
        return cls.PLOT_TYPE_NAME

    @property
    def onset_symbol(self) -> str:
        """Onset symbol, must be one of ['circle', 'square']"""
        return self._onset_symbol

    @onset_symbol.setter
    def onset_symbol(self, onset_symbol: str):
        options = (self.SQUARE, self.CIRCLE)
        if onset_symbol not in options:
            raise ValueError("Unknown onset symbol: '%s'. Choose between: %s" % (onset_symbol, str(options)))
        self._onset_symbol = onset_symbol

    @property
    def track_ids(self) -> bool:
        """Set to False to hide the track id labels"""
        return self._show_track_labels

    @track_ids.setter
    def track_ids(self, text_box: bool):
        self._show_track_labels = bool(text_box)

    @property
    def relative_padding(self) -> float:
        """Relative horizontal padding"""
        return self._relative_padding

    @relative_padding.setter
    def relative_padding(self, relative_padding: float):
        self._relative_padding = float(relative_padding)

    @property
    def background(self) -> tp.Union[tp.Any, tp.Tuple[tp.Any, tp.Any]]:
        return self._background

    @background.setter
    def background(self, background: tp.Union[tp.Any, tp.Tuple[tp.Any, tp.Any]]):
        try:
            _ = to_rgba(background)
            self._background = (background, background)
        except (TypeError, ValueError) as err_to_rgba:
            try:
                color_a, color_b = background
            except (TypeError, ValueError):
                raise err_to_rgba

            _ = to_rgba(color_a)
            _ = to_rgba(color_b)

            self._background = color_a, color_b

    @property
    def line_color(self) -> tp.Any:
        return self._line_color

    @line_color.setter
    def line_color(self, line_color: tp.Any):
        _ = to_rgba(line_color)
        self._line_color = line_color

    @property
    def line_width(self) -> int:
        return self._line_width

    @line_width.setter
    def line_width(self, line_width: int):
        self._line_width = int(line_width)

    def __setup_subplot__(self, rhythm_loop: RhythmLoop, axes: plt.Axes, **kw):
        box_notation_grid_info = plot_box_notation_grid(
            axes, rhythm_loop, self.unit, Point2D(0, 0),
            2 if self.track_ids else 0, 1, "black", beat_colors=self._background
        )

        container = box_notation_grid_info['container']  # type: Rectangle2D

        # get viewport dimensions
        x_padding = (self.rel_pad_x * box_notation_grid_info['container'].width) / 2
        x_bounds = [container.x_bounds[0] - x_padding, container.x_bounds[1] + x_padding]
        y_padding = (abs(x_bounds[0] - x_bounds[1]) - container.height) / 2
        y_bounds = [container.y_bounds[0] - y_padding, container.y_bounds[1] + y_padding]

        # setup viewport
        axes.axis("equal")  # equal axis for perfectly "squared" squares.. :)
        axes.set_xlim(*x_bounds)
        axes.set_ylim(*reversed(y_bounds))  # reversed to have (0, 0) in upper-left position

        return {
            **box_notation_grid_info,
            'viewport': Rectangle2D(
                x=x_bounds[0], y=y_bounds[0],
                width=abs(x_bounds[0] - x_bounds[1]),
                height=abs(y_bounds[0] - y_bounds[1])
            )
        }

    def __draw_track__(self, rhythm_track: Track, axes: plt.Axes, **kw):
        setup_result = kw['setup_ret']
        grid_box = setup_result['grid_box']  # type: Rectangle2D

        onset_positions = self.get_feature_extractor("onset_positions").process(rhythm_track)
        n_steps = rhythm_track.get_duration(self.unit)
        track_ix = kw['track_ix']

        if self.onset_symbol == self.SQUARE:
            get_artist = lambda pos: plt.Rectangle(
                [grid_box.x + onset_pos, grid_box.y + track_ix],
                width=1, height=1, facecolor=kw['color'],
                edgecolor="black", linewidth=0.75, joinstyle="miter"
            )
        elif self.onset_symbol == self.CIRCLE:
            get_artist = lambda pos: plt.Circle(
                [grid_box.x + onset_pos + 0.5, grid_box.y + track_ix + 0.5],
                radius=0.25, facecolor=kw['color'], fill=True, edgecolor='none'
            )
        else:
            raise ValueError("Unknown symbol type: '%s'" % self.onset_symbol)

        # filter out onsets whose position might have been pushed out of the grid (due to
        # a low rhythm plotter unit)
        for onset_pos in filter(lambda pos: pos < n_steps, onset_positions):
            axes.add_artist(get_artist(onset_pos))

        return plt.Rectangle([0, 0], 1, 1)


class PolyphonicSyncopationVectorGraphBase(BoxNotation, metaclass=ABCMeta):

    def __init__(self, unit: UnitType, *_, track_colors: tp.Iterable[str] = DEFAULT_TRACK_COLORS):
        super().__init__(
            unit=unit,
            extra_feature_extractors={'poly_sync_vector': self.__get_sync_extractor__()},
            track_colors=track_colors
        )

        self.spacing = 1
        self.text_box_width = 3
        self.syncopations_color = "black"
        self.cyclic_syncopations_color = "gray"

    def __setup_subplot__(self, rhythm_loop: RhythmLoop, axes: plt.Axes, **kw):
        box_notation_setup_res = super().__setup_subplot__(rhythm_loop, axes, **kw)
        box_notation_viewport = box_notation_setup_res['viewport']
        box_notation_container = box_notation_setup_res['container']
        box_notation_grid = box_notation_setup_res['grid_box']
        box_notation_textbox = box_notation_setup_res['text_box']

        # compute the polyphonic syncopation vector
        poly_sync_extractor = self.__get_sync_extractor__()
        poly_sync_vector = poly_sync_extractor.process(rhythm_loop)

        # retrieve the levels on which syncopations could occur
        time_sig = rhythm_loop.get_time_signature()
        natural_duration_map = time_sig.get_natural_duration_map(self.unit)
        possible_sync_durations = sorted(set(natural_duration_map))
        sync_level_units = tuple(Unit.get(Fraction(
            d, self.unit.get_note_value().denominator)) for d in possible_sync_durations)  # type: tp.Tuple[Unit]
        n_sync_levels = len(possible_sync_durations)

        # main syncopations container
        sync_container = Rectangle2D(
            x=box_notation_container.x,
            y=box_notation_container.y_bounds[1] + self.spacing,
            width=box_notation_container.width,
            height=n_sync_levels
        )

        # syncopations grid box
        sync_grid_box = Rectangle2D(
            x=box_notation_grid.x, y=sync_container.y,
            width=box_notation_grid.width, height=box_notation_grid.height
        )

        # draw main syncopations container
        axes.add_artist(plt.Rectangle(
            sync_container.position, sync_container.width, sync_container.height,
            fill=False, edgecolor=self.line_color, linewidth=self.line_width
        ))

        # add horizontal lines
        for sync_level in range(1, n_sync_levels):
            line_y = sync_container.y + sync_level
            axes.add_artist(plt.Line2D(
                sync_container.x_bounds, [line_y, line_y],
                color=self.line_color, linewidth=self.line_width
            ))

        # draw horizontal header line
        axes.add_artist(plt.Line2D(
            [sync_grid_box.x, sync_grid_box.x], sync_container.y_bounds,
            color=self.line_color, linewidth=self.line_width
        ))

        # draw vertical grid lines (transparent)
        for step in range(1, sync_grid_box.width):
            line_x = sync_grid_box.x + step
            axes.add_artist(plt.Line2D(
                [line_x, line_x], sync_container.y_bounds,
                color=to_rgba(self.line_color, 0.1), linewidth=self.line_width
            ))

        sync_level_label_x = box_notation_textbox.x + (box_notation_textbox.width / 2)

        # draw sync level labels
        for ix, sync_unit in enumerate(sync_level_units):
            sync_unit_label = str(sync_unit.get_note_value())
            axes.text(
                sync_level_label_x, sync_container.y + ix + 0.5, sync_unit_label,
                verticalalignment="center", horizontalalignment="center"
            )

        n_steps = rhythm_loop.get_duration(self.unit, ceil=True)

        # draw the syncopations
        for syncopation in poly_sync_vector:
            pos_from, pos_to = syncopation[1:]
            cyclic = False

            while pos_to < pos_from:
                cyclic = True
                pos_to += n_steps

            sync_duration = pos_to - pos_from
            sync_level_ix = possible_sync_durations.index(sync_duration)
            line_x_bounds = sync_grid_box.x + pos_from + 0.5, sync_grid_box.x + pos_to + 0.5
            line_y = sync_grid_box.y + sync_level_ix + 0.5
            sync_line_color = self.cyclic_syncopations_color if cyclic else self.syncopations_color

            axes.add_artist(plt.Line2D(
                line_x_bounds, [line_y, line_y],
                color=sync_line_color, linewidth=self.line_width, marker="o"
            ))

        # area occupied by both the box notation container and the syncopations container
        combined_area = Rectangle2D(
            x=box_notation_container.x, y=box_notation_container.y,
            width=abs(box_notation_container.x_bounds[0] - sync_container.x_bounds[1]),
            height=abs(box_notation_container.y_bounds[0] - sync_container.y_bounds[1])
        )

        # re-adjust viewport (center the whole thing vertically)
        pad_y = (box_notation_viewport.width - combined_area.height) / 2
        y_bounds = (combined_area.y_bounds[0] - pad_y, combined_area.y_bounds[1] + pad_y)
        viewport = Rectangle2D(
            x=box_notation_viewport.x, y=y_bounds[0],
            width=box_notation_viewport.width, height=abs(y_bounds[0] - y_bounds[1])
        )

        axes.set_xlim(viewport.x_bounds)
        axes.set_ylim(*reversed(viewport.y_bounds))

        return box_notation_setup_res

    @abstractmethod
    def __get_sync_extractor__(self):
        # should return the polyphonic syncopation extractor

        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_plot_type_name(cls):
        raise NotImplementedError


class PolyphonicSyncopationVectorWitekGraph(PolyphonicSyncopationVectorGraphBase):
    PLOT_TYPE_NAME = "Syncopation graph (Witek)"

    def __init__(self, unit: UnitType = Unit.EIGHTH, *_, track_colors: tp.Iterable[str] = DEFAULT_TRACK_COLORS):
        self._extractor = PolyphonicSyncopationVector(unit)
        super().__init__(unit, track_colors=track_colors)

    @property
    def salience_profile_type(self) -> str:
        return self._extractor.salience_profile_type

    @salience_profile_type.setter
    def salience_profile_type(self, salience_profile_type: str):
        self._extractor.salience_profile_type = salience_profile_type

    def __get_sync_extractor__(self) -> PolyphonicSyncopationVector:
        return self._extractor

    @classmethod
    def get_plot_type_name(cls):
        return cls.PLOT_TYPE_NAME


class PolyphonicSyncopationVectorGraph(PolyphonicSyncopationVectorGraphBase):
    PLOT_TYPE_NAME = "Syncopation graph"

    def __init__(self, unit: UnitType = Unit.EIGHTH, *_, track_colors: tp.Iterable[str] = DEFAULT_TRACK_COLORS):
        self._extractor = DistantPolyphonicSyncopationVector(unit)
        super().__init__(unit, track_colors=track_colors)

    @property
    def salience_profile_type(self) -> str:
        return self._extractor.salience_profile_type

    @salience_profile_type.setter
    def salience_profile_type(self, salience_profile_type: str):
        self._extractor.salience_profile_type = salience_profile_type

    @property
    def instrumentation_weight_function(self) -> tp.Callable[[str, tp.Iterable[str]], int]:
        return self._extractor.instrumentation_weight_function

    @instrumentation_weight_function.setter
    def instrumentation_weight_function(self, instr_w_function: tp.Callable[[str, tp.Iterable[str]], int]):
        self._extractor.instrumentation_weight_function = instr_w_function

    @property
    def nested_syncopations(self):
        return self._extractor.nested_syncopations

    @nested_syncopations.setter
    def nested_syncopations(self, nested_syncopations: str):
        self._extractor.nested_syncopations = nested_syncopations

    @property
    def only_uninterrupted_syncopations(self) -> bool:
        return self._extractor.only_uninterrupted_syncopations

    @only_uninterrupted_syncopations.setter
    def only_uninterrupted_syncopations(self, only_uninterrupted_syncopations: bool):
        self._extractor.only_uninterrupted_syncopations = only_uninterrupted_syncopations

    def __get_sync_extractor__(self) -> DistantPolyphonicSyncopationVector:
        return self._extractor

    @classmethod
    def get_plot_type_name(cls):
        return cls.PLOT_TYPE_NAME


class MonophonicMTVGraph(RhythmLoopPlotter):
    PLOT_TYPE_NAME = "MTV (mono)"

    @classmethod
    def get_plot_type_name(cls):
        return cls.PLOT_TYPE_NAME

    def __init__(self, unit: UnitType = Unit.EIGHTH, salience_profile_type: str = "equal_upbeats",
                 *_, track_colors: tp.Iterable[str] = DEFAULT_TRACK_COLORS):
        super().__init__(
            unit=unit,
            subplot_layout=CombinedSubplotLayout(),
            feature_extractors={'mtv': MonophonicMetricalTensionVector(unit, salience_profile_type, normalize=True)},
            snap_to_grid_policy=SnapsToGridPolicy.ALWAYS,
            track_colors=track_colors
        )

    @property
    def salience_profile_type(self) -> str:
        mtv = self.get_feature_extractor("mtv")  # type: MonophonicMetricalTensionVector
        return mtv.salience_profile_type

    @salience_profile_type.setter
    def salience_profile_type(self, salience_profile_type: str):
        mtv = self.get_feature_extractor("mtv")  # type: MonophonicMetricalTensionVector
        mtv.salience_profile_type = salience_profile_type

    def __setup_subplot__(self, rhythm_loop: RhythmLoop, axes: plt.Axes, **kw):
        plot_rhythm_grid(axes, rhythm_loop, self.get_unit())
        axes.set_ylim(0, 1)  # normalize is set to True, which makes the tension go from 0 to 1

    def __draw_track__(self, rhythm_track: Track, axes: plt.Axes, **kw):
        mtv = self.get_feature_extractor("mtv").process(rhythm_track)
        return axes.plot(mtv, ".-", color=kw['color'])


class PolyphonicMTVGraph(RhythmLoopPlotter):
    PLOT_TYPE_NAME = "MTV (poly)"

    @classmethod
    def get_plot_type_name(cls):
        return cls.PLOT_TYPE_NAME

    def __init__(self, unit: UnitType = Unit.EIGHTH, salience_profile_type: str = "equal_upbeats",
                 *_, track_colors: tp.Iterable[str] = DEFAULT_TRACK_COLORS):
        super().__init__(
            unit=unit,
            subplot_layout=CombinedSubplotLayout(),
            feature_extractors={
                'mono_mtv': MonophonicMetricalTensionVector(unit, salience_profile_type, normalize=True),
                'poly_mtv': PolyphonicMetricalTensionVector(unit, salience_profile_type, normalize=True)
            },
            snap_to_grid_policy=SnapsToGridPolicy.ALWAYS,
            track_colors=track_colors
        )

    @property
    def salience_profile_type(self) -> str:
        mono_mtv = self.get_feature_extractor("mono_mtv")  # type: MonophonicMetricalTensionVector
        poly_mtv = self.get_feature_extractor("poly_mtv")  # type: PolyphonicMetricalTensionVector
        assert mono_mtv.salience_profile_type == poly_mtv.salience_profile_type
        return mono_mtv.salience_profile_type

    @salience_profile_type.setter
    def salience_profile_type(self, salience_profile_type: str):
        mono_mtv = self.get_feature_extractor("mono_mtv")  # type: MonophonicMetricalTensionVector
        poly_mtv = self.get_feature_extractor("poly_mtv")  # type: PolyphonicMetricalTensionVector
        mono_mtv.salience_profile_type = salience_profile_type
        poly_mtv.salience_profile_type = salience_profile_type

    def set_instrument_weights(self, weights: tp.Dict[str, float]):
        poly_mtv = self.get_feature_extractor("poly_mtv")  # type: PolyphonicMetricalTensionVector
        poly_mtv.set_instrument_weights(weights)

    def get_instrument_weights(self):
        poly_mtv = self.get_feature_extractor("poly_mtv")  # type: PolyphonicMetricalTensionVector
        return poly_mtv.get_instrument_weights()

    def __setup_subplot__(self, rhythm_loop: RhythmLoop, axes: plt.Axes, **kw):
        plot_rhythm_grid(axes, rhythm_loop, self.get_unit())
        axes.set_ylim(0, 1)

        extractor = self.get_feature_extractor("poly_mtv")
        poly_mtv = extractor.process(rhythm_loop)

        axes.plot(poly_mtv, color="black", linewidth=2)

    def __draw_track__(self, rhythm_track: Track, axes: plt.Axes, **kw):
        extractor = self.get_feature_extractor("mono_mtv")  # type: MonophonicMetricalTensionVector
        mono_mtv = extractor.process(rhythm_track)

        weights = self.get_instrument_weights()
        weights_sum = sum(weights.values())

        try:
            normalized_instr_w = weights[rhythm_track.name] / weights_sum
            weight_known = True
        except KeyError:
            normalized_instr_w = 1
            weight_known = False

        axes.set_axisbelow(True)
        return axes.plot(mono_mtv, "-" if weight_known else ":", color=to_rgba(kw['color'], normalized_instr_w))


class MonophonicVariationGraph(RhythmLoopPlotter):
    PLOT_TYPE_NAME = "Variation (mono)"

    @classmethod
    def get_plot_type_name(cls):
        return cls.PLOT_TYPE_NAME

    def __init__(self, unit: UnitType = Unit.EIGHTH, *_, track_colors: tp.Iterable[str] = DEFAULT_TRACK_COLORS):
        super().__init__(
            unit=unit,
            subplot_layout=CombinedSubplotLayout(),
            feature_extractors={'variation': MonophonicVariabilityVector(unit)},
            snap_to_grid_policy=SnapsToGridPolicy.ALWAYS,
            track_colors=track_colors
        )

    def __setup_subplot__(self, rhythm_loop: RhythmLoop, axes: plt.Axes, **kw):
        plot_rhythm_grid(axes, rhythm_loop, self.get_unit())
        axes.set_ylim(0, 1)

    def __draw_track__(self, rhythm_track: Track, axes: plt.Axes, **kw):
        variation_extractor = self.get_feature_extractor('variation')
        variation_vector = variation_extractor.process(rhythm_track)
        return axes.plot(variation_vector, ".-", color=kw['color'])


def get_rhythm_loop_plotter_classes():
    """Returns RhythmLoopPlotter subclasses

    :return: RhythmLoopPlotter subclasses as a list
    """

    return find_all_concrete_subclasses(RhythmLoopPlotter)


def plot_salience_profile(
        salience_profile: tp.Sequence[int],
        bottom: tp.Optional[int] = None,
        axes: tp.Optional[plt.Axes] = None,
        **kwargs: tp.Dict[str, tp.Any]
) -> plt.Axes:
    """Utility function to plot a metrical salience profile

    This function plots the given salience profile and returns the matplotlib axes object on which the salience profile
    was drawn.

    :param salience_profile: salience profile returned by :meth:`beatsearch.rhythm.TimeSignature.get_salience_profile`
    :param bottom: the metrical salience value from which to draw the lines (defaults to min(salience_profile) - 1)
    :param axes: matplotlib axes object, when given, the salience profile will be drawn on these axes
    :param: kwargs: keyword arguments passed to :meth:`matplotlib.pyplot.Axes.stem`
    :return: the matplotlib axes object on which the salience profile was drawn
    """

    if not axes:
        figure = plt.figure("Salience Profile")
        axes = figure.add_subplot(111)

    if bottom is None:
        bottom = min(salience_profile) - 1

    axes.stem(salience_profile, bottom=bottom, **kwargs)
    return axes


def plot_meter_tree(root: MeterTreeNode, axes: tp.Optional[plt.Axes] = None, center: bool = False) -> plt.Axes:
    """
    Utility function to plot a hierarchical meter tree, given its root.

    :param root: root of the hierarchical meter tree returned by :meth:`beatsearch.rhythm.TimeSignature.get_meter_tree`
    :param axes: matplotlib axes object, when given, the tree be drawn on these axes
    :param center: when set to True, the nodes will be centered horizontally
    :return: the matplotlib axes object on which the hierarchical meter tree was drawn
    """

    if not axes:
        figure = plt.figure("Meter tree")
        axes = figure.add_subplot(111)

    max_depth = 0
    width = root.duration

    axes.set_xlim(-1, width + 1)
    axes.set_xticks([])
    axes.set_yticks([])

    text_kwargs = dict(
        horizontalalignment="center", verticalalignment="center",
        color="black", bbox={'boxstyle': "circle", 'fc': "white", 'ec': "black", 'pad': 0.5, 'lw': 1.5}
    )

    for nodes_curr_depth in anytree.LevelOrderGroupIter(root):
        if len(nodes_curr_depth) == 0:
            continue

        curr_depth = nodes_curr_depth[0].depth
        curr_node_width = nodes_curr_depth[0].duration
        max_depth = max(curr_depth, max_depth)
        x_offset = width * 0.5 / len(nodes_curr_depth) if center else 0

        for node_i, node in enumerate(nodes_curr_depth):
            node.xy = node_i * curr_node_width + x_offset, curr_depth
            if not node.is_root:
                axes.add_artist(plt.Line2D(
                    [node.xy[0], node.parent.xy[0]],
                    [node.xy[1], node.parent.xy[1]],
                    color="black", lw=text_kwargs['bbox']['lw']
                ))

            # Background textbox just for double-edge effect to make the main text circle 'float' from the line :)
            axes.text(*node.xy, curr_node_width, {**text_kwargs, 'bbox': {
                **text_kwargs['bbox'],
                'pad': text_kwargs['bbox']['pad'] + 0.2,
                'fc': "white", 'ec': "white"
            }})

            axes.text(*node.xy, curr_node_width, **text_kwargs)

    axes.set_ylim(max_depth + 1, -1)
    return axes


__all__ = [
    # Rhythm plotters
    'RhythmLoopPlotter', 'get_rhythm_loop_plotter_classes',

    'SchillingerRhythmNotation', 'ChronotonicNotation', 'PolygonNotation',
    'SpectralNotation', 'TEDASNotation', 'IOIHistogram', 'BoxNotation',
    'PolyphonicSyncopationVectorGraph', 'PolyphonicSyncopationVectorWitekGraph',
    'MonophonicMTVGraph', 'PolyphonicMTVGraph',

    # Subplot layouts
    'SubplotLayout', 'CombinedSubplotLayout', 'StackedSubplotLayout', 'Orientation',

    # Misc
    'SnapsToGridPolicy', 'plot_rhythm_grid', 'plot_salience_profile', 'plot_meter_tree'
]

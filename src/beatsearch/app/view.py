import os
import signal
import tkinter.ttk
# noinspection PyPep8Naming
import tkinter as tk
import tkinter.font
import tkinter.filedialog
from tkinter import messagebox
from contextlib import contextmanager
from itertools import zip_longest, repeat
from tkinter import ttk
from tkinter import filedialog
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from functools import wraps, partial
import typing as tp
from collections import OrderedDict
from beatsearch.rhythm import (
    Unit
)
from beatsearch.metrics import MonophonicRhythmDistanceMeasure, TRACK_WILDCARDS
from beatsearch.utils import (
    head_trail_iter,
    no_callback,
    type_check_and_instantiate_if_necessary,
    eat_args,
    color_variant
)
from beatsearch.app.control import BSController, BSRhythmLoopLoader
from beatsearch.graphics.plot import RhythmLoopPlotter
from beatsearch.rhythm import RhythmLoop, Rhythm as Rhythm
import midi  # after beatsearch imports!


class BSAppFrame(tk.Frame):
    def __init__(self, app, **kwargs):
        if not isinstance(app, BSApp):
            raise TypeError("Expected a BSApp but got '%s'" % app)
        tk.Frame.__init__(self, app, **kwargs)
        self.controller = app.controller
        self.dpi = app.winfo_fpixels("1i")

    def redraw(self):
        raise NotImplementedError


# TODO Remove duplicated functionality
class BSAppTtkFrame(ttk.Frame):
    def __init__(self, app, **kwargs):
        if not isinstance(app, BSApp):
            raise TypeError("Expected a BSApp but got '%s'" % app)
        ttk.Frame.__init__(self, app, **kwargs)
        self.controller = app.controller
        self.dpi = app.winfo_fpixels("1i")


class BSMidiRhythmLoopLoader(BSRhythmLoopLoader):
    SOURCE_NAME = "MIDI file"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_source_name(cls):
        return cls.SOURCE_NAME

    def is_available(self):
        return True

    def __load__(self, **kwargs):
        fpath = filedialog.askopenfilename(
            title="Load rhythm from MIDI file",
            filetypes=(("MIDI files", "*.mid"), ("All files", "*"))
        )
        if not fpath:
            return None
        if not os.path.isfile(fpath):
            raise self.LoadingError("No such file: %s" % fpath)
        try:
            pattern = midi.read_midifile(fpath)
        except TypeError as e:
            fname = os.path.basename(fpath)
            raise self.LoadingError("Error loading\"%s\": %s" % (fname, str(e)))
        rhythm = Rhythm.create_from_midi(pattern, fpath)
        return rhythm


class BSSearchForm(BSAppFrame):
    INNER_PAD_X = 6

    COMBO_DISTANCE_MEASURE = "<Combo-DistanceMeasure>"
    COMBO_TRACKS = "<Combo-Tracks>"
    COMBO_QUANTIZE = "<Combo-Quantize>"

    def __init__(self, controller, background=None, **kwargs):
        BSAppFrame.__init__(self, controller, background=background, **kwargs)

        combobox_info = [
            (self.COMBO_DISTANCE_MEASURE, "Distance measure", MonophonicRhythmDistanceMeasure.get_measure_names()),
            (self.COMBO_TRACKS, "Tracks to compare", TRACK_WILDCARDS),
            (self.COMBO_QUANTIZE, "Quantize", Unit.get_unit_names())
        ]

        widgets = []
        self._combo_boxes = {}

        for is_first, is_last, [name, text, values] in head_trail_iter(combobox_info):
            box_container = tk.Frame(self)

            # setup combobox
            combobox = tkinter.ttk.Combobox(box_container, values=values, state="readonly")
            combobox.current(0)
            combobox.pack(side=tk.BOTTOM, fill=tk.X, expand=True)
            self._combo_boxes[name] = combobox

            # setup label
            box_label = tk.Label(box_container, text=text, anchor=tk.W, bg=background)
            box_label.pack(side=tk.TOP, fill=tk.X, expand=True)

            widgets.append(box_container)

        self._btn_search = tk.Button(self, text="Search")
        widgets.append(self._btn_search)

        for is_first, is_last, w in head_trail_iter(widgets):
            w.pack(side=tk.LEFT, padx=(
                BSSearchForm.INNER_PAD_X if is_first else BSSearchForm.INNER_PAD_X / 2.0,
                BSSearchForm.INNER_PAD_X if is_last else BSSearchForm.INNER_PAD_X / 2.0
            ), fill=tk.BOTH, expand=True)

        self._callbacks = dict((combo_name, no_callback) for combo_name in [
            self.COMBO_DISTANCE_MEASURE,
            self.COMBO_TRACKS,
            self.COMBO_QUANTIZE
        ])

    def redraw(self):
        controller = self.controller
        q_combo = self._combo_boxes[self.COMBO_QUANTIZE]
        q_combo.config(state=tk.NORMAL if controller.is_current_distance_measure_quantizable() else tk.DISABLED)
        self._btn_search.config(state=tk.NORMAL if controller.is_target_rhythm_set() else tk.DISABLED)

    @property
    def on_new_measure(self):  # type: () -> tp.Callable[str]
        return self._callbacks[self.COMBO_DISTANCE_MEASURE]

    @property
    def on_new_tracks(self):  # type: () -> tp.Callable[str]
        return self._callbacks[self.COMBO_TRACKS]

    @property
    def on_new_quantize(self):  # type: () -> tp.Callable[str]
        return self._callbacks[self.COMBO_QUANTIZE]

    @on_new_measure.setter
    def on_new_measure(self, callback):
        self._set_combobox_value_callback(self.COMBO_DISTANCE_MEASURE, callback)

    @on_new_tracks.setter
    def on_new_tracks(self, callback):
        self._set_combobox_value_callback(self.COMBO_TRACKS, callback)

    @on_new_quantize.setter
    def on_new_quantize(self, callback):
        self._set_combobox_value_callback(self.COMBO_QUANTIZE, callback)

    def _set_combobox_value_callback(self, combobox_name, callback):
        # type: (str, tp.Callable[str]) -> None
        @wraps(callback)
        def wrapper(event):
            value = event.widget.get()
            return callback(value)

        combobox = self._combo_boxes[combobox_name]
        combobox.bind("<<ComboboxSelected>>", wrapper)

    @property
    def search_command(self):
        return self._btn_search.cget("command")

    @search_command.setter
    def search_command(self, callback):
        if not callable(callback):
            raise TypeError("Expected callable but got '%s'" % str(callback))
        self._btn_search.configure(command=callback)


class ContextMenu(tk.Menu):
    def __init__(self, root, event_show="<Button-3>", **kwargs):
        tk.Menu.__init__(self, root, tearoff=0, **kwargs)
        root.bind(event_show, lambda event: self.show(event.x_root, event.y_root))

    def show(self, x, y):
        try:
            self.tk_popup(x, y, 0)
        finally:
            self.grab_release()

    def add_submenu(self, label):
        submenu = tk.Menu(tearoff=0)
        self.add_cascade(label=label, menu=submenu)
        return submenu


class BSRhythmList(BSAppFrame):

    def __init__(self, app, h_scroll=False, v_scroll=True, background="white", **kwargs):
        BSAppFrame.__init__(self, app, **kwargs)
        column_headers = BSController.get_rhythm_data_attr_names()
        self._tree_view = tkinter.ttk.Treeview(columns=column_headers, show="headings")

        # set treeview background
        style = ttk.Style(self)
        style.configure("Treeview", background=background)

        scrollbars = {
            tk.X: tkinter.ttk.Scrollbar(orient="horizontal", command=self._tree_view.xview),
            tk.Y: tkinter.ttk.Scrollbar(orient="vertical", command=self._tree_view.yview)
        }

        self.bind("<MouseWheel>", self._on_mousewheel)

        self._tree_view.bind("<<TreeviewSelect>>", self._on_tree_select)
        self._tree_view.bind("<Double-Button-1>", self._on_double_click)

        self._tree_view.configure(
            xscrollcommand=scrollbars[tk.X].set,
            yscrollcommand=scrollbars[tk.Y].set
        )

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._tree_view.grid(column=0, row=0, sticky="nsew", in_=self)

        if h_scroll:
            scrollbars[tk.X].grid(column=0, row=1, sticky="ew", in_=self)
        if v_scroll:
            scrollbars[tk.Y].grid(column=1, row=0, sticky="ns", in_=self)

        self._corpus_name = None

        context_menu = ContextMenu(self._tree_view)
        context_menu.add_command(label="Set as target rhythm", command=self._on_set_as_target_rhythm)
        plot_rhythms_menu = context_menu.add_submenu("Plot rhythms")

        for plot_type_name in BSApp.RHYTHM_PLOT_TYPES_BY_NAME.keys():
            plot_rhythms_menu.add_command(
                label=plot_type_name,
                command=partial(self._on_plot_rhythms, plot_type_name=plot_type_name)
            )

        self._context_menu = context_menu

        self._on_request_target_rhythm = no_callback
        self._on_request_rhythm_plot = no_callback

    def redraw(self):
        controller = self.controller

        if self._corpus_changed():
            self._clear_tree_view()
            self._fill_tree_view()
            return

        rhythm_data = tuple(controller.get_rhythm_data())

        tv = self._tree_view
        for item_iid in tv.get_children():
            rhythm_ix = self._get_rhythm_index(item_iid)
            tv.item(item_iid, values=rhythm_data[rhythm_ix])

        self._sort_tree_view("Distance to target", 0)

    @property
    def on_request_target_rhythm(self):  # type: () -> tp.Callable[int]
        return self._on_request_target_rhythm

    @on_request_target_rhythm.setter
    def on_request_target_rhythm(self, callback):  # type: (tp.Callable[int]) -> None
        if not callable(callback):
            raise TypeError("Expected a callback but got \"%s\"" % str(callback))
        self._on_request_target_rhythm = callback

    @property
    def on_request_rhythm_plot(self):
        return self._on_request_rhythm_plot

    @on_request_rhythm_plot.setter
    def on_request_rhythm_plot(self, callback: tp.Callable[[tp.Iterable[int]], None]):
        if not callable(callback):
            raise TypeError("Expected a callback but got \"%s\"" % str(callback))
        self._on_request_rhythm_plot = callback

    def _on_set_as_target_rhythm(self):
        selected_rhythms = self._get_selected_rhythm_indices()
        assert len(selected_rhythms) >= 1
        self.on_request_target_rhythm(selected_rhythms[0])

    def _on_plot_rhythms(self, plot_type_name):
        selected_rhythms = self._get_selected_rhythm_indices()
        self.on_request_rhythm_plot(selected_rhythms, plot_type_name)

    def _fill_tree_view(self):
        controller = self.controller
        column_headers = controller.get_rhythm_data_attr_names()

        for col in column_headers:
            self._tree_view.heading(col, text=col, command=lambda c=col: self._sort_tree_view(c, False))
            # adjust column width to header string
            self._tree_view.column(col, width=tkinter.font.Font().measure(col))

        for item in controller.get_rhythm_data():
            self._tree_view.insert("", tk.END, values=item)

            # adjust column width if necessary to fit each value
            for ix, val in enumerate(item):
                col_w = tkinter.font.Font().measure(val)
                if self._tree_view.column(column_headers[ix], width=None) < col_w:
                    self._tree_view.column(column_headers[ix], width=col_w)

        self._corpus_name = controller.get_corpus_name()

    def _clear_tree_view(self):
        tv = self._tree_view
        tv.delete(*tv.get_children())

    def _corpus_changed(self):  # TODO get rid of assumption that corpora won't be named alike
        corpus_name = self.controller.get_corpus_name()
        return self._corpus_name != corpus_name

    def _on_mousewheel(self, event):
        self._tree_view.yview_scroll(-1 * (event.delta // 15), tk.UNITS)

    def _get_rhythm_index(self, tree_view_item_iid):
        tv = self._tree_view
        tv_item = tv.item(tree_view_item_iid)
        tv_values = tv_item['values']
        return tv_values[0]

    def _get_selected_rhythm_indices(self):
        tv_selection = self._tree_view.selection()
        return [self._get_rhythm_index(item_iid) for item_iid in tv_selection]

    def _on_tree_select(self, _):
        controller = self.controller
        selected_rhythms = self._get_selected_rhythm_indices()
        controller.set_rhythm_selection(selected_rhythms)

    def _on_double_click(self, _):
        selected_rhythms = self._get_selected_rhythm_indices()
        assert len(selected_rhythms) == 1
        self.on_request_target_rhythm(selected_rhythms[0])

    def _sort_tree_view(self, column, descending):  # sorts the rhythms by the given column
        tv = self._tree_view
        data_type = BSController.RHYTHM_DATA_TYPES[column]
        data = [(tv.set(item_iid, column), item_iid) for item_iid in tv.get_children("")]
        data.sort(reverse=descending, key=lambda cell: data_type(cell[0]))
        for i, row_info in enumerate(data):
            tv.move(row_info[1], "", i)
        tv.heading(column, command=lambda col=column: self._sort_tree_view(col, not descending))

    class SingleRhythmContextMenu(ContextMenu):
        def __init__(self, root, **kwargs):
            ContextMenu.__init__(self, root, **kwargs)
            self.add_command(label="Set as target rhythm")
            self.add_command(label="Plot rhythm")
            self.add_separator()
            self.add_command(label="Hola, soy Juan")


class BSTransportControls(BSAppFrame, object):
    def __init__(self, controller, background="#37474F", **kwargs):
        BSAppFrame.__init__(self, controller, background=background, **kwargs)
        self._btn_toggle_play = ToggleButton(self, text=("Play", "Stop"), width=8)
        self._btn_toggle_play.pack(side=tk.LEFT, padx=6, pady=6)
        self._play_command = None

    def redraw(self):
        controller = self.controller
        rhythm_selection = controller.get_rhythm_selection()
        is_playing = controller.is_rhythm_player_playing()
        btn = self._btn_toggle_play
        btn.set_enabled(len(rhythm_selection) > 0)
        btn.set_toggle(is_playing)

    @property
    def toggle_play_command(self):
        return self._btn_toggle_play.cget("command")

    @toggle_play_command.setter
    def toggle_play_command(self, callback):
        if not callable(callback):
            raise TypeError("Expected callable but got '%s'" % str(callback))
        self._btn_toggle_play.configure(command=callback)


class BSRhythmComparisonStrip(BSAppTtkFrame):
    def __init__(self, app, background_left="#E0E0E0", background_right="#EEEEEE", **kwargs):
        super().__init__(app, **kwargs)
        self.rhythm_plotter = RhythmLoopPlotter()
        self._rhythm_plot_function = self.rhythm_plotter.polygon

        # left panel (target rhythm box)
        header_target = tk.Frame(self, bg=background_left)
        label_target = tk.Label(
            header_target, text="Target Rhythm", anchor=tk.W, font=BSApp.FONT['header'], bg=background_left)
        target_rhythm_menu_button = tk.Menubutton(header_target, text="Load", relief=tk.FLAT, borderwidth=0)
        target_rhythm_menu_button.bind("<Button-1>", eat_args(self.redraw_target_rhythm_menu))
        label_target.pack(side=tk.LEFT, padx=3)
        target_rhythm_menu_button.pack(side=tk.RIGHT, padx=4, pady=4)
        frame_target = self.TargetRhythmBox(self, self.dpi, background=background_left)

        # right panel (selected rhythm plots)
        header_selection = tk.Frame(self, bg=background_right)
        label_selection = tk.Label(
            header_selection, text="Selected Rhythms", anchor=tk.W, font=BSApp.FONT['header'], bg=background_right)
        combo_label_selection = tk.Label(header_selection, text="Plot type", bg=background_right)
        combo_selection = tkinter.ttk.Combobox(
            header_selection, values=tuple(BSApp.RHYTHM_PLOT_TYPES_BY_NAME.keys()), state="readonly")
        label_selection.pack(side=tk.LEFT, padx=3)
        combo_selection.pack(side=tk.RIGHT, padx=(0, 6))
        combo_label_selection.pack(side=tk.RIGHT, fill=tk.Y, padx=6)
        frame_selection = BSRhythmComparisonStrip.SelectedRhythmsFrame(
            self, n_boxes=19, dpi=self.dpi, background=background_right)

        # lay out left panel
        frame_target.grid(row=1, column=0, sticky="nsew")
        header_target.grid(row=0, column=0, sticky="nsew")

        # lay out right panel
        frame_selection.grid(row=1, column=1, sticky="nsew")
        header_selection.grid(row=0, column=1, sticky="nsew")
        self.grid_columnconfigure(1, weight=1)  # expand horizontally

        # setup target rhythm menu
        self._target_rhythm_menu = tk.Menu(target_rhythm_menu_button, tearoff=0)
        self._target_rhythm_menu_item_count = 0
        target_rhythm_menu_button['menu'] = self._target_rhythm_menu

        self._frame_target = frame_target
        self._frame_selection = frame_selection

        # find the index of the current plot function
        plot_function_name = self._rhythm_plot_function.__name__
        plot_func_names = BSApp.RHYTHM_PLOT_TYPES_BY_NAME.values()
        current_plot_func_ix = next(i for i, f in enumerate(plot_func_names) if f == plot_function_name)

        # set the combobox to the current plot function
        combo_selection.current(current_plot_func_ix)
        combo_selection.bind("<<ComboboxSelected>>", self._on_plot_type_combobox)

    class RhythmPlottingCanvas(FigureCanvasTkAgg, object):
        def __init__(self, master, dpi, background="white", figsize=(3, 3), **kwargs):
            figure = plt.Figure(dpi=dpi, figsize=figsize, facecolor=background)
            self.background = background
            super().__init__(figure=figure, master=master, **kwargs)

        @property
        def widget(self):
            return self.get_tk_widget()

        @contextmanager
        def figure_update(self):
            figure = self.figure
            figure.clear()
            figure.set_facecolor(self.background)
            yield figure
            self.draw()

        def reset(self):
            with self.figure_update() as _:
                pass

    class SelectedRhythmsFrame(tk.Frame):
        def __init__(self, root, n_boxes=20, dpi=100, background=None, **kwargs):
            super().__init__(root, background=background, **kwargs)
            scrolled_frame = HorizontalScrolledFrame(self, bg=background)
            self._boxes = tuple(box(scrolled_frame.interior, dpi, background=background)
                                for box in repeat(self.SelectedRhythmBox, n_boxes))
            for box in self._boxes:
                box.on_mousewheel = self._handle_canvas_mousewheel

            scrolled_frame.pack(expand=True, fill=tk.BOTH)
            self._scrolled_frame = scrolled_frame

        class SelectedRhythmBox(tk.Frame, object):
            def __init__(self, master, dpi, background="white", **kwargs):
                super().__init__(master, background=background, **kwargs)
                self._plot = BSRhythmComparisonStrip.RhythmPlottingCanvas(self, dpi, background)
                self._prev_redraw_args = ()
                self._plot.widget.pack(expand=True, fill=tk.BOTH)
                self._mousewheel_callback = None
                self._mousewheel_callback_id = None

            def redraw(self, rhythm_loop: RhythmLoop, plot_function: tp.Callable):
                # plot function should be a @plot decorated RhythmLoopPlotter method
                if (rhythm_loop, plot_function) == self._prev_redraw_args:
                    return

                with self._plot.figure_update() as figure:
                    if rhythm_loop is not None:
                        plot_function(rhythm_loop, figure=figure)
                        figure.suptitle(rhythm_loop.name, fontsize="small", y=0.13)

                self._prev_redraw_args = (rhythm_loop, plot_function)

            @property
            def on_mousewheel(self):
                return self._mousewheel_callback

            @on_mousewheel.setter
            def on_mousewheel(self, callback):
                canvas = self._plot.figure.canvas

                if callback is None and self._mousewheel_callback_id is not None:
                    canvas.mpl_disconnect(self._mousewheel_callback_id)
                    self._mousewheel_callback_id = None
                    self._mousewheel_callback = None
                    return

                if not callable(callback):
                    raise TypeError("Expected callable but got \"%s\"" % callback)

                self._mousewheel_callback = callback
                cid = canvas.mpl_connect("scroll_event", callback)
                self._mousewheel_callback_id = cid

        def redraw(self, rhythms: tp.Iterable[Rhythm], plot_function: tp.Callable):
            n_boxes = len(self._boxes)
            is_first = True

            for i, [box, rhythm] in enumerate(zip_longest(self._boxes, rhythms)):
                if i >= n_boxes:
                    break

                box.redraw(rhythm, plot_function)

                # Pack at least one box in order to stretch the parent's y
                # to the correct size
                if is_first or rhythm is not None:
                    box.pack(side=tk.LEFT)
                else:
                    box.pack_forget()

                is_first = False

        def _handle_canvas_mousewheel(self, event):
            scrolled_frame = self._scrolled_frame
            right = event.button == "up"
            scrolled_frame.xview_scroll(1 if right else -1, tk.UNITS)

    class TargetRhythmBox(tk.Frame):
        def __init__(self, master, dpi, background=None, **kwargs):
            super().__init__(master, background=background, **kwargs)
            container = tk.Frame(self)
            container.pack(side="top", fill="both", expand=True)
            container.grid_rowconfigure(0, weight=1)
            container.grid_columnconfigure(0, weight=1)

            self._screen_no_target = tk.Label(container, text="No target rhythm set", bg=background)
            self._screen_main = tk.Frame(container, bg=background)

            plot_canvas = BSRhythmComparisonStrip.RhythmPlottingCanvas(self._screen_main, dpi, background)
            plot_canvas.widget.pack()
            self._plot_canvas = plot_canvas

            for screen in [self._screen_no_target, self._screen_main]:
                screen.grid(row=0, column=0, sticky="nsew")

            container.pack(padx=3)

        def redraw(self, rhythm_loop: RhythmLoop, plot_function: tp.Callable):
            with self._plot_canvas.figure_update() as figure:
                figure.clear()
                if rhythm_loop is not None:
                    plot_function(rhythm_loop, figure=figure)
                    figure.suptitle(rhythm_loop.name, fontsize="small", y=0.13)
                    self._screen_main.tkraise()
                else:
                    self._screen_no_target.tkraise()

    def redraw(self):
        self.redraw_rhythm_plots()
        self.redraw_target_rhythm_menu()

    def set_rhythm_plot_function(self, plot_function_name):
        self._rhythm_plot_function = getattr(self.rhythm_plotter, plot_function_name)
        self.redraw_rhythm_plots()

    def redraw_rhythm_plots(self):
        controller = self.controller
        plot_function = self._rhythm_plot_function

        target_rhythm_frame = self._frame_target
        target_rhythm = controller.get_target_rhythm()
        target_rhythm_frame.redraw(target_rhythm, plot_function)

        rhythm_selection_frame = self._frame_selection
        rhythm_selection = controller.get_rhythm_selection()
        rhythm_selection_frame.redraw(
            iter(controller.get_rhythm_by_index(ix) for ix in rhythm_selection),
            plot_function
        )

    def _on_plot_type_combobox(self, event):
        value = event.widget.get()
        plot_function_name = BSApp.RHYTHM_PLOT_TYPES_BY_NAME[value]
        self.set_rhythm_plot_function(plot_function_name)

    def _on_rhythm_load(self, source_type):
        controller = self.controller
        loader = controller.get_rhythm_loader(source_type)
        rhythm = loader.load()
        if rhythm is not None:
            controller.set_target_rhythm(rhythm)

    def redraw_target_rhythm_menu(self):
        controller = self.controller
        menu = self._target_rhythm_menu

        # reset menu
        menu.delete(0, self._target_rhythm_menu_item_count)
        self._target_rhythm_menu_item_count = 0

        for rhythm_loader_type, rhythm_loader in controller.get_rhythm_loader_iterator():
            rhythm_source_name = rhythm_loader.get_source_name()
            label = "From %s..." % rhythm_source_name
            menu.add_command(
                label=label,
                command=partial(self._on_rhythm_load, rhythm_loader_type),
                state=tk.NORMAL if rhythm_loader.is_available() else tk.DISABLED
            )
            self._target_rhythm_menu_item_count += 1


class BSMainMenu(tk.Menu, object):
    def __init__(self, root, **kwargs):
        tk.Menu.__init__(self, root, **kwargs)
        f_menu = tk.Menu(self, tearoff=0)
        f_menu.add_command(label="Load corpus", command=self._show_open_corpus_file_dialog)
        f_menu.add_separator()
        f_menu.add_command(label="Exit", command=lambda *_: self.on_request_exit())
        self.add_cascade(label="File", menu=f_menu)
        self._on_request_load_corpus = no_callback
        self._on_request_exit = no_callback

    @property
    def on_request_load_corpus(self):  # type: () -> tp.Callable[str]
        return self._on_request_load_corpus

    @on_request_load_corpus.setter
    def on_request_load_corpus(self, callback):
        if not callable(callback):
            raise Exception("Expected callable but got \"%s\"" % str(callback))
        self._on_request_load_corpus = callback

    @property
    def on_request_exit(self):
        return self._on_request_exit

    @on_request_exit.setter
    def on_request_exit(self, callback):
        if not callable(callback):
            raise Exception("Expected callable but got \"%s\"" % str(callback))
        self._on_request_exit = callback

    def _show_open_corpus_file_dialog(self):
        fname = tkinter.filedialog.askopenfilename(
            title="Open rhythm corpus file",
            filetypes=[("Pickle files", "*.pkl")],
            parent=self.master
        )

        if not os.path.isfile(fname):
            return

        self.on_request_load_corpus(fname)


class BSApp(tk.Tk, object):
    WINDOW_TITLE = "BeatSearch search tool"

    STYLES = {
        'inner-pad-x': 0,
        'inner-pad-y': 6.0
    }

    FRAME_RHYTHM_LIST = "<Frame-RhythmList>"
    FRAME_TRANSPORT = "<Frame-Transport>"
    FRAME_SEARCH = "<Frame-Search>"
    FRAME_RHYTHM_COMPARISON_STRIP = "<Frame-RhythmComparisonStrip>"

    RHYTHM_PLOT_TYPES_BY_NAME = OrderedDict((
        ("Chronotonic", "chronotonic"),
        ("Polygon", "polygon"),
        ("Schillinger", "schillinger"),
        ("Spectral", "spectral"),
        ("TEDAS", "tedas"),
        ("IOI Histogram", "inter_onset_interval_histogram"),
    ))

    FONT = {
        'header': ("Helvetica", 14),
        'normal': ("Helvetica", 12)
    }

    def __init__(
            self,
            controller: BSController = BSController(),  # TODO Get rid of mutable default
            search_frame_cls: tp.Type[BSSearchForm] = BSSearchForm,
            rhythms_frame_cls: tp.Type[BSRhythmList] = BSRhythmList,
            transport_frame_cls: tp.Union[tp.Type[BSTransportControls], None] = BSTransportControls,
            rhythm_comparison_strip_frame_cls: tp.Type[BSRhythmComparisonStrip] = BSRhythmComparisonStrip,
            main_menu: tp.Union[BSMainMenu, tp.Type[BSMainMenu], None] = BSMainMenu,
            background="#EEEEEE",
            **kwargs
    ):
        tk.Tk.__init__(self, **kwargs)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Control-c>", eat_args(self.close))
        signal.signal(signal.SIGTERM, eat_args(self.close))
        signal.signal(signal.SIGINT, eat_args(self.close))
        self.is_closed = True

        self.wm_title(BSApp.WINDOW_TITLE)
        self.config(bg=background)
        self._midi_rhythm_loader_by_dialog = BSMidiRhythmLoopLoader()
        self.controller = self._controller = controller
        self.drum_loop_plotter = RhythmLoopPlotter()

        self._menubar = type_check_and_instantiate_if_necessary(main_menu, BSMainMenu, allow_none=True, root=self)
        self.frames = OrderedDict()

        # frame name, frame class, instantiation args, pack args
        frame_info = (
            (
                BSApp.FRAME_SEARCH,
                search_frame_cls,
                dict(background=background),
                dict(expand=False, fill=tk.X, pady=3)
            ),

            (
                BSApp.FRAME_RHYTHM_LIST,
                rhythms_frame_cls,
                dict(background=color_variant(background, 1)),
                dict(expand=True, fill=tk.BOTH, pady=(3, 0))
            ),

            (
                BSApp.FRAME_RHYTHM_COMPARISON_STRIP,
                rhythm_comparison_strip_frame_cls,
                dict(background_left=color_variant(background, -0.055), background_right=background),
                dict(expand=False, fill=tk.X, pady=0),
            ),

            (
                BSApp.FRAME_TRANSPORT,
                transport_frame_cls,
                dict(),
                dict(expand=False, fill=tk.X, pady=(3, 0))
            ),
        )

        padx = BSApp.STYLES['inner-pad-x']

        for frame_name, frame_cls, frame_args, pack_args in frame_info:
            if frame_cls is None:
                continue
            frame = frame_cls(self, **frame_args)
            self.frames[frame_name] = frame
            frame.pack(**{
                'side': tk.TOP,
                'padx': padx,
                **pack_args
            })

        self._setup_menubar()
        self._setup_frames()

        redraw_frames_on_controller_callbacks = {
            BSController.RHYTHM_SELECTION: [BSApp.FRAME_TRANSPORT, BSApp.FRAME_RHYTHM_COMPARISON_STRIP],
            BSController.RHYTHM_PLAYBACK_START: [BSApp.FRAME_TRANSPORT],
            BSController.RHYTHM_PLAYBACK_STOP: [BSApp.FRAME_TRANSPORT],
            BSController.CORPUS_LOADED: [BSApp.FRAME_RHYTHM_LIST],
            BSController.DISTANCES_TO_TARGET_UPDATED: [BSApp.FRAME_RHYTHM_LIST],
            BSController.TARGET_RHYTHM_SET: [BSApp.FRAME_SEARCH, BSApp.FRAME_RHYTHM_COMPARISON_STRIP],
            BSController.DISTANCE_MEASURE_SET: [BSApp.FRAME_SEARCH]
        }

        for action, frames in redraw_frames_on_controller_callbacks.items():
            def get_callback(_frames):
                def callback(*_, **__):
                    self.redraw_frames(*_frames)
                return callback
            self.controller.bind(action, get_callback(frames))

        # set loading error handler on new rhythm loaders
        self.controller.bind(
            BSController.RHYTHM_LOADER_REGISTERED,
            lambda loader: setattr(loader, "on_loading_error", self._on_loading_error)
        )

        self.redraw_frames()

    @property
    def controller(self):  # type: () -> tp.Union[BSController, None]
        return self._controller

    @controller.setter
    def controller(self, controller: tp.Union[BSController, None]):
        self.unbind_all("<space>")

        # reset rhythm loading error handlers
        for _, loader in controller.get_rhythm_loader_iterator():
            if loader.on_loading_error == self._on_loading_error:
                loader.on_loading_error = no_callback

        if controller is None:
            self._controller = None
            return

        if not isinstance(controller, BSController):
            raise TypeError("Expected a BSController but got \"%s\"" % str(controller))

        if controller.is_rhythm_player_set():  # bind space for whole application
            self.bind_all("<space>", eat_args(self._toggle_rhythm_playback))

        # add rhythm loader from MIDI file with dialog
        controller.register_rhythm_loader(self._midi_rhythm_loader_by_dialog)

        # set rhythm loading error handlers
        for _, loader in controller.get_rhythm_loader_iterator():
            loader.on_loading_error = self._on_loading_error

        self._controller = controller

    def redraw_frames(self, *frame_names):
        if not frame_names:
            frame_names = self.frames.keys()
        for name in frame_names:
            try:
                self.frames[name].redraw()
            except KeyError:
                pass

    def get_frame_names(self):
        return list(self.frames.keys())

    def mainloop(self, n=0):
        self.is_closed = False
        super().mainloop(n)

    def close(self):
        self.quit()
        self.destroy()
        self.is_closed = True

    def _setup_frames(self):
        search_frame = self.frames[BSApp.FRAME_SEARCH]
        rhythms_frame = self.frames[BSApp.FRAME_RHYTHM_LIST]
        rhythms_frame.on_request_target_rhythm = self._handle_target_rhythm_request
        rhythms_frame.on_request_rhythm_plot = self._handle_rhythm_plot_request
        search_frame.search_command = self.controller.calculate_distances_to_target_rhythm
        search_frame.on_new_measure = self.controller.set_distance_measure
        search_frame.on_new_tracks = self.controller.set_tracks_to_compare

        def on_new_quantize_unit(unit_name, controller):  # type: (str, BSController) -> None
            unit = Unit.get_unit_by_name(unit_name)
            controller.set_measure_quantization_unit(unit)

        search_frame.on_new_quantize = partial(on_new_quantize_unit, controller=self.controller)

        try:
            transport_frame = self.frames[BSApp.FRAME_TRANSPORT]
            transport_frame.toggle_play_command = self._toggle_rhythm_playback
        except KeyError:
            pass

    def _setup_menubar(self):
        menubar = self._menubar
        if menubar is None:
            return
        menubar.on_request_load_corpus = self.controller.set_corpus
        menubar.on_request_exit = self.close
        self.config(menu=menubar)

    def _toggle_rhythm_playback(self):
        controller = self.controller
        if controller.is_rhythm_player_playing():
            controller.stop_rhythm_playback()
        else:
            controller.playback_selected_rhythms()

    def _handle_target_rhythm_request(self, rhythm_ix):
        controller = self.controller
        rhythm = controller.get_rhythm_by_index(rhythm_ix)
        controller.set_target_rhythm(rhythm)

    def _handle_rhythm_plot_request(self, rhythm_indices, plot_type_name):
        controller = self.controller
        plotter = self.drum_loop_plotter
        plot_function_name = self.RHYTHM_PLOT_TYPES_BY_NAME[plot_type_name]
        plot_function = getattr(plotter, plot_function_name)
        rhythms = iter(controller.get_rhythm_by_index(ix) for ix in rhythm_indices)
        for rhythm in rhythms:
            plot_function(rhythm)
            plt.draw()
        plt.show()

    @staticmethod
    def _on_loading_error(loading_error: BSRhythmLoopLoader.LoadingError):
        messagebox.showerror(
            title="Rhythm loading error",
            message=str(loading_error)
        )


class ToggleButton(tk.Button):
    def __init__(
            self,
            master,
            text,
            background=(None, "#009688"),
            foreground=(None, None),
            **kwargs
    ):
        tk.Button.__init__(self, master, **kwargs)

        if not (0 < len(text) <= 2):
            raise ValueError("Expected a tuple or list containing exactly 1 or 2 elements but got '%s'" % str(text))

        # set false colors to default
        background = [self.cget("background") if not c else c for c in background]
        foreground = [self.cget("foreground") if not c else c for c in foreground]

        self._toggled = False
        self._background = background
        self._foreground = foreground
        self._text = text
        self._enabled = True
        self.set_toggle(False)

    def set_toggle(self, toggle=True):
        self._toggled = bool(toggle)
        self.redraw()

    def redraw(self):
        i = int(self._toggled)
        self.config(
            background=self._background[i],
            fg=self._foreground[i],
            text=self._text[i],
            state=tk.NORMAL if self._enabled else tk.DISABLED
        )

    def toggle(self):
        self.set_toggle(not self._toggled)

    def set_enabled(self, enabled=True):
        self._enabled = bool(enabled)
        self.redraw()


class HorizontalScrolledFrame(tk.Frame, object):
    """https://stackoverflow.com/a/16198198/5508855"""

    def __init__(self, parent, *args, **kw):
        tk.Frame.__init__(self, parent, *args, **kw)

        # create a canvas object and a vertical scrollbar for scrolling it
        scrollbar = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        scrollbar.pack(fill=tk.X, side=tk.BOTTOM, expand=tk.FALSE)
        canvas = tk.Canvas(self, bd=0, highlightthickness=0, xscrollcommand=scrollbar.set, bg=self.cget("bg"))
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.TRUE)
        scrollbar.config(command=canvas.xview)
        self._canvas = canvas

        # reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # create a frame inside the canvas which will be scrolled with it
        self.interior = interior = tk.Frame(canvas)
        canvas.create_window(0, 0, window=interior, anchor=tk.NW)

        # track changes to the canvas and frame height and sync them, also updating the scrollbar
        def configure_interior(_):
            # update the scrollbars to match the size of the inner frame
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqheight() != canvas.winfo_height():
                # update the canvas's height to fit the inner frame
                canvas.config(height=interior.winfo_reqheight())

        interior.bind("<Configure>", configure_interior)

    def xview_scroll(self, number, what):
        canvas = self._canvas
        interior_width = self.interior.winfo_width()
        canvas_width = canvas.winfo_width()
        # reset if interior is smaller than canvas
        if interior_width < canvas_width:
            canvas.xview_moveto(0)
            return
        canvas.xview_scroll(number, what)

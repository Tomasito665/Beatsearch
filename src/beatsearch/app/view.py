import ttk
# noinspection PyPep8Naming
import Tkinter as tk
import tkFont
from functools import wraps
from typing import Type, Callable
from collections import OrderedDict
from beatsearch.data.rhythm import (
    TrackDistanceMeasure,
    TRACK_WILDCARDS,
    Unit
)
from beatsearch.utils import head_trail_iter, no_callback
from beatsearch.app.control import BSController


class BSAppFrame(tk.Frame):
    def __init__(self, app, **kwargs):
        if not isinstance(app, BSApp):
            raise TypeError("Expected a BSApp but got '%s'" % app)
        tk.Frame.__init__(self, app, **kwargs)
        self.controller = app.controller

    def redraw(self):
        raise NotImplementedError


class BSSearchForm(object, BSAppFrame):
    INNER_PAD_X = 6

    COMBO_DISTANCE_MEASURE = "<Combo-DistanceMeasure>"
    COMBO_TRACKS = "<Combo-Tracks>"
    COMBO_QUANTIZE = "<Combo-Quantize>"

    def __init__(self, controller, **kwargs):
        BSAppFrame.__init__(self, controller, **kwargs)

        combobox_info = [
            (self.COMBO_DISTANCE_MEASURE, "Distance measure", TrackDistanceMeasure.get_measure_names()),
            (self.COMBO_TRACKS, "Tracks to compare", TRACK_WILDCARDS),
            (self.COMBO_QUANTIZE, "Quantize", Unit.get_unit_names())
        ]

        widgets = []
        self._combo_boxes = {}

        for is_first, is_last, [name, text, values] in head_trail_iter(combobox_info):
            box_container = tk.Frame(self)

            # setup combobox
            combobox = ttk.Combobox(box_container, values=values, state="readonly")
            combobox.current(0)
            combobox.pack(side=tk.BOTTOM, fill=tk.X, expand=True)
            self._combo_boxes[name] = combobox

            # setup label
            box_label = tk.Label(box_container, text=text, anchor=tk.W)
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
        pass

    @property
    def on_new_measure(self):  # type: () -> Callable[str]
        return self._callbacks[self.COMBO_DISTANCE_MEASURE]

    @property
    def on_new_tracks(self):  # type: () -> Callable[str]
        return self._callbacks[self.COMBO_TRACKS]

    @property
    def on_new_quantize(self):  # type: () -> Callable[str]
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
        # type: (str, Callable[str]) -> None
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


class BSRhythmList(BSAppFrame):

    def __init__(self, app, h_scroll=False, v_scroll=True, **kwargs):
        BSAppFrame.__init__(self, app, **kwargs)
        column_headers = BSController.get_rhythm_info_names()
        self._tree_view = ttk.Treeview(columns=column_headers, show="headings")

        scrollbars = {
            tk.X: ttk.Scrollbar(orient="horizontal", command=self._tree_view.xview),
            tk.Y: ttk.Scrollbar(orient="vertical", command=self._tree_view.yview)
        }

        self.bind_all("<MouseWheel>", self._on_mousewheel)

        self._tree_view.bind("<<TreeviewSelect>>", self._on_tree_select)
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

    def _fill_tree_view(self):
        controller = self.controller
        column_headers = controller.get_rhythm_info_names()

        for col in column_headers:
            self._tree_view.heading(
                col, text=col,
                command=lambda c=col: BSRhythmList.sort_tree_view(self._tree_view, c, False)
            )

            # adjust column width to header string
            self._tree_view.column(col, width=tkFont.Font().measure(col))

        for item in controller.get_rhythm_info():
            self._tree_view.insert("", tk.END, values=item)

            # adjust column width if necessary to fit each value
            for ix, val in enumerate(item):
                col_w = tkFont.Font().measure(val)
                if self._tree_view.column(column_headers[ix], width=None) < col_w:
                    self._tree_view.column(column_headers[ix], width=col_w)

        self._corpus_name = controller.get_corpus_name()

    def _clear_tree_view(self):
        tv = self._tree_view
        tv.delete(*tv.get_children())

    def _corpus_changed(self):  # TODO get rid of assumption that corpora won't be named alike
        corpus_name = self.controller.get_corpus_name()
        return self._corpus_name != corpus_name

    def redraw(self):
        controller = self.controller

        if self._corpus_changed():
            self._clear_tree_view()
            self._fill_tree_view()
            return

        rhythm_info = tuple(controller.get_rhythm_info())

        tv = self._tree_view
        for item_iid in tv.get_children():
            rhythm_ix = self._get_rhythm_index(item_iid)
            tv.item(item_iid, values=rhythm_info[rhythm_ix])

    def _on_mousewheel(self, event):
        self._tree_view.yview_scroll(-1 * (event.delta / 15), tk.UNITS)

    def _get_rhythm_index(self, tree_view_item_iid):
        tv = self._tree_view
        tv_item = tv.item(tree_view_item_iid)
        tv_values = tv_item['values']
        return tv_values[0]

    def _on_tree_select(self, _):
        controller = self.controller
        tv = self._tree_view
        tv_selection = tv.selection()
        selected_rhythms = [self._get_rhythm_index(item_iid) for item_iid in tv_selection]
        controller.rhythm_selection_set(selected_rhythms)

    @staticmethod
    def sort_tree_view(tree_view, column, descending=False):
        column_type = BSController.RHYTHM_INFO_STRUCTURE[column]  # type to cast the data to
        data = [(tree_view.set(child, column), child) for child in tree_view.get_children("")]  # data to sort
        data.sort(reverse=descending, key=lambda cell_data: column_type(cell_data[0]))  # cast for correct sort
        for i, item in enumerate(data):
            tree_view.move(item[1], "", i)
        tree_view.heading(column, command=lambda c=column: BSRhythmList.sort_tree_view(tree_view, c, not descending))


class BSTransportControls(object, BSAppFrame):
    def __init__(self, controller, **kwargs):
        BSAppFrame.__init__(self, controller, **kwargs)
        self._btn_toggle_play = ToggleButton(self, text=("Play", "Stop"), width=20)
        self._btn_toggle_play.pack(side=tk.LEFT, padx=6)
        self._play_command = None

    def redraw(self):
        controller = self.controller
        rhythm_selection = controller.get_rhythm_selection()
        is_playing = controller.are_rhythms_playing_back()
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


class BSApp(tk.Tk):
    WINDOW_TITLE = "BeatSearch search tool"

    STYLES = {
        'inner-pad-x': 0,
        'inner-pad-y': 6.0
    }

    FRAME_RHYTHM_LIST = '<Frame-RhythmList>'
    FRAME_TRANSPORT = '<Frame-Transport>'
    FRAME_SEARCH = '<Frame-Search>'

    def __init__(
            self,
            controller=BSController(),
            search_frame_cls=BSSearchForm,
            rhythms_frame_cls=BSRhythmList,
            transport_frame_cls=BSTransportControls
    ):  # type: (BSController, Type[BSSearchForm], Type[BSRhythmList], Type[BSTransportControls]) -> None

        tk.Tk.__init__(self)
        self.wm_title(BSApp.WINDOW_TITLE)
        self.controller = controller

        self.frames = OrderedDict()

        frame_info = (
            (BSApp.FRAME_SEARCH, search_frame_cls),
            (BSApp.FRAME_RHYTHM_LIST, rhythms_frame_cls),
            (BSApp.FRAME_TRANSPORT, transport_frame_cls)
        )

        for frame_name, frame_cls in frame_info:
            if frame_cls is None:
                continue
            self.frames[frame_name] = frame_cls(self)

        self._setup_frames()

        pady = BSApp.STYLES['inner-pad-y'] / 2.0
        padx = BSApp.STYLES['inner-pad-x']

        for is_first, is_last, frame in head_trail_iter(self.frames.values()):
            frame.pack(
                side=tk.TOP,
                fill=tk.BOTH,
                expand=True,
                padx=padx,
                pady=(pady * 2 if is_first else pady, pady * 2 if is_last else pady)
            )

        redraw_frames_on_controller_callbacks = {
            BSController.RHYTHM_SELECTION: [BSApp.FRAME_TRANSPORT],
            BSController.RHYTHM_PLAYBACK_START: [BSApp.FRAME_TRANSPORT],
            BSController.RHYTHM_PLAYBACK_STOP: [BSApp.FRAME_TRANSPORT],
            BSController.CORPUS_LOADED: [BSApp.FRAME_RHYTHM_LIST],
            BSController.DISTANCES_TO_TARGET_UPDATED: [BSApp.FRAME_RHYTHM_LIST],
        }

        for event_name, frames in redraw_frames_on_controller_callbacks.items():
            def get_callback(_frames):
                def callback(*_, **__):
                    self.redraw_frames(*_frames)
                return callback
            self.controller.bind(event_name, get_callback(frames))

        self.redraw_frames()

    def redraw_frames(self, *frame_names):
        if not frame_names:
            frame_names = self.frames.keys()
        for name in frame_names:
            self.frames[name].redraw()

    def get_frame_names(self):
        return self.frames.keys()

    def _setup_frames(self):
        search_frame = self.frames[BSApp.FRAME_SEARCH]
        transport_frame = self.frames[BSApp.FRAME_TRANSPORT]
        search_frame.search_command = self.controller.update_distances
        search_frame.on_new_measure = self.controller.set_distance_measure
        search_frame.on_new_tracks = self.controller.set_tracks_to_compare
        if transport_frame is not None:
            transport_frame.toggle_play_command = self._toggle_rhythm_playback

    def _toggle_rhythm_playback(self):
        controller = self.controller
        if controller.are_rhythms_playing_back():
            controller.stop_rhythm_playback()
        else:
            controller.playback_selected_rhythms()


class ToggleButton(tk.Button):
    def __init__(
            self,
            master,
            text,
            background=(None, 'green'),
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

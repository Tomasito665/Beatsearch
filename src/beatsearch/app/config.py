import os
import uuid
import configparser
import typing as tp
from contextlib import contextmanager
from beatsearch.utils import get_beatsearch_dir
from beatsearch.rhythm import get_drum_mapping_reducer_implementation


class BSConfigSettingHandle(object):
    class StateError(Exception):
        pass

    def __init__(self, section: str, name: str, default_value: tp.Any, to_type: tp.Callable,
                 to_str: tp.Callable = None, validate: tp.Callable = None):

        if validate and not callable(validate):
            raise TypeError("validate function should be callable")

        self.config_parser = None       # type: tp.Union[configparser.ConfigParser, None]
        self._validate = validate       # type: tp.Callable[[tp.Any], bool]
        self._to_type = to_type         # type: tp.Callable[[str], tp.Any]
        self._to_str = to_str or str    # type: tp.Callable[[tp.Any], str]
        self._default_value = None      # type: tp.Union[tp.Any, None]
        self._listeners = []            # type: tp.List[tp.Callable[[tp.Any, tp.Any], tp.Any], ...]
        self._binding_setters = []      # type: tp.List[tp.Callable[[tp.Any], tp.Any], ...]

        self.section = section  # type: str
        self.name = name        # type: str
        self.default_value = default_value  # type: tp.Any

    def set(self, value: tp.Union[tp.Any, str]) -> None:
        """Sets the value of this setting

        :param value: new value of the setting
        :return: None

        :raises StateError: if config_parser not set
        """

        old_value = self.get()
        self._do_set(value)
        self._notify_bindings()
        self._notify_listeners(old_value)

    def _do_set(self, value: tp.Union[tp.Any, str]) -> None:
        parser = self._check_and_get_parser()
        section = self.section

        if isinstance(value, str):
            try:
                value = self._to_type(value)
            except Exception as e:
                print("couldn't convert \"%s\" with \"%s\"" % (str(value), self._to_type))
                raise e

        if self._validate and not self._validate(value):
            raise ValueError(value)

        if section not in parser:
            parser[section] = {}

        if not self._is_representable_as_string(value):
            raise RuntimeError("value \"%s\" is not representable as a string without information loss" % value)

        parser[section][self.name] = self._to_str(value)

    def get(self) -> tp.Any:
        """Returns the value of this setting or the default if not set

        :return: the value of this setting or the default if not set
        :raises StateError: if config_parser not set
        """

        parser = self._check_and_get_parser()

        try:
            value_as_str = parser[self.section][self.name]
        except KeyError:
            # if not set, set it to the default value and return that
            self._do_set(self.default_value)
            return self.default_value

        return self._to_type(value_as_str)

    def add_binding(self, binding_setter: tp.Callable[[tp.Any], tp.Any]):
        """Adds a binding to this setting

        Bindings are variables that are bound to this setting. The given binding setter should be a callable receiving
        the value of this setting as its single parameter. The binding setter will be called immediately immediately
        after it's bound and later at every change through the set() method.

        :param binding_setter: callable receiving the setting value as its single parameter
        :return: None
        """

        if not callable(binding_setter):
            raise TypeError

        binding_setter(self.get())
        self._binding_setters.append(binding_setter)

    def remove_binding(self, binding_setter: tp.Callable[[tp.Any], tp.Any]) -> bool:
        """Removes a binding

        :param binding_setter: setter of binding to remove
        :return: True if it was removed; False if it not added
        """

        try:
            self._binding_setters.remove(binding_setter)
        except ValueError:
            return False

        return True

    def add_listener(self, callback: tp.Union[tp.Callable[[tp.Any], tp.Any]]) -> None:
        """Adds a change listener to this setting

        Adds a change listener to this setting. The listeners will be called immediately after the execution of set().

        :param callback: callback receiving one or two parameters, consecutively the new value and the old value
        :return: None
        """

        if not callable(callback):
            raise TypeError
        self._listeners.append(callback)

    def remove_listener(self, callback: tp.Callable[[tp.Any], tp.Any]) -> bool:
        """Removes the given change listener of this setting

        :param callback: callback to remove
        :return: True if it was removed; False if it was never added
        """

        try:
            self._listeners.remove(callback)
        except ValueError:
            return False

        return True

    @property
    def default_value(self) -> tp.Any:
        """The default value of this setting returned by get() if this setting was not set yet"""
        return self._default_value

    @default_value.setter
    def default_value(self, default_value: tp.Any):
        if not self._is_representable_as_string(default_value):
            raise ValueError("given default value \"%s\" is not representable as a "
                             "string without information loss" % default_value)
        self._default_value = default_value

    def _notify_bindings(self) -> None:
        value = self.get()
        for binding_setter in self._binding_setters:
            binding_setter(value)

    def _notify_listeners(self, old_value: tp.Any) -> None:
        new_value = self.get()
        for callback in self._listeners:
            callback(old_value, new_value)

    def _is_representable_as_string(self, value_original: tp.Any) -> bool:
        value_as_str = self._to_str(value_original)
        value_converted_back = self._to_type(value_as_str)
        return value_original == value_converted_back

    def _check_and_get_parser(self) -> configparser.ConfigParser:
        # returns config parser and raises a StateError if not yet set
        parser = self.config_parser
        if not parser:
            raise self.StateError("ConfigParser not set")
        return parser


class BSConfigSection(object):
    DEFAULT = "Default"
    SERIALIZED_RHYTHM_CORPORA = "SerializedRhythmCorpora"


def normalize_directory(directory):
    """Normalizes a directory path

    Normalizes the directory path, replaces the backslashes with forward slashes and removes the ending slash.

    :param directory: directory path to normalize
    :return: normalized directory path
    """

    directory = os.path.normpath(directory)
    directory = directory.replace("\\", "/")
    if directory.endswith("/"):
        directory = directory[:-1]
    return directory


class BSConfig(object):
    INI_FILE_NAME = "settings.ini"
    MIDI_DIRECTORY_CACHE_FILE_EXTENSION = ".pkl"

    def __init__(self, config_dir: str = None):
        if not config_dir:
            config_dir = get_beatsearch_dir(True)
            assert os.path.isdir(config_dir)
        elif not os.path.isdir(config_dir):
            raise ValueError("no such directory: %s" % config_dir)

        self._config_directory = config_dir

        # Config parsers use both '=' and ':' per default. We only want '=' because
        # we'll have paths as keys, which may contain an ':' (e.g. C:\...)
        self._config_parser = configparser.ConfigParser(delimiters="=")

        # The ini file is located at the root of the beatsearch configuration directory
        self._ini_file_path = os.path.join(config_dir, self.INI_FILE_NAME)

        ###################
        # Setting handles #
        ###################

        self.midi_root_directory = BSConfigSettingHandle(
            BSConfigSection.DEFAULT, "midi_root_dir", "", str,
            validate=lambda val: not val or os.path.isdir(val)
        )

        self.rhythm_resolution = BSConfigSettingHandle(
            BSConfigSection.DEFAULT, "rhythm_resolution", 240, int,
            validate=lambda val: val >= 0
        )

        self.mapping_reducer = BSConfigSettingHandle(
            BSConfigSection.DEFAULT, "midi_mapping_reducer", None,
            lambda name: None if name == "None" else get_drum_mapping_reducer_implementation(name),
            lambda reducer: "None" if reducer is None else reducer.__name__
        )

        for setting in (self.midi_root_directory, self.rhythm_resolution, self.mapping_reducer):
            setting.config_parser = self._config_parser

        # If there already exists an ini file, load that in
        if os.path.isfile(self._ini_file_path):
            with open(self._ini_file_path, "r", encoding="utf8") as ini_file:
                self._config_parser.read_file(ini_file)

    @contextmanager
    def add_midi_root_directory_cache_file(self, midi_directory: str, file_mode="wb", save=True):
        if not os.path.isdir(midi_directory):
            raise ValueError("not a directory: %s" % midi_directory)

        midi_directory = normalize_directory(midi_directory)
        assert os.path.isdir(midi_directory), "midi dir exists, but normalized midi dir does not"
        parser = self._config_parser

        try:
            cache_ids = set(item[1] for item in parser.items(BSConfigSection.SERIALIZED_RHYTHM_CORPORA))
        except configparser.NoSectionError:
            cache_ids = set()

        # generate unique new cache id
        generate_cache_id = lambda: "midi_dir_%s" % str(uuid.uuid4())
        new_cache_id = generate_cache_id()
        while new_cache_id in cache_ids:
            new_cache_id = generate_cache_id()

        cache_fpath = self._get_midi_directory_cache_fpath(new_cache_id)
        with open(cache_fpath, file_mode) as cache_file:
            yield cache_file

        try:
            parser.add_section(BSConfigSection.SERIALIZED_RHYTHM_CORPORA)
        except configparser.DuplicateSectionError:
            pass

        parser.set(BSConfigSection.SERIALIZED_RHYTHM_CORPORA, midi_directory, new_cache_id)

        if save:
            self.save()

    def get_midi_root_directory_cache_fpath(self, midi_directory):
        midi_directory = normalize_directory(midi_directory)
        cache_id = self._get_midi_directory_cache_id(midi_directory)
        if not cache_id:
            return None
        return self._get_midi_directory_cache_fpath(cache_id)

    def forget_midi_root_directory_cache_file(self, midi_directory, remove_cache_file=True):
        midi_directory = normalize_directory(midi_directory)
        cache_id = self._get_midi_directory_cache_id(midi_directory)
        if not cache_id:
            return False
        if remove_cache_file:
            cache_fpath = self._get_midi_directory_cache_fpath(cache_id)
            try:
                os.remove(cache_fpath)
            except OSError:
                # in case that the file didn't exist
                pass
        parser = self._config_parser
        return parser.remove_option(BSConfigSection.SERIALIZED_RHYTHM_CORPORA, cache_id)

    def get_cache_directory(self, mkdir=True):
        root_dir = self._config_directory
        cache_directory = os.path.join(root_dir, "cache")
        if mkdir and not os.path.isdir(cache_directory):
            os.mkdir(cache_directory)
        return cache_directory

    def reload(self):
        parser = self._config_parser
        ini_fpath = self._ini_file_path

        if os.path.isfile(ini_fpath):
            with open(ini_fpath, "r", encoding="utf8") as ini_file:
                parser.read_file(ini_file)

    def save(self):
        parser = self._config_parser
        ini_fpath = self._ini_file_path

        with open(ini_fpath, "w", encoding="utf-8") as configfile:
            parser.write(configfile)

    def _get_midi_directory_cache_fpath(self, cache_id):
        # returns the midi directory cache file path
        cache_dir = self.get_cache_directory()
        return os.path.join(cache_dir, cache_id + self.MIDI_DIRECTORY_CACHE_FILE_EXTENSION)

    def _get_midi_directory_cache_id(self, midi_directory):
        parser = self._config_parser
        midi_directory = normalize_directory(midi_directory)
        return parser.get(BSConfigSection.SERIALIZED_RHYTHM_CORPORA, midi_directory, fallback=None)


__all__ = ['BSConfigSettingHandle', 'BSConfigSection', 'BSConfig']

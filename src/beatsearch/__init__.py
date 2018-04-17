import os
import sys

__version__ = "0.0.1.dev7"


def find_beatsearch_project_dir(max_levels: int = 5) -> str:
    cd = os.path.dirname(os.path.realpath(__file__))

    def check_directory(directory):
        return all(os.path.isfile(os.path.join(directory, fname)) for fname in [
            "build_dependencies.py", "setup.py"
        ])

    for level in range(max_levels):
        if check_directory(cd):
            return cd
        next_dir = os.path.dirname(cd)
        if next_dir == cd:
            break
        cd = os.path.dirname(cd)

    return ""


BEATSEARCH_PROJECT_DIR = find_beatsearch_project_dir()
if BEATSEARCH_PROJECT_DIR:
    sys.path.append(BEATSEARCH_PROJECT_DIR)
    from build_dependencies import Python3Midi, get_third_party_dir

    python3_midi = Python3Midi(get_third_party_dir(BEATSEARCH_PROJECT_DIR))

    if not python3_midi.has_build():
        print("Python3Midi hasn't built yet, please run build_dependencies.py")
        sys.exit(-1)

    try:
        import midi
    except ModuleNotFoundError:
        sys.path.append(python3_midi.lib_dir)

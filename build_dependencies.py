import os
import sys
import subprocess


def get_third_party_dir(beatsearch_project_dir=None):
    beatsearch_project_dir = beatsearch_project_dir or os.path.dirname(os.path.realpath(__file__))
    return os.path.join(beatsearch_project_dir, "third-party")


class Python3Midi(object):
    PROJ_DIR_NAME = "python3-midi"

    def __init__(self, third_party_dir: str):
        if not os.path.isdir(third_party_dir):
            raise ValueError("No such directory: %s" % third_party_dir)
        self.project_dir = os.path.join(third_party_dir, self.PROJ_DIR_NAME)
        self.lib_dir = os.path.join(self.project_dir, "build", "lib")

    def build(self, force=False):
        if self.has_build() and not force:
            return
        project_dir = self.project_dir
        subprocess.check_call((sys.executable, os.path.join(
            project_dir, "setup.py"), "build"), cwd=project_dir)

    def has_build(self):
        module_dir = os.path.join(self.lib_dir, "midi")
        return os.path.isdir(module_dir)


if __name__ == "__main__":
    python3midi = Python3Midi(get_third_party_dir())
    python3midi.build()

import os
import sys
import subprocess

BEATSEARCH_LIB_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
BEATSEARCH_PROJECT_DIR = os.path.dirname(BEATSEARCH_LIB_DIR)
THIRD_PARTY_DIR = os.path.join(BEATSEARCH_PROJECT_DIR, "third-party")

PYTHON3_MIDI_PROJECT_DIR = os.path.join(THIRD_PARTY_DIR, "python3-midi")
PYTHON3_MIDI_LIB_DIR = os.path.join(PYTHON3_MIDI_PROJECT_DIR, "build", "lib")

sys.path.append(BEATSEARCH_LIB_DIR)

if not os.path.isdir(os.path.join(PYTHON3_MIDI_LIB_DIR, "midi")):
    subprocess.check_call((sys.executable, os.path.join(PYTHON3_MIDI_PROJECT_DIR, "setup.py"), "build"),
                          cwd=PYTHON3_MIDI_PROJECT_DIR)

sys.path.append(PYTHON3_MIDI_LIB_DIR)

import os
import sys
import logging
from setuptools import setup

sys.path.insert(0, os.path.join(".", "src"))
import beatsearch

try:
    from build_dependencies import Python3Midi, get_third_party_dir
    build_dependencies = True
except ModuleNotFoundError:
    build_dependencies = False

if build_dependencies and not Python3Midi(get_third_party_dir()).has_build():
    logging.critical("Python3Midi hasn't built yet, please run build_dependencies.py")
    sys.exit(-1)

with open("README.rst") as f:
    logging.debug("Reading README.rst")
    long_desc = f.read()

setup(
    name="beatsearch",
    version=beatsearch.__version__,
    packages=["beatsearch", "midi"],
    package_dir={
        '': "src",
        'midi': "third-party/python3-midi/build/lib/midi"
    },
    url="https://tomasito665.github.io/beatsearch",
    download_url="https://pypi.org/project/beatsearch/",
    license="MIT License",
    author="Jordi Ortolá Ankum",
    author_email="jordi665@hotmail.com",
    description="Symbolic rhythm analysis",
    long_description=long_desc,
    install_requires=["numpy", "matplotlib", "anytree"]
)

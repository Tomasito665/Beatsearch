import sys
from setuptools import setup

try:
    from build_dependencies import Python3Midi, get_third_party_dir
    build_dependencies = True
except ModuleNotFoundError:
    build_dependencies = False

if build_dependencies and not Python3Midi(get_third_party_dir()).has_build():
    print("Python3Midi hasn't built yet, please run build_dependencies.py")
    sys.exit(-1)

setup(
    name='beatsearch',
    version='0.0.1.dev4',
    packages=['beatsearch', 'midi'],
    package_dir={
        '': "src",
        'midi': "third-party/python3-midi/build/lib/midi"
    },
    url='',
    license='MIT License',
    author='Jordi Ortolá Ankum',
    author_email='jordi665@hotmail.com',
    description='',
    install_requires=['numpy', 'matplotlib']
)

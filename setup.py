import sys
from setuptools import setup
from build_dependencies import Python3Midi, get_third_party_dir

if not Python3Midi(get_third_party_dir()).has_build():
    print("Python3Midi hasn't built yet, please run build_dependencies.py")
    sys.exit(-1)

setup(
    name='beatsearch',
    version='0.0.0',
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

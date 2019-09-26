import codecs
import os
import pathlib
import re

from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent


def read(*parts):
    with codecs.open(os.path.join(HERE, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


__name__ = 'Tea'
__author__ = "DiegoSong, TianjinMouth"
__copyright__ = "Copyright 2019, TianjinMouth"
__credits__ = []
__license__ = "Apache License 2.0"
__maintainer__ = "DiegoSong, TianjinMouth"
__email__ = "ssyshenn@gmail.com, tjszgaosan@sina.cn"

__url__ = 'https://github.com/BrikerMan/Kashgari'
__description__ = 'Automated Modeling in Financial Domain.' \
                  ' Tea is a simple and design friendly automatic modeling learning framework.'

__version__ = find_version('Tea', 'version.py')
README = (HERE / "README.md").read_text(encoding='utf-8')

with codecs.open('requirements.txt', 'r', 'utf8') as reader:
    install_requires = list(map(lambda x: x.strip(), reader.readlines()))

setup(
    name=__name__,
    version=__version__,
    description=__description__,
    python_requires='>3.6',
    long_description=README,
    long_description_content_type="text/markdown",
    author=__author__,
    author_email=__email__,
    url=__url__,
    packages=find_packages(exclude=('tests',)),
    install_requires=install_requires,
    include_package_data=True,
    license=__license__,
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        # 'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)

if __name__ == "__main__":
    print("Hello world")

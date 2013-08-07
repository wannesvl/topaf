import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="pathfollowing",
    version="0.1",
    author="Wannes Van Loock",
    author_email="wannes.vanloock@gmail.com",
    description=("Calculate solutions to time optimal path following problems"),
    license="LGPLv3",
    keywords="optimization",
    url="https://github.com/wannesvl/topaf",
    packages=['pathfollowing'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    ],
)

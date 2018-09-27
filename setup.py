# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
import sys
import os
import glob

import numpy as np
from numpy.distutils.core import setup
from numpy.distutils.extension import Extension

version = 0.1

requirements = [
    "strawberryfields>=0.8.0",
]


extensions = [Extension("torontonian_samples",
    sources=['src/kinds.f90', 'src/structures.f90', 'src/vars.f90', 'src/torontonian_samples.f90'],
    extra_compile_args=["-std=c99 -O3 -Wall -fPIC -shared -fopenmp"],
    extra_link_args=['-fopenmp'])]

info = {
    'name': 'torontonian_samples',
    'version': version,
    'maintainer': 'Xanadu Inc.',
    'maintainer_email': 'brajesh@xanadu.ai',
    'url': 'http://xanadu.ai',
    'license': 'Apache License 2.0',
    'packages': [
                    'torontonian_samples',
                ],
    'description': 'Open source Python package for Torontonian sampling',
    'long_description': open('README.md').read(),
    'provides': ["torontonian_samples"],
    'install_requires': requirements,
    'ext_modules': extensions,
    # 'cmdclass': {'build_ext': build_ext},
    'command_options': {
        'build_sphinx': {
            'version': ('setup.py', version),
            'release': ('setup.py', version)}}
}

classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3 :: Only',
    "Topic :: Scientific/Engineering :: Physics"
]

setup(classifiers=classifiers, **(info))

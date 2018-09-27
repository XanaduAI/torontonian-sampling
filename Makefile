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

PYTHON=python3

all: python

python: torontonian_samples.cpython-*

torontonian_samples.cpython-*:
	mkdir -p torontonian_samples
	$(PYTHON) setup.py build
	cp build/lib*/torontonian_samples.cpython-* .
	rm -r torontonian_samples

examples: fortran
	$(MAKE) -C examples

fortran:
	$(MAKE) -C src
	mkdir -p include
	mv src/*.o include/
	mv src/*.mod include/

.PHONY: clean

clean:
	rm -rf *.so *.dll build bin
	$(MAKE) -C src clean
	$(MAKE) -C examples clean

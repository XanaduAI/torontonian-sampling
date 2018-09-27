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

# compiler
FC := gfortran
# compile flags
FCFLAGS =
# link flags
LDFLAGS = -fopenmp
# include
INCLUDE = ../src

# Program name
PROG = fortran_example
# source files
SRC = $(PROG).f90
# object files
OBJ = $(SRC:.f90=.o) ../src/*.o

all: $(PROG)

$(PROG): $(OBJ)
	$(FC) $(FCFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.f90
	$(FC) $(FCFLAGS) -c $< -I$(INCLUDE)

.PHONY: clean veryclean

clean:
	rm -f *.o $(PROG)

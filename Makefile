

library:
	f2py -c -m TorontonianSamples --f90flags='-fopenmp' -lgomp main.f90
clean:
	rm -rf TorontonianSamples*.so *~

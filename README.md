# TorontonianLib

Fortran code wrapped into python to generate samples from Torontonian.

Requirements: 
1. Python wrapper for fortran programs "f2py" should be installed on your computer.
2. Standard GNU libraries which come with any Linux distribution. 
3. Strawberry Fields software to generate covariance matrices for the Gaussian states. You may need to upgrade SF in order 
    to use some of the function not available in the standard installation. Run the following command: 
            
            pip3 install https://github.com/XanaduAI/strawberryfields/archive/master.zip --upgrade

This should be enough for the library to be called from Python. 

Take a look at the TorontonianSampling.ipynb for an example.

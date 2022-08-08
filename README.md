# Phase-retrieval-algorithm-for-SLM
This is a Python toolkit for finding the phase mask needed to display on the SLM for a given desired image.

main.py is the main module that contains three sections: initialisation, shape of desired image, calculate phase mask, and equalise trap intensities with feedback loop. Need to make sure the Python module PIL is downloaded.

phase_retrieval.py is a base module that contains the two functions used. They are the Gerchberg-Saxton algorithm and a 2D Gaussian function, and both of them are imported to the main module main.py. Need to make sure the Python module PIL is downloaded.

lens.py is a supplementary module that calculates the phase mask of lenses given focal length and wavelength of incoming light. Additional Python module 'diffractio' must be downloaded to run this module.

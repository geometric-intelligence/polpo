POLPO
=====

A `Geometric Intelligence Lab <https://gi.ece.ucsb.edu/>`_'s collection of weakly-related tools.


Installation
------------
To install ``polpo`` and its required dependencies:


::

    pip install polpo@git+https://github.com/geometric-intelligence/polpo.git@main


Or equivalently, first manually clone the repo and proceed with a local installation of dependencies:

:: 

    git clone https://github.com/geometric-intelligence/polpo.git
    cd polpo
    pip install .


Optional dependencies
*********************


``polpo`` depends heavily on external libraries.
To minimize installation load, different sets of optional dependencies are available in `pyproject.toml <./pyproject.toml>`_.
Choose the one that is more convenient to your use case.


For example, to install ``polpo`` and all the optional dependencies for the dash capabilities, you can run:

::

    pip install polpo[dash]@git+https://github.com/geometric-intelligence/polpo@main

NB: Use flag ``-e`` for editable mode.

To install all optional dependencies, you can run:
::
    pip install polpo[all]@git+https://github.com/geometric-intelligence/polpo@main.

This is recommended if you plan to use ``polpo`` for development purposes, as it will install all the optional dependencies required for testing and other features.


Virtual environments
********************


Often, it is convenient to install a package within a virtual environment.
We recommend using `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_:

::

    conda create -n polpo python=3.11
    conda activate polpo
    
after which you can install ``polpo`` as described above.

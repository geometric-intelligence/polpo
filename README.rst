POLPO
=====

A `Geometric Intelligence Lab <https://gi.ece.ucsb.edu/>`_'s collection of weakly-related tools.


Installation
------------

The following are three common ways for installating a Python package from a github repo.

To install ``polpo`` and its required dependencies:


::

    pip install polpo@git+https://github.com/geometric-intelligence/polpo.git@main


To install ``polpo`` and all its optional dependencies:


::

    pip install polpo[all]@git+https://github.com/geometric-intelligence/polpo@main


Or, first manually clone the repo and proceed with a local installation:

:: 

    git clone https://github.com/geometric-intelligence/polpo.git
    cd polpo
    pip install .


NB: Use flag ``-e`` for editable mode.
Optional dependencies can be installed as above (e.g. ``pip install .[all]``).



Virtual environments
********************


Often, it is convenient to install a package within a virtual environment.
We recommend using `conda <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_:

::

    conda create -n polpo python=3.11
    conda activate polpo
    


Optional dependencies
*********************

``polpo`` depends heavily on external libraries.
To minimize installation load, different sets of optional dependencies are available in `pyproject.toml <./pyproject.toml>`_.
Choose the one that is more convenient to your use case.
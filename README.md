# Welcome to CASBI (Chemical Abundance Simulation Based Inference)

 [![arXiv](https://img.shields.io/badge/arXiv-2107.07405-b31b1b.svg)](https://arxiv.org/abs/2411.17269)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/casbi/badge/?version=latest)](https://casbi.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/vepe99/CASBI/branch/main/graph/badge.svg)](https://codecov.io/gh/vepe99/CASBI)

![CASBI Complete](./notebooks/CASBI_complete.png)



## Installation

The Python package `CASBI` can be installed from PyPI:

```
python -m pip install CASBI
```

## Development installation

If you want to contribute to the development of `CASBI`, we recommend
the following editable installation from this repository:

```
git clone https://github.com/vepe99/CASBI.git
cd CASBI
python -m pip install --editable .[tests]
```

Having done so, the test suite can be run using `pytest`:

```
python -m pytest
```



## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).


## Cite

>how to cite this code:
>if you find this code useful for your analyses, please cite the associated paper, as well as:
>
>-NeurIPS paper: [https://arxiv.org/abs/2411.17269](https://arxiv.org/abs/2411.17269)

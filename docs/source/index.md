# Introduction and Installation

`jf1uids` is a one-dimensional radial fluid solver written in `JAX`.


::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Fast and Differentiable

Written in `JAX`, `jf1uids` is fully differentiable - a simulation can be differentiated with respect to any input parameter - and just-in-time compiled for fast execution on CPU, GPU, or TPU.

+++
[Learn more »](notebooks/preprocessing.ipynb)
:::

:::{grid-item-card} {octicon}`shield-check;1.5em;sd-mr-1` Well-considered Numerical Methods

Conservative properties of the flow are maintained in the radial simulations based on the approach of [Crittenden and Balachandar (2018)](https://doi.org/10.1007/s00193-017-0784-y) with a well-tested MUSCL-Hancock Riemann solver at its core.

+++
[Learn more »](notebooks/conservational_properties.ipynb)
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Extensible

Physical modules, like one for stellar winds, can be easily added to the simulation. As we 
are dealing with a differentiable simulator, these modules can include parameters (or neural
network terms) which can be optimized directly in the solver.

+++
[Learn more »](notebooks/wind_parameter_optimization.ipynb)
:::

::::

## Installation

`jf1uids` can be installed via `pip`

```bash
pip install jf1uids
```

Note that if `JAX` is not yet installed, only the CPU version of `JAX` will be installed
as a dependency. For a GPU-compatible installation of `JAX`, please refer to the
[JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

:::{tip} Get started with this [simple example](notebooks/simple_example.ipynb).

```{toctree}
:hidden:
:maxdepth: 2
:caption: Introduction

self
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Notebooks

notebooks/simple_example.ipynb
notebooks/conservational_properties.ipynb
notebooks/gradients_through_stellar_wind.ipynb
notebooks/wind_parameter_optimization.ipynb
```

```{toctree}
:hidden:
:caption: Reference

source/jf1uids.time_stepping.time_integration
apidocs/index
```
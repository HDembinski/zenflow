# zenflow

[![](https://img.shields.io/pypi/v/zenflow.svg)](https://pypi.org/project/zenflow/)
[![Coverage Status](https://coveralls.io/repos/github/HDembinski/zenflow/badge.svg?branch=main)](https://coveralls.io/github/HDembinski/zenflow?branch=main)
[![DOI](https://zenodo.org/badge/757339505.svg)](https://zenodo.org/doi/10.5281/zenodo.13236492)

Fast conditional flow-based generative models which are implemented as FLAX modules. Conditional PDFs are supported.

## License

The source code is released under the MIT license.

## Installation

```sh
pip install zenflow
```

## Documentation

There is currently no online documentation, but the library has useful docstrings. Please use the docstrings and look into the usage examples in the `examples` folder.

## Citation

If you use this package in a scientific work, please cite us. You can generate citations in your preferred format on the [Zenodo website](https://zenodo.org/doi/10.5281/zenodo.13236492).

## History

This project was originally forked from [PZFlow](https://github.com/jfcrenshaw/pzflow) by [John Franklin Crenshaw](jfcrenshaw@gmail.com), but largely rewritten. PZFlow itself draws from other repositories, which are listed in the PZFlow documentation. I needed a code base which is simple to understand and stripped down to the essentials for my use case. The main differences between PZFlow and zenflow:

* zenflow uses generic JAX arrays for data input and output, while PZFlow enforces the use of Pandas dataframes.
* zenflow implements all trainable objects as FLAX modules, while PZFlow uses JAX primitives.
* zenflow follows a clean functional design inspired by FLAX.
* zenflow is minimalistic and focussed on providing trainable conditional PDFs with the neural spline coupling technique.

Because all trainable objects are FLAX modules, including the flow object and all bijectors, one can make new FLAX modules that build on flow objects or bijectors, and train the combination. This, for example, allows one to construct conditional PDFs that use complex models like transformers to provide the conditional input, and train everything together. The examples include an application that uses a Deep Set as conditional input for the PDF.

Features of PZFlow that are not included in zenflow:
* Training on data points with uncertainties.
* Computing marginalized posterior densities. This can be done "by hand", however.
* Bijectors that are not needed for the neural spline coupling approach.
* Support for periodic variables.

Features which are planned:
* Support for periodic variables.

# zenflow

[![](https://img.shields.io/pypi/v/zenflow.svg)](https://pypi.org/project/zenflow/)
[![Coverage Status](https://coveralls.io/repos/github/HDembinski/zenflow/badge.svg?branch=main)](https://coveralls.io/github/HDembinski/zenflow?branch=main)

This library implements a flow-based generative model and bijectors which are implemented as FLAX modules. Conditional flows are supported.

## License

The source code is released under the MIT license.

## Installation

```sh
pip install zenflow
```

## Documentation

There is currently no online documentation, but the library has useful docstrings. Please use the docstrings and look into the usage examples in the `examples` folder.

## History

This project was originally forked from [PZFlow](https://github.com/jfcrenshaw/pzflow) by [John Franklin Crenshaw](jfcrenshaw@gmail.com), but largely rewritten. PZFlow itself draws from other repositories, which are listed in the PZFlow documentation. I needed a code base which is simple to understand and stripped down to the essentials for my use case. Differences between PZFlow and zenflow:

* zenflow uses generic JAX arrays for data input and output, while PZFlow enforces the use of Pandas dataframes.
* zenflow implements all trainable objects as FLAX modules, while PZFlow uses JAX primitives.
* Like FLAX, zenflow follows a clean functional design.
* PZflow supports training on data points with uncertainties, zenflow has no support for that.
* PZflow supports computing marginalized posterior densities, zenflow has no support for that.
* PZflow supports periodic data, which is currently not supported by zenflow (I plan to reintroduce this later).
* PZflow supports more bijectors than zenflow.

Because all trainable objects, like the flow object itself and its bijectors are FLAX modules, one can make new FLAX modules that build on flow objects or bijectors, and train the combination. This, for example, allows one to construct conditional PDFs that use complex models like transformers to provide the conditional input, and train everything together. The examples include an application that uses a Deep Set as conditional input for the PDF.

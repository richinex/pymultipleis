
# pymultipleis

[**Installation**](#installation)
| [**Examples**](https://github.com/richinex/pymultipleis/tree/main/docs/source/examples)
| [**Documentation**](https://pymultipleis.readthedocs.io/en/latest/index.html)
| [**Citing this work**](#citation)


A library for fitting a sequence of electrochemical impedance spectra.

- Implements algorithms for simultaneous and sequential fitting.

- Written in python and based on the [JAX library](https://github.com/google/jax).

- Leverages JAX's in-built automatic differentiation ([autodiff](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)) of Python functions.

- Takes advantage of JAX's just-in-time compilation (JIT) of Python code to [XLA](https://www.tensorflow.org/xla) which runs on GPU or TPU hardware.


## Installation<a id="installation"></a>

pymultipleis requires the following:

-   Python (>=3.9)
-   [JAX](https://jax.readthedocs.io/en/latest/) (>=0.3.17)

Installing JAX on Linux is natively supported by the JAX team and instructions to do so can be found [here](https://github.com/google/jax#installation).

For Windows systems, the officially supported method is building directly from the source code (see [Building JAX from source](https://jax.readthedocs.io/en/latest/developer.html#building-from-source)).
However, it might be easier to use pre-built JAX wheels which can be found in this [Github repo](https://github.com/cloudhan/jax-windows-builder). Further details
on Windows installation is also provided in this [repo](https://github.com/Dipolar-Quantum-Gases/jaxfit/blob/main/README.md).


After installing JAX, you can now install pymultipleis via the following pip command

```bash
pip install pymultipleis
```

[Getting started with pymultipleis](https://pymultipleis.readthedocs.io/en/latest/quick-start-guide.html#) contains a quick start guide to
fitting your data with ``pymultipleis``.


## Examples

Jupyter notebooks which cover several aspects of ``pymultipleis`` can be found in [Examples](https://github.com/richinex/pymultipleis/tree/main/docs/source/examples).

## Documentation

Details about the ``pymultipleis`` API, can be found in the [reference documentation](https://pymultipleis.readthedocs.io/en/latest/index.html).


## Citing this work<a id="citation"></a>

If you use pymultipleis for academic research, you may cite the library as follows:

```
@misc{Chukwu2022,
  author = {Chukwu, Richard},
  title = {pymultipleis: a library for fitting a sequence of electrochemical impedance spectra},
  publisher = {GitHub},
  year = {2022},
  url = {https://github.com/richinex/pymultipleis},
}
```
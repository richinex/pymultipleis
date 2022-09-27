<div align="center">
<img src="https://github.com/richinex/pymultipleis/blob/main/docs/source/_static/z_bode.png" alt="logo"></img>
</div>


pymultipleis
=============

[**Installation**](#installation)
| [**Examples**](https://github.com/richinex/pymultipleis/tree/main/docs/source/examples)
| [**Documentation**](https://pymultipleis.readthedocs.io/en/latest/index.html)
| [**References**](#references)


A library for fitting a sequence of electrochemical impedance spectra.

- Implements algorithms for simultaneous and sequential fitting.

- Written in python and based on the [JAX library](https://github.com/google/jax).

- Leverages JAX's in-built automatic differentiation ([autodiff](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)) of Python functions.

- Takes advantage of JAX's just-in-time compilation (JIT) of Python code to [XLA](https://www.tensorflow.org/xla) which runs on GPU or TPU hardware.


## Installation<a id="installation"></a>

pymultipleis requires the following:

-   Python (>=3.9)
-   [JAX](https://jax.readthedocs.io/en/latest/) (>=0.3.17)
-   [JAXopt](https://github.com/google/jaxopt/blob/main/README.md) (>=0.5)
-   Matplotlib (>=3.6.0)
-   NumPy (>=1.23.3)
-   Pandas (>=1.4.4)
-   SciPy (>=1.9.1)

Installing JAX on Linux is natively supported by the JAX team and instructions to do so can be found [here](https://github.com/google/jax#installation).

For Windows systems, the officially supported method is building directly from the source code (see [Building JAX from source](https://jax.readthedocs.io/en/latest/developer.html#building-from-source)).
However, it might be easier to use pre-built JAX wheels which can be found in this [Github repo](https://github.com/cloudhan/jax-windows-builder). Further details
on Windows installation is also provided in this [repo](https://github.com/Dipolar-Quantum-Gases/jaxfit/blob/main/README.md).


After installing the dependencies, you can now install pymultipleis via the following pip command

```
pip install pymultipleis
```

[Getting started with pymultipleis](https://pymultipleis.readthedocs.io/en/latest/quick-start-guide.html#) contains a quick start guide to
fitting your data with ``pymultipleis``.


## Examples

Detailed tutorials on other aspects of ``pymultipleis`` can be found in [Examples](https://github.com/richinex/pymultipleis/tree/main/docs/source/examples).

## Documentation

Details about the ``pymultipleis`` API, can be found in the [reference documentation](https://pymultipleis.readthedocs.io/en/latest/index.html).


## References

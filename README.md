pymultipleis
=============

   ``"Simplicity is the ultimate sophistication - Leonardo da Vinci"``

``pymultipleis`` offers a robust approach to batch-fitting electrochemical impedance spectra with a dependence.
Dependence implies that the spectra being fitted are gradually varying or similar to each other
and were obtained as a result of continuous change of in the property of the electrochemical system under study.
Such properties include but are not limited to temperature, potential, state of charge and depth of discharge.

The batch-fitting algorithm implemented in pymultieis allows the kinetic parameters of the system
such as the charge transfer resistance, double layer capacitance and Warburg coefficient to be obtained
as curves which vary as a function of the dependent variable under study.

The ``py`` in ``pymultipleis`` represents python while the ``multipleis`` is an abbreviation for ``Multiple Electrochemical Impedance Spectra``.

``pymultipleis`` offers methods modules for model fiting, model validation, visualization,


Installation
*************

   pip install pymultieis

[Getting started with pymultipleis](https://pymultieis.readthedocs.io/en/latest/getting-started.html) contains a step-by-step tutorial
fitting your data with ``pymultipleis``.

Dependencies
**************

impedance.py requires:

-   Python (>=3.9)
-   Jax (>=1.23.3)
-   Jaxopt (>=1.23.3)
-   SciPy (>=1.9.1)
-   NumPy (>=1.23.3)
-   Matplotlib (>=3.6.0)



Several example notebooks are provided in the examples/ directory.
Opening these will require Jupyter notebook or Jupyter lab.

Examples
*********************

Detailed tutorials on several aspects of ``pymultipleis`` can be found on the [examples page](https://pymultipleis.readthedocs.io/en/latest/examples.html).

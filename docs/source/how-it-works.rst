.. _how-it-works-label:

=========================================
How :code:`pymultipleis` works
=========================================

The batch fitting algorithm implemented in :code:`pymultipleis` is described in the paper
On the Analysis of Non-stationary Impedance Spectra by Alberto Battistel, Guoqing Du, and Fabio La Mantia.
Fitting is done via complex non-linear optimization of the model parameters using two approaches - deterministic and stochastic.
The deterministic optimization is based on the TNC/BFGS/L-BFGS solvers provided by the jaxopt library
which uses the real first and second derivatives computed behind the scenes via autograd.
The stochastic option uses Adam optimizer from the the jax library.

Rather than rely on the use prefit and use previous approach to batch-fitting,
the algorithm implemented in pymultieis preserves the correlation between parameters by introducing a custom cost function
which is a combination of the scaled version of the chisquare used in complex nonlinear regression and two additional terms:

- Numerical integral of the second derivative of the parameters with respect to the immittance and
- A smoothing factor.

Minimizing these additional terms allow the algorithm to minimize the number of times the curve changes concavity.
This allows the minimization algorithm to obtain smoothly varying optimal parameters. The parameters thus obtained can
now be used as initial guess while the smoothing factor is set to zero. In this way the dependence between the parameters is
preserved.



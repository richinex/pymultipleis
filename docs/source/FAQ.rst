.. _FAQ-label:

===================================================
Frequently asked questions
===================================================

1. How do I obtain the standard deviation of my impedance measurements?

The straightforward approach to obtain the variance of the error in impedance measurements would be to take replicate measurements;
however, this method is unreliable due to unwanted long-term drifts of the experimental conditions.
If the spectra were obtained via dynamic multi-frequency analysis `DMFA <https://doi.org/10.1002/elan.201600260>`_ under the
assumption that the noise in the voltage and current is constant and independent of time, then
this `paper <https://doi.org/10.1002/celc.202200109>`_ provides an empirical expression for the standard deviation of the admittance derived from the principles of error propagation.
Another approach for obtaining the standard deviation of impedance measurements is the use of measurement models.
Measurement models allow the assessment of the variance of the errors from semi-replicate or sequential impedance measurements.
The `measurement model program <https://ecsarxiv.org/kze9x/>`_ by Watson and Orazem can be used for this purpose.


2. What if I have just one spectra?

While ``pymultieis`` is meant to be used for a sequence of spectra, It can also be tweaked to fit a single spectra.
The trick is to repeat the single spectra up to a certain number, say 10 and use ``fit_simultaneous()`` or ``fit_stochastic()``
with the smoothing factor for all parameters set to ``inf``. For instance:

.. code-block:: python

  Y_single_spectra = Y_her[:, 40]
  Y_single_spectra.shape
  # torch.Size([35])

  Y_repeated = torch.tile(Y_her_single_spectra[:,None], (1, 10))
  Y_repeated.shape
  # torch.Size([35, 10])

  smf = torch.full((len(p0),), torch.inf)
  eis_her = pym.Multieis(p0, F_her, Y_repeated, bounds, smf, her, weight= 'modulus', immittance='admittance')
  popt, perr, chisqr, chitot, AIC = eis_her.fit_stochastic()
  popt, perr, chisqr, chitot, AIC = eis_her.fit_sequential()

See the notebook ``smoothing-factor-effect.ipynb`` under `Examples <https://github.com/richinex/pymultieis/tree/main/docs/source/examples>`_ for more details.

3. How do I choose the right smoothing factor?

From experience with working with multiple datasets, we recommend choosing a smothing factor of 1.0 when the weighting is the modulus and
a smoothing factor higher than 100000.0 when the standard deviation is used as weighting. Keep in mind, however, that the aim of using a smoothing factor is to
obtain a better initial guess after which the minimization algorithm is run again with the smoothing set to zero.



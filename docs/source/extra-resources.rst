.. _extra-resources-label:

===================================================
Extra Resources
===================================================


Distributed Elements
===================================================

Semi-infinite (planar infinite length) Warburg
***************************************************

Describes linear diffusion from a medium with lebgth which can be approximated
as infinite.

.. math:: Z_{W} = \frac{A_W}{\sqrt{w}}(1-j)
    :label: ZW

Or

.. math::
    Z_{W} = \sqrt{\frac{R_d}{s C_d}} (\sqrt{R_{d}~sC_{d}})

Where :math:`s = j \omega` with :math:`j` being the imaginary unit and :math:`\omega` the angular frequency.
:math:`A_{W} has units of :math:`\Omega s^{-0.5}`, :math:`R` has units of Ohms (:math:`\Omega`) and :math:`C` has units of Farads (:math:`F`).

And

.. math::
    A_W = \frac{RT}{F^{2}C_{o}\sqrt{D_o}}

.. code-block:: python

  w = 2 * jnp.pi * freq
  s = 1j * w
  Zw = Aw/jnp.sqrt(w) * (1-1j)

  # Or

  Zw = jnp.sqrt(Rd/s*Cd) * (jnp.sqrt(Rd * s*Cd))


Finite length diffusion with reflective boundary
*****************************************************

Describes the reaction of mobile active species distributed in a layer with finite length,
terminated by an impermeable boundary.

.. math::
    Z_{Wo} = \sqrt{\frac{R_d}{s C_d}} \coth(\sqrt{R_{d}~sC_{d}})

Or

.. math:: Z_{Wo} = R \frac{coth(j \omega \tau)^{\phi}}{(j \omega \tau)_{\phi}}
    :label: ZWo


Where :math:`\phi` = 0.5

.. code-block:: python

  w = 2 * jnp.pi * freq
  s = 1j * w
  ZWs = jnp.sqrt(Rd/s*Cd) * 1/jnp.tanh(jnp.sqrt(Rd * s*Cd))


Finite length diffusion with transmissive boundary
******************************************************

Describes the reaction of mobile active species distributed in a layer with finite length,
terminated by an impermeable boundary.

.. math::
    Z_{Ws} = \sqrt{\frac{R_d}{s C_d}} \tanh(\sqrt{R_{d}~sC_{d}})

Or

.. math:: Z_{Ws} = R \frac{tanh(j \omega \tau)^{\phi}}{(j \omega \tau)_{\phi}}
    :label: ZWs

Where :math:`\phi` = 0.5

.. code-block:: python

  w = 2 * jnp.pi * freq
  s = 1j * w
  ZWs = jnp.sqrt(Rd/s*Cd) * jnp.tanh(jnp.sqrt(Rd * s*Cd))



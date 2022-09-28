import pymultipleis.multieis as pym
import numpy as onp
import jax.numpy as jnp
import pytest

# Load the file containing the frequencies
F = jnp.asarray(onp.load('data/redox_exp_50/freq_50.npy'))
# Load the file containing the admittances (a set of 50 spectra)
Y = jnp.asarray(onp.load('data/redox_exp_50/Y_50.npy'))
# Load the file containing the standard deviation of the admittances
Yerr = jnp.asarray(onp.load('data/redox_exp_50/sigma_Y_50.npy'))


def par(a, b):
    """
    Defines the total impedance of circuit elements in parallel
    """
    return 1/(1/a + 1/b)


def redox(p, f):
    w = 2*jnp.pi*f                      # Angular frequency
    s = 1j*w                            # Complex variable
    Rs = p[0]
    Qh = p[1]
    nh = p[2]
    Rct = p[3]
    Wct = p[4]
    Rw = p[5]
    Zw = Wct/jnp.sqrt(w) * (1-1j)       # Planar infinite length Warburg impedance
    Zdl = 1/(s**nh*Qh)                  # admittance of a CPE
    Z = Rs + par(Zdl, Rct + par(Zw, Rw))
    Y = 1/Z
    return jnp.concatenate((Y.real, Y.imag), axis=0)


p0 = jnp.asarray(
    [
        1.6295e+02,
        3.0678e-08,
        9.3104e-01,
        1.1865e+04,
        4.7125e+05,
        1.3296e+06
        ]
        )

bounds = [
    [1e-15 , 1e15],
    [1e-9 , 1e2],
    [1e-1 , 1e0],
    [1e-15 , 1e15],
    [1e-15 , 1e15],
    [1e-15 , 1e15]
    ]

# Smoothing factor used with the standard deviation
smf_sigma = jnp.asarray(
    [
        100000.,
        100000.,
        100000.,
        100000.,
        100000.,
        100000.
        ]
        )

# Smoothing factor used with the modulus
smf_modulus = jnp.asarray(
    [
        1.,
        1.,
        1.,
        1.,
        1.,
        1.
        ]
        )

# Smoothing factor used with the first parameter set to inf
smf_inf = jnp.asarray(
    [
        jnp.inf,
        1.,
        1.,
        1.,
        1.,
        1.
        ]
        )


# Test for weight type

all_weights = [Yerr, 'modulus', 'proportional', None]

true_weights = ['sigma', 'modulus', 'proportional', 'unit']


@pytest.mark.parametrize("weight, true_weight", list(zip(all_weights, true_weights)))
def test_weight_name(weight, true_weight):
    """Test least-squares problem on unconstrained minimizers."""

    multieis_instance = pym.Multieis(
        p0,
        F,
        Y,
        bounds,
        smf_sigma,
        redox,
        weight=weight,
        immittance='admittance'
        )

    assert (true_weight == multieis_instance.weight_name)


# Test for immittance type

all_immittances = ['admittance', 'impedance']

true_immittances = ['admittance', 'impedance']


@pytest.mark.parametrize(
    "immittance,true_immittance",
    list(zip(all_immittances, true_immittances))
    )
def test_immittance(immittance, true_immittance):
    """Test least-squares problem on unconstrained minimizers."""

    multieis_instance = pym.Multieis(
        p0,
        F,
        Y,
        bounds,
        smf_sigma,
        redox,
        weight='modulus',
        immittance=immittance
        )

    assert (true_immittance == multieis_instance.immittance)


# Test for invalid bounds
def test_invalid_bounds():

    bounds_bad = [
        [1e5 , 1e15],
        [1e-9 , 1e2],
        [1e-1 , 1e0],
        [1e-15 , 1e15],
        [1e-15 , 1e15],
        [1e-15 , 1e15]
        ]
    with pytest.raises(AssertionError) as excinfo:

        pym.Multieis(
            p0,
            F,
            Y,
            bounds_bad,
            smf_sigma,
            redox,
            weight='modulus',
            immittance='admittance'
            )
    print(str(excinfo.value))
    assert str(excinfo.value) == """Initial guess can not be
                                        greater than the upper bound
                                        or less than lower bound"""


# Test for NaN in parameters
def test_invalid_params():

    p0_bad = jnp.asarray(
        [
            1.6295e+02,
            onp.nan,
            9.3104e-01,
            1.1865e+04,
            4.7125e+05,
            1.3296e+06
            ]
            )

    with pytest.raises(Exception) as excinfo:

        pym.Multieis(
            p0_bad,
            F,
            Y,
            bounds,
            smf_sigma,
            redox,
            weight='modulus',
            immittance='admittance'
            )
    print(str(excinfo.value))
    assert str(excinfo.value) == "Values must not contain nan"


# Test for zero in parameters
def test_zero_in_params():

    p0_bad = jnp.asarray(
        [
            1.6295e+02,
            0,
            9.3104e-01,
            1.1865e+04,
            4.7125e+05,
            1.3296e+06
            ]
            )

    with pytest.raises(Exception) as excinfo:

        pym.Multieis(
            p0_bad,
            F,
            Y,
            bounds,
            smf_sigma,
            redox,
            weight='modulus',
            immittance='admittance'
            )
    print(str(excinfo.value))
    assert str(excinfo.value) == "Values must be greater than zero"

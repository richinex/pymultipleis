import os
import scipy
import scipy.sparse as sps
import jax
import jaxopt
import jax.numpy as jnp
from jax.example_libraries import optimizers as jax_opt
import numpy as onp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import logging
import collections
from typing import Callable, Optional, Dict, Union, Sequence, Tuple
from datetime import datetime

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)
plt.style.use("default")
mpl.rcParams["figure.facecolor"] = "white"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.linewidth"] = 1.5
mpl.rcParams["ps.fonttype"] = 42


class Multieis:
    """
    An immittance batch processing class

    :param p0: A 1D or 2D array of initial guess

    :param freq: An (m, ) 1D array containing the frequencies. \
                 Where m is the number of frequencies

    :param Z: An (m, n) 2D array of complex immittances. \
              Where m is the number of frequencies and \
              n is the number of spectra

    :param bounds: A sequence of (min, max) pairs for \
                   each element in p0. The values must be real

    :param smf: A array of real elements same size as p0. \
                when set to inf, the corresponding parameter is kept constant

    :param func: A model e.g an equivalent circuit model (ECM) or \
                 an arbitrary immittance expression composed as python function

    :param weight: A string representing the weighting scheme or \
                   an (m,n) 2-D array of real values containing \
                   the measurement standard deviation. \
                   Defaults to unit weighting if left unspecified.

    :param immittance: A string corresponding to the immittance type

    :returns: A Multieis instance

    """

    def __init__(
        self,
        p0: jnp.ndarray,
        freq: jnp.ndarray,
        Z: jnp.ndarray,
        bounds: Sequence[Union[int, float]],
        smf: jnp.ndarray,
        func: Callable[[float, float], float],
        immittance: str = "impedance",
        weight: Optional[Union[str, jnp.ndarray]] = None,
    ) -> None:

        assert (
            p0.ndim > 0 and p0.ndim <= 2
        ), ("Initial guess must be a 1-D array or 2-D "
            "array with same number of cols as `F`")
        assert (
            Z.ndim == 2 and Z.shape[1] >= 5
        ), "The algorithm requires that the number of spectra be >= 5"
        assert freq.ndim == 1, "The frequencies supplied should be 1-D"
        assert (
            len(freq) == Z.shape[0]
        ), ("Length mismatch: The len of F is {} while the rows of Z are {}"
            .format(len(freq), Z.shape[0]))

        # Create the lower and upper bounds
        try:
            self.lb = self.check_zero_and_negative_values(
                jnp.asarray([i[0] for i in bounds])
            )
            self.ub = self.check_zero_and_negative_values(
                jnp.asarray([i[1] for i in bounds])
            )
        except IndexError:
            print("Bounds must be a sequence of min-max pairs")

        if p0.ndim == 1:
            self.p0 = self.check_zero_and_negative_values(self.check_nan_values(p0))
            self.num_params = len(self.p0)
            assert (
                len(self.lb) == self.num_params
            ), "Shape mismatch between initial guess and bounds"
            if __debug__:
                if not jnp.all(
                    jnp.logical_and(
                        jnp.greater(self.p0, self.lb),
                        jnp.greater(self.ub, self.p0)
                    )
                ):
                    raise AssertionError("""Initial guess can not be
                                        greater than the upper bound
                                        or less than lower bound""")
        elif (p0.ndim == 2) and (1 in p0.shape):
            self.p0 = self.check_zero_and_negative_values(self.check_nan_values(p0.flatten()))
            self.num_params = len(self.p0)
            assert (
                len(self.lb) == self.num_params
            ), "Shape mismatch between initial guess and bounds"
            if __debug__:
                if not jnp.all(
                    jnp.logical_and(
                        jnp.greater(self.p0, self.lb),
                        jnp.greater(self.ub, self.p0)
                    )
                ):
                    raise AssertionError("""Initial guess can not be
                                        greater than the upper bound
                                        or less than lower bound""")
        else:
            assert p0.shape[1] == Z.shape[1], ("Columns of p0 "
                                               "do not match that of Z")
            assert (
                len(self.lb) == p0.shape[0]
            ), ("The len of p0 is {} while that of the bounds is {}"
                .format(p0.shape[0], len(self.lb)))
            self.p0 = self.check_zero_and_negative_values(self.check_nan_values(p0))
            self.num_params = p0.shape[0]

        self.immittance_list = ["admittance", "impedance"]
        assert (
            immittance.lower() in self.immittance_list
        ), "Either use 'admittance' or 'impedance'"

        self.num_freq = len(freq)
        self.num_eis = Z.shape[1]
        self.F = jnp.asarray(freq, dtype=jnp.float64)
        self.Z = self.check_is_complex(Z)
        self.Z_exp = self.Z.copy()
        self.Y_exp = 1 / self.Z_exp.copy()
        self.indices = None
        self.n_fits = None

        self.func = func
        self.immittance = immittance

        self.smf = smf

        self.smf_1 = jnp.where(jnp.isinf(self.smf), 0.0, self.smf)

        self.kvals = list(jnp.cumsum(jnp.insert(
            jnp.where(jnp.isinf(self.smf), 1, self.num_eis), 0, 0)))

        self.gather_indices = jnp.zeros(shape=(self.num_params, self.num_eis), dtype=jnp.int64)
        for i in range(self.num_params):
            self.gather_indices = \
                self.gather_indices.at[i, :].set(jnp.arange(self.kvals[i], self.kvals[i+1]))

        self.d2m = self.get_fd()
        self.dof = (2 * self.num_freq * self.num_eis) - \
            (self.num_params * self.num_eis)
        self.plot_title1 = " ".join(
            [x.title() for x in self.immittance_list if (x == self.immittance)]
        )
        self.plot_title2 = " ".join(
            [x.title() for x in self.immittance_list if x != self.immittance]
        )

        self.lb_vec, self.ub_vec = self.get_bounds_vector(self.lb, self.ub)

        # Define weighting strategies
        if isinstance(weight, jnp.ndarray):
            self.weight_name = "sigma"
            assert (
                Z.shape == weight.shape
            ), "Shape mismatch between Z and the weight array"
            self.Zerr_Re = weight
            self.Zerr_Im = weight
        elif isinstance(weight, str):
            assert weight.lower() in [
                "proportional",
                "modulus",
            ], ("weight must be one of None, "
                "proportional', 'modulus' or an 2-D array of weights")
            if weight.lower() == "proportional":
                self.weight_name = "proportional"
                self.Zerr_Re = self.Z.real
                self.Zerr_Im = self.Z.imag
            else:
                self.weight_name = "modulus"
                self.Zerr_Re = jnp.abs(self.Z)
                self.Zerr_Im = jnp.abs(self.Z)
        elif weight is None:
            # if set to None we use "unit" weighting
            self.weight_name = "unit"
            self.Zerr_Re = jnp.ones(shape=(self.num_freq, self.num_eis))
            self.Zerr_Im = jnp.ones(shape=(self.num_freq, self.num_eis))
        else:
            raise AttributeError(
                ("weight must be one of 'None', "
                 "proportional', 'modulus' or an 2-D array of weights")
            )

    def __str__(self):
        return f"""Multieis({self.p0},{self.F},{self.Z},{self.Zerr_Re},\
                {self.Zerr_Im}, {list(zip(self.lb, self.ub))},\
                {self.func},{self.immittance},{self.weight_name})"""

    __repr__ = __str__

    @staticmethod
    def check_nan_values(arr):
        if jnp.isnan(jnp.sum(arr)):
            raise Exception("Values must not contain nan")
        else:
            return arr

    @staticmethod
    def check_zero_and_negative_values(arr):
        if jnp.all(arr > 0):
            return arr
        raise Exception("Values must be greater than zero")

    @staticmethod
    def try_convert(x):
        try:
            return str(x)
        except Exception as e:
            print(e.__doc__)
            print(e.message)
        return x

    @staticmethod
    def check_is_complex(arr):
        if onp.iscomplexobj(arr):
            return jnp.asarray(arr, dtype=jnp.complex64)
        else:
            return jnp.asarray(arr, dtype=jnp.complex64)

    def get_bounds_vector(self,
                          lb: jnp.ndarray,
                          ub: jnp.ndarray
                          ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Creates vectors for the upper and lower \
        bounds which are the same length \
        as the number of parameters to be fitted.

        :param lb: A 1D array of lower bounds
        :param ub: A 1D array of upper bounds

        :returns: A tuple of bounds vectors

        """
        lb_vec = jnp.zeros(
            self.num_params * self.num_eis
            - (self.num_eis - 1)
            * jnp.sum(jnp.isinf(self.smf))
        )
        ub_vec = jnp.zeros(
            self.num_params * self.num_eis
            - (self.num_eis - 1) * jnp.sum(jnp.isinf(self.smf))
        )
        for i in range(self.num_params):
            lb_vec = lb_vec.at[self.kvals[i]:self.kvals[i + 1]].set(lb[i])
            ub_vec = ub_vec.at[self.kvals[i]:self.kvals[i + 1]].set(ub[i])
        return lb_vec, ub_vec

    def get_fd(self):
        """
        Computes the finite difference stencil \
        for a second order derivative. \
        The derivatives at the boundaries is calculated \
        using special finite difference equations
        derived specifically for just these points \
        (aka higher order boundary conditions).
        They are used to handle numerical problems \
        that occur at the edge of grids.

        :returns: Finite difference stencil for a second order derivative
        """
        self.d2m = (
            sps.diags([1, -2, 1], [-1, 0, 1],
                      shape=(self.num_eis, self.num_eis))
            .tolil()
            .toarray()
        )
        self.d2m[0, :4] = [2, -5, 4, -1]
        self.d2m[-1, -4:] = [-1, 4, -5, 2]
        return jnp.asarray(self.d2m)

    def convert_to_internal(self,
                            p: jnp.ndarray
                            ) -> jnp.ndarray:
        """
        Converts A array of parameters from an external \
        to an internal coordinates (log10 scale)

        :param p: A 1D or 2D array of parameter values

        :returns: Parameters in log10 scale
        """
        assert p.ndim > 0 and p.ndim <= 2
        if p.ndim == 1:
            par = jnp.broadcast_to(
                p[:, None],
                (self.num_params, self.num_eis)
                )
        else:
            par = p
        self.p0_mat = jnp.zeros(
            self.num_params * self.num_eis
            - (self.num_eis - 1) * jnp.sum(jnp.isinf(self.smf))
        )
        for i in range(self.num_params):
            self.p0_mat = self.p0_mat.at[self.kvals[i]:self.kvals[i + 1]].set(par[
                i, : self.kvals[i + 1] - self.kvals[i]
            ])
        p_log = jnp.log10(
            (self.p0_mat - self.lb_vec) / (1 - self.p0_mat / self.ub_vec)
        )
        return p_log

    def convert_to_external(self,
                            P: jnp.ndarray
                            ) -> jnp.ndarray:

        """
        Converts A array of parameters from an internal \
        to an external coordinates

        :param p: A 1D array of parameter values

        :returns: Parameters in normal scale
        """
        par_ext = jnp.zeros(shape=(self.num_params, self.num_eis))
        for i in range(self.num_params):
            par_ext = par_ext.at[i, :].set((
                self.lb_vec[self.kvals[i]:self.kvals[i + 1]]
                + 10 ** P[self.kvals[i]:self.kvals[i + 1]]
            ) / (
                1
                + (10 ** P[self.kvals[i]:self.kvals[i + 1]])
                / self.ub_vec[self.kvals[i]:self.kvals[i + 1]]
            ))
        return par_ext

    def compute_wrss(self,
                     p: jnp.ndarray,
                     f: jnp.ndarray,
                     z: jnp.ndarray,
                     zerr_re: jnp.ndarray,
                     zerr_im: jnp.ndarray
                     ) -> jnp.ndarray:

        """
        Computes the scalar weighted residual sum of squares \
        (aka scaled version of the chisquare or the chisquare itself)

        :param p: A 1D array of parameter values

        :param f: A 1D array of frequency

        :param z: A 1D array of complex immittance

        :param zerr_re: A 1D array of weights for \
                        the real part of the immittance

        :param zerr_im: A 1D array of weights for \
                        the imaginary part of the immittance

        :returns: A scalar value of the \
                  weighted residual sum of squares

        """
        z_concat = jnp.concatenate([z.real, z.imag], axis=0)
        sigma = jnp.concatenate([zerr_re, zerr_im], axis=0)
        z_model = self.func(p, f)
        wrss = jnp.linalg.norm(((z_concat - z_model) / sigma)) ** 2
        return wrss

    def compute_rss(self,
                    p: jnp.ndarray,
                    f: jnp.ndarray,
                    z: jnp.ndarray,
                    zerr_re: jnp.ndarray,
                    zerr_im: jnp.ndarray,
                    lb,
                    ub
                    ) -> jnp.ndarray:
        """
        Computes the vector of weighted residuals. \
        This is the objective function passed to the least squares solver.

        :param p: A 1D array of parameter values

        :param f: A 1D array of frequency

        :param z: A 1D array of complex immittance

        :param zerr_re: A 1D array of weights for \
                        the real part of the immittance

        :param zerr_im: A 1D array of weights for \
                        the imaginary part of the immittance

        :param lb: A 1D array of values for the lower bounds

        :param ub: A 1D array of values for the upper bounds

        :returns: A vector of residuals

        """
        p = (lb + 10 ** (p)) / (1 + 10 ** (p) / ub)
        z_concat = jnp.concatenate([z.real, z.imag], axis=0)
        sigma = jnp.concatenate([zerr_re, zerr_im], axis=0)
        z_model = self.func(p, f)
        residuals = 0.5*((z_concat - z_model) / sigma)
        return residuals

    def compute_wrms(self,
                     p: jnp.ndarray,
                     f: jnp.ndarray,
                     z: jnp.ndarray,
                     zerr_re: jnp.ndarray,
                     zerr_im: jnp.ndarray
                     ) -> jnp.ndarray:
        """
        Computes the weighted residual mean square

        :param p: A 1D array of parameter values

        :param f: A 1D array of frequency

        :param z: A 1D array of complex immittance

        :param zerr_re: A 1D array of weights for \
                        the real part of the immittance

        :param zerr_im: A 1D array of weights for \
                        the imaginary part of the immittance

        :returns: A scalar value of the weighted residual mean square
        """
        z_concat = jnp.concatenate([z.real, z.imag], axis=0)
        sigma = jnp.concatenate([zerr_re, zerr_im], axis=0)
        z_model = self.func(p, f)
        wrss = jnp.linalg.norm(((z_concat - z_model) / sigma)) ** 2
        wrms = wrss / (2 * len(f) - len(p))
        return wrms

    def compute_total_obj(self,
                          P: jnp.ndarray,
                          F: jnp.ndarray,
                          Z: jnp.ndarray,
                          Zerr_Re: jnp.ndarray,
                          Zerr_Im: jnp.ndarray,
                          LB: jnp.ndarray,
                          UB: jnp.ndarray,
                          smf: jnp.ndarray
                          ) -> jnp.ndarray:
        """
        This function computes the total scalar objective function to minimize
        which is a combination of the weighted residual sum of squares
        and the smoothing factor divided by the degrees of freedom

        :param P: A 1D array of parameter values

        :param F: A 1D array of frequency

        :param Z: A 2D array of complex immittance

        :param Zerr_Re: A 2D array of weights for \
                        the real part of the immittance

        :param Zerr_Im: A 2D array of weights for \
                        the imaginary part of the immittance

        :param LB: A 1D array of values for \
                   the lower bounds (for the total parameters)

        :param LB: A 1D array of values for \
                   the upper bounds (for the total parameters)

        :param smf: An array of real elements same size as p0. \
            when set to inf, the corresponding parameter is kept constant

        :returns: A scalar value of the total objective function

        """
        P_log = jnp.take(P, self.gather_indices)

        up = (10 ** P_log)

        P_norm = (
            jnp.take(LB, self.gather_indices)
            + up
        ) / (
            1
            + up
            / jnp.take(UB, self.gather_indices)
        )

        chi_smf = ((((self.d2m @ P_log.T) * (self.d2m @ P_log.T)))
                   .sum(0) * smf).sum()
        wrss_tot = jax.vmap(self.compute_wrss, in_axes=(1, None, 1, 1, 1))(
            P_norm, F, Z, Zerr_Re, Zerr_Im
        )
        return (jnp.sum(wrss_tot) + chi_smf)

    def compute_perr(self,
                     P: jnp.ndarray,
                     F: jnp.ndarray,
                     Z: jnp.ndarray,
                     Zerr_Re: jnp.ndarray,
                     Zerr_Im: jnp.ndarray,
                     LB: jnp.ndarray,
                     UB: jnp.ndarray,
                     smf: jnp.ndarray
                     ) -> jnp.ndarray:

        """
        Computes the error on the parameters resulting from the batch fit
        using the hessian inverse of the parameters at the minimum computed
        via automatic differentiation

        :param P: A 2D array of parameter values

        :param F: A 1D array of frequency

        :param Z: A 2D array of complex immittance

        :param Zerr_Re: A 2D array of weights for \
                        the real part of the immittance

        :param Zerr_Im: A 2D array of weights for \
                        the imaginary part of the immittance

        :param LB: A 1D array of values for \
                   the lower bounds (for the total parameters)

        :param LB: A 1D array of values for \
                   the upper bounds (for the total parameters)

        :param smf: An array of real elements same size as p0. \
            when set to inf, the corresponding parameter is kept constant

        :returns: A 2D array of the standard error on the parameters

        """
        P_log = self.convert_to_internal(P)

        chitot = self.compute_total_obj(P_log, F, Z, Zerr_Re, Zerr_Im, LB, UB, smf)/self.dof
        hess_mat = jax.hessian(self.compute_total_obj)(P_log, F, Z, Zerr_Re, Zerr_Im, LB, UB, smf)
        try:
            # Here we check to see if the Hessian matrix is singular \
            # or ill-conditioned since this makes accurate computation of the
            # confidence intervals close to impossible.
            hess_inv = jnp.linalg.inv(hess_mat)
        except Exception as e:
            print(e.__doc__)
            print(e.message)
            hess_inv = jnp.linalg.pinv(hess_mat)

        # The covariance matrix of the parameter estimates
        # is (asymptotically) the inverse of the hessian matrix
        cov_mat = hess_inv * chitot
        perr = jnp.zeros(shape=(self.num_params, self.num_eis))
        for i in range(self.num_params):
            perr = perr.at[i, :].set((jnp.sqrt(jnp.diag(cov_mat)))[
                self.kvals[i]:self.kvals[i + 1]
            ])
        perr = perr.copy() * P
        # if the error is nan, a value of 1 is assigned.
        return jnp.nan_to_num(perr, nan=1.0e15)

    def compute_perr_QR(self,
                        P: jnp.ndarray,
                        F: jnp.ndarray,
                        Z: jnp.ndarray,
                        Zerr_Re: jnp.ndarray,
                        Zerr_Im: jnp.ndarray
                        ) -> jnp.ndarray:

        """
        Computes the error on the parameters resulting from the batch fit
        using QR decomposition

        :param P: A 2D array of parameter values

        :param F: A 1D array of frequency

        :param Z: A 2D array of complex immittance

        :param Zerr_Re: A 2D array of weights for \
                        the real part of the immittance

        :param Zerr_Im: A 2D array of weights for \
                        the imaginary part of the immittance

        :returns: A 2D tensor of the standard error on the parameters

        Ref
        ----
        Bates, D. M., Watts, D. G. (1988). Nonlinear regression analysis \
        and its applications. New York [u.a.]: Wiley. ISBN: 0471816434

        """
        def grad_func(p,
                      f
                      ):
            return jax.jacfwd(self.func)(p, f)

        perr = jnp.zeros(shape=(self.num_params, self.num_eis))
        for i in range(self.num_eis):
            wrms = self.compute_wrms(P[:, i], F, Z[:, i], Zerr_Re[:, i], Zerr_Im[:, i])
            gradsre = grad_func(P[:, i], F)[:self.num_freq]
            gradsim = grad_func(P[:, i], F)[self.num_freq:]
            diag_wtre_matrix = jnp.diag((1/Zerr_Re[:, i]))
            diag_wtim_matrix = jnp.diag((1/Zerr_Im[:, i]))
            vre = diag_wtre_matrix@gradsre
            vim = diag_wtim_matrix@gradsim
            Q1 , R1 = jnp.linalg.qr(jnp.concatenate([vre , vim] , axis=0))
            try:
                # Here we check to see if the Hessian matrix is singular or
                # ill-conditioned since this makes accurate computation of the
                # confidence intervals close to impossible.
                invR1 = jnp.linalg.inv(R1)
            except Exception as e:
                print(e.__doc__)
                print(e.message)
                print(f"\nHessian Matrix is singular for spectra {i}")
                invR1 = jnp.linalg.pinv(R1)

            perr = perr.at[:, i].set(jnp.linalg.norm(invR1, axis=1)*jnp.sqrt(wrms))
        # if the error is nan, a value of 1 is assigned.
        return jnp.nan_to_num(perr, nan=1.0e15)

    def train_step(self,
                   step_i,
                   opt_state,
                   F,
                   Z,
                   Zerr_Re,
                   Zerr_Im,
                   LB,
                   UB,
                   smf
                   ):
        net_params = self.get_params(opt_state)
        self.loss, self.grads = jax.value_and_grad(
            self.compute_total_obj, argnums=0
            )(
                net_params,
                F,
                Z,
                Zerr_Re,
                Zerr_Im,
                LB,
                UB,
                smf
                )
        return self.loss, self.opt_update(step_i, self.grads, opt_state)

    def compute_aic(self,
                    p: jnp.ndarray,
                    f: jnp.ndarray,
                    z: jnp.ndarray,
                    zerr_re: jnp.ndarray,
                    zerr_im: jnp.ndarray,
                    ) -> jnp.ndarray:
        """
        Computes the Akaike Information Criterion according to
        `M. Ingdal et al <https://www.sciencedirect.com/science/article/abs/pii/S0013468619311739>`_

        :param p: A 1D array of parameter values

        :param f: A 1D array of frequency

        :param z: A 1D array of complex immittance

        :param zerr_re: A 1D array of weights for \
                        the real part of the immittance

        :param zerr_im: A 1D array of weights for \
                        the imaginary part of the immittance


        :returns: A value for the AIC
        """

        wrss = self.compute_wrss(p, f, z, zerr_re, zerr_im)
        if self.weight_name == "sigma":
            m2lnL = (
                (2 * self.num_freq) * jnp.log(2 * jnp.pi)
                + jnp.sum(jnp.log(zerr_re**2))
                + jnp.sum(jnp.log(zerr_im**2))
                + (wrss)
            )
            aic = m2lnL + 2 * self.num_params

        elif self.weight_name == "unit":
            m2lnL = (
                2 * self.num_freq * jnp.log(2 * jnp.pi)
                - 2 * self.num_freq
                * jnp.log(2 * self.num_freq)
                + 2 * self.num_freq
                + 2 * self.num_freq * jnp.log(wrss)
            )
            aic = m2lnL + 2 * self.num_params

        else:
            wt_re = 1 / zerr_re**2
            wt_im = wt_re
            m2lnL = (
                2 * self.num_freq * jnp.log(2 * jnp.pi)
                - 2 * self.num_freq
                * jnp.log(2 * self.num_freq)
                + 2 * self.num_freq
                - jnp.sum(jnp.log(wt_re))
                - jnp.sum(jnp.log(wt_im))
                + 2 * self.num_freq * jnp.log(wrss)
            )  # log-likelihood calculation
            aic = m2lnL + 2 * (self.num_params + 1)
        return aic

    def fit_simultaneous(self,
                         method : str = 'TNC',
                         n_iter : int = 5000,
                         ) -> Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ]:

        """
        Simultaneous fitting routine with an arbitrary smoothing factor.

        :params method: Solver to use (must be one of "'TNC', \
                        'BFGS' or 'L-BFGS-B'")

        :params n_iter: Number of iterations


        :returns: A tuple containing the optimal parameters (popt), \
                  the standard error of the parameters (perr), \
                  the objective function at the minimum (chisqr), \
                  the total cost function (chitot) and the AIC
        """
        self.method = method.lower()
        assert (self.method in ['tnc', 'bfgs', 'l-bfgs-b']), ("method must be one of "
                                                              "'TNC', 'BFGS' or 'L-BFGS-B'")
        if hasattr(self, "popt") and self.popt.shape[1] == self.Z.shape[1]:
            print("\nUsing prefit")

            self.par_log = (
                self.convert_to_internal(self.popt)
            )
        else:

            self.par_log = (
                self.convert_to_internal(self.p0)
            )
            print("\nUsing initial")

        start = datetime.now()

        solver = jaxopt.ScipyMinimize(
            method=self.method,
            fun=jax.jit(self.compute_total_obj),
            dtype='float64',
            tol=1e-14,
            maxiter=n_iter,
            )
        self.sol = solver.run(
            self.par_log,
            self.F,
            self.Z,
            self.Zerr_Re,
            self.Zerr_Im,
            self.lb_vec,
            self.ub_vec,
            self.smf_1
            )

        self.popt = self.convert_to_external(self.sol.params)
        self.chitot = self.sol.state.fun_val/self.dof

        self.perr = self.compute_perr(
            self.popt,
            self.F,
            self.Z,
            self.Zerr_Re,
            self.Zerr_Im,
            self.lb_vec,
            self.ub_vec,
            self.smf_1,
        )

        self.chisqr = (
            jnp.mean(
                jax.vmap(self.compute_wrms, in_axes=(1, None, 1, 1, 1))(
                    self.popt, self.F, self.Z, self.Zerr_Re, self.Zerr_Im
                )
            )
        )
        self.AIC = (
            jnp.mean(
                jax.vmap(self.compute_aic, in_axes=(1, None, 1, 1, 1))(
                    self.popt, self.F, self.Z, self.Zerr_Re, self.Zerr_Im
                )
            )
        )
        print("\nOptimization complete")
        end = datetime.now()
        print(f"total time is {end-start}", end=" ")
        self.Z_exp = self.Z.copy()
        self.Y_exp = 1 / self.Z_exp.copy()
        self.Z_pred, self.Y_pred = self.model_prediction(self.popt, self.F)
        self.indices = [i for i in range(self.Z_exp.shape[1])]
        return self.popt, self.perr, self.chisqr, self.chitot, self.AIC

    def fit_stochastic(self,
                       lr: float = 1e-3,
                       num_epochs: int = 1e5,
                       ) -> Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ]:  # Optimal parameters, parameter error,
        # weighted residual mean square, and the AIC

        """
        Fitting routine which uses the Adam optimizer.
        It is important to note here that even stocahstic search procedures,
        although applicable to large scale problems do not \
        find the global optimum with certainty (Aster, Richard pg 249)

        :param lr: Learning rate

        :param num_epochs: Number of epochs

        :returns: A tuple containing the optimal parameters (popt), \
                  the standard error of the parameters (perr), \
                  the objective function at the minimum (chisqr), \
                  the total cost function (chitot) and the AIC
        """
        self.lr = lr
        self.num_epochs = int(num_epochs)

        if hasattr(self, "popt") and self.popt.shape[1] == self.Z.shape[1]:
            print("\nUsing prefit")

            self.par_log = (
                self.convert_to_internal(self.popt)
            )
        else:
            print("\nUsing initial")
            self.par_log = (
                self.convert_to_internal(self.p0)
            )

        start = datetime.now()
        self.opt_init, self.opt_update, self.get_params = jax_opt.adam(self.lr)
        self.opt_state = self.opt_init(self.par_log)
        self.losses = []
        for epoch in range(self.num_epochs):

            self.loss, self.opt_state = jax.jit(self.train_step)(
                epoch,
                self.opt_state,
                self.F,
                self.Z,
                self.Zerr_Re,
                self.Zerr_Im,
                self.lb_vec,
                self.ub_vec,
                self.smf_1
                )
            self.losses.append(float(self.loss))
            if epoch % int(self.num_epochs/10) == 0:
                print(
                    "" + str(epoch) + ": "
                    + "loss=" + "{:5.3e}".format(self.loss/self.dof)
                    )

        self.popt = self.convert_to_external(self.get_params(self.opt_state))
        self.chitot = self.losses[-1]

        # Computer perr using the fractional covariance matrix

        self.perr = self.compute_perr(
            self.popt,
            self.F,
            self.Z,
            self.Zerr_Re,
            self.Zerr_Im,
            self.lb_vec,
            self.ub_vec,
            self.smf_1,
        )
        self.chisqr = jnp.mean(
            jax.vmap(self.compute_wrms, in_axes=(1, None, 1, 1, 1))(
                self.popt, self.F, self.Z, self.Zerr_Re, self.Zerr_Im
            )
        )
        self.AIC = jnp.mean(
            jax.vmap(self.compute_aic, in_axes=(1, None, 1, 1, 1))(
                self.popt, self.F, self.Z, self.Zerr_Re, self.Zerr_Im
            )
        )
        print("Optimization complete")
        end = datetime.now()
        print(f"total time is {end-start}", end=" ")
        self.Z_exp = self.Z.clone()
        self.Y_exp = 1 / self.Z_exp.clone()
        self.Z_pred, self.Y_pred = self.model_prediction(self.popt, self.F)
        self.indices = [i for i in range(self.Z_exp.shape[1])]

        return self.popt, self.perr, self.chisqr, self.chitot, self.AIC

    def fit_simultaneous_zero(self,
                              method : str = 'TNC',
                              n_iter: int = 5000,
                              ) -> Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ]:

        """
        Simultaneous fitting routine with the smoothing factor set to zero.


        :params method: Solver to use (must be one of "'TNC', \
                        'BFGS' or 'L-BFGS-B'")

        :param n_iter: Number of iterations

        :returns: A tuple containing the optimal parameters (popt), \
                  the standard error of the parameters (perr), \
                  the objective function at the minimum (chisqr), \
                  the total cost function (chitot) and the AIC
        """
        self.method = method.lower()
        assert (self.method in ['tnc', 'bfgs', 'l-bfgs-b']), ("method must be one of "
                                                              "'TNC', 'BFGS' or 'L-BFGS-B'")
        if hasattr(self, "popt") and self.popt.shape[1] == self.Z.shape[1]:
            print("\nUsing prefit")

            self.par_log = (
                self.convert_to_internal(self.popt)
            )
        else:

            self.par_log = (
                self.convert_to_internal(self.p0)
            )
            print("\nUsing initial")

        # Optimizer 1 uses the BFGS algorithm
        start = datetime.now()
        solver = jaxopt.ScipyMinimize(
            method=self.method,
            fun=jax.jit(self.compute_total_obj),
            dtype='float64',
            tol=1e-14,
            maxiter=n_iter
            )
        self.sol = solver.run(
            self.par_log,
            self.F,
            self.Z,
            self.Zerr_Re,
            self.Zerr_Im,
            self.lb_vec,
            self.ub_vec,
            jnp.zeros(self.num_params),
            )

        self.popt = self.convert_to_external(self.sol.params)
        self.chitot = self.sol.state.fun_val/self.dof

        # Check if the hess_inv output from the optimizer is identity.
        # If yes, use the compute_perr function

        self.perr = self.compute_perr_QR(
            self.popt,
            self.F,
            self.Z,
            self.Zerr_Re,
            self.Zerr_Im,
        )

        self.chisqr = (
            jnp.mean(
                jax.vmap(self.compute_wrms, in_axes=(1, None, 1, 1, 1))(
                    self.popt, self.F, self.Z, self.Zerr_Re, self.Zerr_Im
                )
            )
        )
        self.AIC = (
            jnp.mean(
                jax.vmap(self.compute_aic, in_axes=(1, None, 1, 1, 1))(
                    self.popt, self.F, self.Z, self.Zerr_Re, self.Zerr_Im
                )
            )
        )
        print("\nOptimization complete")
        end = datetime.now()
        print(f"total time is {end-start}", end=" ")
        self.Z_exp = self.Z.copy()
        self.Y_exp = 1 / self.Z_exp.copy()
        self.Z_pred, self.Y_pred = self.model_prediction(self.popt, self.F)
        self.indices = [i for i in range(self.Z_exp.shape[1])]
        return self.popt, self.perr, self.chisqr, self.chitot, self.AIC

    def fit_sequential(self,
                       indices: Sequence[int] = None,
                       ) -> Tuple[
                        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
                        ]:
        """
        Fits each spectra individually using the L-M least squares method

        :params indices: List containing the indices of spectra to plot. \
                         If set to None, all spectra are fitted sequentially

        :returns: A tuple containing the optimal parameters (popt), \
                  the standard error of the parameters (perr), \
                  the objective function at the minimum (chisqr), \
                  the total cost function (chitot) and the AIC
        """

        if indices:
            assert all(
                i < self.num_eis for i in indices
            ), ("One or more values in the indices list "
                "are greater the number of spectra supplied")
            self.indices = indices
            self.n_fits = len(self.indices)
        elif indices is None:
            self.indices = [i for i in range(self.num_eis)]
            self.n_fits = len(self.indices)

        else:
            raise AttributeError("""
            Please choose the index or indices of spectra to fit""")

        if hasattr(self, "popt") and self.popt.shape[1] == self.Z.shape[1]:
            print("\nUsing prefit")

            self.par_log = (
                self.convert_to_internal(self.popt)
            )
        else:

            self.par_log = (
                self.convert_to_internal(self.p0)
            )
            print("\nUsing initial")

        popt = jnp.zeros(shape=(self.num_params, self.n_fits))
        perr = jnp.zeros(shape=(self.num_params, self.n_fits))
        params_init = jnp.zeros(shape=(self.num_params, self.num_eis))
        for i in range(self.num_params):
            params_init = params_init.at[i, :].set(
                (self.par_log)[self.kvals[i]:self.kvals[i + 1]]
                )
        chisqr = jnp.zeros(self.n_fits)
        aic = jnp.zeros(self.n_fits)
        start = datetime.now()
        for i, val in enumerate(self.indices):
            if i % 10 == 0:
                print(
                    f"fitting spectra {val}"
                )
            try:
                pfit, chi2 = self.do_minimize_lstsq(
                    params_init[:, val],
                    self.F,
                    self.Z[:, val],
                    self.Zerr_Re[:, val],
                    self.Zerr_Im[:, val],
                    self.lb,
                    self.ub,
                )
            except ValueError:
                pfit = self.encode(params_init[:, val], self.lb, self.ub)
                chi2 = self.compute_rss(
                    params_init[:, val],
                    self.F,
                    self.Z[:, val],
                    self.Zerr_Re[:, val],
                    self.Zerr_Im[:, val],
                    self.lb,
                    self.ub,
                )

            popt = popt.at[:, i].set(self.decode(pfit, self.lb, self.ub))
            chisqr = chisqr.at[i].set(
                jnp.sum((2*chi2)**2) / (2 * self.num_freq - self.num_params)
                )

            aic = aic.at[i].set(self.compute_aic(
                popt[:, i],
                self.F,
                self.Z[:, val],
                self.Zerr_Re[:, val],
                self.Zerr_Im[:, val],
            ))
            jac = self.compute_jac(
                params_init[:, val],
                self.F,
                self.Z[:, val],
                self.Zerr_Re[:, val],
                self.Zerr_Im[:, val],
                self.lb,
                self.ub,
            )
            hess = jac.T @ jac
            try:
                hess_inv = jnp.linalg.inv(hess)
                cov_mat = hess_inv * (chisqr[i])
                perr = perr.at[:, i].set(jnp.sqrt(jnp.diag(cov_mat)) * popt[:, i])
            except Exception as e:
                print(e.__doc__)
                print(e.message)
                print(
                    "Matrix is singular for spectra {}, using QR decomposition"
                    .format(val)
                )
                grads = jax.jacobian(
                    self.func)(popt[:, i], self.F
                               )

                gradsre = grads[:self.num_freq]
                gradsim = grads[self.num_freq:]
                diag_wtre_matrix = jnp.diag((1 / self.Zerr_Re[:, val]))
                diag_wtim_matrix = jnp.diag((1 / self.Zerr_Im[:, val]))
                vre = diag_wtre_matrix @ gradsre
                vim = diag_wtim_matrix @ gradsim
                Q1, R1 = jnp.linalg.qr(jnp.concatenate([vre, vim], axis=0))
                try:
                    invR1 = jnp.linalg.inv(R1)
                    perr = perr.at[:, i].set(
                        jnp.linalg.norm(invR1, axis=1)
                        * jnp.sqrt(chisqr[i])
                        )
                except Exception as e:
                    print(e.__doc__)
                    print(e.message)
                    print(
                        """Matrix is singular for spectra {},
                        perr will be assigned a value of ones"""
                        .format(val)
                    )
                    invR1 = jnp.linalg.pinv(R1)
                    perr = perr.at[:, i].set(
                        jnp.linalg.norm(invR1, axis=1)
                        * jnp.sqrt(chisqr[i])
                        )

        self.popt = popt.copy()
        self.perr = jnp.nan_to_num(perr.copy(), nan=1.0e15)
        self.chisqr = jnp.mean(chisqr)
        self.chitot = self.chisqr.copy()
        self.AIC = jnp.mean(aic)
        self.Z_exp = self.Z[:, self.indices]
        self.Y_exp = 1 / self.Z_exp.copy()
        self.Z_pred, self.Y_pred = self.model_prediction(self.popt, self.F)
        print("\nOptimization complete")
        end = datetime.now()
        print(f"total time is {end-start}", end=" ")
        return self.popt, self.perr, self.chisqr, self.chitot, self.AIC

    def compute_perr_mc(self,
                        n_boots: int = 500,
                        ) -> Tuple[
                            jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
                            ]:

        """
        The bootstrap approach used here is \
        similar to the fixed-X resampling. \
        In this approach we construct bootstrap observations \
        from the fitted values and the residuals. \
        The assumption that the functional form of the model \
        is implicit in this method. We also assume that \
        the errors are identically distributed with constant variance.
        (Bootstrapping Regression Models - J Fox)

        :returns: A tuple containing the optimal parameters (popt), \
                  the standard error of the parameters (perr), \
                  the objective function at the minimum (chisqr), \
                  the total cost function (chitot) and the AIC
        """
        print("""\nPlease run fit_simultaneous() or fit_stochastic()
              on your data before running the compute_perr_mc() method.
              ignore this message if you did.""")

        if hasattr(self, "popt") and self.popt.shape[1] == self.Z.shape[1]:

            par_log = (
                self.convert_to_internal(self.popt)
            )
        else:
            raise ValueError(
                """Please run fit_simultaneous() or fit_stochastic() before using
                the compute_perr_mc() method"""
            )

        self.n_boots = n_boots
        wrms = jax.vmap(self.compute_wrms, in_axes=(1, None, 1, 1, 1))(
            self.popt, self.F, self.Z, self.Zerr_Re, self.Zerr_Im
        )

        self.Z_pred, self.Y_pred = self.model_prediction(self.popt, self.F)

        # Taking the sqrt of the chisquare gives us an
        # estimate of the error in measured immittance values

        key = jax.random.PRNGKey(0)
        main_key, sub_key = jax.random.split(key, num=2)

        rnd_resid_Re = \
            jax.random.normal(
                main_key,
                (self.num_freq, self.num_eis)
                ) * jnp.sqrt(wrms)
        rnd_resid_Im = \
            jax.random.normal(sub_key, (self.num_freq, self.num_eis)) * jnp.sqrt(wrms)

        if self.weight_name == "sigma":
            Zerr_Re_mc = self.Zerr_Re
            Zerr_Im_mc = self.Zerr_Im
        elif self.weight_name == "proportional":
            Zerr_Re_mc = self.Z_pred.real
            Zerr_Im_mc = self.Z_pred.imag
        elif self.weight_name == "modulus":
            Zerr_Re_mc = jnp.abs(self.Z_pred)
            Zerr_Im_mc = jnp.abs(self.Z_pred)
        else:
            Zerr_Re_mc = jnp.ones(shape=(self.num_freq, self.num_eis))
            Zerr_Im_mc = jnp.ones(shape=(self.num_freq, self.num_eis))

        idx = [i for i in range(self.num_freq)]

        # Make containers to hold bootstrapped values
        self.popt_mc = jnp.zeros(
            shape=(
                self.n_boots,
                self.num_params,
                self.num_eis
                )
            )
        self.Z_pred_mc_tot = jnp.zeros(
            shape=(self.n_boots, self.num_freq, self.num_eis),
            dtype=jnp.complex64
            )

        self.chisqr_mc = jnp.zeros(self.n_boots)

        popt_log_mc = jnp.zeros(
            shape=(
                self.n_boots,
                self.num_params
                * self.num_eis
                - (self.num_eis - 1)
                * jnp.sum(jnp.isinf(self.smf))
                ),
                )

        # Here we loop through the number of boots and
        # run the minimization algorithm using the do_minimize function
        start = datetime.now()
        par_log_mc = par_log.copy()
        for i in range(self.n_boots):
            sidx = onp.random.choice(idx, replace=True, size=self.num_freq)
            rnd_resid_Re_boot = rnd_resid_Re[sidx, :]
            rnd_resid_Im_boot = rnd_resid_Im[sidx, :]
            Z_pred_mc = (
                self.Z_pred.real
                + Zerr_Re_mc * rnd_resid_Re_boot
                + 1j * (self.Z_pred.imag + Zerr_Im_mc * rnd_resid_Im_boot)
            )

            res = self.do_minimize(
                par_log_mc,
                self.F,
                Z_pred_mc,
                Zerr_Re_mc,
                Zerr_Im_mc,
                self.lb_vec,
                self.ub_vec,
                self.smf_1,
            )

            popt_log_mc = popt_log_mc.at[i, :].set(res.params)
            self.popt_mc = self.popt_mc.at[i, :, :].set((
                self.convert_to_external(popt_log_mc[i, :])
            ))
            self.chisqr_mc = self.chisqr_mc.at[i].set(res.state.fun_val/self.dof)
            self.Z_pred_mc_tot = self.Z_pred_mc_tot.at[i, :, :].set(
                jnp.asarray(Z_pred_mc, dtype=jnp.complex64)
                )
        self.popt = self.popt.copy()
        self.perr = jnp.std(self.popt_mc, ddof=1, axis=0)
        self.chisqr = jnp.mean(self.chisqr_mc, axis=0)
        self.chitot = self.chisqr.copy()
        print("\nOptimization complete")
        end = datetime.now()
        print(f"total time is {end-start}", end=" ")
        return self.popt, self.perr, self.chisqr, self.chitot, self.AIC

    def do_minimize(self,
                    p: jnp.ndarray,
                    f: jnp.ndarray,
                    z: jnp.ndarray,
                    zerr_re: jnp.ndarray,
                    zerr_im: jnp.ndarray,
                    lb: jnp.ndarray,
                    ub: jnp.ndarray,
                    smf: jnp.ndarray
                    ):
        """

        Fitting routine used in the bootstrap Monte Carlo procedure


        :param p: A 1D array of parameter values

        :param f: A 1D array of frequency

        :param z: A 1D array of complex immittance

        :param zerr_re: A 1D array of weights for \
                        the real part of the immittance

        :param zerr_im: A 1D array of weights for \
                        the imaginary part of the immittance

        :param lb: A 1D array of values for \
                   the lower bounds (for the total parameters)

        :param ub: A 1D array of values for \
                   the upper bounds (for the total parameters)

        :param smf: An array of real elements same size as p0. \
            when set to inf, the corresponding parameter is kept constant
        """
        solver = jaxopt.ScipyMinimize(
            method="TNC",
            fun=jax.jit(self.compute_total_obj),
            dtype='float64',
            tol=1e-14,
            maxiter=5000
            )
        sol = solver.run(
            p,
            f,
            z,
            zerr_re,
            zerr_im,
            lb,
            ub,
            smf
            )
        return sol

    def compute_jac(self,
                    p: jnp.ndarray,
                    f: jnp.ndarray,
                    z: jnp.ndarray,
                    zerr_re: jnp.ndarray,
                    zerr_im: jnp.ndarray,
                    lb: jnp.ndarray,
                    ub: jnp.ndarray,
                    ) -> jnp.ndarray:
        """
        Computes the Jacobian of the least squares \
        objective function w.r.t the parameters

        :param p: A 1D array of parameter values

        :param f: A 1D array of frequency

        :param z: A 1D array of complex immittance

        :param zerr_re: A 1D array of weights for \
                        the real part of the immittance

        :param zerr_im: A 1D array of weights for \
                        the imaginary part of the immittance

        :param lb: A 1D array of values for \
                   the lower bounds (for the total parameters)

        :param ub: A 1D array of values for \
                   the upper bounds (for the total parameters)

        :returns:  Returns the Jacobian matrix
        """
        return jax.jacobian(
            jax.jit(self.compute_rss))(p, f, z, zerr_re, zerr_im, lb, ub)

    def do_minimize_lstsq(self,
                          p: jnp.ndarray,
                          f: jnp.ndarray,
                          z: jnp.ndarray,
                          zerr_re: jnp.ndarray,
                          zerr_im: jnp.ndarray,
                          lb: jnp.ndarray,
                          ub: jnp.ndarray,
                          ) -> Tuple[
        jnp.ndarray, jnp.ndarray
    ]:  #
        """
        Least squares routine - uses residual_func

        :param p: A 1D array of parameter values

        :param f: A 1D array of frequency

        :param z: A 1D array of complex immittance

        :param zerr_re: A 1D array of weights for \
                        the real part of the immittance

        :param zerr_im: A 1D array of weights for \
                        the imaginary part of the immittance

        :param lb: A 1D array of values for \
                   the lower bounds (for the total parameters)

        :param ub: A 1D array of values for \
                   the upper bounds (for the total parameters)

        :returns: Returns the log-scaled optimal parameters \
                  and the weighted residual mean square
        """
        # lm = jaxopt.LevenbergMarquardt(self.residual_func, tol=1e-8, xtol=1e-8,
        # gtol=1e-8, stop_criterion='madsen-nielsen', jit=True)
        # lm_sol = jax.jit(lm.run)(p, f, z, zerr_re, zerr_im, lb, ub)
        # return lm_sol.params, lm_sol.state.residual
        res = scipy.optimize.least_squares(
            jax.jit(self.compute_rss),
            p,
            args=(f, z, zerr_re, zerr_im, lb, ub),
            method='trf',
            tr_solver='lsmr',
            jac=self.compute_jac
            )
        return res.x, res.fun

    def encode(self,
               p: jnp.ndarray,
               lb: jnp.ndarray,
               ub: jnp.ndarray,
               ) -> jnp.ndarray:
        """
        Converts external parameters to internal parameters

        :param p: A 1D array of parameter values

        :param lb: A 1D array of values for \
                   the lower bounds (for the total parameters)

        :param ub: A 1D array of values for \
                   the upper bounds (for the total parameters)

        :returns:  Returns the parameter vector \
                   in log scale (internal coordinates)
        """
        p = jnp.log10((p - lb) / (1 - p / ub))
        return p

    def decode(self,
               p: jnp.ndarray,
               lb: jnp.ndarray,
               ub: jnp.ndarray
               ) -> jnp.ndarray:
        """
        Converts internal parameters to external parameters

        :param p: A 1D array of parameter values

        :param lb: A 1D array of values for \
                   the lower bounds (for the total parameters)

        :param ub: A 1D array of values for \
                   the upper bounds (for the total parameters)

        :returns:  Returns the parameter vector \
                   in normal scale (external coordinates)
        """
        p = (lb + 10 ** (p)) / (1 + 10 ** (p) / ub)
        return p

    def real_to_complex(self,
                        z: jnp.ndarray,
                        ) -> jnp.ndarray:
        """
        :param z: real vector of length 2n \
                  where n is the number of frequencies

        :returns: Returns a complex vector of length n.
        """
        return z[: len(z) // 2] + 1j * z[len(z) // 2:]

    def complex_to_real(self,
                        z: jnp.ndarray,
                        ) -> jnp.ndarray:

        """
        :param z: complex vector of length n \
                  where n is the number of frequencies

        :returns: Returns a real vector of length 2n
        """

        return jnp.concatenate((z.real, z.imag), axis=0)

    def model_prediction(self,
                         P: jnp.ndarray,
                         F: jnp.ndarray
                         ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes the predicted immittance and its inverse

        :param P:

        :param Z:

        :returns: The predicted immittance (Z_pred) \
                  and its inverse(Y_pred)
        """
        Z_pred = jax.vmap(self.real_to_complex, in_axes=0)(
            jax.vmap(self.func, in_axes=(1, None))(P, F)
        ).T
        Y_pred = 1 / Z_pred.copy()
        return Z_pred, Y_pred

    def create_dir(self, dir_name: str):
        """
        Creates a directory. equivalent to using mkdir -p on the command line
        """
        self.dir_name = dir_name
        from errno import EEXIST
        from os import makedirs, path

        try:
            makedirs(self.dir_name)
        except OSError as exc:  # Python >2.5
            if exc.errno == EEXIST and path.isdir(self.dir_name):
                pass
            else:
                raise

    def plot_nyquist(self,
                     steps: int = 1,
                     **kwargs,
                     ):
        """
        Creates the complex plane plots (aka Nyquist plots)

        :param steps: Spacing between plots. Defaults to 1

        :keyword fpath1: Additional keyword arguments \
                         passed to plot (i.e file path)

        :keyword fpath2: Additional keyword arguments \
                    passed to plot (i.e file path)

        :returns: The complex plane plots.

        """

        self.steps = steps
        assert (
            self.steps <= self.Z_exp.shape[1]
        ), (
            """Steps with size {} is greater that
            the number of fitted spectra with size {}"""
            .format(steps, self.Z_exp.shape[1]))

        # If the fit method has not been called,
        # only the plots of the experimental data are presented
        if not hasattr(self, "popt"):
            indices = [i for i in range(self.Z_exp.shape[1])]

            self.n_plots = len(
                onp.arange(0, int(self.Z_exp.shape[1]), self.steps)
            )  # (or however many you programatically figure out you need)
            n_cols = 4
            n_rows = 5

            # If self.immittance is impedance then fig_nyquist1 is the
            # impedance plot while fig_nyquist2 is the admittance plot
            self.fig_nyquist1 = plt.figure(figsize=(15, 12), facecolor="white")

            if self.immittance == "impedance":
                # make a plot of the impedance
                k = 0
                for i in range(0, self.Z_exp.shape[1], self.steps):

                    k = k + 1
                    ax = plt.subplot(n_rows, n_cols, k)
                    ax.set_title("Idx = {:d}".format(indices[i]))
                    ax.plot(
                        self.Z_exp[:, i].real,
                        -self.Z_exp[:, i].imag,
                        "o",
                        color="orange",
                        label="Expt",
                    )
                    ax.set(adjustable="datalim", aspect="equal")
                    ax.set_xlabel(r"$\Re(Z)$" + "[" + r"$\Omega$" + "]")
                    ax.set_ylabel(r"$-\Im(Z)$" + "[" + r"$\Omega$" + "]")
                    ax.legend(loc="best")
                    plt.xticks(rotation=45)
                    if k > n_rows * n_cols - 1:
                        break

            else:
                # Make a plot of the admittance
                k = 0
                for i in range(0, self.Z_exp.shape[1], self.steps):

                    k = k + 1
                    ax = plt.subplot(n_rows, n_cols, k)
                    ax.set_title("Idx = {:d}".format(indices[i]))
                    ax.plot(
                        self.Z_exp[:, i].real,
                        self.Z_exp[:, i].imag,
                        "o",
                        color="orange",
                        label="Expt",
                    )
                    ax.set(adjustable="datalim", aspect="equal")
                    ax.set_xlabel(r"$\Re(Y)$" + "[" + r"$S$" + "]")
                    ax.set_ylabel(r"$\Im(Y)$" + "[" + r"$S$" + "]")
                    ax.legend(loc="best")
                    plt.xticks(rotation=45)
                    if k > n_rows * n_cols - 1:
                        break

            self.fig_nyquist1.suptitle(self.plot_title1, y=1.02)
            self.fig_nyquist1.tight_layout(w_pad=0.5, h_pad=0.5)
            if kwargs:
                fpath1 = kwargs.get("fpath1", None)
                self.fig_nyquist1.savefig(
                    fpath1, dpi=300, transparent=False, bbox_inches="tight"
                )
                plt.close(self.fig_nyquist1)
            else:
                plt.show()

            self.fig_nyquist2 = plt.figure(figsize=(15, 12), facecolor="white")

            if self.immittance == "impedance":
                k = 0
                for i in range(0, self.Z_exp.shape[1], self.steps):

                    k = k + 1
                    ax = plt.subplot(n_rows, n_cols, k)
                    ax.set_title("Idx = {:d}".format(indices[i]))
                    ax.plot(
                        self.Y_exp[:, i].real,
                        self.Y_exp[:, i].imag,
                        "o",
                        color="orange",
                        label="Expt",
                    )
                    ax.set(adjustable="datalim", aspect="equal")
                    ax.set_xlabel(r"$\Re(Y)$" + "[" + r"$S$" + "]")
                    ax.set_ylabel(r"$\Im(Y)$" + "[" + r"$S$" + "]")
                    ax.legend(loc="best")
                    plt.xticks(rotation=45)
                    if k > n_rows * n_cols - 1:
                        break

            else:
                k = 0
                for i in range(0, self.Z_exp.shape[1], self.steps):

                    k = k + 1
                    ax = plt.subplot(n_rows, n_cols, k)
                    ax.set_title("Idx = {:d}".format(indices[i]))
                    ax.plot(
                        self.Y_exp[:, i].real,
                        -self.Y_exp[:, i].imag,
                        "o",
                        color="orange",
                        label="Expt",
                    )
                    ax.set(adjustable="datalim", aspect="equal")
                    ax.set_xlabel(r"$\Re(Z)$" + "[" + r"$\Omega$" + "]")
                    ax.set_ylabel(r"$-\Im(Z)$" + "[" + r"$\Omega$" + "]")
                    ax.legend(loc="best")
                    plt.xticks(rotation=45)
                    if k > n_rows * n_cols - 1:
                        break

            self.fig_nyquist2.suptitle(self.plot_title2, y=1.02)
            self.fig_nyquist2.tight_layout(w_pad=0.5, h_pad=0.5)
            if kwargs:
                fpath2 = kwargs.get("fpath2", None)
                self.fig_nyquist2.savefig(
                    fpath2, dpi=300, transparent=False, bbox_inches="tight"
                )
                plt.close(self.fig_nyquist2)
            else:
                plt.show()

        else:
            # If a fit method has been called then assign self.indices to indices
            indices = self.indices

            n_cols = 4
            n_rows = 5

            self.fig_nyquist1 = plt.figure(figsize=(15, 12), facecolor="white")

            if self.immittance == "impedance":

                k = 0
                for i in range(0, self.Z_exp.shape[1], self.steps):

                    k = k + 1
                    ax = plt.subplot(n_rows, n_cols, k)
                    ax.set_title("Idx = {:d}".format(indices[i]))
                    ax.plot(
                        self.Z_exp[:, i].real,
                        -self.Z_exp[:, i].imag,
                        "o",
                        color="orange",
                        label="Expt",
                    )
                    ax.plot(
                        (self.Z_pred[:, i]).real,
                        -(self.Z_pred[:, i]).imag,
                        "b-",
                        lw=1.5,
                        label="Model",
                    )
                    ax.set(adjustable="datalim", aspect="equal")
                    ax.set_xlabel(r"$\Re(Z)$" + "[" + r"$\Omega$" + "]")
                    ax.set_ylabel(r"$-\Im(Z)$" + "[" + r"$\Omega$" + "]")
                    ax.legend(loc="best")
                    plt.xticks(rotation=45)
                    if k > n_rows * n_cols - 1:
                        break

            else:
                k = 0
                for i in range(0, self.Z_exp.shape[1], self.steps):

                    k = k + 1
                    ax = plt.subplot(n_rows, n_cols, k)
                    ax.set_title("Idx = {:d}".format(indices[i]))
                    ax.plot(
                        self.Z_exp[:, i].real,
                        self.Z_exp[:, i].imag,
                        "o",
                        color="orange",
                        label="Expt",
                    )
                    ax.plot(
                        (self.Z_pred[:, i]).real,
                        (self.Z_pred[:, i]).imag,
                        "b-",
                        lw=1.5,
                        label="Model",
                    )
                    ax.set(adjustable="datalim", aspect="equal")
                    ax.set_xlabel(r"$\Re(Y)$" + "[" + r"$S$" + "]")
                    ax.set_ylabel(r"$\Im(Y)$" + "[" + r"$S$" + "]")
                    ax.legend(loc="best")
                    plt.xticks(rotation=45)
                    if k > n_rows * n_cols - 1:
                        break

            self.fig_nyquist1.suptitle(self.plot_title1, y=1.02)
            self.fig_nyquist1.tight_layout(w_pad=0.5, h_pad=0.5)
            if kwargs:
                fpath1 = kwargs.get("fpath1", None)
                self.fig_nyquist1.savefig(
                    fpath1, dpi=300, transparent=False, bbox_inches="tight"
                )
                plt.close(self.fig_nyquist1)
            else:
                plt.show()

            self.fig_nyquist2 = plt.figure(figsize=(15, 12), facecolor="white")

            if self.immittance == "impedance":
                k = 0
                for i in range(0, self.Z_exp.shape[1], self.steps):

                    k = k + 1
                    ax = plt.subplot(n_rows, n_cols, k)
                    ax.set_title("Idx = {:d}".format(indices[i]))
                    ax.plot(
                        self.Y_exp[:, i].real,
                        self.Y_exp[:, i].imag,
                        "o",
                        color="orange",
                        label="Expt",
                    )
                    ax.plot(
                        (self.Y_pred[:, i]).real,
                        (self.Y_pred[:, i]).imag,
                        "b-",
                        lw=1.5,
                        label="Model",
                    )
                    ax.set(adjustable="datalim", aspect="equal")
                    ax.set_xlabel(r"$\Re(Y)$" + "[" + r"$S$" + "]")
                    ax.set_ylabel(r"$\Im(Y)$" + "[" + r"$S$" + "]")
                    ax.legend(loc="best")
                    plt.xticks(rotation=45)
                    if k > n_rows * n_cols - 1:
                        break

            else:
                k = 0
                for i in range(0, self.Z_exp.shape[1], self.steps):

                    k = k + 1
                    ax = plt.subplot(n_rows, n_cols, k)
                    ax.set_title("Idx = {:d}".format(indices[i]))
                    ax.plot(
                        self.Y_exp[:, i].real,
                        -self.Y_exp[:, i].imag,
                        "o",
                        color="orange",
                        label="Expt",
                    )
                    ax.plot(
                        (self.Y_pred[:, i]).real,
                        -(self.Y_pred[:, i]).imag,
                        "b-",
                        lw=1.5,
                        label="Model",
                    )
                    ax.set(adjustable="datalim", aspect="equal")
                    ax.set_xlabel(r"$\Re(Z)$" + "[" + r"$\Omega$" + "]")
                    ax.set_ylabel(r"$-\Im(Z)$" + "[" + r"$\Omega$" + "]")
                    ax.legend(loc="best")
                    plt.xticks(rotation=45)
                    if k > n_rows * n_cols - 1:
                        break

            self.fig_nyquist2.suptitle(self.plot_title2, y=1.02)
            self.fig_nyquist2.tight_layout(w_pad=0.5, h_pad=0.5)
            if kwargs:
                fpath2 = kwargs.get("fpath2", None)
                self.fig_nyquist2.savefig(
                    fpath2, dpi=300, transparent=False, bbox_inches="tight"
                )
                plt.close(self.fig_nyquist2)
            else:
                plt.show()

    def plot_bode(self,
                  steps: int = 1,
                  **kwargs,
                  ):
        """
        Creates the Bode plots
        The Bode plot shows the phase angle of a
        capacitor's or inductors opptosition to current.
        A capacitor's opposition to current is -90°,
        which means that a capacitor's opposition
        to current is a negative imaginary quantity.


        :param steps: Spacing between plots. Defaults to 1.

        :keyword fpath: Additional keyword arguments \
                         passed to plot (i.e file path)

        :returns: The bode plots.

        Notes
        ---------

        .. math::

            \\theta = arctan2 (\\frac{\\Im{Z}}{\\Re{Z}} \\frac{180}{\\pi})

        """
        assert (
            steps <= self.Z_exp.shape[1]
        ), (
            """Steps with size {} is greater that the
            number of fitted spectra with size {}"""
            .format(steps, self.Z_exp.shape[1]))
        self.steps = steps
        self.Z_mag = jnp.abs(self.Z_exp)
        self.Z_angle = jnp.rad2deg(jnp.arctan2(self.Z_exp.imag, self.Z_exp.real))

        if not hasattr(
            self, "popt"
        ):  # If the fit method has not been called,
            # only the plots of the experimental data are presented

            indices = [
                i for i in range(self.Z_exp.shape[1])
            ]
            # Indices should be determined by Z_exp
            # which changes depending on the routine used
            n_cols = 4
            n_rows = 5

            self.fig_bode = plt.figure(figsize=(15, 12), facecolor="white")
            if self.immittance == "impedance":

                k = 0
                for i in range(0, self.Z_exp.shape[1], self.steps):

                    k = k + 1
                    ax = plt.subplot(n_rows, n_cols, k)
                    ax1 = ax.twinx()
                    ax.set_title("Idx = {:d}".format(indices[i]))
                    (p1,) = ax.plot(
                        self.F,
                        self.Z_mag[:, i],
                        "o",
                        markersize=5,
                        color="blue",
                        label="Expt",
                    )
                    (p2,) = ax1.plot(
                        self.F,
                        self.Z_angle[:, i],
                        "o",
                        markersize=5,
                        color="orange",
                        label="Expt",
                    )
                    ax.set(adjustable="datalim")
                    ax.set_xlim(self.F.min(), self.F.max())
                    ax1.invert_yaxis()
                    ax.relim()
                    ax.autoscale()
                    ax.semilogx()
                    ax.semilogy()
                    ax.set_xlabel("$F$" + "[" + "$Hz$" + "]")
                    ax.set_ylabel(r"$|Z|$" + "[" + r"$\Omega$" + "]")
                    ax1.set_ylabel(r"$\theta$ $(^{\circ}) $")
                    ax.yaxis.label.set_color(p1.get_color())
                    ax1.yaxis.label.set_color(p2.get_color())
                    ax.tick_params(axis="y", colors=p1.get_color(), which="both")
                    ax1.tick_params(axis="y", colors=p2.get_color(), which="both")
                    if k > n_rows * n_cols - 1:
                        break
            else:
                k = 0
                for i in range(0, self.Z_exp.shape[1], self.steps):

                    k = k + 1
                    ax = plt.subplot(n_rows, n_cols, k)
                    ax1 = ax.twinx()
                    ax.set_title("Idx = {:d}".format(indices[i]))
                    (p1,) = ax.plot(
                        self.F,
                        self.Z_mag[:, i],
                        "o",
                        markersize=5,
                        color="blue",
                        label="Expt",
                    )
                    (p2,) = ax1.plot(
                        self.F,
                        -self.Z_angle[:, i],
                        "o",
                        markersize=5,
                        color="orange",
                        label="Expt",
                    )
                    ax.set(adjustable="datalim")
                    ax.set_xlim(self.F.min(), self.F.max())
                    ax1.invert_yaxis()
                    ax.relim()
                    ax.autoscale()
                    ax.semilogx()
                    ax.semilogy()
                    ax.set_xlabel("$F$" + "[" + "$Hz$" + "]")
                    ax.set_ylabel("$|Y|$" + "[" + "$S$" + "]")
                    ax1.set_ylabel(r"$\theta$ $(^{\circ}) $")
                    ax.yaxis.label.set_color(p1.get_color())
                    ax1.yaxis.label.set_color(p2.get_color())
                    ax.tick_params(axis="y", colors=p1.get_color(), which="both")
                    ax1.tick_params(axis="y", colors=p2.get_color(), which="both")
                    if k > n_rows * n_cols - 1:
                        break

            self.fig_bode.suptitle("Bode Plot", y=1.02)
            plt.tight_layout(w_pad=0.5, h_pad=0.5)
            if kwargs:
                fpath = kwargs.get("fpath", None)
                self.fig_bode.savefig(
                    fpath, dpi=300, transparent=False, bbox_inches="tight"
                )
                plt.close(self.fig_bode)
            else:
                plt.show()

        else:
            indices = (
                self.indices
            )  # Indices becomes sel.indices after a fit method has been called.
            self.Z_mag_pred = jnp.abs(self.Z_pred)
            self.Z_angle_pred = jnp.rad2deg(
                jnp.arctan2(self.Z_pred.imag, self.Z_pred.real)
            )

            n_cols = 4
            n_rows = 5

            self.fig_bode = plt.figure(figsize=(15, 12), facecolor="white")
            if self.immittance == "impedance":

                k = 0
                for i in range(0, self.Z_exp.shape[1], self.steps):

                    k = k + 1
                    ax = plt.subplot(n_rows, n_cols, k)
                    ax1 = ax.twinx()
                    ax.set_title("Idx = {:d}".format(indices[i]))
                    (p1,) = ax.plot(
                        self.F,
                        self.Z_mag[:, i],
                        "o",
                        markersize=5,
                        color="blue",
                        label="Expt",
                    )
                    ax.plot(
                        self.F,
                        self.Z_mag_pred[:, i],
                        "-",
                        color="red",
                        lw=1.5,
                        label="Model",
                    )
                    (p2,) = ax1.plot(
                        self.F,
                        self.Z_angle[:, i],
                        "o",
                        markersize=5,
                        color="orange",
                        label="Expt",
                    )
                    ax1.plot(
                        self.F,
                        self.Z_angle_pred[:, i],
                        "-",
                        color="purple",
                        lw=1.5,
                        label="Model",
                    )
                    ax.set(adjustable="datalim")
                    ax.set_xlim(self.F.min(), self.F.max())
                    ax1.invert_yaxis()
                    ax.relim()
                    ax.autoscale()
                    ax.semilogx()
                    ax.semilogy()
                    ax.set_xlabel("$F$" + "[" + "$Hz$" + "]")
                    ax.set_ylabel(r"$|Z|$" + "[" + r"$\Omega$" + "]")
                    ax1.set_ylabel(r"$\theta$ $(^{\circ}) $")
                    ax.yaxis.label.set_color(p1.get_color())
                    ax1.yaxis.label.set_color(p2.get_color())
                    ax.tick_params(axis="y", colors=p1.get_color(), which="both")
                    ax1.tick_params(axis="y", colors=p2.get_color(), which="both")
                    if k > n_rows * n_cols - 1:
                        break
            else:
                k = 0
                for i in range(0, self.Z_exp.shape[1], self.steps):

                    k = k + 1
                    ax = plt.subplot(n_rows, n_cols, k)
                    ax1 = ax.twinx()
                    ax.set_title("Idx = {:d}".format(indices[i]))
                    (p1,) = ax.plot(
                        self.F,
                        self.Z_mag[:, i],
                        "o",
                        markersize=5,
                        color="blue",
                        label="Expt",
                    )
                    ax.plot(
                        self.F,
                        self.Z_mag_pred[:, i],
                        "-",
                        color="red",
                        lw=1.5,
                        label="Model",
                    )
                    (p2,) = ax1.plot(
                        self.F,
                        -self.Z_angle[:, i],
                        "o",
                        markersize=5,
                        color="orange",
                        label="Expt",
                    )
                    ax1.plot(
                        self.F,
                        -self.Z_angle_pred[:, i],
                        "-",
                        color="purple",
                        lw=1.5,
                        label="Model",
                    )
                    ax.set(adjustable="datalim")
                    ax.set_xlim(self.F.min(), self.F.max())
                    ax1.invert_yaxis()
                    ax.relim()
                    ax.autoscale()
                    ax.semilogx()
                    ax.semilogy()
                    ax.set_xlabel("$F$" + "[" + "$Hz$" + "]")
                    ax.set_ylabel("$|Y|$" + "[" + "$S$" + "]")
                    ax1.set_ylabel(r"$\theta$ $(^{\circ}) $")
                    ax.yaxis.label.set_color(p1.get_color())
                    ax1.yaxis.label.set_color(p2.get_color())
                    ax.tick_params(axis="y", colors=p1.get_color(), which="both")
                    ax1.tick_params(axis="y", colors=p2.get_color(), which="both")
                    if k > n_rows * n_cols - 1:
                        break

            self.fig_bode.suptitle("Bode Plot", y=1.02)
            plt.tight_layout(w_pad=0.5, h_pad=0.5)
            if kwargs:
                fpath = kwargs.get("fpath", None)
                self.fig_bode.savefig(
                    fpath, dpi=300, transparent=False, bbox_inches="tight"
                )
                plt.close(self.fig_bode)
            else:
                plt.show()

    def plot_params(self,
                    show_errorbar: bool = False,
                    labels: Dict[str, str] = None,
                    **kwargs,
                    ) -> None:
        """
        Creates the plot of the optimal parameters as a function of the index

        :param show_errorbar: If set to True, \
                              the errorbars are shown on the parameter plot.

        :param labels: A dictionary containing the circuit elements\
                       as keys and the units as values e.g \
                       labels = {
                        "Rs":"$\\Omega$",
                        "Qh":"$F$",
                        "nh":"-",
                        "Rct":"$\\Omega$",
                        "Wct":"$\\Omega\\cdot^{-0.5}$",
                        "Rw":"$\\Omega$"
                        }

        :keyword fpath: Additional keyword arguments \
                         passed to plot (i.e file path)

        :returns: The parameter plots
        """
        if labels is None:
            self.labels = [str(i) for i in range(self.num_params)]
        else:

            assert (isinstance(labels, collections.Mapping)), (
                """labels is not a valid dictionary"""
                )

            assert (len(labels.items()) == self.num_params), (
                """Ths size of the labels is {}
                while the size of the parameters is {}"""
                .format(
                    len(labels.items()), self.num_params
                    )
                )

            self.labels = {
                self.try_convert(k): self.try_convert(v) for k, v in labels.items()
                }

        self.show_errorbar = show_errorbar

        if not hasattr(self, "popt"):
            raise AttributeError("A fit() method has not been called.")
        else:

            self.param_idx = [str(i) for i in self.indices]
            params_df = pd.DataFrame(
                onp.asarray(self.popt.T),
                columns=[i for i in range(self.num_params)]
                )
            params_df['Idx'] = self.param_idx
            params_df['Idx'] = params_df["Idx"].astype('category')
            self.params_df = params_df.fillna(0)
            if self.show_errorbar is True:
                # Plot with error bars
                self.fig_params = (
                    self.params_df.plot(
                        x="Idx",
                        marker="o",
                        linestyle="--",
                        subplots=True,
                        layout=(5, 5),
                        yerr=onp.asarray(self.perr),
                        figsize=(15, 12),
                        rot=45,
                        legend=False,
                    )
                    .ravel()[0]
                    .get_figure()
                )
            else:
                # Plot without error bars
                self.fig_params = (
                    self.params_df.plot(
                        x="Idx",
                        marker="o",
                        linestyle="--",
                        subplots=True,
                        layout=(5, 5),
                        figsize=(15, 12),
                        rot=45,
                        legend=False
                    )
                    .ravel()[0]
                    .get_figure()
                )
            plt.suptitle("Evolution of parameters ", y=1.01)
            plt.gcf().set_facecolor("white")
            all_axes = plt.gcf().get_axes()
            if labels is not None:
                for i, (k, v) in enumerate(self.labels.items()):
                    all_axes[i].set_ylabel(k + " " + "/" + " " + v, rotation=90)
            else:
                for i, v in enumerate(self.labels):
                    all_axes[i].set_ylabel(v, rotation=90)

            plt.tight_layout()
            if kwargs:
                fpath = kwargs.get("fpath", None)
                self.fig_params.savefig(
                    fpath, dpi=300, transparent=False, bbox_inches="tight"
                )
                plt.close(self.fig_params)
            else:
                plt.show()

    def get_img_path(self,
                     fname:
                     str = None
                     ):
        """
        Creates a path name for saving images
        """
        if fname is None:
            img_path = os.path.join(os.path.abspath(os.getcwd()), "fit")
            img_folder = os.path.join(img_path, "images")
            self.create_dir(img_folder)
            path_name = os.path.join(img_folder, "fit")

        elif isinstance(fname, str):
            img_path = os.path.join(os.path.abspath(os.getcwd()), fname)
            img_folder = os.path.join(img_path, "images")
            self.create_dir(img_folder)
            path_name = os.path.join(img_folder, fname)

        else:
            raise TypeError(
                f"Oops! {fname} is not valid. fname should be None or a valid string"
            )

        return path_name

    def get_results_path(self,
                         fname: str = None
                         ):
        """
        Creates a path name for saving the results
        """
        if fname is None:
            results_path = os.path.join(os.path.abspath(os.getcwd()), "fit")
            results_folder = os.path.join(results_path, "results")
            self.create_dir(results_folder)
            path_name = os.path.join(results_folder, "fit")

        elif isinstance(fname, str):
            results_path = os.path.join(os.path.abspath(os.getcwd()), fname)
            results_folder = os.path.join(results_path, "results")
            self.create_dir(results_folder)
            path_name = os.path.join(results_folder, fname)

        else:
            raise TypeError(
                f"Oops! {fname} is not valid. fname should be None or a valid string"
            )

        return path_name

    def save_plot_nyquist(self,
                          steps: int = 1,
                          *,
                          fname: str = None,
                          ) -> None:
        """
        Saves the Nyquist plots in the current working directory
        with the fname provided.

        :param steps: Spacing between plots. Defaults to 1.

        :keyword fname: Name assigned to the directory, generated plots and data

        :returns: A .png image of the the complex plane plots
        """

        self.img_path_name = self.get_img_path(fname)
        try:
            self.plot_nyquist(
                steps,
                fpath1=self.img_path_name + "_" + self.plot_title1.lower() + ".png",
                fpath2=self.img_path_name + "_" + self.plot_title2.lower() + ".png",
            )

        except AttributeError as e:
            logging.exception("", e, exc_info=True)

    def save_plot_bode(self,
                       steps: int = 1,
                       *,
                       fname: str = None,
                       ) -> None:
        """
        Saves the Bode plots in the current working directory
        with the fname provided

        :param steps: Spacing between plots. Defaults to 1.

        :keyword fname: Name assigned to the directory, generated plots and data

        :returns: A .png image of the the bode plot
        """
        self.img_path_name = self.get_img_path(fname)
        try:
            self.plot_bode(steps, fpath=self.img_path_name + "_bode" + ".png")

        except AttributeError as e:
            logging.exception("", e, exc_info=True)

    def save_plot_params(self,
                         show_errorbar: bool = False,
                         labels: Dict[str, str] = None,
                         *,
                         fname: str = None,
                         ) -> None:
        """
        Saves the parameter plots in the current working directory
        with the fname provided.

        :param show_errorbar: If set to True, \
                              the errorbars are shown on the parameter plot.


        :param labels: A dictionary containing the circuit elements\
                       as keys and the units as values e.g \
                       labels = {
                        "Rs":"$\\Omega$",
                        "Qh":"$F$",
                        "nh":"-",
                        "Rct":"$\\Omega$",
                        "Wct":"$\\Omega\\cdot^{0.5}$",
                        "Rw":"$\\Omega$"
                        }

        :keyword fname: Name assigned to the directory, generated plots and data

        :returns: A .png image of the the parameter plot
        """
        if labels is None:
            self.labels = None
        else:
            assert (isinstance(labels, collections.Mapping)), (
                """labels is not a valid dictionary"""
                )

            assert (len(labels.items()) == self.num_params), (
                """Ths size of the labels is {}
                while the size of the parameters is {}"""
                .format(
                    len(labels.items()), self.num_params
                    )
                )

            self.labels = {
                self.try_convert(k): self.try_convert(v) for k, v in labels.items()
                }

        if not hasattr(self, "popt"):
            raise AttributeError("A fit() method has not been called.")
        else:
            self.img_path_name = self.get_img_path(fname)
            try:
                self.plot_params(
                    show_errorbar=show_errorbar,
                    labels=self.labels,
                    fpath=self.img_path_name + "_params" + ".png"
                )

            except AttributeError as e:
                logging.exception("", e, exc_info=True)

    def save_results(self,
                     *,
                     fname: str = None,
                     ):  # The complex plane, bode and the parameter plots.
        """
        Saves the results (popt, perr, and Z_pred) in the current working directory
        with the fname provided

        :keyword fname: Name assigned to the directory, generated plots and data

        :returns: A .png image of the the complex plane plots
        """
        if not hasattr(self, "popt"):
            raise AttributeError("A fit() method has not been called.")
        else:
            self.results_path_name = self.get_results_path(fname)
            onp.save(self.results_path_name + "_popt.npy", onp.asarray(self.popt))
            onp.save(self.results_path_name + "_perr.npy", onp.asarray(self.perr))
            onp.save(self.results_path_name + "_Z_pred.npy", onp.asarray(self.Z_pred))
            with open(self.results_path_name + "_metrics.txt", "w") as fh:
                fh.write(
                    "%s %s %s %s %s\r"
                    % ("Immittance", "Weight", "AIC", "chisqr", "chitot")
                )
                fh.write(
                    "%s %s %.2f %.2e %.2e\r"
                    % (
                        self.immittance,
                        self.weight_name,
                        onp.asarray(self.AIC),
                        onp.asarray(self.chisqr),
                        onp.asarray(self.chitot),
                    )
                )

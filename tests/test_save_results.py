from unittest.mock import MagicMock
import numpy
np = numpy
np.load = MagicMock(return_value=(45, 50))
popt_shape = np.load('source/examples/example_results/results/example_results_popt.npy')

assert numpy.allclose(popt_shape, (45, 50))

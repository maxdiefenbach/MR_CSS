import numpy as np
import matplotlib.pyplot as plt
from CSmodel import CSmodel
from copy import deepcopy
import residual
import sim
import pytest


@pytest.fixture
def testobj():
    m = CSmodel()
    m.pm = np.array([[  1, 0.75, 100, 5],
                     [0.5, 0.75, 120, 5],
                     [0.5, 0.75, 150, 5]])
    m.Cm = np.array([[1,   0],
                     [0, 0.5],
                     [0, 0.5]])
    m.Cp = np.array([[1, 0],
                     [0, 1],
                     [0, 1]])
    m.Cf = np.array([[1],
                     [1],
                     [1]])
    m.Cr = np.array([[1],
                     [1],
                     [1]])
    m.constraints_matrices = [m.Cm, m.Cp, m.Cf, m.Cr]
    m.build_signal()
    return m


def test_constructor():
    m = CSmodel()
    assert isinstance(m, CSmodel)
    assert set(['Cm', 'Cp', 'Cr', 'Cf', 'SNR', 'TE1_s', 'TE_s', 'acq_shift',
               'centerfreq_Hz', 'dTE_s', 'dTE_s_eff', 'fieldstrength_T',
                'nAcq', 'nTE', 'pm', 'sigamp', 't_s'])\
               .issubset(set(m.__dict__.keys()))


def test_with_constraints_matrices(testobj):
    m = testobj
    # m = CSmodel()
    # m.pm = np.array([[  1, 0.75, 100, 5],
    #                  [0.5, 0.75, 120, 5],
    #                  [0.5, 0.75, 150, 5]])
    # m.Cm = np.array([[1,   0],
    #                  [0, 0.5],
    #                  [0, 0.5]])
    # m.Cp = np.array([[1, 0],
    #                  [0, 1],
    #                  [0, 1]])
    # m.Cf = np.array([[1],
    #                  [1],
    #                  [1]])
    # m.Cr = np.array([[1],
    #                  [1],
    #                  [1]])
    m.build_signal()
    m.plot_signal()
    plt.savefig('test.png')


def test_fixture(testobj):
    print(testobj)
    m = testobj
    m.plot_signal()
    plt.savefig('test2.png')


def test_normalize_amplitudes(testobj):
    m = testobj
    m.pm = np.array([[  1, 0.75, 100, 5],
                     [0.5, 0.75, 120, 5],
                     [0.5, 0.75, 150, 5]])
    m.normalize_amplitudes()
    assert sum(m.pm[:, 0]) == 1


def test_set_centerfreq_Hz(testobj):
    m = testobj
    exp = 42.58e6 * 1.5
    m.set_centerfreq_Hz(exp)
    assert m.centerfreq_Hz == exp


def test_set_TE_s(testobj):
    m = testobj
    m.set_TE_s(6, 1.e-3, 1.2e-3)
    print(m.TE_s)
    assert len(m.TE_s) == 6
    assert m.TE_s[0] == 1.e-3
    assert np.allclose(np.diff(m.TE_s)[0], 1.2e-3)


def test_build_signal(testobj):
    testobj.build_signal()
    assert testobj.signal_samp is not None


def test_set_residual(testobj):
    pm_init = deepcopy(testobj.pm)
    pm_init[:, 0:2] = 0
    pm_init[:, 2] -= pm_init[0, 2]
    pm_init[:, 3] = 0
    testobj.pm_init = pm_init

    nlparams_in = [[testobj.pm[0, 1]], [testobj.pm[1, 1]],
                   list(np.arange(-10, 20, 1)),
                   list(np.arange(0, 20, 1))]
    shape = tuple([len(x) for x in nlparams_in])
    nlparams = np.concatenate(nlparams_in).ravel()

    testobj.set_residual(nlparams, shape)
    assert isinstance(testobj.residual, np.ndarray)


def test_set_Fisher_matrix(testobj):
    testobj.set_Fisher_matrix()
    assert isinstance(testobj.Fisher_matrix, np.ndarray)


def test_estimate(testobj):
    pass


if __name__ == '__main__':
    m = CSmodel()
    m.pm = np.array([[  1, 0.75, 100, 5],
                     [0.5, 0.75, 120, 5],
                     [0.5, 0.75, 150, 5]])
    m.Cm = np.array([[1,   0],
                     [0, 0.5],
                     [0, 0.5]])
    m.Cp = np.array([[1, 0],
                     [0, 1],
                     [0, 1]])
    m.Cf = np.array([[1],
                     [1],
                     [1]])
    m.Cr = np.array([[1],
                     [1],
                     [1]])
    m.constraints_matrices = [m.Cm, m.Cp, m.Cf, m.Cr]
    m.build_signal()

    pm_init = deepcopy(m.pm)
    pm_init[:, 0:2] = 0
    pm_init[:, 2] -= pm_init[0, 2]
    pm_init[:, 3] = 0
    m.pm_init = pm_init

    nlparams_in = [[m.pm[0, 1]], [m.pm[1, 1]],
                   list(np.arange(-10, 20, 1)),
                   list(np.arange(0, 20, 1))]
    shape = tuple([len(x) for x in nlparams_in])
    nlparams = np.concatenate(nlparams_in).ravel()

    m.set_residual(nlparams, shape)

    assert (m.residual == residual.get_residual_array(m.TE_s, m.signal_samp,
                                                      pm_init,
                                                      m.Cm, m.Cp, m.Cf, m.Cr,
                                                      nlparams, shape)).all()

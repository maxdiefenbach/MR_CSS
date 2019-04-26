import pytest
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import residual
import css
import sim
import nbutils


tol = 1e-6
itermax = 100

# nTE = np.arange(5, 25, 5)
# TE1_s = np.arange(0., 2.0, 0.5) * 1e-3
# dTE_s = np.array([0.6, 2.4, 0.6]) * 1e-3
nTE = [5]
TE1_s = [1.e-3]
dTE_s = [1.2e-3]
TE_params = np.array(np.meshgrid(nTE, TE1_s, dTE_s)).T.reshape(-1, 3)


@pytest.fixture(params=TE_params)
def TE_s(request):
    nTE, TE1_s, dTE_s = request.param
    return sim.get_TE_s(int(nTE), TE1_s, dTE_s, 1, [0.5])


fieldmap_Hz = [0]
R2s_Hz = [5]
nl_params = np.array(np.meshgrid(fieldmap_Hz, R2s_Hz)).T.reshape(-1, 2)


@pytest.fixture(params=nl_params)
def pm(request):
    fieldmap_Hz, R2s_Hz = request.param
    pm = np.ones((3, 4))
    pm[:, 0] = np.array([1, 0.5, 0.5]) * 100
    pm[:, 1] = np.pi/4
    pm[:, 2] = fieldmap_Hz + np.array([0, 340, 440])
    pm[:, 3] = R2s_Hz
    return pm


@pytest.fixture()
def Cmats():
    Cm = np.array([[1, 0], [0, 0.5], [0, 0.5]])
    Cp = np.array([[1, 0], [0, 1], [0, 1]])
    Cf = np.atleast_2d([1, 1, 1]).T
    Cr = deepcopy(Cf)
    return Cm, Cp, Cf, Cr


def test_get_residual(pm, TE_s, Cmats):
    # FIXME: understand the failure
    Cm = Cmats[0]
    sig = css.build_signal(TE_s, pm)
    res = residual.get_residual(TE_s, pm, sig, Cm)
    assert res == 0


def test_list_residuals(pm, TE_s, Cmats):
    Cm, Cp, Cf, Cr = Cmats
    sig = css.build_signal(TE_s, pm)

    nlparams_in = [[0], [0],
                   list(np.arange(-100, 100, 5)),
                   list(np.arange(0, 300, 30))]
    shape = np.array([len(x) for x in nlparams_in])
    nlparams = np.concatenate(nlparams_in).ravel()

    Pm = sim.assemble_Pm(nlparams, shape, Cp, Cf, Cr, pm)

    residual_list = residual.list_residuals(TE_s, sig, Pm, Cm)
    print(Pm[residual_list.argmin(), :, :])

    assert residual_list.ndim == 1
    assert len(residual_list) == np.prod(shape)


def test_get_residual_array(pm, TE_s, Cmats):
    Cm, Cp, Cf, Cr = Cmats
    sig = css.build_signal(TE_s, pm)

    nlparams_in = [[0], [0],
                   list(np.arange(-100, 100, 5)),
                   list(np.arange(0, 300, 30))]
    shape = tuple([len(x) for x in nlparams_in])
    nlparams = np.concatenate(nlparams_in).ravel()

    pm_init = deepcopy(pm)
    pm_init[:, 0:2] = 0
    pm_init[:, 2] -= pm_init[0, 2]
    pm_init[:, 3] = 0

    R = residual.get_residual_array(TE_s, sig, pm_init,
                                    Cm, Cp, Cf, Cr,
                                    nlparams, shape)
    print(pm)
    print(R.argmin())
    assert R.shape == shape


def test_get_residual_minima(pm, TE_s, Cmats):

    sig = css.build_signal(TE_s, pm)

    pm_init = deepcopy(pm)
    pm_init[:, 0:2] = 0
    pm_init[:, 1] = 0
    pm_init[:, 2] -= pm_init[0, 2]
    pm_init[:, 3] = 0
    print(pm_init)

    Cm, Cp, Cf, Cr = Cmats

    nlparams_in = [[pm[0, 1]], [pm[1, 1]],
                   list(np.arange(-10, 20, 1)),
                   list(np.arange(0, 20, 1))]
    shape = tuple([len(x) for x in nlparams_in])
    nlparams = np.concatenate(nlparams_in).ravel()

    params_at_minima = residual.get_residual_minima(TE_s, sig, pm_init,
                                                    Cm, Cp, Cf, Cr,
                                                    nlparams, shape)

    print(pm, params_at_minima.shape, params_at_minima)
    assert params_at_minima[3, 0] == pm[0, 2]
    assert params_at_minima[4, 0] == pm[0, 3]


def test_get_global_residual_minimum(pm, TE_s, Cmats):
    sig = css.build_signal(TE_s, pm)

    pm_init = deepcopy(pm)
    pm_init[:, 0:2] = 0
    pm_init[:, 1] = 0
    pm_init[:, 2] -= pm_init[0, 2]
    pm_init[:, 3] = 0
    print(pm_init)

    Cm, Cp, Cf, Cr = Cmats

    nlparams_in = [[pm[0, 1]], [pm[1, 1]],
                   list(np.arange(-10, 20, 1)),
                   list(np.arange(0, 20, 1))]
    shape = tuple([len(x) for x in nlparams_in])
    nlparams = np.concatenate(nlparams_in).ravel()

    global_residual_minimum = residual.get_global_residual_minimum(TE_s, sig, pm_init,
                                                                   Cm, Cp, Cf, Cr,
                                                                   nlparams, shape)

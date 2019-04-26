import numpy as np
import pandas as pd
from copy import deepcopy
import css
import sim
import pytest
from Fatmodel import Fatmodel
import matplotlib.pyplot as plt


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


fieldmap_Hz = np.arange(-50, 55, 25)
R2s_Hz = np.arange(0, 350, 100)
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


def test_get_TE_s():
    nTE = 3
    TE1_s = 1.
    dTE_s = 1.
    nAcq = 3
    acq_shift = np.array([0.25, 0.5])

    TE_s = sim.get_TE_s(nTE, TE1_s, dTE_s, nAcq, acq_shift)
    print(TE_s)
    assert isinstance(TE_s, np.ndarray)
    assert (TE_s ==
            np.array([1., 1.25, 1.75, 2., 2.25, 2.75, 3., 3.25, 3.75])).all()

    TE_s = sim.get_TE_s(nTE, TE1_s, dTE_s, nAcq, [0.25])
    print(TE_s)
    assert isinstance(TE_s, np.ndarray)
    assert (TE_s ==
            np.array([1., 1.25, 1.5, 2., 2.25, 2.5, 3., 3.25, 3.5])).all()


def test_get_dTEeff_s():
    nTE = 3
    TE1_s = 1.
    dTE_s = 1.
    nAcq = 3
    acq_shift = np.array([0.25, 0.5])

    TE_s = sim.get_TE_s(nTE, TE1_s, dTE_s, nAcq, acq_shift)
    dTE_s = sim.get_dTEeff_s(TE_s)
    print(dTE_s)

    assert dTE_s[0] == 0.25
    assert dTE_s[1] == 0.5


def test_get_params_combinations():
    nlparams_in = [[0], [0],
                   list(np.arange(-100, 100, 5)),
                   list(np.arange(0, 300, 30))]
    shape = np.array([len(x) for x in nlparams_in])
    nlparams = np.concatenate(nlparams_in).ravel()
    params_combinations = sim.get_params_combinations(nlparams, shape)
    print(params_combinations)

    nparams = len(shape)
    assert params_combinations.shape == (np.prod(shape), nparams)
    for n in range(nparams):
        assert (np.unique(params_combinations[:, n]) == nlparams_in[n]).all()


def test_assemble_Pm(pm, Cmats):
    _, Cp, Cf, Cr = Cmats
    nlparams_in = [[0], [0],
                   list(np.arange(-100, 100, 5)),
                   list(np.arange(0, 300, 30))]
    shape = np.array([len(x) for x in nlparams_in])
    nlparams = np.concatenate(nlparams_in).ravel()

    pm_init = deepcopy(pm)
    pm_init[:, 0:2] = 0
    pm_init[:, 2] -= pm_init[0, 2]
    pm_init[:, 3] = 0

    Pm = sim.assemble_Pm(nlparams, shape, Cp, Cf, Cr, pm_init)

    assert (np.unique(Pm[:, 0, 2]) == np.array(nlparams_in[2])).all()
    assert Pm.shape == (np.prod(shape), Cp.shape[0], 4)


def test_assemble_TEarray():
    TEparams_in = [np.arange(3, 10),  # nTE, TE1_s, dTE_s, nAcq, acq_shift
                   np.arange(0.8, 1.2, 0.1) * 1e-3,
                   np.arange(0.8, 1.2, 0.1) * 1e-3,
                   np.array([1]),
                   np.array([0.5])]
    shape = np.array([len(x) for x in TEparams_in])
    TEparams = np.concatenate(TEparams_in).ravel()

    params_combinations = sim.get_params_combinations(TEparams, shape)
    for p in params_combinations:
        nTE, TE1_s, dTE_s, nAcq, acq_shift = p
        TE_s = sim.get_TE_s(int(nTE), TE1_s, dTE_s, int(nAcq), [acq_shift])
        assert len(TE_s) == nTE
        assert TE_s[0] == TE1_s


def test_estimate_bias(pm, TE_s, Cmats):
    Cm, Cp, Cf, Cr = Cmats
    sig = css.build_signal(TE_s, pm)

    nlparams_in = [[0], [0],
                   list(np.arange(-100, 100, 5)),
                   list(np.arange(0, 300, 30))]
    shape = tuple([len(x) for x in nlparams_in])
    nlparams = np.concatenate(nlparams_in).ravel()

    pm_init = deepcopy(pm)
    pm_init[:, 0:2] = 0
    pm_init[:, 1] = 0
    pm_init[:, 2] -= pm_init[0, 2]
    pm_init[:, 3] = 0
    print(pm_init)

    bias = sim.estimate_bias(TE_s, pm, pm_init,
                             Cm, Cp, Cf, Cr, nlparams, shape,
                             tol, itermax)

    print(bias)
    assert (bias < tol).all()


def test_compute_FIMs():

    F = Fatmodel(fatfraction_percent=80)
    F.set_params_matrix()
    F.set_constraints_matrices()
    F.build_signal()
    # F.plot_signal()

    nVoxel = 10
    Pm = np.tile(F.pm.copy(), [nVoxel, 1, 1])

    TE_s = F.TE_s
    Cm = F.Cm
    Cp = F.Cp
    Cf = F.Cf
    Cr = F.Cr

    FIM = sim.get_Fisher_matrix(TE_s, F.pm, Cm, Cp, Cf, Cr)
    print(FIM)

    CRLBs, NSAs, Invariants, FIMs, FIMinvs = sim.compute_FIMparams(TE_s, Pm,
                                                                   Cm, Cp, Cf, Cr)

    assert FIMs.shape == (nVoxel, FIM.shape[0], FIM.shape[1])
    assert FIMinvs.shape == (nVoxel, FIM.shape[0], FIM.shape[1])
    assert (np.squeeze(FIMs[0, ...]) == FIM).all()


def test_mc_css_varpro():

    F = Fatmodel()

    r2s = F.R2s_Hz
    TE_s = F.TE_s.copy()
    nTE = len(TE_s)

    F.set_constraints_matrices()
    Cm, Cp, Cf, Cr = F.constraints_matrices

    Ninr = int(1e4)
    snr = 100
    tol = 1e-4
    itermax = 50

    cols = ['pdff', 'r2s', 'CRLBs', 'NSAs', 'mcNSAs', 'trF', 'detF', 'trFinv', 'detFinv', 'snr', 'mean', 'var']
    df = pd.DataFrame(columns=cols)
    for i, pdff in enumerate(np.linspace(1, 99, 20)):
        F.fatfraction_percent = pdff
        F.set_params_matrix()
        F.build_signal()
        pm0 = F.pm
        sig = F.signal_samp

        mean, var = sim.mc_css_varpro(Ninr, snr, TE_s, sig, pm0,
                                      Cm, Cp, Cf, Cr, tol, itermax)
        print(i, pdff, mean)

        F.set_Fisher_matrix()
        FIM = F.Fisher_matrix

        FIMinv = np.linalg.inv(FIM)

        trF = np.trace(FIM)
        detF = np.linalg.det(FIM)
        CRLBs = np.diagonal(FIMinv)
        trFinv = np.trace(FIMinv)
        detFinv = np.linalg.det(FIMinv)

        NSAs = nTE / np.diag(FIM) / CRLBs
        mcNSAs = (1/snr)**2 * nTE / np.diag(FIM) / var

        df = df.append(
            pd.DataFrame(data=[[pdff, r2s, CRLBs, NSAs, mcNSAs, trF, detF, trFinv, detFinv,
                                snr, mean, var]],
                         columns=cols))

    plt.close('all')
    fig, axs = plt.subplots(3, 2)
    for i, ax in enumerate(axs.ravel()):
        # ax.plot(df['pdff'], df['CRLBs'].str[i], 'o-')
        # ax.plot(df['pdff'], df['var'].str[i]*snr**2, 'o-')
        ax.plot(df['pdff'], df['NSAs'].str[i], 'o-')
        ax.plot(df['pdff'], df['mcNSAs'].str[i], 'o')
    ax.legend()


    plt.figure()
    plt.plot(df['pdff'], df['CRLBs'].apply(np.sum))
    plt.plot(df['pdff'], df['trFinv'])

    plt.figure()
    plt.plot(df['pdff'], df['detF'])




def compare_toPineda():
    F = Fatmodel(fieldstrength_T=1.5, fieldmap_Hz=110, R2s_Hz=50)
    F.deshielding_ppm = np.array([-3.80, -3.40, -2.60, -1.94, -0.39, 0.60])
    F.relamps_percent = 100 * np.array([0.087, 0.693, 0.128, 0.004, 0.039, 0.048])
    F.deshielding_ppm = np.array([-3.4])
    F.relamps_percent = np.array([100])

    F.set_params_matrix()
    F.TE_s = np.array([1.5500, 3.8200, 6.0900]) * 1e-3
    TE_s = F.TE_s
    nTE = len(TE_s)

    F.set_constraints_matrices()
    Cm, Cp, Cf, Cr = F.constraints_matrices

    Ninr = int(100) # int(1e5)
    snr = 200
    tol = 1e-4
    itermax = 50

    N = 101


    cols = ['pdff', 'r2s', 'CRLBs', 'NSAs', 'mcNSAs', 'trF', 'detF', 'trFinv', 'detFinv', 'snr', 'mean', 'var']
    df = pd.DataFrame(columns=cols)
    for i, pdff in enumerate(np.linspace(0.1, 99.9, N)):
        F.fatfraction_percent = pdff
        F.set_params_matrix()
        F.build_signal()
        pm0 = F.pm
        sig = F.signal_samp

        mean, var = mc_css_varpro(Ninr, snr, TE_s, sig, pm0,
                                  Cm, Cp, Cf, Cr, tol, itermax)

        print(i, pdff, mean)

        F.set_Fisher_matrix()
        FIM = F.Fisher_matrix

        FIMinv = np.linalg.inv(FIM)

        trF = np.trace(FIM)
        detF = np.linalg.det(FIM)
        CRLBs = np.diagonal(FIMinv)
        trFinv = np.trace(FIMinv)
        detFinv = np.linalg.det(FIMinv)

        NSAs = nTE / np.diag(FIM) / CRLBs

        mcNSAs = (1/snr)**2 * nTE / np.diag(FIM) / var

        df = df.append(
            pd.DataFrame(data=[[pdff, r2s, CRLBs, NSAs, mcNSAs, trF, detF, trFinv, detFinv,
                                snr, mean, var]],
                         columns=cols))


    print(df)

    plt.close('all')
    for i in range(6):
        plt.figure()
        # plt.plot(df['pdff'], df['CRLBs'].str[i], 'o-')
        # plt.plot(df['pdff'], (1/snr)**2 * df['var'].str[i], 'o-')
        # plt.plot(df['pdff'], (snr/F.sigamp) * df['NSAs'].str[i], 'o-')
        plt.plot(df['pdff'], df['NSAs'].str[i], 'o-')
        plt.plot(df['pdff'], df['mcNSAs'].str[i], 'o-')
        # plt.xscale('log')
        plt.legend()



    F = Fatmodel(fieldstrength_T=1.5, fieldmap_Hz=110, R2s_Hz=50, sigamp=100)
    F.deshielding_ppm = np.array([-3.4])
    F.relamps_percent = np.array([100])
    F.set_constraints_matrices()
    # print(F.Cm)
    # F.Cm = F.Cm / np.diag(F.Cm.T.astype(bool).dot(F.Cm) / F.Cm.sum(0)).sum()
    # F.constraints_matrices[0] = F.Cm
    # F.constraints_matrices[1] = F.Cm
    print(F.constraints_matrices)
    # print(F.Cm)
    # F.Cp = F.Cm
    # F.Cp = F.Cp / np.diag(F.Cp.T.astype(bool).dot(F.Cp) / F.Cp.sum(0)).sum()
    # F.Cm /= F.Cm.shape[-1]
    # F.Cp /= F.Cp.shape[-1]
    # F.Cf /= F.Cf.shape[-1]
    # F.Cr /= F.Cr.shape[-1]
    F.TE_s = np.array([1.5500, 3.8200, 6.0900]) * 1e-3
    F.set_params_matrix()
    F.set_Fisher_matrix()
    J = css.get_Jacobian(F.TE_s, F.pm, F.Cm, F.Cp, F.Cf, F.Cr)
    # FIM = F.Fisher_matrix
    FIM = J.T.conj().dot(J).real
    crlb = np.diag(np.linalg.inv(FIM))
    np.set_printoptions(precision=2)
    print(FIM)
    print()
    print(crlb)
    print(crlb/2)

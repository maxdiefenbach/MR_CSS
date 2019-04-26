import pytest
from copy import deepcopy
import numpy as np
import css
import sim
import h5io
import wfi
from Fatmodel import Fatmodel


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


def test_get_Amatrix(pm, TE_s):
    A = css.get_Amatrix(TE_s, pm)
    assert A.shape == (len(TE_s), pm.shape[0])


def test_build_signal(pm, TE_s):
    sig = css.build_signal(TE_s, pm)
    assert sig is not None
    assert sig.dtype == complex


def test_get_paramsCount(Cmats):
    Cm, Cp, Cf, Cr = Cmats
    M, Ntot, Nm, Np, Nf, Nr = css.get_paramsCount(Cm, Cp, Cf, Cr)
    exp = css._nonzero_diag_elements(Cm) + \
        css._nonzero_diag_elements(Cp) + \
        css._nonzero_diag_elements(Cf) + \
        css._nonzero_diag_elements(Cr)
    print(Ntot, exp)
    assert Ntot == exp
    assert M == Cm.shape[0]
    assert Nm == css._nonzero_diag_elements(Cm)
    assert Np == css._nonzero_diag_elements(Cp)
    assert Nf == css._nonzero_diag_elements(Cf)
    assert Nr == css._nonzero_diag_elements(Cr)


def test_get_Jacobian(pm, TE_s, Cmats):
    Cm, Cp, Cf, Cr = Cmats
    J = css.get_Jacobian(TE_s, pm, Cm, Cp, Cf, Cr)
    paramsCount = css.get_paramsCount(Cm, Cp, Cf, Cr)
    nparams = paramsCount[1]
    assert J.shape == (len(TE_s), nparams)


def test_varpro(pm, TE_s, Cmats):
    Cm, Cp, Cf, Cr = Cmats
    sig = css.build_signal(TE_s, pm)

    tol = 1e-6
    itermax = 100
    pm0 = deepcopy(pm)

    pme, resnorm, i = css.varpro(TE_s, sig, pm0, Cm, Cp, Cf, Cr, tol, int(itermax))

    print('i =', i, 'resnorm =', resnorm)
    print('result: pm =\n', pme, '\nerror =', pme - pm)
    assert np.allclose(pm, pme)
    assert isinstance(resnorm, float)
    assert isinstance(i, int)
    assert i == 1


def test_map_varpro(pm, TE_s, Cmats):
    Cm, Cp, Cf, Cr = Cmats
    sig = css.build_signal(TE_s, pm)

    nVoxel = 100
    Sig = np.tile(sig.T, [nVoxel, 1])
    pm0 = deepcopy(pm)
    pm0[:, 0:2] = 0
    pm0[:, 3] = 0
    Pm0 = np.tile(pm, [nVoxel, 1, 1])

    Pme, Resnorm, Iterations = css.map_varpro(TE_s, Sig, Pm0,
                                                Cm, Cp, Cf, Cr,
                                                tol, itermax)
    assert np.allclose(Pm0, Pme)


def test_css_varpro():
    filename = '/Users/mnd/Projects/FatParameterEstimation/data/DiagnostikBilanz/20170609_125718_0402_ImDataParams.mat'
    imDataParams = load_ImDataParams_mat(filename)
    imDataParams = imDataParams['ImDataParams']
    iz = slice(35, 37)
    imDataParams['signal'] = imDataParams['signal'][:, :, iz, :]
    imDataParams['TE_s'] = imDataParams['TE_s'].ravel()

    tissue_mask = wfi.calculate_tissue_mask(imDataParams['signal'])

    filename = '/Users/mnd/Projects/FatParameterEstimation/data/DiagnostikBilanz/20170609_125718_0402_WFIparams_CSS_GANDALF2D_VL.mat'
    h5file = h5.File(filename, 'r')
    path = '/WFIparams'
    f = h5.File(filename, 'r')
    fieldmap_Hz = np.transpose(f['/WFIparams/fieldmap_Hz'][...])[:, :, iz]
    fieldmap_Hz_equalized = wfi.equalize_fieldmap_periods(fieldmap_Hz, imDataParams['TE_s'])
    fieldmap_Hz_equalized = tissue_mask * \
        wfi.equalize_fieldmap_periods(fieldmap_Hz, imDataParams['TE_s'])


    F = Fatmodel()
    F.set_fatmodel('Hamilton VAT')
    F.set_constraints_matrices()
    Cm, Cp, Cf, Cr = F.constraints_matrices

    options = {}
    options['Cm'] = Cm
    options['Cp'] = Cp
    options['Cf'] = Cf
    options['Cr'] = Cr
    options['Pm0'] = wfi.build_Pm0(F.get_chemical_shifts_Hz(), fieldmap_Hz_equalized[tissue_mask])
    options['mask'] = tissue_mask
    options['tol'] = 1e-5
    options['itermax'] = 100
    outParams = css.css_varpro(imDataParams, options)

    assert set(outParams.keys()) == {'elapsed_time_s', 'iterations',
                                     'param_maps', 'resnorm'}
    assert outParams['param_maps'][0].shape == tissue_mask.shape


def test_add_noise():
    TE_s = np.array([1, 2, 3, 4]) * 1e-3
    pm = np.ones((3, 4))
    pm[:, 0] = [1, 0.5, 0.5]
    pm[:, 1] = np.pi/4
    pm[:, 2] = 100 + np.array([0, 340, 440])
    pm[:, 3] = 5
    sig = css.build_signal(TE_s, pm)
    SNR = 20
    noisy_sig = css.add_noise(sig, SNR)
    assert (noisy_sig != sig).all()
    assert noisy_sig.dtype == complex


def test_varpro_phaseconstrained():
    from Fatmodel import Fatmodel
    F = Fatmodel()
    F.set_params_matrix()
    F.build_signal()
    # F.plot_signal()
    F.pm[1:, 1] = np.pi/2
    F.build_signal()
    # F.plot_signal()
    # F.sample_signal(SNR=30)
    F.set_constraints_matrices()

    sig = F.signal_samp
    tol = 1e-10
    itermax = 100
    pm0 = deepcopy(F.pm)
    pm0[:, :2] = 0
    pm0[:, 2] -= pm0[0, 2]
    pm0[:, 3] = 0
    Cp = np.zeros((len(pm0), 1))
    Cp[:, 0] = 1.

    print(pm0)

    pme, resnorm, i = css.varpro(F.TE_s, sig, pm0, F.Cm, F.Cp, F.Cf, F.Cr, tol, itermax)
    print(pme, '\n', css.extract_param_type(1, pme, [F.Cm, F.Cp, F.Cf, F.Cr]))
    assert pme[0, 1] != pme[1, 1]

    pm0 = deepcopy(F.pm)
    pm0[:, :2] = 0
    pm0[:, 2] -= pm0[0, 2]
    pm0[:, 3] = 0
    pme2, resnorm2, i2 = css.varpro(F.TE_s, sig, pm0, F.Cm, Cp, F.Cf, F.Cr, tol, itermax)
    print(pme2, '\n', css.extract_param_type(1, pme2, [F.Cm, Cp, F.Cf, F.Cr]))
    assert np.isclose(pme2[0, 1], pme2[1, 1])


def test_fucl():
    F = Fatmodel(modelname='Berglund 10 peaks',
                 fatfraction_percent=80,
                 nTE=20, ndb=2.73, cl=18, nmidb=0.76)

    F.compute_fatmodel()
    F.set_params_matrix()
    F.build_signal()
    F.plot_signal()
    F.set_constraints_matrices()

    TE_s = F.TE_s.copy()
    sig = F.signal_samp.copy()

    Cm = np.array([[1, 0,   0, 0,   0],
                   [0, 1,   0, 0,   0],
                   [0, 0,   1, 0,   0],
                   [0, 2/3, 0, 0,   0],
                   [0, 0,   0, 1,   0],
                   [0, 2/3, 0, 0,   0],
                   [0, 0,   0, 0,   1],
                   [0, 2/9, 0, 0,   0],
                   [0, 2/9, 0, 0,   0],
                   [0, 1/9, 0, 0,   0],
                   [0, 0,   0, 1/2, 1]])
    Cp = (Cm > 0).astype(float)
    Cf = F.Cf
    Cr = F.Cr

    tol = 1e-5
    itermax = 100

    pm = np.zeros((Cm.shape[0], 4))
    pm[:, 2] = F.get_chemical_shifts_Hz() + F.pm[0, 2]
    pm[:, 3] = 0

    pme, resnorm, i = css.varpro(TE_s, sig, pm, Cm, Cp, Cf, Cr, tol, int(itermax))

    print('result: pm =\n', pme, '\nerror =', pme - F.pm)
    assert np.allclose(pm, pme)
    assert isinstance(resnorm, float)
    assert isinstance(i, int)


def test_extract_intercepts():

    F = Fatmodel()
    F.set_params_matrix()
    F.set_constraints_matrices()
    Cs = F.constraints_matrices
    nVoxel = 100
    Pm = np.tile(F.pm.copy(), [nVoxel, 1, 1])

    variables = css.extract_variables(Pm, F.constraints_matrices)

    assert len(variables) == css.get_paramsCount(F.Cm, F.Cp, F.Cf, F.Cr)[1]
    assert (variables[0] == F.pm[0, 0]).all()
    assert (variables[1] == F.pm[1:, 0].sum()).all()
    assert (variables[2] == F.pm[0, 1].sum()).all()
    assert (variables[3] == F.pm[1, 1].sum()).all()
    assert (variables[4] == F.pm[0, 2].sum()).all()
    assert (variables[5] == F.pm[0, 3].sum()).all()

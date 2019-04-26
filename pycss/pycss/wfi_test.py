import pytest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from copy import deepcopy
from utils.utils import *
import h5io
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pprint import pprint
from Fatmodel import Fatmodel
import css
import wfi
import datetime
import h5io
import sim


def test_build_Pm0():
    F = Fatmodel()
    F.set_fatmodel('Hamilton VAT')
    init_fieldmap_Hz = np.zeros((224, 224, 88))
    Pm0  = wfi.build_Pm0(F.get_chemical_shifts_Hz(), init_fieldmap_Hz)

    assert (Pm0[:, 0, 2] == init_fieldmap_Hz.ravel()).all()


def test_wfi_css_varpro():
    filename = '/Users/mnd/Projects/FatParameterEstimation/data/DiagnostikBilanz/20170609_125718_0402_ImDataParams.mat'
    imDataParams = h5io.load_ImDataParams_mat(filename)
    imDataParams = imDataParams['ImDataParams']

    filename = '/Users/mnd/Projects/FatParameterEstimation/data/DiagnostikBilanz/20170609_125718_0402_WFIparams_CSS_GANDALF2D_VL.mat'
    wfiParams = h5io.load_WFIparams_mat(filename)
    fieldmap_Hz = np.transpose(wfiParams['WFIparams']['fieldmap_Hz'])
    fieldmap_Hz_equalized = wfi.equalize_fieldmap_periods(fieldmap_Hz,
                                                          imDataParams['TE_s'])

    options = {'init_fieldmap_Hz': fieldmap_Hz_equalized,
               'iSlice': slice(35, 37)}

    wfiParams = wfi.wfi_css_varpro(imDataParams, options)

    assert {'water', 'fat', 'fieldmap_Hz', 'R2s_Hz', 'pdff_percent'}.\
        issubset(wfiParams.keys())

    close_all()
    show_arr3d(np.clip(wfiParams['pdff_percent'], 0, 100))


def test_wfi_css_varpro_wuq():
    filename = '/Users/mnd/Projects/FatParameterEstimation/data/FatParamEstimation/BMR/BMR106_EN/20171013_131406_1302_ImDataParams.mat'
    imDataParams = h5io.load_ImDataParams_mat(filename)
    imDataParams = imDataParams['ImDataParams']

    filename = '/Users/mnd/Projects/FatParameterEstimation/data/FatParamEstimation/BMR/BMR106_EN/20171013_131406_1302_WFIparams_GANDALF3D_VL.mat'
    wfiParams = h5io.load_WFIparams_mat(filename)
    fieldmap_Hz = np.transpose(wfiParams['WFIparams']['fieldmap_Hz'])
    fieldmap_Hz_equalized = wfi.equalize_fieldmap_periods(fieldmap_Hz,
                                                          imDataParams['TE_s'])

    options = {'init_fieldmap_Hz': fieldmap_Hz_equalized,
               'iSlice': slice(35, 37),
               'iTE': slice(2, 20),
               'uncertainty_quant': True}

    wfiParams = wfi.wfi_css_varpro(imDataParams, options)

    show_arr3d(wfi.calculate_echoMIP(imDataParams['signal']))

    print(wfiParams.keys())

    crlb = wfiParams['CRLBs'][0]

    assert len(wfiParams['CRLBs']) == 6

    close_all()
    show_arr3d(np.log(wfiParams['CRLBs'][0]))
    show_arr3d(np.log(wfiParams['CRLBs'][1]))
    show_arr3d(np.log(wfiParams['CRLBs'][2]))
    show_arr3d(np.log(wfiParams['CRLBs'][3]))
    show_arr3d(np.log(wfiParams['CRLBs'][4]))
    show_arr3d(np.log(wfiParams['CRLBs'][5]))

    close_all()
    show_arr3d(wfiParams['NSAs'][0])
    show_arr3d(wfiParams['NSAs'][1])
    show_arr3d(wfiParams['NSAs'][2])
    show_arr3d(wfiParams['NSAs'][3])
    show_arr3d(wfiParams['NSAs'][4])
    show_arr3d(wfiParams['NSAs'][5])

    trF = wfiParams['trFIM']
    trInvF = wfiParams['trInvFIM']
    detF = wfiParams['detFIM']
    detInvF = np.divide(1, detF, out=-np.ones_like(detF), where=detF!=0)

    close_all()
    show_arr3d(np.log(trF))
    show_arr3d(-np.log(trInvF))
    show_arr3d(np.log(detF))
    show_arr3d(-np.log(detInvF))


def test_wfi_css_varpro_wuq_dualR2s():
    filename = '/Users/mnd/Projects/FatParameterEstimation/data/FatParamEstimation/BMR/BMR106_EN/20171013_131406_1302_ImDataParams.mat'
    imDataParams = h5io.load_ImDataParams_mat(filename)
    imDataParams = imDataParams['ImDataParams']
    imDataParams['TE_s'] = imDataParams['TE_s'].ravel()

    filename = '/Users/mnd/Projects/FatParameterEstimation/data/FatParamEstimation/BMR/BMR106_EN/20171013_131406_1302_WFIparams_GANDALF3D_VL.mat'
    wfiParams = h5io.load_WFIparams_mat(filename)
    fieldmap_Hz = np.transpose(wfiParams['WFIparams']['fieldmap_Hz'])
    fieldmap_Hz_equalized = wfi.equalize_fieldmap_periods(fieldmap_Hz,
                                                          imDataParams['TE_s'])


    options = {'init_fieldmap_Hz': fieldmap_Hz_equalized,
               'iSlice': slice(35, 37),
               'iTE': slice(4, 16),
               'uncertainty_quant': True}

    wfiParams = wfi.wfi_css_varpro(imDataParams, options)

    options2 = options.copy()
    options2.update({'Cr': np.array([[1., 0.],
                                     [0., 1.],
                                     [0., 1.],
                                     [0., 1.],
                                     [0., 1.],
                                     [0., 1.],
                                     [0., 1.],
                                     [0., 1.],
                                     [0., 1.],
                                     [0., 1.],
                                     [0., 1.]])})

    wfiParams2 = wfi.wfi_css_varpro(imDataParams, options2)

    FIMparams = css.compute_FIMmaps(imDataParams['TE_s'],
                                    wfiParams['params_matrices'],
                                    wfiParams['Cm'],
                                    wfiParams['Cp'],
                                    wfiParams['Cf'],
                                    wfiParams['Cr'],
                                    wfiParams['mask'])

    FIMparams2 = css.compute_FIMmaps(imDataParams['TE_s'],
                                     wfiParams['params_matrices'],
                                     wfiParams['Cm'],
                                     wfiParams['Cp'],
                                     wfiParams['Cf'],
                                     wfiParams2['Cr'],
                                     wfiParams['mask'])

    close_all()
    show_arr3d(np.log(FIMparams['CRLBs'][0]))
    show_arr3d(np.log(FIMparams2['CRLBs'][0]))

    close_all()
    show_arr3d(np.log(FIMparams['trFIM']))
    show_arr3d(np.log(FIMparams2['trFIM']))
    show_arr3d(np.log(FIMparams['trFIM']) - np.log(FIMparams2['trFIM']))

    echoMIP = wfi.calculate_echoMIP(imDataParams['signal'][:, :, 35:37, 4:16])

    diff_dict = {}
    for k in ['CRLBs', 'NSAs']:
        diff_dict[k] = np.array(wfiParams[k]) - np.array(wfiParams2[k][:-1])
    for k in ['trFIM', 'trInvFIM', 'detFIM']:
        diff_dict[k] = np.array(wfiParams[k]) - np.array(wfiParams2[k])

    show_arr3d(diff_dict['CRLBs'][0])
    show_arr3d(diff_dict['CRLBs'][1])
    show_arr3d(diff_dict['CRLBs'][2])
    show_arr3d(diff_dict['CRLBs'][3])
    show_arr3d(diff_dict['CRLBs'][4])
    show_arr3d(diff_dict['CRLBs'][5])

    show_arr3d(diff_dict['trFIM'])

    close_all()
    maptofuse = 'pdff_percent'
    # maptofuse = 'fieldmap_Hz'
    vmin, vmax = 0, 100
    # vmin, vmax = -400, 200
    wheretofuse = FIMparams['trFIM'] >= FIMparams2['trFIM']
    # wheretofuse = FIMparams['detFIM'] >= FIMparams2['detFIM']
    # wheretofuse = FIMparams['trInvFIM'] <= FIMparams2['trInvFIM']
    # wheretofuse = FIMparams['detInvFIM'] <= FIMparams2['detInvFIM']
    # wheretofuse = FIMparams['NSAs'][5] >= FIMparams2['NSAs'][5]
    frankenstein = np.where(wheretofuse,
                            wfiParams[maptofuse],
                            wfiParams2[maptofuse])
    show_arr3d(wfiParams[maptofuse])
    show_arr3d(wfiParams2[maptofuse])
    show_arr3d(wheretofuse * echoMIP)
    show_arr3d(np.clip(frankenstein, vmin, vmax))
    show_arr3d(np.clip(frankenstein, vmin, vmax) -
               np.clip(wfiParams[maptofuse], vmin, vmax))


    show_arr3d(wfiParams['resnorm'])


    close_all()
    maptofuse = 'fat'
    vmin, vmax = 0, 200
    wheretofuse = FIMparams['detFIM'] >= FIMparams2['detFIM']
    frankenstein = np.where(wheretofuse,
                            wfiParams[maptofuse],
                            wfiParams2[maptofuse])
    show_arr3d(wheretofuse * echoMIP)
    show_arr3d(np.clip(wfiParams[maptofuse], vmin, vmax))
    show_arr3d(np.clip(frankenstein, vmin, vmax))
    show_arr3d(np.clip(frankenstein, vmin, vmax) -
               np.clip(wfiParams[maptofuse], vmin, vmax))

    close_all()
    maptofuse = 'R2s_Hz'
    maptofuse2 = 'waterR2s_Hz'
    vmin, vmax = 0, 200
    wheretofuse = FIMparams['detFIM'] >= FIMparams2['detFIM']
    frankenstein = np.where(wheretofuse,
                            wfiParams[maptofuse],
                            wfiParams2[maptofuse2])
    show_arr3d(wheretofuse * echoMIP)
    show_arr3d(np.clip(wfiParams[maptofuse], vmin, vmax))
    show_arr3d(np.clip(frankenstein, vmin, vmax))
    show_arr3d(np.clip(frankenstein, vmin, vmax) -
               np.clip(wfiParams[maptofuse], vmin, vmax))

    close_all()
    maptofuse = 'R2s_Hz'
    maptofuse2 = 'fatR2s_Hz'
    vmin, vmax = 0, 200
    wheretofuse = FIMparams['detFIM'] >= FIMparams2['detFIM']
    frankenstein = np.where(wheretofuse,
                            wfiParams[maptofuse],
                            wfiParams2[maptofuse2])
    show_arr3d(wheretofuse * echoMIP)
    show_arr3d(np.clip(wfiParams[maptofuse], vmin, vmax))
    show_arr3d(np.clip(frankenstein, vmin, vmax))
    show_arr3d(np.clip(frankenstein, vmin, vmax) -
               np.clip(wfiParams[maptofuse], vmin, vmax))

    close_all()
    show_arr3d(np.log(FIMparams['detFIM']))
    show_arr3d(np.log(FIMparams2['detFIM']))

    show_arr3d(np.log(FIMparams['detFIM']) - np.log(FIMparams2['detFIM']))

    show_arr3d(np.log(FIMparams['trFIM']) - np.log(FIMparams2['trFIM']))

    show_arr3d(np.log(FIMparams['trInvFIM']) - np.log(FIMparams2['trInvFIM']))

    show_arr3d(FIMparams['NSAs'][0] - FIMparams2['NSAs'][0])
    show_arr3d(FIMparams['NSAs'][1] - FIMparams2['NSAs'][1])

    show_arr3d(FIMparams['NSAs'][4] - FIMparams2['NSAs'][4])


    show_arr3d(wfiParams['fieldmap_Hz'] - wfiParams2['fieldmap_Hz'])


def test_wfi_dualR2s():
    filename = '/Users/mnd/Projects/FatParameterEstimation/data/DiagnostikBilanz/20170609_125718_0402_ImDataParams.mat'
    imDataParams = h5io.load_ImDataParams_mat(filename)
    imDataParams = imDataParams['ImDataParams']

    filename = '/Users/mnd/Projects/FatParameterEstimation/data/DiagnostikBilanz/20170609_125718_0402_WFIparams_CSS_GANDALF2D_VL.mat'
    wfiParams = h5io.load_WFIparams_mat(filename)
    fieldmap_Hz = np.transpose(wfiParams['WFIparams']['fieldmap_Hz'])
    fieldmap_Hz_equalized = wfi.equalize_fieldmap_periods(fieldmap_Hz,
                                                          imDataParams['TE_s'])

    options = {'init_fieldmap_Hz': fieldmap_Hz_equalized,
               'iSlice': slice(35, 37)}

    wfiParams = wfi.wfi_css_varpro(imDataParams, options)

    assert {'water', 'fat', 'fieldmap_Hz', 'R2s_Hz', 'pdff_percent'}.\
        issubset(wfiParams.keys())

    Cr = np.eye(len(wfiParams['Cr']), 2)
    Cr[1:, 1] = 1
    options2 = wfiParams.copy()
    options2.pop('water')
    options2.pop('fat')
    options2.pop('pdff_percent')
    options2.pop('fieldmap_Hz')
    options2.pop('R2s_Hz')
    options2.pop('param_maps')
    options2.pop('resnorm')
    options2.pop('iterations')
    options2.update({'Cr': Cr})

    wfiParams2 = wfi.wfi_css_varpro(imDataParams, options2)

    close_all()
    show_arr3d(np.clip(wfiParams['pdff_percent'], 0, 100))
    show_arr3d(np.clip(wfiParams2['pdff_percent'], 0, 100))

    close_all()
    show_arr3d(np.clip(wfiParams['R2s_Hz'], 0, 400))
    show_arr3d(np.clip(wfiParams2['R2s_Hz'], 0, 400))
    show_arr3d(np.clip(wfiParams2['param_maps'][6], 0, 400))

    close_all()
    show_arr3d(np.clip(wfiParams['fieldmap_Hz'], -400, 200))
    show_arr3d(np.clip(wfiParams2['fieldmap_Hz'], -400, 200))

    close_all()
    show_arr3d(np.clip(wfiParams['resnorm'], 0, 1e2))
    show_arr3d(np.clip(wfiParams2['resnorm'], 0, 1e2))

    close_all()
    show_arr3d(np.clip(wfiParams['iterations'], 0, 1e2))
    show_arr3d(np.clip(wfiParams2['iterations'], 0, 1e2))


def test_contrainedPhase():
    filename = '/Users/mnd/Projects/FatParameterEstimation/data/DiagnostikBilanz/20170609_125718_0402_ImDataParams.mat'
    imDataParams = h5io.load_ImDataParams_mat(filename)
    imDataParams = imDataParams['ImDataParams']

    filename = '/Users/mnd/Projects/FatParameterEstimation/data/DiagnostikBilanz/20170609_125718_0402_WFIparams_CSS_GANDALF2D_VL.mat'
    wfiParams = h5io.load_WFIparams_mat(filename)
    fieldmap_Hz = np.transpose(wfiParams['WFIparams']['fieldmap_Hz'])
    fieldmap_Hz_equalized = wfi.equalize_fieldmap_periods(fieldmap_Hz,
                                                          imDataParams['TE_s'])

    options = {'init_fieldmap_Hz': fieldmap_Hz_equalized,
               'iSlice': slice(35, 37)}

    wfiParams = wfi.wfi_css_varpro(imDataParams, options)

    assert {'water', 'fat', 'fieldmap_Hz', 'R2s_Hz', 'pdff_percent'}.\
        issubset(wfiParams.keys())

    options2 = options.copy()
    Cp = np.zeros((10, 1))
    Cp[:, 0] = 1.
    options2['Cp'] = Cp

    wfiParams2 = wfi.wfi_css_varpro(imDataParams, options2)

    close_all()
    show_arr3d(np.clip(wfiParams['pdff_percent'], 0, 100))
    show_arr3d(np.clip(wfiParams2['pdff_percent'], 0, 100))
    show_arr3d(np.clip(wfiParams2['pdff_percent'], 0, 100) - \
               np.clip(wfiParams['pdff_percent'], 0, 100))

    close_all()
    show_arr3d(np.angle(wfiParams['water']))
    show_arr3d(np.angle(wfiParams['fat']))

    show_arr3d(np.angle(wfiParams2['water']))
    show_arr3d(np.angle(wfiParams2['fat']))

    assert ~np.allclose(np.angle(wfiParams['water']),
                        np.angle(wfiParams['fat']))

    assert np.allclose(np.angle(wfiParams2['water']),
                       np.angle(wfiParams2['fat']))

    close_all()
    show_arr3d(np.clip(wfiParams['R2s_Hz'], 0, 400))
    show_arr3d(np.clip(wfiParams2['R2s_Hz'], 0, 400))
    show_arr3d(np.clip(wfiParams2['R2s_Hz'], 0, 400) - \
               np.clip(wfiParams['R2s_Hz'], 0, 400))


    close_all()
    show_arr3d(np.clip(wfiParams['fieldmap_Hz'], -400, 200))
    show_arr3d(np.clip(wfiParams2['fieldmap_Hz'], -400, 200))
    show_arr3d(np.clip(wfiParams2['R2s_Hz'], -400, 200) - \
               np.clip(wfiParams['R2s_Hz'], -400, 200))

    close_all()
    show_arr3d(np.clip(wfiParams['resnorm'], 0, 1e2))
    show_arr3d(np.clip(wfiParams2['resnorm'], 0, 1e2))

    close_all()
    show_arr3d(wfiParams['iterations'], )
    show_arr3d(wfiParams2['iterations'])


def test_bydder():

    from Fatmodel import Fatmodel
    F = Fatmodel()
    F.set_params_matrix()
    # F.build_signal()
    # F.plot_signal()
    # F.pm[1:, 1] = np.pi/2
    F.build_signal()
    # F.plot_signal()
    # F.sample_signal(SNR=30)
    F.set_constraints_matrices()
    Cm, Cp, Cf, Cr = F.Cm, F.Cp, F.Cf, F.Cr
    Cmc = Cm.astype(np.complex128)

    sig = F.signal_samp
    tol = 1e-10
    itermax = 100
    pm = deepcopy(F.pm)
    pm[:, :2] = 0
    pm[:, 2] -= pm0[0, 2]
    pm[:, 3] = 0
    Cp = np.zeros((len(pm0), 1))
    Cp[:, 0] = 1.

    TE_s = F.TE_s

    M, Ntot, Nm, Np, Nf, Nr = css.get_paramsCount(Cm, Cp, Cf, Cr)

    eps = 2 * tol
    i = 0
    cond_thres = 1e6

    while eps > tol and i < itermax:
        # update linear parameters
        A = css.get_Amatrix(TE_s, pm)
        if np.isnan(A).any() or np.isinf(A).any():
            break
        if np.linalg.cond(A) > cond_thres:
            break

        # P = np.diag(np.exp(1.j * pm[:, 1]))
        # Atb = A.conj().T.dot(sig)
        # AtArealinv = np.linalg.inv(A.conj().T.dot(A).real)
        # pm[:, 0] = AtArealinv.dot(np.real(P.conj().dot(Atb)))
        # print(pm)
        # phi = 0.5 * np.angle(Atb.conj().T.dot(AtAinv).dot(P.conj().dot(Atb)))

        rho = np.linalg.lstsq(np.dot(A, Cmc), sig, rcond=-1)[0]
        Cm_rho = np.dot(Cmc, rho)
        pm[:, 0] = np.abs(Cm_rho)
        # pm[:, 1] = np.angle(Cm_rho)
        # pm[:, 0] = css.extract_param_type(0, pm, [Cm, Cp, Cf, Cr])
        # pm[:, 1] = css.extract_param_type(1, pm, [Cm, Cp, Cf, Cr])

        # update nonlinear parameters
        r = sig - css.build_signal(TE_s, pm)
        J = css.get_Jacobian(TE_s, pm, Cm, Cp, Cf, Cr)
        rr = np.concatenate((r.real, r.imag), axis=0)
        JJ = np.concatenate((J.real, J.imag), axis=0)
        if np.isnan(JJ).any() or np.isinf(JJ).any():
            break
        if np.linalg.cond(JJ) > cond_thres:
            break
        updates = np.linalg.lstsq(JJ, rr, rcond=-1)[0]
        print(updates)
        eps = np.linalg.norm(updates)  # for convergence criterium

        pm_update = np.zeros_like(pm)
        pm_update[:, 1] = np.dot(Cp, updates[Nm:Nm+Np])
        pm_update[:, 2] = np.dot(Cf, updates[Nm+Np:Nm+Np+Nf])
        pm_update[:, 3] = np.dot(Cr, updates[Nm+Np+Nf:Nm+Np+Nf+Nr])
        pm += pm_update

        i += 1

    resnorm = np.linalg.norm(sig - css.build_signal(TE_s, pm)) # residual norm

    print(pm0, F.pm, pm, pm - F.pm, resnorm, i, sep='\n')

    pme, resnorm, i = css.varpro(F.TE_s, sig, pm0, F.Cm, F.Cp, F.Cf, F.Cr, tol, itermax)

    print(pme)
    A = css.get_Amatrix(TE_s, pme)
    print(A)
    Atb = A.conj().T.dot(sig)
    print(Atb)
    AtArealinv = np.linalg.inv(A.conj().T.dot(A).real)
    print(AtArealinv)
    phi = 0.5 * np.angle(Atb.conj().T.dot(AtAinv).dot(Atb))
    rho = AtArealinv.dot((np.exp(-1.j * phi) * Atb).real)
    print(rho, phi)


def test_fucl_wat():
    filename = '/Users/mnd/Projects/FatParameterEstimation/data/FatParamEstimation/BMR/BMR106_EN/20171013_131406_1302_ImDataParams.mat'
    imDataParams = h5io.load_ImDataParams_mat(filename)
    imDataParams = imDataParams['ImDataParams']

    filename = '/Users/mnd/Projects/FatParameterEstimation/data/FatParamEstimation/BMR/BMR106_EN/20171013_131406_1302_WFIparams_GANDALF3D_VL.mat'
    wfiParams = h5io.load_WFIparams_mat(filename)
    for k in wfiParams['WFIparams'].keys():
        print(k)
    fieldmap_Hz = np.transpose(wfiParams['WFIparams']['fieldmap_Hz'])

    echoMIP = wfi.calculate_echoMIP(imDataParams['signal'])

    options = {'init_fieldmap_Hz': fieldmap_Hz,
               'iSlice': slice(34, 40),
               'fatmodel': 'Hamilton sSAT'}

    wfiParams = wfi.wfi_css_varpro(imDataParams, options)

    close_all()
    show_arr3d(echoMIP)
    show_arr3d(np.clip(wfiParams['pdff_percent'], -10, 110))
    show_arr3d(np.clip(wfiParams['R2s_Hz'], 0, 200))
    show_arr3d(np.clip(wfiParams['fieldmap_Hz'], -400, 200))


    F = Fatmodel(modelname='Berglund 10 peaks')
    F.compute_fatmodel(cl=18.5, ndb=2.73, nmidb=0.76)
    F.set_constraints_matrices()

    Cm = np.array([[1, 0,   0, 0, 0,   0, 0],
                   [0, 1,   0, 0, 0,   0, 0],
                   [0, 0,   1, 0, 0,   0, 0],
                   [0, 2/3, 0, 0, 0,   0, 0],
                   [0, 0,   0, 0, 1,   0, 0],
                   [0, 2/3, 0, 0, 0,   0, 0],
                   [0, 0,   0, 0, 0,   0, 1],
                   [0, 2/9, 0, 0, 0,   0, 0],
                   [0, 2/9, 0, 0, 0,   0, 0],
                   [0, 1/9, 0, 0, 0,   0, 0],
                   [0, 0,   0, 0, 1/2, 0, 1]])
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

    options['fatmodel'] = 'Berglund 10 peaks'
    options['Cm'] = Cm
    options['Cp'] = Cp

    fuclParams = wfi.wfi_css_varpro(imDataParams, options)

    print(len(fuclParams['param_maps']))
    close_all()
    W = fuclParams['param_maps'][0] * np.exp(1.j * fuclParams['param_maps'][5])
    F1 = fuclParams['param_maps'][1] * np.exp(1.j * fuclParams['param_maps'][6])
    F2 = fuclParams['param_maps'][2] * np.exp(1.j * fuclParams['param_maps'][7])
    F3 = fuclParams['param_maps'][3] * np.exp(1.j * fuclParams['param_maps'][8])
    F4 = fuclParams['param_maps'][4] * np.exp(1.j * fuclParams['param_maps'][9])

    F1nonzero = F1 != 0
    null = np.zeros_like(F1nonzero, dtype=np.complex128)
    CL = 4 + np.abs(np.divide((F2 + 4 * F3 + 3 * F4) / 3, F1,
                              out=null, where=F1nonzero))
    UD = np.abs(np.divide(F3 + F4, F1, out=null, where=F1nonzero))
    SF = 1 - np.abs(np.divide(F3 / 3, F1, out=null, where=F1nonzero))
    PUD = np.abs(np.divide(F4, F1, out=null, where=F1nonzero))
    PUF = PUD / 3
    UF = np.abs(np.divide(F3 / 3, F1, out=null, where=F1nonzero))
    MUF = UF - PUF

    show_arr3d(CL)
    show_arr3d(UD)
    show_arr3d(SF)
    show_arr3d(PUD)
    show_arr3d(PUF)
    show_arr3d(UF)
    show_arr3d(MUF)


def test_fucl_bat():
    filename = '/Users/mnd/Projects/FatParameterEstimation/data/FatParamEstimation/BMR/BMR106_EN/20171013_124742_0502_ImDataParams.mat'
    imDataParams = h5io.load_ImDataParams_mat(filename)
    imDataParams = imDataParams['ImDataParams']

    filename = '/Users/mnd/Projects/FatParameterEstimation/data/FatParamEstimation/BMR/BMR106_EN/20171013_124742_0502_WFIparams_GANDALF3D_VL.mat'
    wfiParams = h5io.load_WFIparams_mat(filename)
    for k in wfiParams['WFIparams'].keys():
        print(k)
    fieldmap_Hz = np.transpose(wfiParams['WFIparams']['fieldmap_Hz'])

    echoMIP = wfi.calculate_echoMIP(imDataParams['signal'])

    options = {'init_fieldmap_Hz': fieldmap_Hz,
               'iSlice': slice(17, 21),
               'fatmodel': 'Hamilton sSAT',
               'tol': 1e-3,
               'itermax': 20}

    F = Fatmodel(modelname='Berglund 10 peaks')
    F.compute_fatmodel(cl=18.5, ndb=2.73, nmidb=0.76)
    F.set_constraints_matrices()

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

    options['fatmodel'] = 'Berglund 10 peaks'
    options['Cm'] = Cm
    options['Cp'] = Cp

    fuclParams = wfi.wfi_css_varpro(imDataParams, options)

    close_all()

    show_arr3d(echoMIP)

    W = fuclParams['param_maps'][0] * np.exp(1.j * fuclParams['param_maps'][5])
    F1 = fuclParams['param_maps'][1] * np.exp(1.j * fuclParams['param_maps'][6])
    F2 = fuclParams['param_maps'][2] * np.exp(1.j * fuclParams['param_maps'][7])
    F3 = fuclParams['param_maps'][3] * np.exp(1.j * fuclParams['param_maps'][8])
    F4 = fuclParams['param_maps'][4] * np.exp(1.j * fuclParams['param_maps'][9])

    show_arr3d(np.abs(W))
    show_arr3d(np.abs(F1))
    show_arr3d(np.abs(F2))
    show_arr3d(np.abs(F3))
    show_arr3d(np.abs(F4))

    fieldmap_Hz = fuclParams['param_maps'][10]
    R2s_Hz = fuclParams['param_maps'][11]
    show_arr3d(fieldmap_Hz)
    show_arr3d(R2s_Hz)

    show_arr3d(fuclParams['resnorm'])
    show_arr3d(fuclParams['iterations'])

    F1nonzero = F1 != 0
    null = np.zeros_like(F1nonzero, dtype=np.complex128)
    CL = 4 + np.abs(np.divide((F2 + 4 * F3 + 3 * F4) / 3, F1,
                              out=null, where=F1nonzero))
    UD = np.abs(np.divide(F3 + F4, F1, out=null, where=F1nonzero))
    SF = 1 - np.abs(np.divide(F3 / 3, F1, out=null, where=F1nonzero))
    PUD = np.abs(np.divide(F4, F1, out=null, where=F1nonzero))
    PUF = PUD / 3
    UF = np.abs(np.divide(F3 / 3, F1, out=null, where=F1nonzero))
    MUF = UF - PUF

    show_arr3d(CL)
    show_arr3d(UD)
    show_arr3d(SF)
    show_arr3d(PUD)
    show_arr3d(PUF)
    show_arr3d(UF)
    show_arr3d(MUF)


def test_dualR2s_wat():
    filename = '/Users/mnd/Projects/FatParameterEstimation/data/FatParamEstimation/BMR/BMR106_EN/20171013_131406_1302_ImDataParams.mat'
    imDataParams = h5io.load_ImDataParams_mat(filename)
    imDataParams = imDataParams['ImDataParams']

    filename = '/Users/mnd/Projects/FatParameterEstimation/data/FatParamEstimation/BMR/BMR106_EN/20171013_131406_1302_WFIparams_GANDALF3D_VL.mat'
    wfiParams = h5io.load_WFIparams_mat(filename)
    fieldmap_Hz = np.transpose(wfiParams['WFIparams']['fieldmap_Hz'])
    fieldmap_Hz_equalized = wfi.equalize_fieldmap_periods(fieldmap_Hz,
                                                          imDataParams['TE_s'])

    options = {'init_fieldmap_Hz': fieldmap_Hz_equalized,
               'iSlice': slice(35, 37)}

    wfiParams = wfi.wfi_css_varpro(imDataParams, options)

    Cr = np.eye(len(wfiParams['Cr']), 2)
    Cr[1:, 1] = 1

    wfiParams2 = wfi.wfi_css_varpro(imDataParams, options2)

    close_all()
    show_arr3d(np.clip(wfiParams['pdff_percent'], 0, 100))
    show_arr3d(np.clip(wfiParams2['pdff_percent'], 0, 100))

    close_all()
    show_arr3d(np.clip(wfiParams['R2s_Hz'], 0, 400))
    show_arr3d(np.clip(wfiParams2['R2s_Hz'], 0, 400))
    show_arr3d(np.clip(wfiParams2['param_maps'][6], 0, 400))

    close_all()
    show_arr3d(np.clip(wfiParams['fieldmap_Hz'], -400, 200))
    show_arr3d(np.clip(wfiParams2['fieldmap_Hz'], -400, 200))

    close_all()
    show_arr3d(np.clip(wfiParams['resnorm'], 0, 1e2))
    show_arr3d(np.clip(wfiParams2['resnorm'], 0, 1e2))

    close_all()
    show_arr3d(np.clip(wfiParams['iterations'], 0, 1e2))
    show_arr3d(np.clip(wfiParams2['iterations'], 0, 1e2))

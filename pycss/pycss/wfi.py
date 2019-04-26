import numpy as np
import scipy.linalg as la
from numba import njit, prange
import matplotlib.pyplot as plt
from copy import deepcopy
import utils.customio as io
from Fatmodel import Fatmodel
import css
import time


def wfi_css_varpro(imDataParams, options=None):

    imDataParams = imDataParams.copy()
    options = options.copy()

    if options is None:
        options = {}

    # adjust imDataParams
    TE_s = imDataParams['TE_s'].ravel()
    iTE = options.get('iTE', slice(0, len(TE_s)))
    imDataParams['TE_s'] = TE_s[iTE]

    signal = imDataParams['signal']
    iSlice = options.get('iSlice', slice(0, signal.shape[2]))
    imDataParams['signal'] = signal[:, :, iSlice, iTE]

    # default options
    mask = calculate_tissue_mask(imDataParams['signal'],
                                 options.get('airsignal_threshold_percent', 5))

    init_fieldmap_Hz = options.get('init_fieldmap_Hz')
    if init_fieldmap_Hz is None:
        init_fieldmap_Hz = np.zeros_like(mask, dtype=float)
    else:
        init_fieldmap_Hz = init_fieldmap_Hz[:, :, iSlice]

    F = Fatmodel()
    F.set_fatmodel(options.get('fatmodel', 'Berglund 10 peaks'))
    F.compute_fatmodel(cl=options.get('cl', 18.5),
                       ndb=options.get('ndb', 2.73,),
                       nmidb=options.get('nmidb', 0.76))
    F.set_constraints_matrices()

    Pm0 = build_Pm0(F.get_chemical_shifts_Hz(), init_fieldmap_Hz[mask].ravel())

    default_options = {'mask': mask,
                       'Cm': F.Cm,
                       'Cp': F.Cp,
                       'Cf': F.Cf,
                       'Cr': F.Cr,
                       'Pm0': Pm0,
                       'tol': 1e-6,
                       'itermax': 100,
                       'verbose': True}
    default_options.update(options)
    options = default_options

    outParams = css.css_varpro(imDataParams, options)

    model_dict = structure_param_maps(outParams['param_maps'],
                                      [options['Cm'],
                                       options['Cp'],
                                       options['Cf'],
                                       options['Cr']])
    outParams.update(model_dict)
    outParams.update(options)

    return outParams


def structure_param_maps(param_maps, constrained_matrices):
    model_dict = {}
    M, Ntot, Nm, Np, Nf, Nr = css.get_paramsCount(*constrained_matrices)

    if Np == 1:
        model_dict['model_desc'] = 'constrained phase, '
        model_dict['water'] = param_maps[0] * np.exp(1.j * param_maps[2])
        model_dict['fat'] = param_maps[1] * np.exp(1.j * param_maps[2])
        nextind = 3
    elif Np == 2:
        model_dict['model_desc'] = '(unconstrained phase), '
        model_dict['water'] = param_maps[0] * np.exp(1.j * param_maps[2])
        model_dict['fat'] = param_maps[1] * np.exp(1.j * param_maps[3])
        nextind = 4
    else:
        print('unknown model')
        return model_dict

    model_dict['fieldmap_Hz'] = param_maps[nextind]

    if Nr == 1:
        model_dict['model_desc'] += 'single R2*'
        model_dict['R2s_Hz'] = param_maps[nextind+1]
    elif Nr == 2:
        model_dict['model_desc'] += 'dual R2*'
        model_dict['waterR2s_Hz'] = param_maps[nextind+1]
        model_dict['fatR2s_Hz'] = param_maps[nextind+2]
    else:
        print('unknown model')
        return model_dict

    model_dict['pdff_percent'] = \
        calculate_pdff_percent(model_dict['water'], model_dict['fat'])

    return model_dict


@njit
def build_Pm0(chemical_shifts_Hz, fieldmap_Hz):
    fieldmap_Hz = fieldmap_Hz.ravel()
    nVoxel = len(fieldmap_Hz)
    Pm0 = np.zeros((nVoxel, len(chemical_shifts_Hz), 4))
    for i in range(nVoxel):
        Pm0[i, :, 2] = chemical_shifts_Hz + fieldmap_Hz.ravel()[i]
    return Pm0


def wrap_fieldmap_Hz(fieldmap_Hz, TE_s):
    dTE_s = np.diff(TE_s.ravel())[0]
    return np.angle(np.exp(2.j * np.pi * fieldmap_Hz * dTE_s)) / \
                    (2. * np.pi * dTE_s)


def equalize_fieldmap_periods(fieldmap_Hz, TE_s):
    dTE_s = np.diff(TE_s.ravel())[0]
    period_length = 1 / dTE_s
    fieldmap_Hz_wrapped = wrap_fieldmap_Hz(fieldmap_Hz, TE_s)
    nperiods = np.round(
        (fieldmap_Hz - fieldmap_Hz_wrapped).mean(0).mean(0) * dTE_s)
    sz = fieldmap_Hz.shape
    return fieldmap_Hz - period_length * np.tile(nperiods, [sz[0], sz[1], 1])


def calculate_pdff_percent(W, F):
    WF = np.abs(W + F)
    Wr = np.divide(np.abs(W), WF, out=np.zeros_like(WF), where=WF!=0)
    Fr = np.divide(np.abs(F), WF, out=np.zeros_like(WF), where=WF!=0)
    wf = np.abs(W) + np.abs(F)
    pdff_abs = 100 * np.divide(np.abs(F), wf,
                               out=np.zeros_like(wf), where=wf!=0)
    return 100 * np.where((0 < pdff_abs) & (pdff_abs <= 50), (1 - Wr), Fr)


def calculate_echoMIP(signal):
    return np.abs(signal).sum(-1)


def calculate_tissue_mask(signal, threshold_percent=5):
    echoMIP = calculate_echoMIP(signal)
    return echoMIP > threshold_percent/100 * echoMIP.max()

import numpy as np
from numba import njit, jit
import nbutils
from nbutils import ind2sub, sub2ind, list_multi_indexes, find_local_minima
import sim
import css


# @jit
# def map_find_residual_minima(TE_s, Sig, Cm, Cp, Cf, Cr, nlparams, shape):
#     """FIXME! briefly describe function

#     :param TE_s:
#     :param Sig:
#     :param pm:
#     :param fieldmapRange_Hz:
#     :param r2sRange_Hz:
#     :param nMinima: int, number of minima to be extracted from the residual
#                     spanned by fieldmapRange_Hz and r2sRange_Hz
#     :returns: ResidualXvals and ResidualYvals
#     :rtype: tuple of two np.arrays with shapes nVoxels x nMinima (x 2)

#     """
#     nTE = len(TE_s)
#     nVoxel = Sig.shape[0]
#     assert Sig.shape[-1] == nTE

#     # ResidualMinInds = np.zeros((nVoxel, nMinima))
#     ResidualXvals = np.zeros((nVoxel, nMinima, 2))
#     ResidualYvals = np.zeros((nVoxel, nMinima))
#     for i in range(nVoxel):

#         sig = Sig[i, ...].reshape(Sig.shape[1:])

#         R = get_residual_array(TE_s, sig, Cm, Cp, Cf, Cr, nlparams, shape)

#         isMinimum = find_local_minima(R)

#         min_fieldmap_ind, min_r2s_ind = np.where(isMinimum)

#         ResidualXvals[i, :nMinima, 0] = fieldmapRange_Hz\
#                                         [min_fieldmap_ind[:nMinima]]
#         ResidualXvals[i, :nMinima, 1] = r2sRange_Hz[min_r2s_ind[:nMinima]]
#         ResidualYvals[i, :nMinima] = residual[min_fieldmap_ind, min_r2s_ind]
#     return ResidualXvals, ResidualYvals


@njit
def get_residual_minima(TE_s, sig, pm_init, Cm, Cp, Cf, Cr, nlparams, shape):
    """return varpro residuals as ndarray for
    different parameter value combinations defined by nlparams and shape

    (for the njitted np.reshape the parameter shape needs to be a tuple,
    but for the nbutils functions shape is converted to a np.array)

    :param TE_s: 1d np.array, with echo times in seconds
    :param sig: 1d np.array, measured complex MR signal
    :param Cm: 2d np.ndarray, constraints matrix for magnitudes
    :param Cp: 2d np.ndarray, constraints matrix for phases
    :param Cf: 2d np.ndarray, constraints matrix for resonance frequencies
    :param Cr: 2d np.ndarray, constraints matrix for relaxation rates
    :param nlparams: 1d iterable, flattend parameter types and values
    :param shape: tuple (!), numbers of unique values per type
    :returns:
    :rtype:

    """
    residuals = get_residual_array(TE_s, sig, pm_init,
                                   Cm, Cp, Cf, Cr, nlparams, shape)
    isMinimum = find_local_minima(residuals)
    param_indices = np.where(isMinimum)

    Nparams = len(shape)
    Nminima = (isMinimum > 0).sum()

    params_at_minima = np.zeros((Nparams+1, Nminima))
    for m in range(Nminima):
        sub = [p[m] for p in param_indices]
        ind = sub2ind(sub, shape)
        params_at_minima[0, m] = residuals.ravel()[ind]

    sz = np.array(shape)
    cumsum = np.cumsum(sz)
    sep_inds = np.zeros((Nparams, 2), dtype=np.int64)
    sep_inds[:, 1] = cumsum
    sep_inds[1:, 0] = cumsum[:-1]

    for i in range(sep_inds.shape[0]):
        prange = nlparams[sep_inds[i, 0]:sep_inds[i, 1]]
        params_at_minima[i+1, :] = prange[param_indices[i]]

    return params_at_minima


@njit
def get_global_residual_minimum(TE_s, sig, pm_init, Cm, Cp, Cf, Cr, nlparams, shape):

    Pm = sim.assemble_Pm(nlparams, np.array(shape), Cp, Cf, Cr, pm_init)
    N = Pm.shape[0]
    residual_minimum = 1e5
    pm_minimum = np.zeros(Pm.shape[1:])
    for i in range(N):
        pm = Pm[i]
        residual = get_residual(TE_s, pm, sig, Cm)
        if residual < residual_minimum:
            residual_minimum = residual
            pm_minimum = pm

    return pm_minimum, residual_minimum


@njit
def get_residual_array(TE_s, sig, pm_init, Cm, Cp, Cf, Cr, nlparams, shape):
    """return varpro residuals as ndarray

    (for the njitted np.reshape the parameter shape needs to be a tuple,
    but for the nbutils functions shape is converted to a np.array)

    :param TE_s: 1d np.array, with echo times in seconds
    :param sig: 1d np.array, measured complex MR signal
    :param pm_init: 2d ndarray, initial parameter matrix, (holds chemical shifts)
    :param Cm: 2d np.ndarray, constraints matrix for magnitudes
    :param Cp: 2d np.ndarray, constraints matrix for phases
    :param Cf: 2d np.ndarray, constraints matrix for resonance frequencies
    :param Cr: 2d np.ndarray, constraints matrix for relaxation rates
    :param nlparams: 1d iterable, flattend parameter types and values
    :param shape: tuple (!), numbers of unique values per type
    :returns:
    :rtype:

    """
    Pm = sim.assemble_Pm(nlparams, np.array(shape), Cp, Cf, Cr, pm_init)

    residual_list = list_residuals(TE_s, sig, Pm, Cm)

    return residual_list.reshape(shape)


@njit
def list_residuals(TE_s, sig, Pm, Cm):
    """compute varpor residual for different parameter value combinations
    and return them in a 1d arrary

    :param TE_s: 1d np.array, with echo times in seconds
    :param sig: 1d np.array, measured complex MR signal
    :param Pm: 3d numpy.ndarray, #params_combinations x #species x #4
    :param Cm: 2d np.ndarray, constraints matrix for magnitudes
    :returns: varpro residual list
    :rtype: 1d np.array of lenght Pm.shape[0]

    """
    N = len(Pm)
    residual = np.zeros(N)
    for i in range(N):
        pm = Pm[i]
        residual[i] = get_residual(TE_s, pm, sig, Cm)

    return residual


@njit
def get_residual(TE_s, pm, sig, Cm):
    A = css.get_Amatrix(TE_s, pm)
    P = np.diag(np.exp(1.j * pm[:, 1]))
    APCm = np.dot(np.dot(A, P), Cm.astype(A.dtype))
    return np.linalg.norm(np.dot(1 - np.dot(APCm, np.linalg.pinv(APCm)), sig))

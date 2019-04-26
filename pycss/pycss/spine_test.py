import pytest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from copy import deepcopy
# import css
from utils.utils import *
from utils.customio import *
import SimpleITK as sitk
import matplotlib.pyplot as plt
# from pywfi import wfi
from pprint import pprint
from Fatmodel import Fatmodel
import css
import wfi


filename = '/Users/mnd/Projects/FatParameterEstimation/data/DiagnostikBilanz/20170609_125718_0402_ImDataParams.mat'
ImDataParams = load_ImDataParams_mat(filename)
ImDataParams = ImDataParams['ImDataParams']
iz = slice(35, 37)
# iz = slice(0,  ImDataParams['signal'].shape[2])
ImDataParams['signal'] = ImDataParams['signal'][:, :, iz, :]
sz = ImDataParams['signal'].shape[:-1]

pprint(ImDataParams.keys())

tissue_mask = wfi.calculate_tissue_mask(ImDataParams['signal'])
# show_arr3d(tissue_mask)

Sig = ImDataParams['signal'][tissue_mask]
print(Sig.shape)

filename = '/Users/mnd/Projects/FatParameterEstimation/data/DiagnostikBilanz/20170609_125718_0402_WFIparams_CSS_GANDALF2D_VL.mat'
h5file = h5.File(filename, 'r')
path = '/WFIparams'
f = h5.File(filename, 'r')

pprint(list(f['/WFIparams'].keys()))

TE_s = ImDataParams['TE_s'].ravel()
fieldmap_Hz = np.transpose(f['/WFIparams/fieldmap_Hz'][...])
fieldmap_Hz = fieldmap_Hz[:, :, iz]

fieldmap_Hz_equalized = tissue_mask * wfi.equalize_fieldmap_periods(fieldmap_Hz, TE_s)
fieldmap_Hz_equalized = fieldmap_Hz_equalized[tissue_mask]



F = Fatmodel(modelname='Hamilton VAT')
F.set_constraints_matrices()
Cm, Cp, Cf, Cr = F.constraints_matrices
F.set_params_matrix()

tol, itermax = 1e-3, 100

# nVoxel = Sig.shape[0]
chemical_shifts_Hz = np.concatenate(([0], F.centerfreq_Hz * 1e-6 * F.deshielding_ppm))
wfi.build_Pm0(chemical_shifts_Hz, fieldmap_Hz_equalized)

Pm0 = np.zeros((nVoxel, len(chemical_shifts_Hz), 4))
for i in range(nVoxel):
    Pm0[i, :, 2] = chemical_shifts_Hz + fieldmap_Hz_equalized.ravel()[i]


print(Sig.shape, Pm0.shape, TE_s.shape)


res = css.map_varpro(TE_s, Sig, Pm0, Cm, Cp, Cf, Cr, tol, itermax)

import sys
sys.path.append('../../')
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import *
import SimpleITK as sitk
import matplotlib.pyplot as plt
from Fatmodel import Fatmodel
import css
import wfi
import h5io
import sim
import deepdish as dd
from matplotlib2tikz import save as tikzsave
import subprocess


filename = '/Users/mnd/Projects/FatParameterEstimation/data/FatParamEstimation/BMR/BMR106_EN/20171013_131406_1302_ImDataParams_corr.mat'
imDataParams = h5io.load_ImDataParams_mat(filename)
imDataParams = imDataParams['ImDataParams']
# imDataParams = dd.io.load('imDataParams.h5')
echoMIP = wfi.calculate_echoMIP(imDataParams['signal'])

# filename = '/Users/mnd/Projects/FatParameterEstimation/data/FatParamEstimation/BMR/BMR106_EN/20171013_131406_1302_WFIparams_GANDALF3D_VL.mat'
# wfiParams = h5io.load_WFIparams_mat(filename)
# fieldmap_Hz = np.transpose(wfiParams['WFIparams']['fieldmap_Hz'])

# options = {'init_fieldmap_Hz': fieldmap_Hz,
#            'iSlice': slice(39, 40),
#            # 'iTE': slice(3, 18),
#            'fatmodel': 'Hamilton sSAT'}

# # wfiParams = wfi.wfi_css_varpro(imDataParams, options)
# # dd.io.save('wfiParams.h5', wfiParams)
wfiParams = dd.io.load('wfiParams.h5')


# F = Fatmodel(modelname='Berglund 10 peaks')
# F.compute_fatmodel(cl=18.5, ndb=2.73, nmidb=0.76)
# F.set_constraints_matrices()

# Cm = np.array([[1, 0, 0, 0, 0],
#                [0, 9, 0, 0, 0],
#                [0, 0, 2, 0, 0],
#                [0, 6, 0, 0, 0],
#                [0, 0, 0, 4, 0],
#                [0, 6, 0, 0, 0],
#                [0, 0, 0, 0, 2],
#                [0, 2, 0, 0, 0],
#                [0, 2, 0, 0, 0],
#                [0, 1, 0, 0, 0],
#                [0, 0, 0, 2, 2]])
# Cp = (Cm > 0).astype(float)

# options['fatmodel'] = 'Berglund 10 peaks'
# options['Cm'] = Cm
# options['Cp'] = Cp

# fuclParams = wfi.wfi_css_varpro(imDataParams, options)
# dd.io.save('fuclParams.h5', fuclParams)
fuclParams = dd.io.load('fuclParams.h5')


def div(arr1, arr2, out=0.):
    return np.divide(arr1, arr2, out=out*np.ones_like(arr1), where=arr2!=0)


mask = fuclParams['mask']
Pme = fuclParams['params_matrices']

W = np.squeeze(css.construct_map(Pme[:, 0, 0] *
                                  np.exp(1j * Pme[:, 0, 1]), mask))
F1 = np.squeeze(css.construct_map(Pme[:, 1, 0] / 9 *
                                  np.exp(1j * Pme[:, 1, 1]), mask))
F2 = np.squeeze(css.construct_map(Pme[:, 2, 0] / 2 *
                                  np.exp(1j * Pme[:, 2, 1]), mask))
F3 = np.squeeze(css.construct_map((Pme[:, 4, 0] / 4 + Pme[:, 10, 0] / 2) *
                                  np.exp(1j * Pme[:, 1, 1]), mask))
F4 = np.squeeze(css.construct_map((Pme[:, 6, 0] / 2 + Pme[:, 10, 0] / 2) *
                                  np.exp(1j * Pme[:, 1, 1]), mask))
fieldmap_Hz = np.squeeze(css.construct_map(Pme[:, 0, 2], mask))
R2s_Hz = np.squeeze(css.construct_map(Pme[:, 0, 3], mask))

totalFat = F1 + F2 + F3 + F4
WA = 100 * np.abs(div(W, W + totalFat))
pdff_percent = 100 * np.abs(div(totalFat, W + totalFat))
FA1 = 100 * np.abs(div(F1, totalFat))
FA2 = 100 * np.abs(div(F2, totalFat))
FA3 = 100 * np.abs(div(F3, totalFat))
FA4 = 100 * np.abs(div(F4, totalFat))

CL = 4. + np.abs(div((F2 + 4 * F3 + 3 * F4) / 3., F1))
UD = np.abs(div(F3 + F4, F1))
SF = 1 - np.abs(div(F3 / 3., F1))
PUD = np.abs(div(F4, F1))
PUF = PUD / 3.
UF = np.abs(div(F3 / 3., F1))
MUF = UF - PUF

##############

subprocess.call(['rm', '*.png'])

plt.close('all')
fig, axs = plt.subplots(3, 3)
axs[0][0].imshow(np.squeeze(np.clip(wfiParams['pdff_percent'], 0, 100))[32:-50, 15:-15],
                 cmap='inferno')
im = axs[0][1].imshow(np.clip(np.squeeze(pdff_percent), 0, 100)[32:-50, 15:-15],
                      cmap='inferno')
plt.colorbar(im, ax=axs[0][1])
im = axs[0][2].imshow(np.squeeze(np.clip(wfiParams['pdff_percent'], 0, 100))[32:-50, 15:-15] -
                      np.clip(np.squeeze(pdff_percent), 0, 100)[32:-50, 15:-15],
                      cmap='inferno',
                      vmin=-5, vmax=10)
plt.colorbar(im, ax=axs[0][2])

axs[1][0].imshow(np.squeeze(np.clip(wfiParams['fieldmap_Hz'], -200, 300))[32:-50, 15:-15], cmap='plasma')
im = axs[1][1].imshow(np.squeeze(np.clip(fieldmap_Hz, -200, 300))[32:-50, 15:-15], cmap='plasma')
plt.colorbar(im, ax=axs[1][1])
im = axs[1][2].imshow(np.squeeze(wfiParams['fieldmap_Hz'])[32:-50, 15:-15] -
                      np.squeeze(fieldmap_Hz)[32:-50, 15:-15],
                      cmap='plasma',
                      vmin=-5, vmax=5)
plt.colorbar(im, ax=axs[1][2])

axs[2][0].imshow(np.squeeze(np.clip(wfiParams['R2s_Hz'], 0, 100))[32:-50, 15:-15],
                 cmap='magma')
im = axs[2][1].imshow(np.clip(np.squeeze(R2s_Hz), 0, 100)[32:-50, 15:-15],
                      cmap='magma')
plt.colorbar(im, ax=axs[2][1])
im = axs[2][2].imshow(np.squeeze(np.clip(wfiParams['R2s_Hz'], 0, 100))[32:-50, 15:-15] -
                      np.clip(np.squeeze(R2s_Hz), 0, 100)[32:-50, 15:-15],
                      cmap='magma',
                      vmin=-5, vmax=5)
plt.colorbar(im, ax=axs[2][2])
plt.tight_layout()
tikzsave('PdffFieldmapR2s.tex')


# plt.close('all')
# fig, axs = plt.subplots(1, 6)
# axs[0].imshow(np.squeeze(np.abs(W))[32:-50, 15:-15], vmin=0, vmax=100)
# axs[1].imshow(np.squeeze(np.abs(totalFat))[32:-50, 15:-15], vmin=0, vmax=100)
# axs[2].imshow(np.squeeze(np.abs(F1))[32:-50, 15:-15], vmin=0, vmax=100)
# axs[3].imshow(np.squeeze(np.abs(F2))[32:-50, 15:-15], vmin=0, vmax=100)
# axs[4].imshow(np.squeeze(np.abs(F3))[32:-50, 15:-15], vmin=0, vmax=100)
# axs[5].imshow(np.squeeze(np.abs(F4))[32:-50, 15:-15], vmin=0, vmax=100)
# plt.tight_layout()
# tikzsave('fatamps.tex')


# plt.close('all')
# fig, axs = plt.subplots(1, 5)
# im = axs[0].imshow(np.squeeze(WA)[32:-50, 15:-15], vmin=0, vmax=100)
# plt.colorbar(im, ax=axs[0])
# im = axs[1].imshow(np.squeeze(FA1)[32:-50, 15:-15], vmin=0, vmax=100)
# plt.colorbar(im, ax=axs[1])
# im = axs[2].imshow(np.squeeze(FA2)[32:-50, 15:-15], vmin=0, vmax=100)
# plt.colorbar(im, ax=axs[2])
# im = axs[3].imshow(np.squeeze(FA3)[32:-50, 15:-15], vmin=0, vmax=100)
# plt.colorbar(im, ax=axs[3])
# im = axs[4].imshow(np.squeeze(FA4)[32:-50, 15:-15], vmin=0, vmax=100)
# plt.colorbar(im, ax=axs[4])
# plt.tight_layout()
# tikzsave('fatfracs.tex')

plt.close('all')
fig, axs = plt.subplots(1, 3)
im = axs[0].imshow(np.squeeze(CL)[32:-50, 15:-15], vmin=20, vmax=30)
plt.colorbar(im, ax=axs[0])
im = axs[1].imshow(np.squeeze(PUD)[32:-50, 15:-15], vmin=2, vmax=6)
plt.colorbar(im, ax=axs[1])
im = axs[2].imshow(np.squeeze(UD)[32:-50, 15:-15], vmin=6, vmax=12)
plt.colorbar(im, ax=axs[2])
plt.tight_layout()
tikzsave('CL_PUD_UD.tex')

plt.close('all')
msk = np.square(np.transpose(sitk.GetArrayFromImage(sitk.ReadImage('mask.nii.gz')), [1, 2, 0]))
msk = np.squeeze(msk[32:-50, 15:-15, 39])
cl = np.squeeze(CL)[32:-50, 15:-15]
pud = np.squeeze(PUD)[32:-50, 15:-15]
ud = np.squeeze(UD)[32:-50, 15:-15]
em = np.squeeze(echoMIP[32:-50, 15:-15, 39])
m = np.ma.MaskedArray(em, 1-msk)

plt.figure()
plt.axis('off')
plt.imshow(cl, vmin=20, vmax=30)
plt.imshow(m, cmap='gray')
plt.savefig('CL.png')

plt.figure()
plt.axis('off')
plt.imshow(pud, vmin=2, vmax=6)
plt.imshow(m, cmap='gray')
plt.savefig('PUD.png')

plt.figure()
plt.axis('off')
plt.imshow(ud, vmin=6, vmax=12)
plt.imshow(m, cmap='gray')
plt.savefig('UD.png')

subprocess.call(['convert', '-trim', 'CL.png', 'CL.png'])
subprocess.call(['convert', '-trim', 'PUD.png', 'PUD.png'])
subprocess.call(['convert', '-trim', 'UD.png', 'UD.png'])

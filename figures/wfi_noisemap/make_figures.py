import sys
sys.path.append('../../')
import numpy as np
import css
import h5io
import wfi
import matplotlib.pyplot as plt
from utils.utils import *
from matplotlib2tikz import save as tikzsave
import deepdish as dd


# filename = '/Users/mnd/Projects/FatParameterEstimation/data/FatParamEstimation/DiagnostikBilanz/20170609_125718_0402_ImDataParams.mat'
# imDataParams = h5io.load_ImDataParams_mat(filename)
# imDataParams = imDataParams['ImDataParams']
# dd.io.save('imDataParams.h5', imDataParams)
imDataParams = dd.io.load('imDataParams.h5')
echoMIP = wfi.calculate_echoMIP(imDataParams['signal'])

filename = '/Users/mnd/Projects/FatParameterEstimation/data/FatParamEstimation/DiagnostikBilanz/20170609_125718_0402_WFIparams_CSS_GANDALF2D_VL.mat'
wfiParams = h5io.load_WFIparams_mat(filename)
fieldmap_Hz = np.transpose(wfiParams['WFIparams']['fieldmap_Hz'])
fieldmap_Hz_equalized = wfi.equalize_fieldmap_periods(fieldmap_Hz,
                                                      imDataParams['TE_s'])


# single R2*
options = {'init_fieldmap_Hz': fieldmap_Hz_equalized,
           'iSlice': slice(35, 36),
           'uncertainty_quant': True}

wfiParams = wfi.wfi_css_varpro(imDataParams, options)
dd.io.save('wfiParams.h5', wfiParams)
# wfiParams = dd.io.load('wfiParams.h5')
# show_arr3d(np.clip(wfiParams['pdff_percent'], 0, 100))

# double R2*
Cr = np.eye(len(wfiParams['Cr']), 2)
Cr[1:, 1] = 1
options['Cr'] = Cr

wfiParams2 = wfi.wfi_css_varpro(imDataParams, options)
dd.io.save('wfiParams2.h5', wfiParams2)
# wfiParams2 = dd.io.load('wfiParams2.h5')

plt.close('all')
fig, axs = plt.subplots(4, 2)
axs[0][0].imshow(np.squeeze(np.clip(wfiParams['pdff_percent'], 0, 100))[38:, 45:],
                 cmap='inferno')
axs[1][0].imshow(np.squeeze(wfiParams['fieldmap_Hz'])[38:, 45:],
                 vmin=-200,vmax=300,cmap='magma')
axs[2][0].imshow(np.squeeze(np.clip(wfiParams['R2s_Hz'], 0, 400))[38:, 45:],
                 cmap='plasma')
im = axs[0][1].imshow(np.squeeze(np.clip(wfiParams2['pdff_percent'], 0, 100))[38:, 45:],
                      cmap='inferno')
plt.colorbar(im, ax=axs[0][1])
im = axs[1][1].imshow(np.squeeze(wfiParams2['fieldmap_Hz'])[38:, 45:],
                      vmin=-200,vmax=300,cmap='magma')
plt.colorbar(im, ax=axs[1][1])
im = axs[2][1].imshow(np.squeeze(np.clip(wfiParams2['waterR2s_Hz'], 0, 400))[38:, 45:],
                      cmap='plasma')
plt.colorbar(im, ax=axs[2][1])
im = axs[3][1].imshow(np.squeeze(np.clip(wfiParams2['fatR2s_Hz'], 0, 400))[38:, 45:],
                      cmap='plasma')
plt.colorbar(im, ax=axs[3][1])
axs[3][0].imshow(np.squeeze(echoMIP[38:, 45:, 35]), cmap='gray')
fig.tight_layout()
tikzsave('singlR2s.tex')


# ROI values
show_arr3d(echoMIP[38:, 45:, 35:36])
labelmap = np.squeeze(np.transpose(sitk.GetArrayFromImage(sitk.ReadImage('label2.nii.gz')),
                                   [1, 2, 0])).astype(bool)

print('mean')
print('PDFF =',
      np.squeeze(np.clip(wfiParams['pdff_percent'], 0, 100))[38:, 45:][labelmap].mean(),
      np.squeeze(np.clip(wfiParams2['pdff_percent'], 0, 100))[38:, 45:][labelmap].mean())

print('fieldmap = ',
      np.squeeze(wfiParams['fieldmap_Hz'])[38:, 45:][labelmap].mean(),
      np.squeeze(wfiParams2['fieldmap_Hz'])[38:, 45:][labelmap].mean())

print("R2* =",
      np.squeeze(np.clip(wfiParams['R2s_Hz'], 0, 400))[38:, 45:][labelmap].mean(),
      np.squeeze(np.clip(wfiParams2['waterR2s_Hz'], 0, 400))[38:, 45:][labelmap].mean(),
      np.squeeze(np.clip(wfiParams2['fatR2s_Hz'], 0, 400))[38:, 45:][labelmap].mean())

print('std')
print('PDFF =',
      np.squeeze(np.clip(wfiParams['pdff_percent'], 0, 100))[38:, 45:][labelmap].std(),
      np.squeeze(np.clip(wfiParams2['pdff_percent'], 0, 100))[38:, 45:][labelmap].std())

print('fieldmap = ',
      np.squeeze(wfiParams['fieldmap_Hz'])[38:, 45:][labelmap].std(),
      np.squeeze(wfiParams2['fieldmap_Hz'])[38:, 45:][labelmap].std())

print("R2* =",
      np.squeeze(np.clip(wfiParams['R2s_Hz'], 0, 400))[38:, 45:][labelmap].std(),
      np.squeeze(np.clip(wfiParams2['waterR2s_Hz'], 0, 400))[38:, 45:][labelmap].std(),
      np.squeeze(np.clip(wfiParams2['fatR2s_Hz'], 0, 400))[38:, 45:][labelmap].std())


import scipy.ndimage as ndi
from skimage.measure import regionprops

rp = regionprops(ndi.label(labelmap)[0])
print(rp[0].centroid, rp[0].equivalent_diameter/2)


FIMparams2 = css.compute_FIMmaps(imDataParams['TE_s'].ravel(),
                                 wfiParams['params_matrices'],
                                 wfiParams['Cm'],
                                 wfiParams['Cp'],
                                 wfiParams['Cf'],
                                 wfiParams2['Cr'],
                                 wfiParams['mask'])

plt.close('all')
fig, axs = plt.subplots(7, 2)
im = axs[0][0].imshow(np.squeeze(wfiParams['NSAs'][0]),vmax=6)
plt.colorbar(im, ax=axs[0][0])
im = axs[1][0].imshow(np.squeeze(wfiParams['NSAs'][1]),vmax=6)
plt.colorbar(im, ax=axs[1][0])
im = axs[2][0].imshow(np.squeeze(wfiParams['NSAs'][2]),vmax=6)
plt.colorbar(im, ax=axs[2][0])
im = axs[3][0].imshow(np.squeeze(wfiParams['NSAs'][3]),vmax=6)
plt.colorbar(im, ax=axs[3][0])
im = axs[4][0].imshow(np.squeeze(wfiParams['NSAs'][4]),vmax=2)
plt.colorbar(im, ax=axs[4][0])
im = axs[5][0].imshow(np.squeeze(wfiParams['NSAs'][5]),vmax=2)
plt.colorbar(im, ax=axs[5][0])
axs[6][0].axis('off')

im = axs[0][1].imshow(np.squeeze(FIMparams2['NSAs'][0]),vmax=2)
plt.colorbar(im, ax=axs[0][1])
im = axs[1][1].imshow(np.squeeze(FIMparams2['NSAs'][1]),vmax=2)
plt.colorbar(im, ax=axs[1][1])
im = axs[2][1].imshow(np.squeeze(FIMparams2['NSAs'][2]),vmax=6)
plt.colorbar(im, ax=axs[2][1])
im = axs[3][1].imshow(np.squeeze(FIMparams2['NSAs'][3]),vmax=6)
plt.colorbar(im, ax=axs[3][1])
im = axs[4][1].imshow(np.squeeze(FIMparams2['NSAs'][4]),vmax=2)
plt.colorbar(im, ax=axs[4][1])
im = axs[5][1].imshow(np.squeeze(FIMparams2['NSAs'][5]),vmax=2)
plt.colorbar(im, ax=axs[5][1])
im = axs[6][1].imshow(np.squeeze(FIMparams2['NSAs'][6]),vmax=2)
plt.colorbar(im, ax=axs[6][1])

# im = axs[0][2].imshow(np.squeeze(wfiParams['NSAs'][0]) - np.squeeze(FIMparams2['NSAs'][0]), cmap='coolwarm')
# plt.colorbar(im, ax=axs[0][2])
# im = axs[1][2].imshow(np.squeeze(wfiParams['NSAs'][1]) - np.squeeze(FIMparams2['NSAs'][1]), cmap='coolwarm')
# plt.colorbar(im, ax=axs[1][2])
# im = axs[2][2].imshow(np.squeeze(wfiParams['NSAs'][2]) - np.squeeze(FIMparams2['NSAs'][2]), cmap='coolwarm')
# plt.colorbar(im, ax=axs[2][2])
# im = axs[3][2].imshow(np.squeeze(wfiParams['NSAs'][3]) - np.squeeze(FIMparams2['NSAs'][3]), cmap='coolwarm')
# plt.colorbar(im, ax=axs[3][2])
# im = axs[4][2].imshow(np.squeeze(wfiParams['NSAs'][4]) - np.squeeze(FIMparams2['NSAs'][4]), cmap='coolwarm')
# plt.colorbar(im, ax=axs[4][2])
# im = axs[5][2].imshow(np.squeeze(wfiParams['NSAs'][5]) - np.squeeze(FIMparams2['NSAs'][5]), cmap='coolwarm')
# plt.colorbar(im, ax=axs[5][2])
# im = axs[6][2].imshow(np.squeeze(wfiParams['NSAs'][5]) - np.squeeze(FIMparams2['NSAs'][6]), cmap='coolwarm')
# plt.colorbar(im, ax=axs[6][2])
plt.tight_layout()

tikzsave('NSAmaps.tex')


plt.close('all')
fig, axs = plt.subplots(7, 2)
im = axs[0][0].imshow(np.squeeze(np.clip(wfiParams['CRLBs'][0], 0, 5)))
plt.colorbar(im, ax=axs[0][0])
im = axs[1][0].imshow(np.squeeze(np.clip(wfiParams['CRLBs'][1], 0, 5)))
plt.colorbar(im, ax=axs[1][0])
im = axs[2][0].imshow(np.squeeze(np.clip(wfiParams['CRLBs'][2], 0, 0.01)))
plt.colorbar(im, ax=axs[2][0])
im = axs[3][0].imshow(np.squeeze(np.clip(wfiParams['CRLBs'][3], 0, 0.01)))
plt.colorbar(im, ax=axs[3][0])
im = axs[4][0].imshow(np.squeeze(np.clip(wfiParams['CRLBs'][4], 0, 10)))
plt.colorbar(im, ax=axs[4][0])
im = axs[5][0].imshow(np.squeeze(np.clip(wfiParams['CRLBs'][5], 0, 200)))
plt.colorbar(im, ax=axs[5][0])
axs[6][0].axis('off')

im = axs[0][1].imshow(np.squeeze(np.clip(FIMparams2['CRLBs'][0], 0, 5)))
plt.colorbar(im, ax=axs[0][1])
im = axs[1][1].imshow(np.squeeze(np.clip(FIMparams2['CRLBs'][1], 0, 5)))
plt.colorbar(im, ax=axs[1][1])
im = axs[2][1].imshow(np.squeeze(np.clip(FIMparams2['CRLBs'][2], 0, 0.01)))
plt.colorbar(im, ax=axs[2][1])
im = axs[3][1].imshow(np.squeeze(np.clip(FIMparams2['CRLBs'][3], 0, 0.01)))
plt.colorbar(im, ax=axs[3][1])
im = axs[4][1].imshow(np.squeeze(np.clip(FIMparams2['CRLBs'][4], 0, 10)))
plt.colorbar(im, ax=axs[4][1])
im = axs[5][1].imshow(np.squeeze(np.clip(FIMparams2['CRLBs'][5], 0, 1000)))
plt.colorbar(im, ax=axs[5][1])
im = axs[6][1].imshow(np.squeeze(np.clip(FIMparams2['CRLBs'][6], 0, 3000)))
plt.colorbar(im, ax=axs[6][1])
plt.tight_layout()

tikzsave('CRLBmaps.tex')


plt.close('all')
fig, axs = plt.subplots(3, 2)
im = axs[0][0].imshow(np.log(np.squeeze(wfiParams['trInvFIM'])), vmin=0, vmax=10)
plt.colorbar(im, ax=axs[0][0])
im = axs[1][0].imshow(np.log(np.squeeze(wfiParams['trFIM'])), vmin=5, vmax=12)
plt.colorbar(im, ax=axs[1][0])
im = axs[2][0].imshow(np.log(np.squeeze(wfiParams['detFIM'])), vmin=-5, vmax=20)
plt.colorbar(im, ax=axs[2][0])

im = axs[0][1].imshow(np.log(np.squeeze(FIMparams2['trInvFIM'])), vmin=0, vmax=10)
plt.colorbar(im, ax=axs[0][1])
im = axs[1][1].imshow(np.log(np.squeeze(FIMparams2['trFIM'])), vmin=5, vmax=12)
plt.colorbar(im, ax=axs[1][1])
im = axs[2][1].imshow(np.log(np.squeeze(FIMparams2['detFIM'])), vmin=-5, vmax=20)
plt.colorbar(im, ax=axs[2][1])
plt.tight_layout()

tikzsave('FIMmaps.tex')


print(imDataParams['TE_s'].ravel())
print('TE1_s =', imDataParams['TE_s'].ravel()[0])
print('nTE =', len(imDataParams['TE_s'].ravel()))
print('dTE_s =', np.diff(imDataParams['TE_s'].ravel())[0])


# plt.close('all')
# fig, axs = plt.subplots(3, 2)
# im = axs[0][0].imshow(np.squeeze(wfiParams['detFIM']))
# plt.colorbar(im, ax=axs[0][0])
# im = axs[1][0].imshow(np.squeeze(wfiParams['trFIM']))
# plt.colorbar(im, ax=axs[1][0])
# im = axs[2][0].imshow(np.squeeze(wfiParams['trInvFIM']))
# plt.colorbar(im, ax=axs[2][0])

# im = axs[0][1].imshow(np.squeeze(FIMparams2['detFIM']))
# plt.colorbar(im, ax=axs[0][1])
# im = axs[1][1].imshow(np.squeeze(FIMparams2['trFIM']))
# plt.colorbar(im, ax=axs[1][1])
# im = axs[2][1].imshow(np.squeeze(FIMparams2['trInvFIM']))
# plt.colorbar(im, ax=axs[2][1])

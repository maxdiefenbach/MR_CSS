import sys
sys.path.append('../../')
import numpy as np
import residual
import css
import sim
import pandas as pd
from Fatmodel import Fatmodel
from CSmodel import CSmodel
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikzsave
from time import time


# single R2*
F = Fatmodel(modelname='Hamilton VAT')

F.set_constraints_matrices()

dTEstep_ms = 0.01
TEparams_in = [[6], # np.arange(6, 31),  # nTE, TE1_s, dTE_s, nAcq, acq_shift
               np.arange(0.35, 2.35, dTEstep_ms) * 1e-3, # np.array([1.1e-3]),
               np.arange(0.35, 2.35, dTEstep_ms) * 1e-3,
               np.array([1]),
               np.array([0.5])]
shape = np.array([len(x) for x in TEparams_in])
TEparams = np.concatenate(TEparams_in).ravel()

cols = ['nTE', 'TE1_s', 'dTE_s', 'pdff', 'r2s', 'CRLBs', 'NSAs', 'trF', 'detF', 'trFinv', 'detFinv']
df = pd.DataFrame(columns=cols)
t0 = time()
print('start 1')
for p in sim.get_params_combinations(TEparams, shape):
    nTE, TE1_s, dTE_s, nAcq, acq_shift = p
    pdff = F.fatfraction_percent
    r2s = F.R2s_Hz
    # print(nTE, TE1_s, dTE_s)

    F.set_TE_s(int(nTE), TE1_s, dTE_s, int(nAcq), [acq_shift])
    # F.build_signal()

    F.set_params_matrix()
    F.set_Fisher_matrix()

    FIM = F.Fisher_matrix
    FIMinv = np.linalg.inv(FIM)

    CRLBs = np.diagonal(FIMinv)
    NSAs = nTE / np.diag(FIM) / CRLBs
    trF = np.trace(FIM)
    detF = np.linalg.det(FIM)
    trFinv = np.trace(FIMinv)
    detFinv = np.linalg.det(FIMinv)

    df = df.append(
        pd.DataFrame(data=[[nTE, TE1_s, dTE_s,
                            pdff, r2s,
                            CRLBs, NSAs,
                            trF, detF, trFinv, detFinv]],
                     columns=cols))
print('done 1', time() - t0)

nParams = len(df['CRLBs'].values[0])

crbcols = ['crlb'+str(i) for i in range(nParams)]
df[crbcols] = pd.DataFrame(df['CRLBs'].values.tolist(), index=df.index)

nsacols = ['nsa'+str(i) for i in range(nParams)]
df[nsacols] = pd.DataFrame(df['NSAs'].values.tolist(), index=df.index)

df.to_csv('df_1R2s.csv', index=False)
print('wrote csv 1')


# double R2*
F = Fatmodel(modelname='Hamilton VAT')

F.set_constraints_matrices()

F.Cr = np.eye(F.Cm.shape[0], 2)
F.Cr[1:, 1] = 1
F.constraints_matrices[-1] = F.Cr

F.R2s_Hz = 35

cols = ['nTE', 'TE1_s', 'dTE_s', 'pdff', 'r2s', 'CRLBs', 'NSAs', 'trF', 'detF', 'trFinv', 'detFinv']
df2 = pd.DataFrame(columns=cols)
print('start 2')
for p in sim.get_params_combinations(TEparams, shape):
    nTE, TE1_s, dTE_s, nAcq, acq_shift = p
    pdff = F.fatfraction_percent
    r2s = F.R2s_Hz
    # print(nTE, TE1_s, dTE_s)

    F.set_TE_s(int(nTE), TE1_s, dTE_s, int(nAcq), [acq_shift])
    # F.build_signal()

    F.set_params_matrix()
    F.set_Fisher_matrix()

    FIM = F.Fisher_matrix
    FIMinv = np.linalg.inv(FIM)

    CRLBs = np.diagonal(FIMinv)
    NSAs = nTE / np.diag(FIM) / CRLBs
    trF = np.trace(FIM)
    detF = np.linalg.det(FIM)
    trFinv = np.trace(FIMinv)
    detFinv = np.linalg.det(FIMinv)

    df2 = df2.append(
        pd.DataFrame(data=[[nTE, TE1_s, dTE_s,
                            pdff, r2s,
                            CRLBs, NSAs,
                            trF, detF, trFinv, detFinv]],
                     columns=cols))
print('done 2', time() - t0)

nParams = len(df2['CRLBs'].values[0])

crbcols = ['crlb'+str(i) for i in range(nParams)]
df2[crbcols] = pd.DataFrame(df2['CRLBs'].values.tolist(), index=df2.index)

nsacols = ['nsa'+str(i) for i in range(nParams)]
df2[nsacols] = pd.DataFrame(df2['NSAs'].values.tolist(), index=df2.index)

df2.to_csv('df_2R2s.csv', index=False)
print('wrote csv 2')



# df = pd.read_csv('df_1R2s.csv')
# df2 = pd.read_csv('df_2R2s.csv')


plt.close('all')
clips = [10, 10, 0.005, 0.005, 5, 100, 1000]
fig, axs = plt.subplots(7, 2)
i = 0
for i in range(6):
    ax = axs[i][0]
    data = np.clip(df[f'crlb{i}'].values.reshape(nTE1, ndTE), 0, clips[i])
    im = ax.imshow(data)
    plt.colorbar(im, ax=ax)
    inds = np.argsort(data.ravel())[:10]
    miny, minx = np.unravel_index(inds, data.shape)
    ax.plot(minx, miny, 'r.')
    # maxx, maxy = np.unravel_index(np.argsort(data.ravel())[:3], data.shape)
    # ax.plot(maxx, maxy, 'r*')
    # ax.set_title(str(list(zip(TEparams_in[1][minx],
    #                           TEparams_in[2][miny],
    #                           data.ravel()[inds]))).replace('),', ')\n'))
    ax.invert_yaxis()
axs[6][0].axis('off')

for i in range(7):
    ax = axs[i][1]
    data = np.clip(df2[f'crlb{i}'].values.reshape(nTE1, ndTE), 0, clips[i])
    im = ax.imshow(data)
    plt.colorbar(im, ax=ax)
    inds = np.argsort(data.ravel())[:10]
    miny, minx = np.unravel_index(inds, data.shape)
    ax.plot(minx, miny, 'r.')
    ax.invert_yaxis()
plt.tight_layout()
tikzsave('crlb_dTE.tex')



plt.close('all')
clips = [2, 6, 2, 6, 2, 2]
fig, axs = plt.subplots(7, 2)
i = 0
for i in range(6):
    ax = axs[i][0]
    data = np.clip(df[f'nsa{i}'].values.reshape(nTE1, ndTE), 0, clips[i])
    im = ax.imshow(data,)
    plt.colorbar(im, ax=ax)
    inds = np.argsort(data.ravel())[-10:]
    maxy, maxx = np.unravel_index(inds, data.shape)
    ax.plot(maxx, maxy, 'g.')
    ax.invert_yaxis()
axs[6][0].axis('off')

clips = [2, 2, 2, 6, 2, 2, 2]
for i in range(7):
    ax = axs[i][1]
    data = np.clip(df2[f'nsa{i}'].values.reshape(nTE1, ndTE), 0, clips[i])
    im = ax.imshow(data)
    plt.colorbar(im, ax=ax)
    inds = np.argsort(data.ravel())[-10:]
    maxy, maxx = np.unravel_index(inds, data.shape)
    ax.plot(maxx, maxy, 'g.')
    ax.invert_yaxis()
plt.tight_layout()
tikzsave('nsa_dTE.tex')



plt.close('all')
fig, axs = plt.subplots(3, 2)

ax = axs[0][0]
data = np.log(df['trFinv'].values.reshape(nTE1, ndTE))
im = ax.imshow(data)
plt.colorbar(im, ax=ax)
inds = np.argsort(data.ravel())[:10]
miny, minx = np.unravel_index(inds, data.shape)
ax.plot(minx, miny, 'r.')
ax.invert_yaxis()

ax = axs[1][0]
data = np.log(df['trF'].values.reshape(nTE1, ndTE))
im = ax.imshow(data)
plt.colorbar(im, ax=ax)
inds = np.argsort(data.ravel())[-10:]
maxy, maxx = np.unravel_index(inds, data.shape)
ax.plot(maxx, maxy, 'g.')
ax.invert_yaxis()

ax = axs[2][0]
data = np.log(df['detF'].values.reshape(nTE1, ndTE))
im = ax.imshow(data)
plt.colorbar(im, ax=ax)
inds = np.argsort(data.ravel())[-10:]
maxy, maxx = np.unravel_index(inds, data.shape)
ax.plot(maxx, maxy, 'g.')
ax.invert_yaxis()

ax = axs[0][1]
data = np.log(df2['trFinv'].values.reshape(nTE1, ndTE))
im = ax.imshow(data)
plt.colorbar(im, ax=ax)
inds = np.argsort(data.ravel())[:10]
miny, minx = np.unravel_index(inds, data.shape)
ax.plot(minx, miny, 'r.')
ax.invert_yaxis()

ax = axs[1][1]
data = np.log(df2['trF'].values.reshape(nTE1, ndTE))
im = ax.imshow(data)
plt.colorbar(im, ax=ax)
inds = np.argsort(data.ravel())[-10:]
maxy, maxx = np.unravel_index(inds, data.shape)
ax.plot(maxx, maxy, 'g.')
ax.invert_yaxis()

ax = axs[2][1]
data = np.log(df2['detF'].values.reshape(nTE1, ndTE))
im = ax.imshow(data)
plt.colorbar(im, ax=ax)
inds = np.argsort(data.ravel())[-10:]
maxy, maxx = np.unravel_index(inds, data.shape)
ax.plot(maxx, maxy, 'g.')
ax.invert_yaxis()
plt.tight_layout()

tikzsave('FIM_dTE.tex')





# im = axs[0][0].imshow(np.clip(df['crlb0'].values.reshape(nTE1, ndTE), 0, 10))
# plt.colorbar(im, ax=axs[0][0])
# im = axs[1][0].imshow(np.clip(df['crlb1'].values.reshape(nTE1, ndTE), 0, 10))
# plt.colorbar(im, ax=axs[1][0])
# im = axs[2][0].imshow(np.clip(df['crlb2'].values.reshape(nTE1, ndTE), 0, 0.005))
# plt.colorbar(im, ax=axs[2][0])
# im = axs[3][0].imshow(np.clip(df['crlb3'].values.reshape(nTE1, ndTE), 0, 0.005))
# plt.colorbar(im, ax=axs[3][0])
# im = axs[4][0].imshow(np.clip(df['crlb4'].values.reshape(nTE1, ndTE), 0, 5))
# plt.colorbar(im, ax=axs[4][0])
# im = axs[5][0].imshow(np.clip(df['crlb5'].values.reshape(nTE1, ndTE), 0, 100))
# plt.colorbar(im, ax=axs[5][0])
# axs[6][0].axis('off')

# im = axs[0][1].imshow(np.clip(df2['crlb0'].values.reshape(nTE1, ndTE), 0, 10))
# plt.colorbar(im, ax=axs[0][1])
# im = axs[1][1].imshow(np.clip(df2['crlb1'].values.reshape(nTE1, ndTE), 0, 10))
# plt.colorbar(im, ax=axs[1][1])
# im = axs[2][1].imshow(np.clip(df2['crlb2'].values.reshape(nTE1, ndTE), 0, 0.005))
# plt.colorbar(im, ax=axs[2][1])
# im = axs[3][1].imshow(np.clip(df2['crlb3'].values.reshape(nTE1, ndTE), 0, 0.005))
# plt.colorbar(im, ax=axs[3][1])
# im = axs[4][1].imshow(np.clip(df2['crlb4'].values.reshape(nTE1, ndTE), 0, 5))
# plt.colorbar(im, ax=axs[4][1])
# im = axs[5][1].imshow(np.clip(df2['crlb5'].values.reshape(nTE1, ndTE), 0, 100))
# plt.colorbar(im, ax=axs[5][1])
# im = axs[6][1].imshow(np.clip(df2['crlb6'].values.reshape(nTE1, ndTE), 0, 1000))
# plt.colorbar(im, ax=axs[6][1])
# plt.tight_layout()


# plt.close('all')
# fig, axs = plt.subplots(7, 2)
# im = axs[0][0].imshow(df['nsa0'].values.reshape(nTE1, ndTE), vmax=nTE//3)
# plt.colorbar(im, ax=axs[0][0])
# im = axs[1][0].imshow(df['nsa1'].values.reshape(nTE1, ndTE), vmax=nTE)
# plt.colorbar(im, ax=axs[1][0])
# im = axs[2][0].imshow(df['nsa2'].values.reshape(nTE1, ndTE), vmax=nTE//3)
# plt.colorbar(im, ax=axs[2][0])
# im = axs[3][0].imshow(df['nsa3'].values.reshape(nTE1, ndTE), vmax=nTE)
# plt.colorbar(im, ax=axs[3][0])
# im = axs[4][0].imshow(df['nsa4'].values.reshape(nTE1, ndTE), vmax=nTE//3)
# plt.colorbar(im, ax=axs[4][0])
# im = axs[5][0].imshow(df['nsa5'].values.reshape(nTE1, ndTE), vmax=nTE//3)
# plt.colorbar(im, ax=axs[5][0])
# axs[6][0].axis('off')

# im = axs[0][1].imshow(df2['nsa0'].values.reshape(nTE1, ndTE), vmax=nTE//3)
# plt.colorbar(im, ax=axs[0][1])
# im = axs[1][1].imshow(df2['nsa1'].values.reshape(nTE1, ndTE), vmax=nTE//3)
# plt.colorbar(im, ax=axs[1][1])
# im = axs[2][1].imshow(df2['nsa2'].values.reshape(nTE1, ndTE), vmax=nTE//3)
# plt.colorbar(im, ax=axs[2][1])
# im = axs[3][1].imshow(df2['nsa3'].values.reshape(nTE1, ndTE), vmax=nTE)
# plt.colorbar(im, ax=axs[3][1])
# im = axs[4][1].imshow(df2['nsa4'].values.reshape(nTE1, ndTE), vmax=nTE//3)
# plt.colorbar(im, ax=axs[4][1])
# im = axs[5][1].imshow(df2['nsa5'].values.reshape(nTE1, ndTE), vmax=nTE//3)
# plt.colorbar(im, ax=axs[5][1])
# im = axs[6][1].imshow(df2['nsa6'].values.reshape(nTE1, ndTE), vmax=nTE//3)
# plt.colorbar(im, ax=axs[6][1])
# plt.tight_layout()

import sys
sys.path.append('../../../../code')
import numpy as np
import pandas as pd
import css
import sim
import pytest
from Fatmodel import Fatmodel
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikzsave


F = Fatmodel()

r2s = F.R2s_Hz
TE_s = F.TE_s.copy()
nTE = len(TE_s)

F.set_constraints_matrices()
Cm, Cp, Cf, Cr = F.constraints_matrices
# Cr = np.zeros_like(Cm)
# F.Cr = Cr
# F.constraints_matrices[-1] = Cr

N = 40
Ninr = int(1e3)
snr = 100
tol = 1e-4
itermax = 50

cols = ['pdff', 'r2s',
        'CRLBs', 'NSAs', 'mcNSAs',
        'trF', 'detF', 'trFinv', 'detFinv',
        'snr', 'mean', 'var']
df = pd.DataFrame(columns=cols)
for i, pdff in enumerate(np.linspace(1, 99, N)):
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

    df = df.append(pd.DataFrame(data=[[pdff, r2s,
                                       CRLBs, NSAs, mcNSAs,
                                       trF, detF, trFinv, detFinv,
                                       snr, mean, var]],
                                columns=cols))


plt.close('all')
fig, axs = plt.subplots(3, 2)
for i, ax in enumerate(axs.ravel()):
    # ax.plot(df['pdff'], df['CRLBs'].str[i])
    # ax.plot(df['pdff'], df['var'].str[i]*snr**2, 'o')
    ax.plot(df['pdff'], df['NSAs'].str[i], '-')
    ax.plot(df['pdff'], df['mcNSAs'].str[i], 'o', mfc='none')
    ax.set_ylim([0, nTE])
axs[0][0].legend()
axs[2][0].set_ylim([0, 2])
axs[2][1].set_ylim([0, 2])
fig.tight_layout()
# tikzsave('wfi1R2s.tex')

plt.figure()
plt.plot(df['pdff'], df['CRLBs'].apply(np.sum), 'o', mfc='none')
plt.plot(df['pdff'], df['trFinv'], '-')

plt.figure()
plt.plot(df['pdff'], df['detF'])

plt.figure()
plt.plot(df['pdff'], df['trF'])

# plt.close('all')
# plt.figure()
# plt.plot(df['pdff'], df['trF'])
# plt.plot(df['pdff'], df['var'].apply(lambda x: np.mean(1/x)))
# plt.plot(df['pdff'], df['var'].apply(lambda x: np.sum(1/x)))
# plt.plot(df['pdff'], df['CRLBs'].apply(lambda x: np.sum(1/x)))
# plt.plot(df['pdff'], df['CRLBs'].apply(lambda x: 1/np.sum(x)))

# plt.plot(df['pdff'], df['detFinv'])

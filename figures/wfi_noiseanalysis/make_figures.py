import sys
sys.path.append('../../')
import numpy as np
import pandas as pd
from Fatmodel import Fatmodel
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikzsave

# single R2*
F = Fatmodel(modelname='Hamilton VAT',
             nTE=6,
             TE1_s=1.e-3,
             dTE_s=1.2e-3)
F.set_constraints_matrices()
nTE = len(F.TE_s)

cols = ['pdff', 'r2s', 'CRLBs', 'NSAs', 'trF', 'detF', 'trFinv', 'detFinv']
df = pd.DataFrame(columns=cols)
N = 50
r2s_range = [20, 25, 35, 50]
for pdff in np.linspace(1, 99, N):
    F.fatfraction_percent = pdff

    for r2s in r2s_range:

        F.R2s_Hz = r2s

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
            pd.DataFrame(data=[[pdff, r2s,
                                CRLBs, NSAs,
                                trF, detF, trFinv, detFinv]],
                         columns=cols))


nParams = len(df['CRLBs'].iloc[0])

crbcols = ['crlb'+str(i) for i in range(nParams)]
df[crbcols] = pd.DataFrame(df['CRLBs'].values.tolist(), index=df.index)

nsacols = ['nsa'+str(i) for i in range(nParams)]
df[nsacols] = pd.DataFrame(df['NSAs'].values.tolist(), index=df.index)


df = df[['pdff', 'r2s', 'trF', 'detF', 'trFinv', 'detFinv']+crbcols+nsacols]

for r2s in r2s_range:
    df[df['r2s']==r2s].to_csv(f'wfi1r2s{r2s}.csv', index=False)


# double R2*
F = Fatmodel(modelname='Hamilton VAT',
             nTE=6,
             TE1_s=1.e-3,
             dTE_s=1.2e-3)
F.set_constraints_matrices()
F.Cr = np.eye(F.Cm.shape[0], 2)
F.Cr[1:, 1] = 1
F.constraints_matrices[-1] = F.Cr
nTE = len(F.TE_s)

cols = ['pdff', 'r2s', 'CRLBs', 'NSAs', 'trF', 'detF', 'trFinv', 'detFinv']
df = pd.DataFrame(columns=cols)
N = 50
r2s_range = [20, 25, 35, 50]
for pdff in np.linspace(1, 99, N):
    F.fatfraction_percent = pdff

    for r2s in r2s_range:

        F.R2s_Hz = r2s

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
            pd.DataFrame(data=[[pdff, r2s,
                                CRLBs, NSAs,
                                trF, detF, trFinv, detFinv]],
                         columns=cols))


nParams = len(df['CRLBs'].iloc[0])

crbcols = ['crlb'+str(i) for i in range(nParams)]
df[crbcols] = pd.DataFrame(df['CRLBs'].values.tolist(), index=df.index)

nsacols = ['nsa'+str(i) for i in range(nParams)]
df[nsacols] = pd.DataFrame(df['NSAs'].values.tolist(), index=df.index)


df = df[['pdff', 'r2s', 'trF', 'detF', 'trFinv', 'detFinv']+crbcols+nsacols]

for r2s in r2s_range:
    df[df['r2s']==r2s].to_csv(f'wfi2r2s{r2s}.csv', index=False)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from pywfi.css import css
from CSmodel import CSmodel


# plt.style.use('ggplot')
plt.style.use('default')

DEFAULT_FATMODELS_CSV = os.path.join(os.path.dirname(__file__), 'fatmodel.csv')


class Fatmodel(CSmodel):
    '''
    convenience class to handle signal simulations in water-fat mixtures
    '''

    def __init__(self, **kwargs):
        CSmodel.__init__(self, **kwargs)

        self.set_fatmodels_df()

        self.waterPeakLocation_ppm = kwargs.get('waterPeakLocation_ppm', 4.7)

        self.modelname = kwargs.get('modelname', 'Hamilton liver')

        self.fatfraction_percent = kwargs.get('fatfraction_percent', 30.0)
        self.fieldmap_Hz = kwargs.get('fieldmap_Hz', 10.0)
        self.R2s_Hz = kwargs.get('R2s_Hz', 5.0)
        self.phase_rad = kwargs.get('phase_rad', np.pi/4)

        self.cl = kwargs.get('cl', 18)
        self.ndb = kwargs.get('ndb', 2.8)
        self.nmidb = kwargs.get('nmidb', 0.8)

        self.set_fatmodel()
        # needs relamps first
        # self.set_params_matrix()
        # self.set_constraints_matrices()

    def set_fatmodels_df(self, filename=DEFAULT_FATMODELS_CSV):
        df = pd.read_csv(filename)
        df = df[['model name', 'parameter name', 'unit',
                 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']].dropna()
        df['lists'] = df.iloc[:, 3:].values.tolist()
        df = df[['model name', 'parameter name', 'unit', 'lists']]
        df = df.pivot(index='model name', columns='parameter name',
                      values='lists').reset_index()
        df = df[['model name', 'chemical shift', 'relative amplitude']]
        df.columns = ['model', 'shielding_ppm', 'relamps_percent']
        self.fatmodels_df = df

    def set_fatmodel(self, name=None):
        if name is None:
            name = self.modelname
        self.set_deshielding_ppm(name)
        self.set_relative_peak_amplitudes(name)
        if self.relamps_percent is not None:
            self._remove_zero_peaks()
            self.normalize_relamps()
        self.modelname = name

    def set_deshielding_ppm(self, name):
        if name not in list(self.fatmodels_df['model']):
            print('Error: modelname not in fatmodels_df.')
            return
        df = self.fatmodels_df[self.fatmodels_df['model'] == name]
        shielding_ppm = np.array([float(f)
                                  for f in df['shielding_ppm'].as_matrix()[0]])
        self.deshielding_ppm = shielding_ppm - self.waterPeakLocation_ppm

    def set_relative_peak_amplitudes(self, name):
        df = self.fatmodels_df[self.fatmodels_df['model'] == name]
        relamps_percent = np.squeeze(df['relamps_percent'])
        if relamps_percent is not None:
            relamps_percent = \
                np.array([float(f) for f in np.squeeze(relamps_percent)])
            self.relamps_percent = relamps_percent
        else:
            print(f'Warning: relamps_percent for fat model "{name}" is None.')
            self.relamps_percent = None

    def _remove_zero_peaks(self):
        assert self.relamps_percent is not None
        peakIsPresent = self.relamps_percent != 0
        self.deshielding_ppm = self.deshielding_ppm[peakIsPresent]
        self.relamps_percent = self.relamps_percent[peakIsPresent]

    def set_params_matrix(self):
        self.pm = build_paramsmatrix(self.fatfraction_percent,
                                     self.deshielding_ppm,
                                     self.relamps_percent,
                                     fieldmap_Hz=self.fieldmap_Hz,
                                     R2s_Hz=self.R2s_Hz,
                                     sigamp=self.sigamp,
                                     phases=self.phase_rad,
                                     centerfreq_Hz=self.centerfreq_Hz)


    def set_constraints_matrices(self):
        Cm, Cp, Cf, Cr = build_constraintsmatrices(self.deshielding_ppm,
                                                   self.relamps_percent,
                                                   self.fieldmap_Hz,
                                                   self.centerfreq_Hz)
        self.Cm = Cm
        self.Cp = Cp
        self.Cf = Cf
        self.Cr = Cr
        self.constraints_matrices = [Cm, Cp, Cf, Cr];


    def compute_fatmodel(self, cl=None, ndb=None, nmidb=None, modelname=None):
        # fat model propeties
        if cl is None:
            cl = self.cl
        if ndb is None:
            ndb = self.ndb
        if nmidb is None:
            nmidb = self.nmidb

        self.cl = cl
        self.ndb = ndb
        self.nmidb = nmidb

        # relamps_percent
        if modelname is None:
            modelname = self.modelname
        self.modelname = modelname

        if modelname == 'Berglund 10 peaks':
            ABC2J = [9, 6 * (cl - 4) - 8 * ndb + 2 * nmidb, 6,
                     4 * (ndb - nmidb), 6, 2 * nmidb, 2, 2, 1, 2 * ndb]

        elif modelname == 'Hamilton 9 peaks':
            ABC2J = [9, 6 * (cl - 4) - 8 * ndb + 2 * nmidb,
                     6, 4 * (ndb - nmidb), 6, 2 * nmidb, 4, 0, 1, 2 * ndb]

        elif modelname == 'Peterson 8 peaks':
            ABC2J = [9, 6 * (cl - 4) - 8 * ndb + 2 * nmidb, 6,
                     4 * (ndb - nmidb), 6, 2 * nmidb, 4, 0, 2 * ndb + 1, 0]

        elif modelname == 'Hamilton 6 peaks':
            ABC2J = [9, 6 * (cl - 4) - 8 * ndb + 2 * nmidb + 6, 0,
                     4 * (ndb - nmidb) + 6, 0, 2 * nmidb, 4, 0, 0, 2 * ndb + 1]

        else:
            print('Fat model {} not implemented.'.format(modelname))
            return

        ABC2J = np.array(ABC2J)
        self.relamps_percent = ABC2J[ABC2J != 0]
        self.normalize_relamps()

    def normalize_relamps(self):
        self.relamps_percent = 100 * (self.relamps_percent /
                                      np.sum(self.relamps_percent))

    def get_chemical_shifts_Hz(self):
        return np.concatenate(
            ([0], self.centerfreq_Hz * 1e-6 * self.deshielding_ppm))

    def plot_spectrum(self):
        plt.figure()
        markerline, stemlines, baseline = plt.stem(self.deshielding_ppm,
                                                   self.relamps_percent)
        plt.setp(baseline, color='k', linewidth=0.1)

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        while len(colors) < 10:
            colors = np.concatenate((colors, prop_cycle.by_key()['color']))

        for i, l in enumerate(stemlines):
            plt.setp(l, linewidth=1, c=colors[i])

        plt.legend(stemlines, np.round(self.deshielding_ppm, 2))
        plt.ylabel('relative amplitude [%]')
        plt.xlabel('deshielding [ppm]')

    def plot_vectors(self):
        theta = 2 * np.pi * self.deshielding_ppm * 1e-6 * \
                (self.centerfreq_Hz + self.fieldmap_Hz) * \
                self.TE_s[:, None]
        radius = self.sigamp * self.relamps_percent/100 * \
                 np.exp(-self.R2s_Hz * self.TE_s[:, None])

        fig, axs = plt.subplots(self.nTE // (self.nTE//2), self.nTE//2,
                                subplot_kw=dict(projection='polar'))
        fig.subplots_adjust(hspace=0.7, wspace=0.7)

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        rmax = np.max(radius)
        rmax = rmax if rmax > self.sigamp * (1 -
                                             self.fatfraction_percent/100) \
               else self.sigamp * (1 - self.fatfraction_percent/100)
        # rmax = 1, might want to change to see all fat spin vectors
        for iTE, ax in enumerate(np.ravel(axs)):
            ax.set_rmax(rmax)
            # water
            wm = self.sigamp * (1 - self.fatfraction_percent/100) * \
                 np.exp(-self.R2s_Hz * self.TE_s[iTE])
            wp = 2 * np.pi * self.fieldmap_Hz * self.TE_s[iTE]
            ax.arrow(0, 0, wp, wm,
                     alpha=0.5, width=0.015, lw=2,
                     length_includes_head=True, color='black')
            # fat
            for i in range(len(self.relamps_percent)):
                ax.arrow(0, 0, theta[iTE, i], radius[iTE, i],
                         alpha=0.5, width=0.015, lw=2,
                         length_includes_head=True, color=colors[i])


def build_paramsmatrix(fatfraction_percent, deshielding_ppm,
                       relamps_percent, fieldmap_Hz=0,
                       R2s_Hz=0, sigamp=100, phases=0, centerfreq_Hz=128e6):
    pm = np.ones((len(deshielding_ppm)+1, 4))
    pm[:, 0] = sigamp * \
               np.concatenate([[(1 - fatfraction_percent/100)],
                               np.array(fatfraction_percent)/100 * \
                               np.array(relamps_percent)/100])
    pm[:, 1] = phases
    pm[:, 2] = np.concatenate([[fieldmap_Hz],
                               fieldmap_Hz + centerfreq_Hz * \
                               np.array(deshielding_ppm) * 1e-6])
    pm[:, 3] = R2s_Hz
    return pm


def build_constraintsmatrices(deshielding_ppm, relamps_percent,
                              fieldmap_Hz=0, centerfreq_Hz=128e6):
    N = len(deshielding_ppm) + 1
    Cm = np.eye(N, 2)
    Cm[1:, 1] = np.array(relamps_percent)/100
    Cp = np.eye(N, 2)
    Cp[1:, 1] = 1
    Cf = np.eye(N, 1)
    Cf[1:, 0] = 1
    Cr = np.eye(N, 1)
    Cr[1:, 0] = 1
    return Cm, Cp, Cf, Cr

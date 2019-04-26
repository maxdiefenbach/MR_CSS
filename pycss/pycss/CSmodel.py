import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import css
import residual
import sim

# plt.style.use('ggplot')
plt.style.use('default')


class CSmodel(object):
    '''
    convenience class to handle MR signal simulations in mixtures
    of different chemical species

    '''

    def __init__(self, **kwargs):
        self.fieldstrength_T = kwargs.get('fieldstrength_T', 3)
        self.set_centerfreq_Hz(kwargs.get('centerfreq_Hz', None))
        self.sigamp = kwargs.get('sigamp', 100)

        self.pm = kwargs.get('pm', None)
        self.pm_init = kwargs.get('pm_init', None)

        self.Cm = kwargs.get('Cm', None)
        self.Cp = kwargs.get('Cp', None)
        self.Cf = kwargs.get('Cf', None)
        self.Cr = kwargs.get('Cr', None)
        self.constraints_matrices = self.Cm, self.Cp, self.Cf, self.Cr

        self.nTE = kwargs.get('nTE', 6)
        self.dTE_s = kwargs.get('dTE_s', 1.2e-3)
        self.TE1_s = kwargs.get('TE1_s', 1e-3)
        self.nAcq = kwargs.get('nAcq', 1)
        self.acq_shift = kwargs.get('acq_shift', [0.5])
        self.set_TE_s(self.nTE, self.TE1_s, self.dTE_s,
                      self.nAcq, self.acq_shift)

        self.SNR = kwargs.get('SNR', None)
        self.update_all()

    def normalize_amplitudes(self):
        self.pm[:, 0] /= sum(self.pm[:, 0])

    def set_centerfreq_Hz(self, centerfreq_Hz=None):
        if centerfreq_Hz is None:
            self.centerfreq_Hz = 42.58e6 * self.fieldstrength_T
        else:
            self.centerfreq_Hz = centerfreq_Hz

    def set_TE_s(self, nTE=None, TE1_s=None, dTE_s=None,
                 nAcq=None, acq_shift=None):
        if nTE is None:
            nTE = self.nTE
        if TE1_s is None:
            TE1_s = self.TE1_s
        if dTE_s is None:
            dTE_s = self.dTE_s
        if nAcq is None:
            nAcq = self.nAcq
        if acq_shift is None:
            acq_shift = self.acq_shift

        self.nTE = nTE
        self.TE1_s = TE1_s
        self.dTE_s = dTE_s
        self.nAcq = nAcq
        self.acq_shift = acq_shift

        self.TE_s = sim.get_TE_s(self.nTE, self.TE1_s, self.dTE_s,
                                 self.nAcq, self.acq_shift)
        self.dTE_s_eff = np.diff(self.TE_s)
        self.t_s = np.linspace(0, self.TE_s.max()+0.5e-3, 1000)

    def build_signal(self):
        self.signal = css.build_signal(self.t_s, self.pm)
        self.sample_signal(self.SNR)

    def sample_signal(self, SNR=None):
        if SNR is None:
            self.signal_samp = css.build_signal(self.TE_s, self.pm)
        else:
            self.signal_samp = \
                css.add_noise(css.build_signal(self.TE_s, self.pm), SNR)

    def set_residual(self, nlparams, shape):
        Cm, Cp, Cf, Cr = self.constraints_matrices
        self.residual = residual.get_residual_array(self.TE_s, self.signal_samp,
                                                    self.pm_init,
                                                    Cm, Cp, Cf, Cr,
                                                    nlparams, shape)
        self.nlparams = nlparams
        self.shape = shape

    def set_params_at_minima(self):
        Cm, Cp, Cf, Cr = self.constraints_matrices
        self.params_at_minima = \
            residual.get_residual_minima(self.TE_s, self.sig, self.pm_init,
                                         Cm, Cp, Cf, Cr,
                                         nlparams, shape)

        isMinimum = css.find_residual_minima(self.residual)
        self.residual_minima = {'fieldmap_Hz': self.fieldmaprange_Hz[isMinimum],
                                'residual': self.residual[isMinimum]}

    def set_Fisher_matrix(self):
        Cm, Cp, Cf, Cr = self.constraints_matrices
        self.Fisher_matrix = sim.get_Fisher_matrix(self.TE_s, self.pm,
                                                   Cm, Cp, Cf, Cr)

    def update_all(self):
        pass

    def set_df(self):
        pass

    def plot_signal(self):
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(self.t_s, np.abs(self.signal))
        axs[0].plot(self.TE_s, np.abs(self.signal_samp), 'o')
        # axs[0].set_xlabel('time [s]')
        axs[0].set_ylabel('magnitude [a.u.]')
        axs[1].plot(self.t_s, np.angle(self.signal))
        axs[1].plot(self.TE_s, np.angle(self.signal_samp), 'o')
        axs[1].set_xlabel('time [s]')
        axs[1].set_ylabel('phase [rad]')

    def plot_residual(self):
        isMinimum = css.find_residual_minima(self.residual)
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.fieldmaprange_Hz, self.residual)
        ax.plot(self.fieldmaprange_Hz[isMinimum],
                self.residual[isMinimum],
                'r.')
        ax.set_xlabel('field map [Hz]')
        ax.set_ylabel('residual [a.u.]')

if __name__ == '__main__':
    m = CSmodel()

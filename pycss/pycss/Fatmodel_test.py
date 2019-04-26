import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from CSmodel import CSmodel
from Fatmodel import Fatmodel


@pytest.fixture
def testobj():
    F = Fatmodel()
    return F


def test_constructor(testobj):
    F = testobj
    assert isinstance(F, Fatmodel)
    assert set(['waterPeakLocation_ppm',
                'fatfraction_percent', 'fieldmap_Hz', 'R2s_Hz', 'phase_rad',
                'cl', 'ndb', 'nmidb',
                'modelname']).issubset(set(F.__dict__.keys()))


def test_constructor2():
    F = Fatmodel({'modelname': 'Hamilton 9 peaks'})


def test_set_fatmodels_df(testobj):
    testobj.set_fatmodels_df()
    assert isinstance(testobj.fatmodels_df, pd.DataFrame)


def test_set_deshielding_ppm(testobj):
    testobj.set_deshielding_ppm('Hamilton liver')
    assert isinstance(testobj.deshielding_ppm, np.ndarray)


def test_set_relative_peak_amplitudes(testobj):
    name = 'Hamilton liver'
    testobj.set_relative_peak_amplitudes(name)
    assert isinstance(testobj.relamps_percent, np.ndarray)

    name = 'Hamilton 9 peaks'
    testobj.set_relative_peak_amplitudes(name)
    assert isinstance(testobj.relamps_percent, np.ndarray)


def test_set_fatmodel(testobj):
    testobj.set_fatmodel('Hamilton liver')
    assert len(testobj.deshielding_ppm) == len(testobj.relamps_percent)

    testobj.set_fatmodel('Hamilton 9 peaks')
    print(testobj.deshielding_ppm)
    print(testobj.relamps_percent)
    assert testobj.relamps_percent is None


def test_build_paramsmatrix(testobj):
    pass


def test_set_params_matrix(testobj):
    testobj.set_params_matrix()
    assert isinstance(testobj.pm, np.ndarray)


def test_build_constraintsmatrices(testobj):
    pass


def test_set_constraints_matrices(testobj):
    testobj.set_constraints_matrices()
    assert isinstance(testobj.constraints_matrices, list)
    assert isinstance(testobj.constraints_matrices[0], np.ndarray)


def test_compute_fatmodel(testobj):
    testobj.compute_fatmodel(18.0, 2.73, 0.7)
    testobj.compute_fatmodel(18.0, 2.73, 0.7, 'Berglund 10 peaks')
    assert np.sum(testobj.relamps_percent) == 100


if __name__ == '__main__':
    F = Fatmodel()
    print(vars(F).keys())

    # print(F.fatmodels_df)
    # F.set_relative_peak_amplitudes('Hamilton 9 peaks')

    name = 'Hamilton 9 peaks'
    df = F.fatmodels_df[F.fatmodels_df['model'] == name]
    np.array([float(f) for f in np.array(df['relamps_percent'])])
    print(df)
    # plt.close('all')

    # F.build_signal()
    # F.plot_signal()

    # F.plot_spectrum()
    # F.plot_vectors()

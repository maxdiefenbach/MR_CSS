import numpy as np
import pylab as plt
from matplotlib2tikz import save as tikzsave


def get_TE_s(nTE, TE1_s, dTE_s):
    TE_s = []
    for t in dTE_s:
        # print(t)
        TE_s.append(TE1_s + np.arange(nTE) * t)
    return np.array(TE_s)


def compute_Z0(R2s_Hz, TE_s):
    res = np.zeros((len(TE_s), len(R2s_Hz)))
    for i, t in enumerate(TE_s):
        for j, r in enumerate(R2s_Hz):
            res[i, j] = np.sum(np.exp(-r * t))
    return res


def compute_Z1(R2s_Hz, TE_s):
    res = np.zeros((len(TE_s), len(R2s_Hz)))
    for i, t in enumerate(TE_s):
        for j, r in enumerate(R2s_Hz):
            res[i, j] = np.sum(t * np.exp(-r * t))
    return res


def compute_Z2(R2s_Hz, TE_s):
    res = np.zeros((len(TE_s), len(R2s_Hz)))
    for i, t in enumerate(TE_s):
        for j, r in enumerate(R2s_Hz):
            res[i, j] = np.sum(t**2 * np.exp(-r * t))
    return res


if __name__ == '__main__':

    nTE = 6
    TE1_s = 1e-3
    # dTE_s = 1.2e-3
    # TE_s = get_TE_s(nTE, TE1_s, dTE_s)

    dTE_s = np.arange(0.2e-3, 60e-3, 0.2e-3)
    TE_s = get_TE_s(nTE, TE1_s, dTE_s)
    print(dTE_s, '\n', TE_s)

    R2s_Hz = [20, 25, 35, 50]
    Z0 = compute_Z0(R2s_Hz, TE_s)
    Z1 = compute_Z1(R2s_Hz, TE_s)
    Z2 = compute_Z2(R2s_Hz, TE_s)

    plt.close('all')
    plt.figure()
    plt.plot(dTE_s, Z0)
    plt.legend(R2s_Hz)

    plt.figure()
    plt.plot(dTE_s, Z1)
    plt.legend(R2s_Hz)

    plt.figure()
    plt.plot(dTE_s, Z2)
    plt.legend(R2s_Hz)

    sigma = 1
    rho = 100
    tr_F = sigma**2 * (Z0 + rho**2 * (Z1 + 2 * Z2))
    det_F = sigma**2 * rho**4 * Z1 * (Z0 * Z2 - Z1**2) * (rho**2 * Z2 - Z1)
    tr_invF = sigma**2 * ((rho**2 * Z2 + Z0) / (rho**2 * (Z0 * Z2 - Z1**2)) + \
                          (Z1 + Z2) / (Z1 * (rho**2 * Z2 - Z1)))
    F00 = Z2 / (Z0 * Z2 - Z1**2)
    F11 = Z2 / (Z1 * (Z1 * (rho**2 * Z2 - Z1)))
    F22 = 1 / (rho**2 * Z2 - Z1)
    F33 = Z0 / (rho**2 * (Z0 * Z2 - Z1**2))

    plt.close('all')

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    ax0.plot(dTE_s, F00)
    ax0.set_ylim(0.6, 1.2)
    ax0.set_title('$F^{-1}_{0 0}$')
    ax0.legend(['$R_2^*='+str(r)+'$' for r in R2s_Hz])

    ax1.plot(dTE_s, F11)
    ax1.set_ylim(0., 0.3)
    ax1.set_title('$F^{-1}_{1 1}$')

    # box = ax1.get_position()
    # ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax1.legend(['$R_2^*='+str(r)+'$' for r in R2s_Hz],
    #            loc='center left', bbox_to_anchor=(1, 0.5))

    # inv F maybe not corrects
    ax2.plot(dTE_s, F22)
    ax2.set_ylim(0., 0.3)
    ax2.set_title('$F^{-1}_{2 2}$')
    ax2.set_xlabel('$\Delta$ TE')

    ax3.plot(dTE_s, F33)
    ax3.set_ylim(0., 0.3)
    ax3.set_title('$F^{-1}_{3 3}$')
    ax3.set_xlabel('$\Delta$ TE')

    fig.tight_layout()
    fig.suptitle('CRLB')
    plt.savefig('CRLB.pdf')
    tikzsave('CRLB.tex')


    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
    # inv F maybe not corrects
    ax0.plot(dTE_s, tr_invF)
    ax0.set_ylim(0.6, 2)
    ax0.set_title('$tr(F^{-1})$')
    ax0.legend(['$R_2^*='+str(r)+'$' for r in R2s_Hz])

    ax1.plot(dTE_s, 1/det_F)
    ax1.set_ylim(0, 1.5e-4)
    ax1.set_title('$det(F^{-1})$')

    ax2.plot(dTE_s, tr_F)
    ax2.set_title('$tr(F)$')
    ax2.set_xlabel('$\Delta$ TE')

    ax3.plot(dTE_s, det_F)
    ax3.set_title('$det(F)$')
    ax3.set_xlabel('$\Delta$ TE')

    fig.tight_layout()
    fig.suptitle('optimal design criteria')
    plt.savefig('opt_design.pdf')
    tikzsave('opt_design.tex')

    # box = ax1.get_position()
    # ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax1.legend(['$R_2^*='+str(r)+'$' for r in R2s_Hz],
    #            loc='center left', bbox_to_anchor=(1, 0.5))

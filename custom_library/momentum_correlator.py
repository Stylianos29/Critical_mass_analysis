import numpy as np


def symmetrization(values_list):
    reverse = values_list[::-1]

    return 0.5*(values_list + np.roll(reverse, shift=+1))


def plateau_periodic_correlator(t, p):
    T = 48
    return p[0]*( np.exp(-p[1]*t) + np.exp(-p[1]*(T-t)) )


def two_state_periodic_correlator(t, p):
    T = 48
    # return p[0]*(np.exp(-p[1]*t -p[2]*t*t) + np.exp(-p[1]*(T-t) -p[2]*(T-t)*(T-t)))
    return p[0]*( np.exp(-p[1]*t) + p[2]*np.exp(-p[3]*t) + np.exp(-p[1]*(T-t)) + p[2]*np.exp(-p[3]*(T-t)) )



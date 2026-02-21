import numpy as np
from mousse.utils import vcorrcoef, Mcorrcoef


def test_vcorrcoeff():
    X = np.arange(24).reshape(6,4)
    y = np.array([3,4,1,2]).reshape(1,4)
    X[1,:] = 0
    X[2,:] = 13
    X[3,:] = y
    X[4,:] = 1-y
    X[5,1] = 13
    r = vcorrcoef(X, y)

    if ((np.corrcoef(X[0,:], y)[1,0] - r[0]) > 1e-10):
        print('Error 0')
        return False

    if (0 != r[1]):
        print('Error 1')
        return False

    if (0 != r[2]):
        print('Error 2')
        return False

    if ((np.corrcoef(X[3,:], y)[1,0] - r[3]) > 1e-10):
        print('Error 3')
        return False

    if ((np.corrcoef(X[4,:], y)[1,0] - r[4]) > 1e-10):
        print('Error 4')
        return False

    if ((np.corrcoef(X[5,:], y)[1,0] - r[5]) > 1e-10):
        print('Error 5')
        return False

    return True


def test_Mcorrcoeff():
    X = np.arange(24).reshape(6,4)
    Y = np.arange(24).reshape(6,4)
    X[:,2] = 5
    X[1,:] = 0
    X[2,:] = 13
    X[3,:] = Y[3,:]
    X[4,:] = 1-Y[4,:]
    r = Mcorrcoef(X, Y)

    if ((np.corrcoef(X[0,:], Y[0,:])[1,0] - r[0]) > 1e-10):
        print('Error 0')
        return False

    if (0 != r[1]):
        print('Error 1')
        return False

    if (0 != r[2]):
        print('Error 2')
        return False

    if ((np.corrcoef(X[3,:], Y[3,:])[1,0] - r[3]) > 1e-10):
        print('Error 3')
        return False

    if ((np.corrcoef(X[4,:], Y[4,:])[1,0] - r[4]) > 1e-10):
        print('Error 4')
        return False

    if ((np.corrcoef(X[5,:], Y[5,:])[1,0] - r[5]) > 1e-10):
        print('Error 5')
        return False

    return True
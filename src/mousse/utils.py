
import numpy as np

def vcorrcoef(X,y, zero_var_out=0):
    # correlation between the rows of the matrix X with dimensions (N x k) and a vector y of size (1 x k)
    # zero_var_out: is the output where the variance is 0
    # can be either 0 or np.nan
    # about 200 times faster than calculating correlations row by row
    Xm = np.reshape(np.nanmean(X,axis=1),(X.shape[0],1))
    vec_nonzero = np.sum((X - Xm)**2, axis=1) != 0 # we select the rows with nonzera VARIANCE...

    ym = np.nanmean(y)
    r_num = np.nansum((X-Xm)*(y-ym),axis=1)
    r_den = np.sqrt(np.nansum((X-Xm)**2,axis=1)*np.nansum((y-ym)**2))
    out_vec = np.zeros_like(r_num)
    out_vec[:] = zero_var_out
    r = np.divide(r_num, r_den, out=out_vec, where=vec_nonzero)
    return r


def Mcorrcoef(X,Y, zero_var_out=0):
    # correlation between the rows of two matrices X and Y with dimensions (N x k)
    # zero_var_out: is the output where the variance is 0
    # can be either 0 or np.nan

    Xm = np.reshape(np.nanmean(X,axis=1),(X.shape[0],1))
    Ym = np.reshape(np.nanmean(Y,axis=1),(Y.shape[0],1))
    vec_nonzero = (np.sum((X - Xm)**2, axis=1) != 0) & (np.sum((Y - Ym)**2, axis=1) != 0) # we select the rows with nonzera VARIANCE...

    r_num = np.nansum((X-Xm)*(Y-Ym),axis=1)
    r_den = np.sqrt(np.nansum((X-Xm)**2,axis=1)*np.nansum((Y-Ym)**2,axis=1))
    out_vec = np.zeros_like(r_num)
    out_vec[:] = zero_var_out
    r = np.divide(r_num, r_den, out=out_vec, where=vec_nonzero)
    return r

def nan_divide(a, b, where=True):
    'division function that returns np.nan where the division is not defined'
    x = np.zeros_like(a)
    x.fill(np.nan)
    x = np.divide(a, b, out=x, where=where)
    return x

def nan_add(a, b):
    'addition function that handles NANs by replacing them with zero - USE with CAUTION!'
    aa = a.copy()
    bb = b.copy()
    aa[np.isnan(aa)] = 0
    bb[np.isnan(bb)] = 0
    x = np.array(aa + bb)
    return x

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

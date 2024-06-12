from scipy.stats import gaussian_kde
import numpy as np

from sklearn.datasets import load_breast_cancer, fetch_california_housing, load_diabetes
from .data_loaders import *
from scipy.stats import wasserstein_distance, energy_distance
def load_data(name):
    if name == "breast_cancer":
        return load_breast_cancer(), True
    elif name == "california":
        return fetch_california_housing(download_if_missing=True), False
    elif name == "diabetes":
        return load_diabetes(), False
    elif name == "credit":
        return load_credit(), True
    elif name == "wages":
        return load_wages(), False
    elif name == "adult":
        return load_adult(), True
    elif name == "insurance":
        return load_insurance(), False
    elif name == "student":
        return load_student(), False
    elif name == "life":
        return load_life(), False
    elif name == "bike":
        return load_bike(), False
    elif name == "wine":
        return load_wine(), False
    elif name == "forest":
        return load_forest(), False
    elif name == "mpg":
        return load_mpg(), False
    elif name == "boston":
        return load_boston(), False
    elif name == "cpu":
        return load_cpu(), False
    elif name == "abalone":
        return load_abalone(), False
    elif name == "automobile":
        return load_automobiles(), False
    elif name == "airquality":
        return load_air_quality(), False
    elif name == "liver":
        return load_liver(), False
    elif name == "gold":
        return load_gold(), False
    else:
        raise ValueError("Unknown dataset "+str(name))
    
def wkl(Y, Y_subgroup, binary,a=1/2):
    if binary:
        y0 = np.sum(Y==0)/Y.shape[0]+1e-6
        y1 = np.sum(Y==1)/Y.shape[0]+1e-6
        ys0 = np.sum(Y_subgroup==0)/Y_subgroup.shape[0]+1e-6
        ys1 = np.sum(Y_subgroup==1)/Y_subgroup.shape[0]+1e-6
        kl = ys0*np.log(ys0/y0) + ys1*np.log(ys1/y1)
        return kl*(Y_subgroup.shape[0]/Y.shape[0])
    else:
        density_y_subgroup = gaussian_kde(Y_subgroup.T)
        density_y = gaussian_kde(Y.T)
        log_y_s = density_y_subgroup.logpdf(Y_subgroup.T)
        log_y = density_y.logpdf(Y_subgroup.T)
        p_y_s = density_y_subgroup.pdf(Y_subgroup.T)
        n_subgroup = Y_subgroup.shape[0]
        kl = np.sum(p_y_s*(log_y_s-log_y))/np.sum(p_y_s)
        if kl < 0:
            kl = 0
        return kl*(n_subgroup/Y.shape[0])**a

def evaluate_overlap(subgroup_labels):
    overlap = 0
    n = 0
    for i in range(len(subgroup_labels)):
        for j in range(i+1,len(subgroup_labels)):
            label1 = subgroup_labels[i]
            label2 = subgroup_labels[j]
            overlap += np.sum(np.logical_and(label1,label2))/np.sum(np.logical_or(label1,label2))
            n += 1
    if n == 0:
        return 0
    return overlap/n

def compute_tv(Y, Y_subgroup,a=1/2):
    density_y_subgroup = gaussian_kde(Y_subgroup.T)
    density_y = gaussian_kde(Y.T)
    p_y_s = density_y_subgroup.pdf(Y_subgroup.T)
    p_y = density_y.pdf(Y_subgroup.T)
    tv = np.sum(np.abs(p_y_s-p_y))
    return tv*(Y_subgroup.shape[0]/Y.shape[0])**a

def compute_wd(Y, Y_subgroup,a=1/2):
    density_y_subgroup = gaussian_kde(Y_subgroup.T)
    density_y = gaussian_kde(Y.T)
    rYs = density_y_subgroup.resample(size=(20000,))
    Ys = density_y.resample(size=(20000,))
    rYs = rYs.reshape((20000,))
    Ys = Ys.reshape((20000,))
    #weights_sub = (rYs < Y_subgroup.max()) + (rYs > Y_subgroup.min())
    #weights = (Ys < Y_subgroup.max()) + (Ys > Y_subgroup.min())
    mask  = np.logical_and(rYs <= Y_subgroup.max(), rYs >= Y_subgroup.min())
    wd = wasserstein_distance(rYs[mask],Ys[mask])
    n_subgroup = Y_subgroup.shape[0]
    return wd *(n_subgroup/Y.shape[0])**a

def compute_ed(Y, Y_subgroup,a=1/2):
    density_y_subgroup = gaussian_kde(Y_subgroup.T)
    density_y = gaussian_kde(Y.T)
    rYs = density_y_subgroup.resample(size=(20000,))
    Ys = density_y.resample(size=(20000,))
    rYs = rYs.reshape((20000,))
    Ys = Ys.reshape((20000,))
    #weights_sub = (rYs < Y_subgroup.max()) + (rYs > Y_subgroup.min())
    #weights = (Ys < Y_subgroup.max()) + (Ys > Y_subgroup.min())
    mask  = np.logical_and(rYs <= Y_subgroup.max(), rYs >= Y_subgroup.min())
    ed = energy_distance(rYs[mask],Ys[mask])
    n_subgroup = Y_subgroup.shape[0]
    return ed *(n_subgroup/Y.shape[0])**a

from ..measures.subgroup_measures import kullbackleibler_gaussian_paramters
from ..rulelistmodel.categoricalmodel.prediction_categorical import point_value_categorical
#from ..rulelistmodel.gaussianmodel.gaussianstatistic import compute_RSS
from ..rulelistmodel.gaussianmodel.mdl_gaussian import gaussian_fixed_encoding
from ..rulelistmodel.gaussianmodel.prediction_gaussian import point_value_gaussian
from ..rulelistmodel.rulesetmodel import RuleSetModel
import pandas as pd
import numpy as np
from functools import reduce
import scipy.stats

from math import log2

from ..util.build.extra_maths import log2_0

point_value_estimation = {
    "gaussian" : point_value_gaussian,
    "categorical": point_value_categorical
}

def predict_rulelist(X : pd.DataFrame, rulelist: RuleSetModel):
    if X is not pd.DataFrame: Exception('X needs to be a DataFrame')
    n_predictions = X.shape[0]
    n_targets = rulelist.default_rule_statistics.number_targets
    instances_covered = np.zeros(n_predictions, dtype=bool)
    predictions = np.empty((n_predictions,n_targets),dtype=object)
    for subgroup in rulelist.subgroups:
        instances_subgroup = ~instances_covered &\
                             reduce(lambda x,y: x & y, [item.activation_function(X).values for item in subgroup.pattern])
        predictions[instances_subgroup,:] = point_value_estimation[rulelist.target_model](subgroup.statistics)
        instances_covered |= instances_subgroup

    # default rule
    predictions[~instances_covered, :] = point_value_estimation[rulelist.target_model](rulelist.default_rule_statistics)
    if n_targets == 1:
        predictions = predictions.flatten()
    return predictions

def swkl_subgroup_discovery(X : pd.DataFrame, Y:pd.DataFrame, rulelist: RuleSetModel):
    """ Compute the Sum of Weighted Kullback-Leibler divergence

    TODO: it only works for single target variable
    """
    if X is not pd.DataFrame: Exception('X needs to be a DataFrame')
    n_predictions = X.shape[0]
    n_targets = rulelist.default_rule_statistics.number_targets
    instances_covered = np.zeros(n_predictions, dtype=bool)
    predictions = np.empty((n_predictions,n_targets),dtype=object)
    loss = 0
    for subgroup in rulelist.subgroups:
        instances_subgroup = reduce(lambda x,y: x & y, [item.activation_function(X).values for item in subgroup.pattern])
        instances_subgroup_in_list = ~instances_covered & instances_subgroup
        for target in Y.columns:
            if rulelist.target_model == 'categorical':
                epsilon = 0.5  # adding Jeffreys prior for unseen data
                counts_subgroup = subgroup.statistics.usage_per_class[target]
                usage_subgroup = subgroup.statistics.usage
                n_classes = subgroup.statistics.number_classes[target]
                probs = {cl: (counts_subgroup.get(cl)+epsilon)/(usage_subgroup + n_classes*epsilon)
                         for cl in Y[target].unique()}
                loss += log_loss_nominal(Y[instances_subgroup_in_list], target, probs)
            else:
                mean_sd = subgroup.statistics.mean[0]
                var_sd = subgroup.statistics.variance[0]
                loss += log_loss_numeric(Y[instances_subgroup_in_list], target, mean_sd, var_sd)

        instances_covered |= instances_subgroup_in_list

    predictions[~instances_covered, :] = point_value_estimation[rulelist.target_model](rulelist.default_rule_statistics)
    for target in Y.columns:
        if rulelist.target_model == 'categorical':
            loss += log_loss_nominal(Y[~instances_covered], target, rulelist.default_rule_statistics.prob_per_classes[target])
        else:
            mean_sd = rulelist.default_rule_statistics.mean[0]
            var_sd = rulelist.default_rule_statistics.variance[0]
            loss += log_loss_numeric(Y[~instances_covered], target, mean_sd, var_sd)

    loss_norm = loss/Y.shape[0]
    # default rule does not require swkl as it
    return loss, loss_norm


def log_loss_numeric(Y:pd.DataFrame, target: str, mean, var):
    c = Y[target] - mean
    rss = np.dot(c, c)
    n = Y[target].shape[0]
    loss = gaussian_fixed_encoding(n, rss, var)

    return loss



def log_loss_nominal(Y:pd.DataFrame,target: str, probs):

    loss = 0
    for val in Y[target].values:
        prob = probs.get(val)
        loss += -log2(prob)
    return loss



def wkl_nominal(Y:pd.DataFrame, target: str, instances_subgroup):

    prob_default = {cl: sum(Y[target] == cl) / Y.shape[0] for cl in Y[target].unique()}
    Y_subgroup = Y[target][instances_subgroup]
    prob_subgroup = {cl: sum(Y_subgroup == cl) / Y_subgroup.shape[0]
                     for cl in Y[target].unique()}
    usage = Y_subgroup.shape[0]
    print(f"prob subgroup {prob_subgroup}")
    print(f"prob_default {prob_default}")

    kl =0
    for cl in Y[target].unique():
        prob_sg = prob_subgroup.get(cl)
        prob_def= prob_default.get(cl)
        kl += prob_sg*log2_0(prob_sg/prob_def)
    wkl= usage*kl
    return wkl
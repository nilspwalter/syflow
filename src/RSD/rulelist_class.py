# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:13:18 2019

@author: gathu
"""
from functools import reduce
from time import time
from typing import Literal
import numpy as np

from .rulelistmodel.prediction import predict_rulelist, swkl_subgroup_discovery
from .search.iterative_rule_search import _fit_rulelist

from .util.bitset_operations import bitset2indexes


class MDLRuleList():
    """Subgroup list discovery for single-numeric targets with an MDL formulation.
    It resorts to greedy and beam search to find the the subgroup list that
    best fits the data
    Parameters
    ----------
    target_type : string, mandatory
        (possible values: "single-numeric" or "nominal")
        choose the appropriate target_type (no default value) for the type of
        rule/subgroup search.
    max_depth : int, optional (default=4)
        defines the maximum size that subgroup description can take based
        on the number of variables that the beam search accepts to refine.
        For example, if 'max_depth = 4' the maximum size of a pattern found is
        4.
    beam_width : int, optional (default=100)
        defines the width of the beam in the beam search, i.e., the number of
        patterns that are selected at each iteration to be expanded.
    min_support : int or float
        defines the minimum support that a rule/subgroup can cover in the training data.
        if positive int, it defines an absolute value
        if smaller than one float, it defines a relative value, i.e., min_support*number_instances_data
    iterative_beam_width
    n_cutpoints : int, optional (default=5)
       number of cut points used to discretize a single-numeric attribute/variable.
       Note 1: this algorithm creates for each cutpoint a binary split, and
       the combination of all cutpoints. As an example of the former, if the
       cut point is x_cut = 5, it will create both the condition x<5 and x>5.
       In relation to the latter, if two of the cut points are x_cut1=3, and
       x_cut2=5, it will also create  3<x<5.
    task : string (default = "discovery")
       (possible values: "discovery" or "prediction")
       - "discovery": performs subgroup discovery by assuming the last rule of
       the model as a constant rule and equal to the dataset distribution.
       - "prediction": finds a rule list for prediction by assuming that the
       last rule changes with other rules in the dataset.
    discretization : string (default="static")
       (possible values: "static" or "dynamic")
       - "static" - performs a priori discretization of all single-numeric variables
       - "dynamic" - at each iteration of the beam search it conditionally
       discretizes all single-numeric variables based on the given pattern.
    max_rules : int, optional (default=0)
       Maximum number of subgroups/rules to mine. If max_rules=0 is given it
       continues finding subgroups/rules until no more compression is achieved.
    alpha_gain : int, optional (default="normalized")
       (possible values: "absolute" or "normalized")
       Type of score used to expand the beam search and to add a rule/subgroup
       at each iteration.
       - "absolute" - adds the rule/subgroup at each iteration that maximizes
       the normalized alpha_gain, i.e., that difference between the length of the
       existing model minus the length of that model with the candidate
       subgroup added.
       - "normalized" - adds the rule/subgroup at each iteration that maximizes
       the "absolute" alpha_gain normalized by the number of instances covered
       (usage) by that rule/subgroup.

    Attributes
    ----------
    number_rules: int
        Number of rules of the list excluding the default rule.
    antecedent_description: list of strings
        String of each rule antecedent description.
    consequent_description: list of strings
        String of each rule consequents.
    """

    def __init__(self,target_model : Literal["gaussian", "categorical"], task : Literal["prediction", "discovery"],
                 max_depth=5, beam_width = 100, min_support = 1, iterative_beam_width=1, n_cutpoints = 5, discretization = "static",
                 max_rules = np.inf, alpha_gain = 1.0):

        if target_model not in ["categorical", "gaussian"]:
            raise ValueError("Target model incorrectly selected, please select either \"categorical\" or \"gaussian\".")

        if task not in ["prediction", "discovery"]:
            raise ValueError("Task incorrectly selected, please select either \"prediction\" or \"discovery\".")

        if not isinstance(max_depth, (int, np.integer)) or max_depth < 1:
            raise ValueError("max_depth incorrectly selected, please select a "
                             "positive integer greater or equal to 1.")

        if not isinstance(beam_width, (int, np.integer)) or beam_width < 1:
            raise ValueError("beam_width incorrectly selected, please select a "
                             "positive integer greater or equal to 1.")

        if not isinstance(n_cutpoints, (int, np.integer)) or n_cutpoints < 2:
            raise ValueError("n_cutpoints incorrectly selected, please select a "
                             "positive integer greater or equal to 2.")

        if discretization not in ["static"]:
            raise ValueError("At this moment we only support \"static\" discretization.")

        if not isinstance(n_cutpoints, (int, np.integer)) or max_rules < 0:
            raise ValueError("max_rules incorrectly selected, please select a "
                             "zero or a positive integer.")

        if alpha_gain < 0 or alpha_gain > 1:
            raise ValueError("alpha_gain incorrectly selected, please select a "
                             "between zero and 1 inclusive.")



        self.target_model = target_model
        self.alpha_gain = alpha_gain
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.min_support = min_support
        self.iterative_beam_width= iterative_beam_width
        self.n_cutpoints = n_cutpoints
        self.discretization = discretization
        self.task = task
        self.number_rules = 0
        self.max_rules = max_rules
        self.rulelist = None

    #TODO:  def __repr__
    def __str__(self):
        text2print = self._rulelist.description if self.number_rules > 0 else "Model not fitted"
        return text2print

    def fit(self,X,Y):
        """Fit the model according to the given training data.
        Parameters
        ----------
        df : pandas dataframe with name variables with last column as target
        variable.
        Returns
        -------
        self : object
        """




        start_time = time()
        self._rulelist = _fit_rulelist(
                X,Y, self.target_model, self.max_depth,self.beam_width,self.min_support,
                self.iterative_beam_width, self.n_cutpoints,
                self.task,self.discretization,self.max_rules,self.alpha_gain)
        self.runtime = time() - start_time
        self.number_rules = self._rulelist.number_rules
        self.rule_sets = [bitset2indexes(bitset) for bitset in self._rulelist.bitset_rules]
        return self



    def predict(self,X):
        """ Predict for new data what is it going to be the performance
        if rule list not fit it does not work
        ----------
        X : a numpy array or pandas dataframe with the variables in the same
            poistion (column number) as given in "fit" function.

        Returns a numpy array y with the predicted values according to the
        fitted rule list (obtained using the "fit" function above). y has the
        same length as X.shape[0] (number of rows).
        -------
        self : object
        """
        y_hat = predict_rulelist(X, self._rulelist)
        return y_hat

    def swkl_generalise(self,X,Y):
        """ Compute the Sum of weighted Kullback-Leibler Divergences (SWKL) for a test set.
        ----------
        X : a numpy array or pandas dataframe with the variables in the same
            position (column number) as given in "fit" function.
        Y:  a numpy array or pandas dataframe with the variables in the same
            position (column number) as given in "fit" function.

        Returns the SWKL measure for X,Y, given a fitted rulelist.
        -------
        self : object
        """
        swkl, swkl_norm = swkl_subgroup_discovery(X, Y, self._rulelist)
        return swkl, swkl_norm
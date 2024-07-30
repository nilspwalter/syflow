import torch
from .syflow import *
from sklearn.preprocessing import StandardScaler
import pysubgroup as ps
from collections import namedtuple
import pandas as pd
import numpy as np
import warnings
import time
import argparse
from .RSD.rulelist_class import MDLRuleList, reduce

warnings.filterwarnings("ignore")
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#       device = "mps"
verbose = False
    
def run_method(method,X,Y,alpha,config,n_subgroups,feature_names):
    subgroups = 123
    rules = 123
    if method == "syflow":
        subgroups, rules = run_syflow(X,Y,config,n_subgroups,feature_names)
    elif method == "sd-mean":
        subgroups, rules = run_sd_mean(X,Y,config,alpha,n_subgroups,feature_names)
    elif method == "sd-kl":
        subgroups, rules = run_sd_kl(X,Y,config,alpha,n_subgroups,feature_names)
    elif method == "rsd":
        subgroups, rules = run_rsd(X,Y,config,alpha,n_subgroups,feature_names)
    elif method == "bh":
        pass
        #subgroups, rules = run_bh(X,Y,config,alpha,n_subgroups,feature_names)
    return subgroups, rules

def run_syflow(X, Y, config, n_subgroups, feature_names, return_flows=False):
    cut_points = torch.zeros((X.shape[1],2))
    scaler_x = StandardScaler()
    X = scaler_x.fit_transform(X)
    scaler_y = StandardScaler()
    Y = scaler_y.fit_transform(Y)
    X_tensor = torch.tensor(X,dtype=torch.float64)
    Y_tensor = torch.tensor(Y,dtype=torch.float64)

    subgroups = []
    priors = []
    rules = []
    pop_flow = None
    print("Logging Hyperparameters")
    print("Alpha:",config.alpha)
    print("Lambda:",config.lambd)
    print("Temperature:",config.temperature,"\n")
    for n in range(n_subgroups):
        print("Discovering Subgroup #{}".format(n+1))
        for i in range(X.shape[1]):
            cut_points[i,0] = torch.quantile(X_tensor[:,i],0)
            cut_points[i,1] = torch.quantile(X_tensor[:,i],1)
        cut_points = torch.sort(cut_points,dim=1)[0]
        classifier = And_Finder(cut_points,temperature=config.temperature,use_weights=config.use_weights,bin_deviation=config.bin_deviation)
        flows, classifier = syflow(X_tensor,Y_tensor,classifier,flow_population=pop_flow,subgroup_priors=priors,
                                            pop_train_epochs=config.pop_train_epochs,subgroup_train_epochs=config.subgroup_train_epochs,final_fit_epochs=config.final_fit_epochs,
                                        device=device,verbose=verbose,lr_flow=config.lr_flow,alpha=config.alpha,
                                    lr_classifier=config.lr_classifier,lambd=config.lambd,config=config)
        pop_flow = flows[0]
        priors.append(flows[1])
        classifier = classifier.to(torch.device("cpu"))
        subgroup = torch.argmax(classifier(X_tensor),dim=1).detach().numpy()==1
        subgroups.append(subgroup)
        rules.append(classifier.get_rules(cut_points,scaler=scaler_x,feature_names=feature_names,X=X))
    if return_flows:
        return subgroups, rules, pop_flow, scaler_y,classifier
    return subgroups, rules

def run_sd_mean(X,Y,config,alpha,n_subgroups,feature_names):
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    Y.columns = ["Y"]
    X.columns = [f"X{i}" for i in range(X.shape[1])]
    data = pd.concat([X,Y],axis=1)
    target = ps.NumericTarget("Y")
    search_space = ps.create_selectors(data, ignore=["Y"],nbins=config.ncutpoints, intervals_only=False)
    task = ps.SubgroupDiscoveryTask (
        data, 
        target, 
        search_space, 
        result_set_size=n_subgroups, 
        depth=config.sd_depth, 
        qf=ps.StandardQFNumeric(alpha))

    result = ps.BeamSearch(beam_width=config.beam_width,beam_width_adaptive=False).execute(task)
    result.to_dataframe()
    subgroups = []
    rules = []
    for i in range(n_subgroups):
        result_string = str(result.to_dataframe().iloc[i]["subgroup"])
        rules.append(replace_feature_names(result_string,feature_names))
        parts = result_string.split(" AND ")
        conditions = []
        for part in parts:
            # parse this: "X0>=0.80" or "X0<0.80"
            if "==" in part:
                var, val = part.split("==")
                var = int(var[1:])
                #if isinstance(val,str):
                val = convert(data,var, val)
                conditions.append((var,val,val))
                continue
            elif "<" in part:
                var, high = part.split("<")
                low = - np.infty
                var = int(var[1:])
                high = float(high)
            else:
                var, low = part.split(">=")
                high = np.infty
                var = int(var[1:])
                low = float(low)
            conditions.append((var,low,high))
            
        subgroup_member = np.ones((X.shape[0],),dtype=bool)
        for cond in conditions:
            var, low, high = cond
            var = int(var)
            subgroup_member = np.logical_and(subgroup_member, np.logical_and(X.iloc[:,var]>=low, X.iloc[:,var]<=high))
        subgroups.append(subgroup_member)
    return subgroups, rules

def convert(df,var, val):
     if val.replace(".", "").isnumeric():
          return float(val)
     else:
          if str(df.dtypes[var]) == 'bool':
               return float(val=='True')
          
def run_sd_kl(X,Y,config,alpha,n_subgroups,feature_names):
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    Y.columns = ["Y"]
    X.columns = [f"X{i}" for i in range(X.shape[1])]
    data = pd.concat([X,Y],axis=1)
    
    target = ps.NumericTarget("Y")
    search_space = ps.create_selectors(data, ignore=["Y"],nbins=config.ncutpoints, intervals_only=False)
    task = ps.SubgroupDiscoveryTask (
        data, 
        target, 
        search_space, 
        result_set_size=n_subgroups, 
        depth=config.sd_depth, 
        qf=QF_WKL(alpha))

    result = ps.BeamSearch(beam_width=config.beam_width,beam_width_adaptive=False).execute(task)
    result.to_dataframe()
    subgroups = []
    rules = []
    for i in range(n_subgroups):
        result_string = str(result.to_dataframe().iloc[i]["subgroup"])
        rules.append(replace_feature_names(result_string,feature_names))
        parts = result_string.split(" AND ")
        conditions = []
        for part in parts:
            # parse this: "X0>=0.80" or "X0<0.80"
            if "==" in part:
                var, val = part.split("==")
                var = int(var[1:])
                val = val = convert(data,var, val)
                conditions.append((var,val,val))
                continue
            elif "<" in part:
                var, high = part.split("<")
                low = - np.infty
                var = int(var[1:])
                high = float(high)
            else:
                var, low = part.split(">=")
                high = np.infty
                var = int(var[1:])
                low = float(low)
            conditions.append((var,low,high))
            
        subgroup_member = np.ones((X.shape[0],),dtype=bool)
        for cond in conditions:
            var, low, high = cond
            var = int(var)
            subgroup_member = np.logical_and(subgroup_member, np.logical_and(X.iloc[:,var]>=low, X.iloc[:,var]<=high))
        subgroups.append(subgroup_member)
    return subgroups, rules


# KL divergence quality measure assuming normal distribution

class QF_WKL(ps.BoundedInterestingnessMeasure):
    tpl = namedtuple('StandardQFNumeric_parameters', ('size_sg', 'mean', "std", 'estimate'))

    def __init__(self, a, invert=False, estimator='sum'):
        self.a = a
        self.invert = invert
        self.required_stat_attrs = ('size_sg', 'mean')
        self.dataset_statistics = None
        self.all_target_values = None
        self.has_constant_statistics = False

    def calculate_constant_statistics(self, data, target):
        self.all_target_values = data[target.target_variable].to_numpy()
        target_mean = np.mean(self.all_target_values)
        data_size = len(data)
        std = np.std(self.all_target_values)
        self.dataset_statistics = QF_WKL.tpl(data_size, target_mean, std,None)
        self.has_constant_statistics = True

    def evaluate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        size_sg = statistics.size_sg
        mean_sg = statistics.mean
        std_sg = statistics.std + 0.0000001
        mean_dataset = dataset.mean
        std_dataset = dataset.std
        if size_sg < 2:
            return 0
        kl = np.log2(std_dataset/std_sg) + (std_sg**2+(mean_sg-mean_dataset)**2)/(2*std_dataset**2)
        w = size_sg/dataset.size_sg
        return w**(self.a)*kl

    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover_arr, sg_size = ps.get_cover_array_and_size(subgroup, len(self.all_target_values), data)
        sg_mean = 0
        sg_target_values = 0
        sg_std = 0
        if sg_size > 1:
            sg_target_values = self.all_target_values[cover_arr]
            sg_mean = np.mean(sg_target_values)
            sg_std = np.std(sg_target_values)
        return QF_WKL.tpl(sg_size, sg_mean, sg_std, None)
    
def run_rsd(X,Y,config,alpha,n_subgroups,feature_names):
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()


    X = scaler_x.fit_transform(X)
    Y = scaler_y.fit_transform(Y)

    X = pd.DataFrame(X)
    X.columns = feature_names
    Y = pd.DataFrame(Y)
    Y.columns = ["Y"]
    target_model = "gaussian"
    task = "discovery"
    model = MDLRuleList(task = task, target_model = target_model,max_rules=n_subgroups, n_cutpoints=config.ncutpoints)
    model.fit(X, Y)
    
    subgroups = []
    for subgroup in model._rulelist.subgroups:
        subgroup_member = reduce(lambda x,y: x & y, [item.activation_function(X).values for item in subgroup.pattern])
        subgroups.append(subgroup_member)
    rules = model._rulelist.get_rules()
    return subgroups, rules

def replace_feature_names(rule,feature_names):
    # replace X0<=... with feature_names[0]<=...
    for i in reversed(range(len(feature_names))):
        if "X"+str(i) in rule:
            rule = rule.replace("X"+str(i),feature_names[i])
    return rule
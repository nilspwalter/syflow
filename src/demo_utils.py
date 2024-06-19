import sys
sys.path.append("../")
from src import *
from src.methods import run_method
from sklearn.metrics import f1_score
from pathlib import Path
import matplotlib.pyplot as plt
from cycler import cycler
import argparse
import csv
import time

plt.rcParams['font.family'] = 'serif'

def setup_path(name):
    base = "../results/{}/".format(name)
    Path(base).mkdir(parents=True, exist_ok=True)
    return base

def plot_target(target, xlabel, ylabel, title, bins=50, density=True):
    plt.hist(target, bins=bins, density=density)
    plt.xlabel(xlabel, fontsize=14, labelpad=16)
    plt.ylabel(ylabel, fontsize=14, labelpad=16)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()

def plot_subgroups(target, subgroups, rules, which_sgs):
    default_cycler = cycler(color=['g', 'b', 'r', 'y'])
    plt.rc('lines', linewidth=4)
    plt.rc('axes', prop_cycle=default_cycler)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    for sg in which_sgs:
        subgroup = target[subgroups[sg]]
        plt.hist(subgroup, label=rules[sg], alpha=1.0, bins=50)
    plt.legend()


class SyflowConfig:

    def __init__(self, alpha=0.3, lamb=2):
        self.lr_flow = 5e-2
        self.lr_classifier = 2e-2
        self.alpha = alpha
        self.lambd = lamb
        self.pop_train_epochs = 1000
        self.subgroup_train_epochs = 1000
        self.final_fit_epochs = 0
        self.temperature = 0.2
        self.bin_deviation = 0.2
        self.use_weights = True
        self.seed = 10
        def flow_gen():
            #return bij.Compose([bij.Spline(count_bins=12), bij.Spline(count_bins=12)])
            return bij.Spline(count_bins=12)
        self.flow_gen = flow_gen

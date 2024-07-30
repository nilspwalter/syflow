import sys
sys.path.append("../")

from src import *
from src.methods import *
from src.demo_utils import setup_path, f1_score, csv

import matplotlib.pyplot as plt

config = Config_Real_World()
parser = argparse.ArgumentParser()

parser.add_argument("--method","-m", type=str, default="syflow", help="method to run")
parser.add_argument("--dataset","-d",  type=str, default="gold", help="dataset")
parser.add_argument("--alpha","-a",  type=float, default=1.0, help="alpha")

args = parser.parse_args()
dataset_name = args.dataset
config = conf_dict[dataset_name]

np.random.seed(args.seed)

if args.dataset == "gold":
    dataset, is_classification = load_gold("sd" in args.method), False
elif args.dataset == "higgs":
    dataset, is_classification = load_higgs(s_portion=0.5, mask=False), False
elif args.dataset == "vdw":
    dataset, is_classification = load_goldvdw("sd" in args.method), False

X = dataset["data"]
Y = dataset["target"]
if len(Y.shape) == 1:
    Y = Y[:,None]
feature_names = dataset["feature_names"]
start = time.time()
subgroups, rules = run_method(args.method,X.copy(),Y.copy(),args.alpha,config,config.n_subgroups,feature_names)
end = time.time()
runtime = end-start
path = "results/real_world/physics/{}/{}-{}.{}"
f2 = open(path.format("rules",args.method,dataset_name,"txt"),"w")
for rule in rules:
    f2.write(rule+"\n")
 
f2.write(str(runtime))
f2.close()

wkls = []
means = []
stds = []
coverage = []
for result in subgroups:
    if np.sum(result) == 0:
        wkl_score = 0
    else:
        wkl_score = 0#wkl(Y, Y[result], is_classification, a=args.alpha)
    wkls.append(wkl_score)
    means.append(np.abs(np.mean(Y[result])-np.mean(Y)))
    stds.append(np.abs(np.std(Y[result])-np.std(Y)))
    coverage.append(np.sum(result)/len(result))
overlap = evaluate_overlap(subgroups)
file = path.format("subgroups",args.method,dataset_name,"npy")
np.save(file,np.array(subgroups))
print("Dataset: "+dataset_name,"Method: "+args.method,"WKL",np.max(wkls),np.mean(wkls),"Overlap",overlap)

if dataset_name == "gold" or dataset_name == "vdw":
    rang=[Y.min(),Y.max()]
    bins=300
else:
    rang=[0,300]
    bins=100

f, axes = plt.subplots(1, len(rules), sharey=True,figsize=(25,5))
for i in range(len(rules)):
    axes[i].hist(Y[subgroups[i]],bins=bins,density=False, range=rang)
    print(subgroups[i].sum())
f.savefig(path.format("plots",args.method,dataset_name,"png"), dpi=f.dpi, bbox_inches='tight')
import sys
sys.path.append("../")

from src import *
from src.methods import *
from src.demo_utils import setup_path, f1_score, csv

parser = argparse.ArgumentParser()

parser.add_argument("--method", type=str, default="sd-mean", help="method to run")
parser.add_argument("--alpha", type=float, default=1, help="alpha")

args = parser.parse_args()

config = Config_Synthetic()
config_exp = Config_Binning()


base = setup_path("binning")
path = base + "{}.csv"

with open(path.format(args.method),"w") as f:
    writer = csv.writer(f)
    writer.writerow(["N-bins","F1","F1-STD","Runtime","Runtime-STD"])
    f.close()

for bins in config_exp.bins:
    feature_names = ["X"+str(i) for i in range(config_exp.n_variables)]
    config.ncutpoints = bins
    f1s = []
    runtimes = []
    for rep in range(config_exp.n_reps):
        X,Y, subgroup_gt = gen_synth(config_exp.n_samples, config_exp.n_variables, config_exp.n_conditions, config_exp.target_dist,seed=rep)
        start = time.time()
        subgroups, rules = run_method(args.method,X,Y,args.alpha,config,config_exp.n_subgroups,feature_names)
        end = time.time()
        runtimes.append(end-start)
        scores = []
        for result in subgroups:
            scores.append(f1_score(subgroup_gt,result))
        f1s.append(np.max(scores))
    f1 = np.mean(f1s)
    f1_std = np.std(f1s)
    runtime = np.mean(runtimes)
    runtime_std = np.std(runtimes)
    print("N-bins: "+str(bins),"Method: "+args.method,"F1",f1,"Runtime",runtime)
    with open(path.format(args.method),"a") as f:
        writer = csv.writer(f)
        writer.writerow([bins,f1,f1_std,runtime,runtime_std])
        f.close()
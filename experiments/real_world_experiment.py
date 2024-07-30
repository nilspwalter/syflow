import sys
sys.path.append("../")

from src import *
from src.methods import *
from src.demo_utils import setup_path, f1_score, csv, Path
from src.utils.utils import load_data

parser = argparse.ArgumentParser()

parser.add_argument("--method", type=str, default="syflow", help="method to run")
parser.add_argument("--alpha", type=float, default=1, help="alpha")
parser.add_argument("--seed", type=int, default=0, help="seed")

args = parser.parse_args()

np.random.seed(0)

config = Config_Real_World()
config.seed = args.seed

base = setup_path("real_world/seeds/")
path = base + "{}_seed_{}.csv"

rule_base = Path(base+"/rules/").mkdir(parents=True, exist_ok=True)
rule_path = base+"/rules/{}-{}-seed-{}.txt"

sg_base = Path(base+"/subgroups/").mkdir(parents=True, exist_ok=True)
sg_path = base+"/subgroups/{}-{}-seed-{}.npy"

with open(path.format(args.method, config.seed),"w") as f:
    writer = csv.writer(f)
    writer.writerow(["Dataset","Top-WKL","Avg-WKL","Top-Mean","Average-Mean","Top-Std","Average-Std","Top-Coverage","Avg-Coverage","Overlap","Runtime"])
    f.close()

for dataset_name in config.datasets:
    dataset, is_classification = load_data(dataset_name)
    X = dataset["data"]
    Y = dataset["target"]

    if len(Y.shape) == 1:
        Y = Y[:,None]
    feature_names = dataset["feature_names"]

    start = time.time()
    subgroups, rules = run_method(args.method,X.copy(),Y.copy(),args.alpha,config,config.n_subgroups,feature_names)

    end = time.time()
    runtime = end-start
    f2 = open(rule_path.format(args.method,dataset_name, config.seed),"w")
    for rule in rules:
        f2.write(rule+"\n")
    f2.close()

    wkls = [0]
    means = [0]
    stds = [0]
    coverage = [0]
    wds = [0]
    eds = [0]
    tvs = [0]
    for result in subgroups:
        if np.sum(result) == 0:
            wkl_score = 0
        else:
            wkl_score = 0
        wkls.append(wkl_score)
        means.append(np.abs(np.mean(Y[result])-np.mean(Y)))
        stds.append(np.abs(np.std(Y[result])-np.std(Y)))
        coverage.append(np.sum(result)/len(result))

    overlap = evaluate_overlap(subgroups)

    file = sg_path.format(args.method,dataset_name,config.seed)
    np.save(file,np.array(subgroups))

    print("Dataset: "+dataset_name,"Method: "+args.method,"WKL",np.max(wkls),np.mean(wkls),"Overlap",overlap,"TV", np.max(tvs), "WD", np.max(wds), "ED", np.max(eds))
    
    with open(path.format(args.method,config.seed),"a") as f:
        writer = csv.writer(f)
        writer.writerow([dataset_name,np.max(wkls),np.mean(wkls),np.max(means),np.mean(means),np.max(stds),np.mean(stds),np.max(coverage),np.mean(coverage), overlap,runtime])
        f.close()

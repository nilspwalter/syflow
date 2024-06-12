from exp_utils import *


parser = argparse.ArgumentParser()

parser.add_argument("--method", type=str, default="syflow", help="method to run")
parser.add_argument("--alpha", type=float, default=1, help="alpha")

args = parser.parse_args()

config = Config_Parameter_Targetdists()
config_exp = Config_Target_Distribution()

config.seed = 0

base = setup_path("target_distribution")
path = base + "{}.csv"

with open(path.format(args.method),"w") as f:
    writer = csv.writer(f)
    writer.writerow(["Distribution","F1","F1-STD","Runtime","Runtime-STD"])
    f.close()
for dist in config_exp.target_dist:
    feature_names = ["X"+str(i) for i in range(config_exp.n_variables)]

    f1s = []
    runtimes = []

    for rep in range(config_exp.n_reps):
        X,Y, subgroup_gt = gen_synth(config_exp.n_samples, config_exp.n_variables, config_exp.n_conditions, dist,seed=rep)
        start = time.time()
        subgroups, rules = run_method(args.method,X,Y,args.alpha,config,config_exp.n_subgroups,feature_names)
        end = time.time()
        runtimes.append(end-start)
        scores = []
        for result in subgroups:
            scores.append(f1_score(subgroup_gt,result))
        f1s.append(np.max(scores))
        print(np.max(scores))
    f1 = np.mean(f1s)
    f1_std = np.std(f1s)
    runtime = np.mean(runtimes)
    runtime_std = np.std(runtimes)

    print("Distribution: "+dist,"Method: "+args.method,"F1",f1,"Runtime",runtime)
    with open(path.format(args.method),"a") as f:
        writer = csv.writer(f)
        writer.writerow([dist,f1,f1_std,runtime,runtime_std])
        f.close()

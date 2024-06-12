#install.packages("prim")

library(prim)

set.seed(0)


dims = list(10,25,50,100,250,500,750,1000, 1500, 2000)
amount = 0.2
min_value = 0
max_value = 1
num_rows = 10000
n_reps = 5
path = "../clean_code/results/scaling/bh.csv"
#f = file(path,open="w")
write("Dimension,F1,F1-STD,Runtime,Runtime-STD",path)
for(j in 1:length(dims)) {
    
    f1_scores = rep(0, n_reps)
    runtimes = rep(0, n_reps)
    for(d in 1:n_reps){
        starting_time = Sys.time()
        num_cols = dims[[j]]
        data <- read.csv(paste0("../data/synthetic/scaling/dim_", num_cols, "_nc_",4 ,"_seed_",d-1,".csv"))
        X = data[,1:num_cols]
        Y = data[,"Y"]
        condition = data[,"condition"]

        bumps <- prim.box(x=X, y=Y,threshold.type = 1)
        runtime = Sys.time() - starting_time
        runtimes[d] = as.numeric(runtime, units="mins")[1]
        f1 = 0
        for(i in 1:4){
            is_subgroup = predict(bumps, X) == i
            # precision
            if(sum(is_subgroup) > 0){
                precision = sum(is_subgroup & condition) / sum(is_subgroup)
            }else{
                precision = 0
            }
            
            # recall
            recall = sum(is_subgroup & condition) / sum(condition)

            # f1 score
            if((precision + recall) > 0){
                f1_temp = 2 * precision * recall / (precision + recall)
            }else{
                f1_temp = 0
            }
            #print(precision)
            #print(recall)
            #print(f1_temp)
            #print(f1)
            if(f1_temp>f1){
                f1 = f1_temp
            }
        }

        f1_scores[d] = f1

    }
    std_f1 = sd(f1_scores)
    mean_f1 = mean(f1_scores)
    mean_time = mean(runtimes)
    std_time = sd(runtimes)
    print(sprintf("%d And-Clauses, %d Dimensions, %f F1", 4, num_cols, mean_f1))
    path = "../clean_code/results/scaling/bh.csv"
    write(sprintf("%d,%f,%f,%f,%f", num_cols, mean_f1, std_f1, mean_time, std_time),path,append=TRUE)

}





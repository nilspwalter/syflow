set.seed(0)
library(rsubgroup)

args <- commandArgs(trailingOnly = TRUE)
n_conditions <- as.integer(args[1])

dims = list(8,16,32,64,128,256,512,1024)
amount = 0.2
min_value = 0
max_value = 1
num_rows = 3000
n_reps = 30
path = sprintf("results/beamsearch_%d_conjunction.csv",n_conditions)
#f = file(path,open="w")
write("Dim,F1 Score,Std,Runtime",path)

for(j in 1:length(dims)) {
    
    f1_scores = rep(0, n_reps)
    runtime = Sys.time()
    for(d in 1:n_reps){

        
        num_cols = dims[[j]]
        per_variable = amount^(1/n_conditions)
        X = matrix(runif(num_rows * num_cols, min = min_value, max = max_value), nrow = num_rows, ncol = num_cols)
        Y = runif(num_rows, min = min_value, max = max_value)
        


        condition = rep(TRUE, num_rows)
        for(a in 1:n_conditions) {
            threshold_lower = runif(1, min = min_value, max = 1-per_variable)
            threshold_upper = threshold_lower + per_variable
            local_condition = X[,a] > threshold_lower & X[,a] < threshold_upper
            condition = condition & local_condition
        }
        
        Y[condition] = Y[condition] + 0.5


        # make data frame
        df = data.frame(X,Y)
        result1 <- DiscoverSubgroups(
            df, as.target("Y"), new("SDTaskConfig",nbins=10,k=5,maxlen=8,method="beam",qf="ps"),as.df=FALSE)
        # make boolean zero vector
        is_subgroup = rep(FALSE, num_rows)

        for(k in 1:length(result1)) {
            pattern <- result1[[k]]
            for(l in 1:num_rows){
                if(is_subgroup[l]) {
                    next
                }
                data = df[l,]
                # check if x in interval
                
                selectors <- pattern@selectors
                matching <- TRUE
                for (sel in names(selectors)) {
                    

                    value <- data[[sel]]
                    rule <- selectors[[sel]]
                    interval = strsplit(rule, ";")
                    lower_bound = interval[[1]][1]
                    lower_bound = substr(lower_bound,2,nchar(lower_bound))
                    upper_bound = interval[[1]][2]
                    upper_bound = substr(upper_bound,1,nchar(upper_bound)-1)
                    
                    
                    # Function to convert string to numeric considering 'Inf' and '-Inf'
                    convert_to_numeric <- function(x) {
                    if (x == "∞") {
                        return(Inf)
                    } else if (x == "-∞") {
                        return(-Inf)
                    } else {
                        return(as.numeric(x))
                    }
                    }

                    # Convert the extracted numbers to numeric values
                    lower_bound <- convert_to_numeric(lower_bound)
                    upper_bound <- convert_to_numeric(upper_bound)

                    if (value < lower_bound | value >= upper_bound) {
                        matching <- FALSE
                        break
                    }
                }
                if(matching) {
                    is_subgroup[l] = TRUE
                }
            }
            
        }
        # parse thresholds


        # precision
        precision = sum(is_subgroup & condition) / sum(is_subgroup)
        # recall
        recall = sum(is_subgroup & condition) / sum(condition)

        # f1 score 
        f1 = 2 * precision * recall / (precision + recall)
        f1_scores[d] = f1
        # hunt bumps with prim
        # prim1 <- prim.box(x=X, y=Y)

        # print(prim1)
    }
    std_f1 = sd(f1_scores)
    mean_f1 = mean(f1_scores)
    total_time = Sys.time() - runtime
    total_time = as.numeric(total_time, units = "secs")
    total_time = total_time / n_reps
    print(sprintf("%d And-Clauses, %d Dimensions, %f F1", n_conditions, num_cols, mean_f1))
    path = sprintf("results/beamsearch_%d_conjunction.csv",n_conditions)
    #f = file(path,open="a")
    write(sprintf("%d,%f,%f,%f", num_cols, mean_f1, std_f1, total_time),path,append=TRUE)
    #close(f)
}





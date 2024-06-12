library("prim")
for (dataset in c("automobile","airquality","abalone","insurance","wages","mpg","bike","california","wine","student")){
    # read csv
    data <- read.csv(paste0("data/r_data/",dataset,".csv"))
    Y = data[,"target"]
    X= data[,1:(ncol(data)-1)]
    bumps <- prim.box(x=X, y=Y,threshold.type = 1)
    is_subgroup = predict(bumps, X)
    # save results
    path = paste0("clean_code/results/real_world/r_subgroups/prim_sg_",dataset,".csv")
    write.table(is_subgroup,path,sep=",",row.names=FALSE,col.names=FALSE)
}
import numpy as np

target_size = 0.1

def get_subgroup_dist(Y,condition,dist):
    if dist == "normal":
        return np.random.normal(loc=1.5, scale=0.5,size=(np.sum(condition)))
    if dist == "bi_modal":
        mask = np.random.choice(2,size=(np.sum(condition)))
        mode1 = np.random.normal(loc=-1.5, scale=0.5,size=(np.sum(condition)))*mask
        mode2 = np.random.normal(loc=1.5, scale=0.5,size=(np.sum(condition)))*(1-mask)
        return mode1+mode2
    if dist == "beta":
        return np.random.beta(a=0.2,b=0.2,size=(np.sum(condition)))*1.2
    if dist == "rayleigh":
        return np.random.rayleigh(scale=2, size=(np.sum(condition)))
    if dist == "uniform":
        return np.random.uniform(low=0.5, high=1.5, size=(np.sum(condition)))
    if dist == "exponential":
        return np.random.exponential(scale=0.5,size=(np.sum(condition)))
    if dist=="cauchy":
        return np.random.standard_cauchy(size=(np.sum(condition)))
    

def gen_synth(n_samples, dim, n_conditions, target_dist, seed = 0):
    print(target_dist)
    assert target_dist in ["rayleigh","cauchy","normal", "bi_modal", "beta", "uniform", "exponential"]
    assert n_conditions <= dim
    
    np.random.seed(seed)
    
    variables = np.random.choice(dim,n_conditions,replace=False)
    interval_size = target_size**(1/n_conditions)
    intervals = np.zeros((n_conditions,2))
    
    for i in range(n_conditions):
        start = np.random.uniform(0,1-interval_size)
        intervals[i,0] = start
        intervals[i,1] = start+interval_size
    X = np.random.uniform(0,1,size=(n_samples,dim))
    # resample the subgroup and non subgroup variables
    X_subgroup = np.zeros((n_samples,n_conditions))
    for i in range(n_conditions):
        X_subgroup[:,i] = np.random.uniform(intervals[i,0],intervals[i,1],size=(n_samples,))
    # resample data strictly outside the intervals
    X_non_subgroup = np.random.uniform(0,1,size=(n_samples,n_conditions))
    in_box = np.ones((n_samples,),dtype=bool)
    for i in range(n_conditions):
        in_box = np.logical_and(in_box,np.logical_and(X_non_subgroup[:,i]>=intervals[i,0],X_non_subgroup[:,i]<intervals[i,1]))
    # resample subgroup and non subgroup variables
    while np.sum(in_box) > 0:
        X_non_subgroup[in_box,:] = np.random.uniform(0,1,size=(np.sum(in_box),n_conditions))
        for i in range(n_conditions):
            in_box = np.logical_and(in_box,np.logical_and(X_non_subgroup[:,i]>=intervals[i,0],X_non_subgroup[:,i]<intervals[i,1]))
    subgroup_labels = np.random.choice(2,size=(n_samples,),p=[1-target_size,target_size])
    subgroup_labels = subgroup_labels.astype(bool)

    for i,v in enumerate(variables):
        X[subgroup_labels,v] = X_subgroup[subgroup_labels,i]
        X[~subgroup_labels,v] = X_non_subgroup[~subgroup_labels,i]

    Y = np.random.uniform(size=(n_samples,))
    Y_subgroup = get_subgroup_dist(0,subgroup_labels,target_dist)
    Y[subgroup_labels] = Y_subgroup
    Y = Y[:,None]

    return X,Y,subgroup_labels


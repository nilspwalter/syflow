import flowtorch.bijectors as bij
class Config_Real_World:
    def __init__(self):
        self.lr_flow = 5e-2
        self.lr_classifier = 2e-2
        self.alpha = 0.3
        self.lambd = 2.0
        self.pop_train_epochs = 1000
        self.subgroup_train_epochs = 1000
        self.final_fit_epochs = 0
        self.temperature = 0.2
        self.bin_deviation = 0.2
        self.use_weights = True
        # subgroup discovery parameters
        self.ncutpoints = 20
        self.beam_width = 100
        self.sd_depth = 8
        def flow_gen():
            return bij.Spline(count_bins=12)
        self.flow_gen = flow_gen
        self.n_subgroups = 5
        self.datasets = ["automobile","airquality","abalone","insurance","wages","mpg","bike","california","wine","student"]


class Config_Gold:
    def __init__(self):
        self.lr_flow = 5e-2
        self.lr_classifier = 2e-2
        self.alpha = 0.2
        self.lambd = 10
        self.pop_train_epochs = 7000
        self.subgroup_train_epochs = 3000
        self.final_fit_epochs = 0
        self.temperature = 0.2
        self.bin_deviation = 0.2
        self.use_weights = True
        # subgroup discovery parameters
        self.ncutpoints = 20
        self.beam_width = 100
        self.sd_depth = 8
        self.n_subgroups = 5
        def flow_gen():
            return bij.Compose([bij.Spline(count_bins=12), bij.Spline(count_bins=12)])
            #return bij.Spline(count_bins=12)
        self.flow_gen = flow_gen

class Config_Higgs:
    def __init__(self):
        self.lr_flow = 5e-2
        self.lr_classifier = 2e-2
        self.alpha = 0.2
        self.lambd = 20
        self.pop_train_epochs = 5000
        self.subgroup_train_epochs = 5000
        self.final_fit_epochs = 0
        self.temperature = 0.2
        self.bin_deviation = 0.2
        self.use_weights = True
        # subgroup discovery parameters
        self.ncutpoints = 20
        self.beam_width = 100
        self.sd_depth = 8
        self.n_subgroups = 5
        def flow_gen():
            return bij.Compose([bij.Spline(count_bins=12), bij.Spline(count_bins=12), bij.Spline(count_bins=12)])
        self.flow_gen = flow_gen

conf_dict = {"gold":Config_Gold(), "higgs": Config_Higgs(),"vdw":Config_Gold() }

class Config_Synthetic:
    def __init__(self):
        self.lr_flow = 5e-2
        self.lr_classifier = 2e-2
        self.alpha = 0.5
        self.lambd = 1
        self.pop_train_epochs = 2000
        self.subgroup_train_epochs = 1500
        self.final_fit_epochs = 0
        self.temperature = 0.2
        self.bin_deviation = 0.2
        self.use_weights = True
        # subgroup discovery parameters
        self.ncutpoints = 20
        self.beam_width = 100
        self.sd_depth = 8
        self.seed = 0
        def flow_gen_default():
            return bij.Spline(count_bins=8)
        self.flow_gen = flow_gen_default

class Config_Synthetic_Hyperparameters:
    def __init__(self):
        self.lr_flow = 5e-2
        self.lr_classifier = 2e-2
        self.alpha = 0.5
        self.alphas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
        self.lambds = [0,0.25,0.5,1.0,2.0,3.0]
        self.temperatures = [0.01,0.1,0.2,0.3,0.4,0.5]
        self.lambd = 1
        self.pop_train_epochs = 2000
        self.subgroup_train_epochs = 1500
        self.final_fit_epochs = 0
        self.temperature = 0.2
        self.bin_deviation = 0.2
        self.use_weights = True
        # subgroup discovery parameters
        self.ncutpoints = 20
        self.beam_width = 100
        self.sd_depth = 8
        def flow_gen_default():
            return bij.Spline(count_bins=8)
        self.flow_gen = flow_gen_default

class Config_Parameter_Targetdists:
    def __init__(self):
        self.lr_flow = 5e-2
        self.lr_classifier = 2e-2
        self.alpha = 0.5
        self.lambd = 1
        self.pop_train_epochs = 2000
        self.subgroup_train_epochs = 1500
        self.final_fit_epochs = 0
        self.temperature = 0.2
        self.use_weights = True
        self.bin_deviation = 0.2
        # subgroup discovery parameters
        self.ncutpoints = 20
        self.beam_width = 100
        self.sd_depth = 8

class Config_Scaling:
    def __init__(self):
        # experiment parameters
        self.n_samples = 20000
        self.n_reps = 1
        self.n_conditions = 4
        self.n_variables = [10,25,50,100] # 
        self.target_dist = "normal"
        self.n_subgroups = 3
    
class Config_Rule_Complexity:
    def __init__(self):
        # experiment parameters
        self.n_samples = 20000
        self.n_reps = 5
        self.n_conditions = [2,4,6,8,10]
        self.n_variables = 100
        self.target_dist = "normal"
        self.n_subgroups = 3

class Config_Target_Distribution:
    def __init__(self):
        # experiment parameters
        self.n_samples = 20000
        self.n_reps = 2
        self.n_conditions = 4
        self.n_variables = 10
        self.target_dist = ["rayleigh","cauchy","normal","uniform","beta","exponential","bi_modal"]
        self.n_subgroups = 3

class Config_Binning:
    def __init__(self):
        # experiment parameters
        self.n_samples = 20000
        self.n_reps = 5
        self.n_conditions = 4
        self.n_variables = 50
        self.target_dist = "normal"
        self.n_subgroups = 3
        self.bins = [2,5,10,20,30,40,50]
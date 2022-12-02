# Imports
import numpy as np

# -----------------------------------------------------------------------------
# Model Parameters
# -----------------------------------------------------------------------------

# Implemented options: inception, fcn, simple_transformer


# mlp options
# ~ 420,000 parameters
model_params = {
    "model_type" : "mlp", 
    "layers" : [312, 256, 256, 128]
}


# ----------------------------------------------------------------------------- 
# Data Parameters
# ----------------------------------------------------------------------------- 

# Analytes Names 
analytes_names = ['DA', '5HT', 'pH', 'NE'] 

# Where to look for the probes and what they are called 
data_folders = [['/mnt/nfs/proj/in-vitro/iterate/results_012/model_style_008/training-A-10Hz-tri/*CF*/', '/mnt/nfs/proj/in-vitro/iterate/results_012/model_style_008/post-surg-10Hz-tri/*CF*/', '/mnt/nfs/proj/in-vitro/iterate/results_011/model_style_008/training-A-10Hz-tri/*CF*/', '/mnt/nfs/proj/in-vitro/Tom/data/testing_3/cf/400Vs/10Hz/*CF*/'],]

probes = [['CF175','CF172','CFO_CCDBS10_2','CF184','CF066','CF081','CF_MSDBS277','CF206','CFO_CCDBS18_2','CF153','CF057','CF094','CFO_CCDBS021','CF168','CF163','CF_MSDBS287','CF190','CF169','CF_MSDBS292','CF_MSDBS282_2','CF_MSDBS307','CF_MSDBS290_2','CF214','CF160','CF099','CF_MSDBS286_2','CF152','CF_MSDBS292_2','CF205','CF_CCDBS010','CF174','CF027','CF119','CF_MSDBS297','CFO_CCDBS025','CF198','CF_MSDBS288','CFO_CCDBS017','CF124','CF189','CF120','CF064','CF161','CF147','CF_MSDBS289_2','CF_MSDBS294','CF188','CF_MSDBS295','CF197','CF_MSDBS286','CF_MSDBS280_2','CF141','CF200','CF_MSDBS283','CF131','CF207','CF_MSDBS282','CF025','CF_MSDBS284','CF150','CF078','CF_MSDBS296','CF173','CF165','CF_MSDBS285','CF103','CF_MSDBS289','CF181','CFO_CCDBS018','CF186','CF177','CF_MSDBS280','CF082','CF_MSDBS290','CF166','CF_MSDBS274_2']]
all_probes = sum(probes, [])
val_probes = []

# Create test_sets
test_sets = [[] for i in range(10)] # a place to store them
shuffled_probes = np.random.default_rng(42).permuted(all_probes)
for i, p in enumerate(shuffled_probes):
    test_sets[i%10].append(p)

fold = 1
test_probes = test_sets[fold]

# Where to save the model
output_directory = f'/mnt/nfs/proj/in-vitro/Tom/dev/trained_models/cf/npd/76_PE/10-F-CV/fold_{fold}/models/{model_params["model_type"]}_big/' # Reference implementation

dataset_directory = f'/mnt/nfs/proj/in-vitro/Tom/dev/trained_models/cf/npd/76_PE/10-F-CV/fold_{fold}/datasets/'

# Portion of data to be in training set vs validation set
# Ignored if val_probes is not none or empty list 
data_split = 0.9

# Number of records per UNIQUE CONCENTRATIONS per probe
# -1 specifies all avalible
n_records_per_probe = 150

# k_fold parameters
k_fold_cv = False
num_folds = -1

# Specify y-normalization strategy
norm = 'Shift-Z'

sample_indices = None

# Select label input range
trim = 2500


# -----------------------------------------------------------------------------
# Training Parameters
# -----------------------------------------------------------------------------

# How many 'repetitions' of the model to train
# Refers to model with same train/val sets
repetitions = 1

# Uses different train/val split for repetitions if using data_split parameter
unique_repetitions = True

# Patience for decreasing learning rate
patience = 5

#Set the minimum learing rate (None for default)
min_lr = None

# Specify number of epochs
epochs = 35
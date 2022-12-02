"""
parametric_fit_function.py

Thomas Twomey
Montague Lab
Fralin Biomedical Institute at VTC
09/03/2021

Modified:
11/20/2022
"""

# Imports
import sys
import os
import argparse
import numpy as np
import traceback
import datetime
import subprocess
import tensorflow as tf
from scipy import stats
import pathlib

# Import to allow for dynamic importing for param file
import importlib
import importlib.util

# Find project folder, this might be able to be done with parameter file
project_folder = os.path.abspath(__file__)
project_folder = project_folder[:project_folder[:project_folder.rfind('/')].rfind('/')+1]
sys.path.append(project_folder)

# Local Imports
from data_recorder.data_recorder import create_shard_dir, append_shard_dir, _spec_file_exists, _load_spec_file, load_dataset_from_dir, create_or_append_shard_dir
from data_loader.data_loader import FilePairDataLoader
from data_slicer.data_slicer import TrainValTestDataSlicer
from utils.model_metadata import ModelMetadata
from utils.data_metadata import DataMetadata
from utils.metrics import shifted_zscore_factory, inverse_shifted_zscore_factory, tf_pmse_factory, tf_mean_pmse_factory, multiply_by_mean_factory, divide_by_mean_factory
from classifiers import inception_tfrecord, fcn_tfrecord, simple_transformer_tfrecord, ssvep_former_tfrecord, fb_ssvep_former_tfrecord, eeg_transformer_reduce_tfrecord, mlp_tfrecord, mlp_dropout_tfrecord, fcn_1d_tfrecord

# Define the models that we can source from
model_sources = {
    "inception" : inception_tfrecord, 
    "fcn" : fcn_tfrecord, 
    "fcn_1d" : fcn_1d_tfrecord,
    "simple_transformer" : simple_transformer_tfrecord,
    "ssvep_former" : ssvep_former_tfrecord,
    "fb_ssvep_former" : fb_ssvep_former_tfrecord,
    # "eeg_transformer" : eeg_transformer_tfrecord,
    "eeg_transformer_reduce" : eeg_transformer_reduce_tfrecord,
    "mlp" : mlp_tfrecord,
    "mlp_dropout" : mlp_dropout_tfrecord
    }

###############################################################################
# Define some Functions
###############################################################################

# Error handling
class CLIError(Exception):
    """Generic exception to raise and log different fatal errors."""

    def __init__(self, msg):
        super(CLIError).__init__(type(self))
        self.msg = f'error: {msg}'

    def __str__(self):
        return self.msg

    def __unicode__(self):
        return self.msg

class Logger(object):
    """
    Logger that redirects std_out to enable print to be dumped to a file and std_out.

    Taken from:
    https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
    """

    def __init__(self, output_file):
        self.terminal = sys.stdout
        # Setup output directory if needed
        os.makedirs(pathlib.Path(output_file).parent, exist_ok=True)
        self.log = open(output_file, "a")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def descriptive_model_path(output_directory, patience, epochs):
    """
    Making this a function so it can be changed from a central location
    """

    return f"{output_directory}P{patience}/E{epochs}/"

def load_module_from_path(input_path, name='params'):
    """
    Loads a module from a full path location.
    """

    spec = importlib.util.spec_from_file_location(name, input_path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    
    return foo

def regulate_parameter_file(input_path, return_out_path=False):
    """
    The goal of this function is to read the parameter file (a .py) in as text.
    It will then add:
        - Any missing variables/parameters (With Warnings)
        - The time that the file is being executed
        - The hash of the latest git commit
        - The computer that it is being run on. 
    """

    def gen_runtime_info():
        """
        Gets the git hash of the most recent commit, the computer, and the datetime
        """

        host = os.uname()[1] # Gets the computer that python is running on
        dt = str(datetime.datetime.now())
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

        hash_line = "################################################################################"

        out = hash_line + "\n# File run on host: " + host + "\n# At date/tim: " + dt + "\n# Most recent GIT commit: " + git_hash + "\n" + hash_line + "\n\n"
        return out

    def existing_param_search(input_path):
        """
        Loads the module from input_path
        determines if any parameters are missing,
        for each missing parameter:
            if no default exists raises error
            else adds parameter assigned to default

        returns:
            new_def_param_block : str
                 string to be added to parameter file
            temp_param_dict : dict 
        """

        # Dictionary to be updated and returned
        temp_param_dict = {}

        # Load the module to find the attributes that are needed
        params = load_module_from_path(input_path, "params")
        
        # Required parameters that cannot be reasonably assigned a default value
        req_params = [
            'output_directory', 
            'data_folders', 
            'probes', 
            'analytes_names'
        ]

        # Find if any of the required parameters are missing
        missing_params = [p for p in req_params if not hasattr(params, p)]

        for p in req_params:
            # Add all the required params to the temp_param_dict
            temp_param_dict[p] = getattr(params, p)

        # If any of the required parameters are missing, generate a error message and raise the error.
        if len(missing_params) > 0:
            raise ValueError(f"The parameter file is missing the parameter{'s' if len(missing_params)>1 else ''}: {' '.join(missing_params)}")
        
        # Parameters that can be reasonably assigned a default value
        # The idea here is to maintain backwards compatibility as new features are added.
        # As old defaults need to be specified, they are placed here.
        def_param_dict = {
            'val_probes' : None,
            'test_probes' : None,
            'patience' : 5,
            'min_lr' : None,
            'epochs' : 100,
            'repetitions' : 5,
            'unique_repetitions' : True,
            'n_records_per_probe' : -1,
            'data_split' : 0.9,
            'num_folds' : -1,
            'k_fold_cv' : False,
            'norm' : "'Shift-Z'",
            'trim' : None,
            'sample_indices' : None,
        }

        # Find if we need to add any default values
        additions = []
        for key in def_param_dict.keys():
            if not hasattr(params, key):
                # In this case we are missing a parameter that we have a default value for.
                # We can thus add it, but we also need to be loud when we do and warn those 
                # who are listening
                default = def_param_dict[key]
                additions.append(f"{key} = {default}")
                print(f"Warning: {key} missing from parameter file. Assigned to default value of {default}")
                temp_param_dict[key] = default
            else:
                temp_param_dict[key] = getattr(params, key)

        
        # Generate a block of text with the default values for the missing parameters
        if len(additions) >0:
            # Add a block of text
            hash_line = "################################################################################"
            new_def_param_block = hash_line + "\n# The following block is parameters that were automatically added with default values:\n" + '\n'.join(additions) + "\n" + hash_line + "\n\n"
        else:
            new_def_param_block = ""

        return new_def_param_block, temp_param_dict

    with open(input_path, 'r') as f:
        f_string = f.read() # Read all the lines

        # Search for all the parameters that are needed and construct the block to be added.
        def_param_block, temp_param_dict = existing_param_search(input_path)

        # Construct our block of run-time information
        run_time_block = gen_runtime_info()

        # Add our blocks of info to the top
        f_string = run_time_block+def_param_block+f_string

        # Determine the output_path from the f_string
        dmp = descriptive_model_path(temp_param_dict["output_directory"], temp_param_dict["patience"], temp_param_dict["epochs"])
        output_path = f"{dmp}/{pathlib.Path(input_path).name}"

        # Setup output_path if needed
        os.makedirs(pathlib.Path(output_path).parent, exist_ok=True)

        # Save the modified file to the output_path if it does not already exist
        if not os.path.exists(output_path):
            print(f"Parameter file generated and saved to output directory: {output_path}")
            with open(output_path, 'w') as f_o:
                f_o.write(f_string)
        else:
            print(f"Parameter file found and loaded from output directory: {output_path}")

    if return_out_path:
        return load_module_from_path(output_path), dmp

    return load_module_from_path(output_path)

def format_dir(dir):
    """
    Appends a trailing "/" if not present in the directory path and returns the modified version
    """

    if dir is None or len(dir)<2:
        raise ValueError(f"Cannot format directory: {dir}")
    else:
        if dir[-1] != "/":
            dir.append("/")
    return dir
    
def within_sweep_normalize(data_dict):
    """
    This function has fallen out of favor.

    Applies a within-sweep zscore

    args:
        data_dict : dict
            A dictionary with the key being the probe name and the the value 
            being a tuple (sweeps, labels)

    returns:
        data_dict (modified)
    """

    for probe in data_dict.keys():
        sweeps = data_dict[probe][0]
        sweeps = stats.zscore(sweeps, axis=1)
        labels = data_dict[probe][1]
        data_dict[probe] = (sweeps, labels)
    return data_dict

def maintain_150_sweeps_per_label(data_dict, n_records_per_probe):
    """
    This function has fallen out of favor

    Duplicates the data such that there are 150 sweeps per label per probe

    args:
        data_dict : dict
            A dictionary with the key being the probe name and the the value 
            being a tuple (sweeps, labels)
        n_record_per_probe : int
            The number of sweeps that currently exist for each label

    returns:
        data_dict (modified)
    """

    for probe in data_dict.keys():
        sweeps = data_dict[probe][0]
        sweeps = np.repeat(sweeps, 150/n_records_per_probe, axis=0)
        labels = data_dict[probe][1]
        labels = np.repeat(labels, 150/n_records_per_probe, axis=0)
        data_dict[probe] = (sweeps, labels)
    return data_dict

def slice_single_probe_k_fold(data_dict, all_probes, test_probes, val_ratio, num_folds, val_set_index=0, fold_index=0):
    """
    Wraps the TrainValDataSlicer function k_fold_single_probe

    args:
        data_dict : dict
            A dictionary with the key being the probe name and the the value 
            being a tuple (sweeps, labels)
        all_probes : list of str
            A list of all the probes
        test_probes : list of str
            A list of the test probes
        val_ratio : float
            The ratio of unique concentrations to be used as the validation set
        num_folds : int
            The number of folds to create (The k in k-fold)
        val_set_index : int (Default: 0)
            If using an ensemble which index in the ensemble is this? 
            Used to select a different train/val split.
        fold_index : int (Default: 0)
            Which fold to create the datasets for.

    returns:
        ((x_train, y_train), (x_val, y_val), (x_test, y_test))
    """

    # Load the general purpose data slicer
    ds = TrainValTestDataSlicer()

    train_probes = list((set(all_probes)-set(test_probes)))

    return ds.k_fold_single_probe(data_dict, train_probes[0], val_ratio, num_folds, val_set_index, fold_index)

def slice_all_val_test_by_probe(data_dict, all_probes, val_probes, test_probes, verbose=True):
    """
    Wraps the TrainValDataSlicer function train_val_test_by_probe

    Uses all_probes, val_probes and test_probes to determine train_probes and then calls train_val_test_by_probe

    args:
        data_dict : dict
            A dictionary with the key being the probe name and the the value 
            being a tuple (sweeps, labels)
        all_probes : list of str
            A list of all the probes
        val_probes : list of str
            A list of the val probes
        test_probes : list of str
            A list of the test probes
        verbose : Boolean
            Unused

    returns:
        ((x_train, y_train), (x_val, y_val), (x_test, y_test))
    """

    # Load the general purpose data slicer
    ds = TrainValTestDataSlicer()

    train_probes = (set(all_probes)-set(test_probes))-set(val_probes)
    
    if verbose:
        print(f"Train Probes: {train_probes}")
        print(f"Val Probes: {val_probes}")
        print(f"Test Probes: {test_probes}")
    
    return ds.train_val_test_by_probe(data_dict, train_probes, val_probes, test_probes)
    
def slice_all_test_by_ratio(data_dict, all_probes, test_probes, val_ratio, val_set_index=0, verbose=True):
    """
    Wraps the TrainValDataSlicer function train_val_test_by_probe_stratified

    Uses all_probes, test_probes and val_ratio to create train, val and test sets

    args:
        data_dict : dict
            A dictionary with the key being the probe name and the the value 
            being a tuple (sweeps, labels)
        all_probes : list of str
            A list of all the probes
        test_probes : list of str
            A list of the test probes
        val_ratio : float
            The ratio of unique concentrations to be used as the validation set
        val_set_index : int (Default: 0)
            If using an ensemble which index in the ensemble is this? 
            Used to select a different train/val split.
        verbose : boolean

    returns:
        ((x_train, y_train), (x_val, y_val), (x_test, y_test))
    """

    # Load the general purpose data slicer
    ds = TrainValTestDataSlicer()
    test_probes = test_probes if not None else []
    train_val_probes = list(set(all_probes)-set(test_probes))
    if verbose:
        print(f"Train/Val Probes: {train_val_probes}")
        print(f"Test Probes: {test_probes}")
    return ds.train_val_test_by_probe_stratified(data_dict, train_val_probes, test_probes, val_ratio, val_set_index)

def load_local_datasets(local_dataset_dir):
    """
    Load the data from local cache.
    Note the assumption that if the output directory has not been changed, the datasets have not been changed.
    Also assumes that test data exists in pairs of x and y.

    args:
        local_dataset_dir : str
            Where to load the data from.
    returns:
        ((x_train, y_train), (x_val, y_val), (x_test, y_test))
        Value set to None if not present
    """

    x_train = np.load(f"{local_dataset_dir}x_train.npy", allow_pickle=True)
    y_train = np.load(f"{local_dataset_dir}y_train.npy", allow_pickle=True)
    if not os.path.isfile(local_dataset_dir+"x_val.npy"):
        x_val = None
        y_val = None
    else:
        x_val = np.load(f"{local_dataset_dir}x_val.npy", allow_pickle=True)
        y_val = np.load(f"{local_dataset_dir}y_val.npy", allow_pickle=True)
    if not os.path.isfile(local_dataset_dir+"x_test.npy"):
        x_test = None
        y_test = None
    else:
        x_test = np.load(f"{local_dataset_dir}x_test.npy", allow_pickle=True)
        y_test = np.load(f"{local_dataset_dir}y_test.npy", allow_pickle=True)
    print('Data loaded from cached local copy.\nWarning: Will not reflect any changes to arguments if provided.')

    return x_train, y_train, x_val, y_val, x_test, y_test

def save_local_datasets(local_dataset_dir, x_train, y_train, x_val, y_val, x_test, y_test):
    """
    Save data locally for posterity and error tracing

    args:
        local_dataset_dir : str
            The directory to save the datasets in.
        x_train : np.ndarray
            The x training data
        y_train : np.ndarray
            The y training data. If None, nothing is saved.
        x_val : np.ndarray
            The x validation data If None, nothing is saved.
        y_val : np.ndarray
            The y validation data. If None, nothing is saved.
        x_test : np.ndarray
            The x test data. If None, nothing is saved.

    """

    np.save(f"{local_dataset_dir}x_train.npy", x_train)
    np.save(f"{local_dataset_dir}y_train.npy", y_train)
    if x_val is not None:
        np.save(f"{local_dataset_dir}x_val.npy", x_val)
        np.save(f"{local_dataset_dir}y_val.npy", y_val)
    if x_test is not None:
        np.save(f"{local_dataset_dir}x_test.npy", x_test)
        np.save(f"{local_dataset_dir}y_test.npy", y_test)
    print('Data loaded from source folder and saved to local cache')


def dict_trim(data_dict, trim):
    """
    Trim dict with tuple (low, high) or just high value
    """
    dl = FilePairDataLoader()
    # New workflow, allow trim to be a tuple with (low,high),
    # if only a value treated as high
    # Trim the data as specified in the parameters
    if isinstance(trim, tuple):
        if len(trim) == 1:
            h_val = trim[0]
        else:
            # Assume this is the case with (low,high)
            l_val, h_val = trim
        if h_val is not None and h_val > 0:
            data_dict = dl.trim(data_dict, h_val)
        if l_val is not None and l_val > 0:
            data_dict = dl.trim_low(data_dict, l_val)
    else:
        # Just high value case
        if trim is not None and trim > 0:
            data_dict = dl.trim(data_dict, trim)

    return data_dict

def flatten_list(nd_list):
    """
    For nested lists, returns a flattened list
    """

    working_list = nd_list

    while isinstance(working_list[0], list) and not isinstance(working_list[0], str):
        working_list = [item for sublist in working_list for item in sublist]
    
    return working_list

def create_load_modify_dict(data_folders, probes, n_records_per_probe, analytes_names, trim, sample_indices):
    # Setup Data-Loader to use standard file pattern
    dl = FilePairDataLoader()

    # Load the data
    data_dict = {}

    # data_folder is a list of lists
    # probes is a parallel list of lists with probes corresponding to data_folder

    data = dl.load(folders = data_folders, probes = probes, n_records_per_label_per_probe=n_records_per_probe, analyte_names=analytes_names)
    data_dict.update(data)

    data_dict = dict_trim(data_dict, trim)

    # Subsample as specified in the parameters
    if sample_indices is not None:
        data_dict = dl.subsample_dict(data_dict, sample_indices)

    # Take the diff
    data_dict = dl.dict_diff(data_dict)

    # Reshape the dict
    data_dict = dl.reshape_dict(data_dict)

    return data_dict

def gen_norm_funcs_and_metrics(norm, label_mean, label_std, analytes_names):
    """ 
    Normalization functions for y values
    Used in classifier
    """
    if "shift" in norm.lower() and 'z' in norm.lower():
        normalize_data = shifted_zscore_factory(label_mean, label_std)
        revert_data = inverse_shifted_zscore_factory(label_mean, label_std)

        # Build the metrics with the metric factory
        # tensorflow projected root mean squared error
        model_metrics = tf_pmse_factory(analytes_names, label_std)

    elif "frac" in norm.lower() or "divide" in norm.lower():
        normalize_data = divide_by_mean_factory(label_mean)
        revert_data = multiply_by_mean_factory(label_mean)
        model_metrics = tf_mean_pmse_factory(analytes_names, label_std)
    
    else:
        raise ValueError("Norm must be a valid string.")
    
    return normalize_data, revert_data, model_metrics

def gen_tfrecord_norm_funcs(labels, norm, analytes_names):
    """
    Calculate the mean and STD of the whole pool.
    """
    labels_sample_mean = np.mean(labels, axis=0)
    labels_sample_std = np.std(labels, axis=0)

    # We need to save these function for posterity
    return gen_norm_funcs_and_metrics(norm, labels_sample_mean, labels_sample_std, analytes_names)

def create_test_tf_dataset(local_dataset_dir, data_folders, test_probes, n_records_per_probe, analytes_names, trim, sample_indices, tfrecord_norm_func, max_rows_in_pool, shard_rows):
    """
    
    """

    data_folders = flatten_list(data_folders)

    print(f"Create_test.. data_folders: {data_folders}")

    test_pool_sweeps = None
    test_pool_labels = None

    for tp in test_probes:
        test_data_dict = create_load_modify_dict(data_folders, [tp], n_records_per_probe, analytes_names, trim, sample_indices)

        # Load the general purpose data slicer
        ds = TrainValTestDataSlicer()
        test_sweeps, test_labels = ds.pool_probes(test_data_dict, [tp])

        print(f"Shape of test_sweeps: {test_sweeps.shape}")
        print(f"Shape of test_labels: {test_labels.shape}")

        if test_pool_sweeps is None:
            test_pool_sweeps = test_sweeps
            test_pool_labels = test_labels
        else:
            test_pool_sweeps = np.append(test_pool_sweeps, test_sweeps, axis=0)
            test_pool_labels = np.append(test_pool_labels, test_labels, axis=0)

        if test_pool_sweeps.shape[0] > max_rows_in_pool:

            print(f"Shape of test_pool_sweeps: {test_pool_sweeps.shape}")
            print(f"Shape of test_pool_labels: {test_pool_labels.shape}")

            shard_test_pool_sweeps = test_pool_sweeps[:int(shard_rows)]
            shard_test_pool_labels = test_pool_labels[:int(shard_rows)]

            print(f"Shape of shard_test_pool_labels: {shard_test_pool_labels.shape}")

            normed_shard_test_pool_labels = tfrecord_norm_func(shard_test_pool_labels)

            test_pool_sweeps = test_pool_sweeps[int(shard_rows):]
            test_pool_labels = test_pool_labels[int(shard_rows):]
            
            create_or_append_shard_dir(f"{local_dataset_dir}/test_dataset/", shard_test_pool_sweeps, normed_shard_test_pool_labels)

    create_or_append_shard_dir(f"{local_dataset_dir}/test_dataset/", test_pool_sweeps, tfrecord_norm_func(test_pool_labels))

def create_val_tf_dataset(local_dataset_dir, data_folders, val_probes, n_records_per_probe, analytes_names, trim, sample_indices, tfrecord_norm_func):
    """
    """

    # Load the general purpose data slicer
    ds = TrainValTestDataSlicer()

    data_folders = flatten_list(data_folders)

    # in this case the val probes can be put in a tfrecord on their own
    val_data_dict = create_load_modify_dict(data_folders, val_probes, n_records_per_probe, analytes_names, trim, sample_indices)

    val_sweeps, val_labels = ds.pool_probes(val_data_dict, val_probes)

    normed_val_labels = tfrecord_norm_func(val_labels)

    create_shard_dir(f"{local_dataset_dir}/val_dataset/", val_sweeps, normed_val_labels)

def create_train_val_tf_dataset(local_dataset_dir, data_folders, train_probes, n_records_per_probe,
                                analytes_names, trim, sample_indices, tfrecord_norm_func, 
                                max_rows_in_pool, shard_rows, data_split, use_val_probes, norm, 
                                repetitions_unique, repetition):
    """
    
    """

    print(f"Create_train_val.. probes: {train_probes}")

    data_folders = flatten_list(data_folders)

    print(f"Create_train_val.. data_folders: {data_folders}")

    # Y Norming Placeholders
    tfrecord_norm_func = None
    tfrecord_revert_func = None
    tfrecord_model_metrics = None

    # Load the general purpose data slicer
    ds = TrainValTestDataSlicer()

    # The pool that is filled as probes are loaded and randomly partially emptied when it is full
    train_pool_sweeps = None
    train_pool_labels = None
    val_pool_sweeps = None
    val_pool_labels = None

    # Export the shapes
    input_shape = None
    output_shape = None

    # Friendly Random Number Generator
    rng = np.random.default_rng(42)

    for p in train_probes:

        probe_data_dict = create_load_modify_dict(data_folders, [p], n_records_per_probe, analytes_names, trim, sample_indices)

        if not use_val_probes:
            # Split the probe into a train and val sets
            if repetitions_unique:
                (p_train_sweeps, p_train_labels), (p_val_sweeps, p_val_labels), (_, _) = slice_all_test_by_ratio(probe_data_dict, [p], test_probes=[], val_ratio=(1-data_split), val_set_index=repetition)
            else:
                (p_train_sweeps, p_train_labels), (p_val_sweeps, p_val_labels), (_, _) = slice_all_test_by_ratio(probe_data_dict, [p], test_probes=[], val_ratio=(1-data_split), val_set_index=repetition)

        else:
            (p_train_sweeps, p_train_labels) = ds.pool_probes(probe_data_dict, [p])

        output_shape = p_train_labels.shape[1]
        input_shape = p_train_sweeps.shape[1:]

        # Stack the sets into the pool
        if train_pool_sweeps is None:
            train_pool_sweeps = p_train_sweeps
            train_pool_labels = p_train_labels
            if not use_val_probes:
                val_pool_sweeps = p_val_sweeps
                val_pool_labels = p_val_labels
        else:
            train_pool_sweeps = np.append(train_pool_sweeps, p_train_sweeps, axis=0)
            train_pool_labels = np.append(train_pool_labels, p_train_labels, axis=0)
            if not use_val_probes:
                val_pool_sweeps = np.append(val_pool_sweeps, p_val_sweeps, axis=0)
                val_pool_labels = np.append(val_pool_labels, p_val_labels, axis=0)

        if train_pool_sweeps.shape[0] > max_rows_in_pool:
            # Randomly select indices from here remove from memory and save to a record
            # We want the probes and concentrations to be randomly distributed across the
            # shards because of the way it is loaded for stochastic mini-batch gradient 
            # decent.
            indices = rng.integers(low=0, high=train_pool_sweeps.shape[0], size=int(shard_rows))

            # In the tfrecord case, to avoid loading and writing the dataset twice, we use 
            # a sample, the first full "pool", to determine a mean and std. Using those 
            # values we transform the rest of the data as it is written out.
            if tfrecord_norm_func is None:

                tfrecord_norm_func, tfrecord_revert_func, tfrecord_model_metrics = gen_tfrecord_norm_funcs(train_pool_labels, norm, analytes_names)

            shard_train_pool_sweeps = train_pool_sweeps[indices]
            shard_train_pool_labels = train_pool_labels[indices]

            # Apply the norm function
            normed_shard_train_pool_labels = tfrecord_norm_func(shard_train_pool_labels)

            train_pool_sweeps = np.delete(train_pool_sweeps, indices, axis=0)
            train_pool_labels = np.delete(train_pool_labels, indices, axis=0)

            create_or_append_shard_dir(f"{local_dataset_dir}/train_dataset/", shard_train_pool_sweeps, normed_shard_train_pool_labels, extendable=True)

        if val_pool_sweeps.shape[0] > max_rows_in_pool:
            # Dump a portion of the indices to a record. Does not have to be random.
            shard_val_pool_sweeps = val_pool_sweeps[:int(shard_rows)]
            shard_val_pool_labels = val_pool_labels[:int(shard_rows)]

            normed_shard_val_pool_labels = tfrecord_norm_func(shard_val_pool_labels)

            val_pool_sweeps = val_pool_sweeps[int(shard_rows):]
            val_pool_labels = val_pool_labels[int(shard_rows):]

            create_or_append_shard_dir(f"{local_dataset_dir}/val_dataset/", shard_val_pool_sweeps, normed_shard_val_pool_labels, extendable=True)
    
    # So now we have the full or partially full pool. These pools can be handled with 
    # the append shard dir function.
    if tfrecord_norm_func is None:
        tfrecord_norm_func, tfrecord_revert_func, tfrecord_model_metrics = gen_tfrecord_norm_funcs(train_pool_labels, norm, analytes_names)
    create_or_append_shard_dir(f"{local_dataset_dir}/train_dataset/", train_pool_sweeps, tfrecord_norm_func(train_pool_labels), extendable=False)
    create_or_append_shard_dir(f"{local_dataset_dir}/val_dataset/", val_pool_sweeps, tfrecord_norm_func(val_pool_labels), extendable=False)

    return tfrecord_norm_func, tfrecord_revert_func, tfrecord_model_metrics, input_shape, output_shape

###############################################################################
# The function of interest
###############################################################################
def parametric_fit(output_directory,
                   data_folders, 
                   probes,
                   val_probes,
                   test_probes, 
                   analytes_names,
                   patience = 5,
                   min_lr = 0.00001,
                   epochs = 100,  
                   repetitions_unique = True, 
                   n_records_per_probe = -1,
                   data_split = 0.9, 
                   norm = "Shift-Z",
                   trim = None,
                   sample_indices = None,
                   model_params = None,
                   dataset_dir = None,
                   ):
    """
    The top level function for parametric fitting

    args:
        output_directory : str
            Where to save the model and its supporting files
        data_folders : list of list of str
            List of list of string paths to be searched for the probes in "probes"
        probes : list of list of str
            List of list of probes names to be searched in the corresponding data_folders
        val_probes : list of str
            List of string names of the probes to be used for validation
        test_probes : list of str
            List of string names of the probes to be held out and then used for testing
        analytes_names : list of str
            The analyte labels to be loaded and used in trained.
        patience : int (Default: 5)
            How many epochs without betterment of validation loss to wait before reducing learning rate.
            Should be adjusted to account for the amount of data in an epoch
        min_lr : float (Default: 1e-5)
            Minimum learning rate to use reduce to with reduceLRonPlateau 
        epochs : int (Default: 100)
            Number of epochs to run.
        instance_index : int (Default: -1)
            What instance of the function is this? Used for parallelizing on repetitions.
        num_instances : int (Default: -1)
            How many instances of the function are there? Used for parallelizing on repetitions.
        repetitions : int (Default: 4)
            How many repetitions of the model should be trained on the same data.
        repetitions_unique : Boolean (Default: True)
            If the repetitions should be trained on unique data. Applies only if using data_split. 
            Not applied if using val_probes.
        n_records_per_probe : int (Default: -1)
            How many sweeps per label per probe should be loaded.
        data_split : float (Default: 0.9)
            What portion of the data to be used as the training the set.
        num_folds : int (Default: -1)
            Should k-fold be used if num_folds > 1, and if so the k to use in k-fold cv. 
        norm : str (Default: 'Shift-Z')
            The y-normalization function 'Shift-Z' or 'frac'
        trim : int or tuple
            If tuple: (low, high)
            If int: high
        sample_indices : list (Default: None)
            If not None. Only the indices of the sweep will be used as the input. 
            Subsampling occurs before diff if enabled.
        model_params : dict (Default: None)
            A dictionary of parameters to be passed to the model. Typically contains model hyperparameters.
        dataset_dir : str (Default: None)
            If not None. The directory to load the train, val and test datasets from instead of generating them.
    """

    # Log the start-time for posterity
    start_time = datetime.datetime.now()
    print(f"parametric_fit() called at:\n{start_time}")
    print(f"Executing Eagerly: {tf.executing_eagerly()}")

    # Add a little information to the file path
    output_directory = format_dir(output_directory)
    model_parent_dir = descriptive_model_path(output_directory, patience, epochs)

    # To enable compatibility with legacy code.
    repetition = 0

    # Setup output directory if needed
    if not (os.path.exists(model_parent_dir)):
        os.makedirs(model_parent_dir, exist_ok=True)
    
    # set all_probes based on probes
    all_probes = sum(probes, [])
    fold_parent_dir = model_parent_dir

    # Use the dataset provided or create it as needed.
    # Setup directory to save data used for training, validation, and testing
    if dataset_dir is None:
        local_dataset_dir = f"{fold_parent_dir}UR{repetition}/datasets/"
    else:
        local_dataset_dir = dataset_dir

    if not (os.path.exists(local_dataset_dir)):
        os.makedirs(local_dataset_dir, exist_ok=True)

    # Y Norming Placeholders
    tfrecord_norm_func = None
    tfrecord_revert_func = None
    tfrecord_model_metrics = None

    # Create the datasets if they do not exist
    if not _spec_file_exists(f"{local_dataset_dir}/train_dataset/"):

        # This is the tfrecord path. In this path we don't load all the probes at the same time.
        print(f"tfrecord all_probes: {all_probes}")
        print(f"tfrecord val_probes: {val_probes}")
        print(f"tfrecord test_probes: {test_probes}")

        # Determine the training probes
        train_probes = list((set(all_probes)-set(val_probes))-set(test_probes))

        print(f"tfrecord train_probes: {train_probes}")

        # The number of training points in the pool at which it is randomly emptied.
        max_rows_in_pool = 5e5
        # The number of rows put in each shard
        shard_rows = 2.5e5

        # Variable to clean up this logic
        use_val_probes = val_probes is not None and len(val_probes) >1

        tfrecord_norm_func, tfrecord_revert_func, tfrecord_model_metrics, input_shape, output_shape = create_train_val_tf_dataset(local_dataset_dir, data_folders, 
                                            train_probes, n_records_per_probe, analytes_names, trim, sample_indices, 
                                            tfrecord_norm_func, max_rows_in_pool, shard_rows, data_split, use_val_probes, norm,
                                            repetitions_unique, repetition) 
        
        # Save dataset metadata
        dm = DataMetadata(name=f'9F_CV: ', metrics=tfrecord_model_metrics, normalization_function=tfrecord_norm_func, unnormalization_function=tfrecord_revert_func, 
                        analytes=analytes_names, means=None, stds=None, data_folders=data_folders, all_probes=probes, test_probes=test_probes,
                        val_probes=val_probes, val_ratio=data_split)
        dm.save(f'{local_dataset_dir}/metadata.pkl')

        if use_val_probes:
            create_val_tf_dataset(local_dataset_dir, data_folders, val_probes, n_records_per_probe,
                                    analytes_names, trim, sample_indices, tfrecord_norm_func)

        
        create_test_tf_dataset(local_dataset_dir, data_folders, test_probes, n_records_per_probe, 
                                analytes_names, trim, sample_indices, tfrecord_norm_func, 
                                max_rows_in_pool, shard_rows)
        
        # Load the val dataset
        if _spec_file_exists(f"{local_dataset_dir}/val_dataset/"):
            ds_val = load_dataset_from_dir(f"{local_dataset_dir}/val_dataset/")

        # Load the train dataset
        ds_train = load_dataset_from_dir(f"{local_dataset_dir}/train_dataset/")

        # Load the test dataset
        if len(test_probes) > 0:
            ds_test = load_dataset_from_dir(f"{local_dataset_dir}/test_dataset/")

    else:
        # Load dataset info
        dm = DataMetadata()
        dm.load(f'{local_dataset_dir}/metadata.pkl')

        tfrecord_norm_func = dm.normalization_function
        tfrecord_revert_func = dm.unnormalization_function
        tfrecord_model_metrics = dm.metrics

        # Need to get input shape and output shape
        input_shape, output_shape = _load_spec_file(f"{local_dataset_dir}/train_dataset/")
        output_shape = output_shape[0]
        ds_train = load_dataset_from_dir(f"{local_dataset_dir}/train_dataset/")
        if _spec_file_exists(f"{local_dataset_dir}/val_dataset/"):
            ds_val = load_dataset_from_dir(f"{local_dataset_dir}/val_dataset/")
        if _spec_file_exists(f"{local_dataset_dir}/test_dataset/"):
            ds_test = load_dataset_from_dir(f"{local_dataset_dir}/test_dataset/")

    normalize_data, revert_data, model_metrics = tfrecord_norm_func, tfrecord_revert_func, tfrecord_model_metrics

    # Prints for manual verification
    print(f'Input shape: {input_shape}')
    print(f'Output shape: {output_shape}')

    print(f"Running repetition {repetition}")
    # Set up directory for repetition
    if repetitions_unique:
        output_directory = f"{fold_parent_dir}UR{repetition}/"
    else:
        output_directory = f"{fold_parent_dir}R{repetition}/"

    # Setup output directory if needed
    if not (os.path.exists(output_directory)):
        os.makedirs(output_directory, exist_ok=True)

    # Check if there is a fit model in the output_directory
    model_path = output_directory + 'best_model.hdf5'
    model_fit = os.path.isfile(model_path)

    # What type of model do we want to load?
    model_source = model_sources[model_params["model_type"]]

    if model_fit:
        print('Best model already fit: %s'%model_path)

        # if the model is fit we need to load the metadata file to get its specific functions
        # Load the metadata
        mm = ModelMetadata()
        mm.load(output_directory+'metadata.pkl')



        reg_model = model_source.Regression_Model(output_directory, 
                                        input_shape, 
                                        output_shape, 
                                        verbose = 2, 
                                        build = False, 
                                        nb_epochs = mm.epochs, 
                                        metrics = mm.metrics, 
                                        patience= mm.patience,
                                        normalize_y = (mm.normalization_function, mm.unnormalization_function),
                                        analytes_names=mm.analytes,
                                        min_lr=min_lr if min_lr is not None else 0.00001,
                                        model_params = model_params)

        reg_model.load_best_model()

        if ds_val is not None:
            print('Computing prediction of validation data')
            metrics = reg_model.predict_ds(ds_val, return_df_metrics=True)
            print(f"Validation Results: \n{metrics}")
    else:
        print('Model not fit yet')

        # This is the model
        reg_model = model_source.Regression_Model(output_directory, 
                                        input_shape, 
                                        output_shape, 
                                        verbose = 2, 
                                        build = True, 
                                        nb_epochs = epochs, 
                                        metrics = model_metrics, 
                                        patience= patience,
                                        normalize_y = (normalize_data, revert_data),
                                        analytes_names=analytes_names,
                                        min_lr=min_lr if min_lr is not None else 0.00001,
                                        model_params = model_params)

        print('Creating metadata')
        # Saves a pickled version and a text version of the parameters
        model_metadata = ModelMetadata(name=f'{model_params["model_type"]}__repetition_{repetition}', 
                                        metrics= model_metrics, 
                                        normalization_function = normalize_data, 
                                        unnormalization_function=revert_data,
                                        analytes = analytes_names, 
                                        means = None,
                                        stds = None,
                                        data_folder = data_folders,
                                        probe_list = probes,
                                        patience = patience,
                                        epochs = epochs,
                                        batch_size = 64,
                                        all_probes = all_probes,
                                        val_probes = val_probes,
                                        test_probes = test_probes)
        model_metadata.save(reg_model.output_directory+'metadata.pkl')

        print('Fitting new model...')
        reg_model.fit_ds(ds_train, ds_val, plot_test_acc=True)
        reg_model.load_best_model()

    # Save prediction of the test set
    if ds_test is not None:
        test_pred, test_metrics = reg_model.predict_ds(ds_test, None, None, both=True)
        print("Test Metrics:")
        print(test_metrics)
        reverted_test_pred = np.apply_along_axis(reg_model.revert_data, axis=1, arr=test_pred)
        np.save(reg_model.output_directory + 'test_y_pred.npy', reverted_test_pred)
        # Get the labels from the test ds?
        test_labels = np.concatenate([y for _, y in ds_test], axis=0)
        # Revert them back to real space
        test_labels = np.apply_along_axis(reg_model.revert_data, axis=1, arr=test_labels)
        np.save(reg_model.output_directory + 'test_y_true.npy', test_labels)
    
    print(f"parametric_fit() finished at:\n{datetime.datetime.now()}")
    print(f"Duration:\n{datetime.datetime.now()-start_time}")

###############################################################################
# CLI input
###############################################################################

def cli_func():
    """
    The Command Line Interface Function

    Parses the command line arguments,
    Copies the parameter file,
    Calls the parametric_fit function,
    and does some error handling
    """

    # For Error reporting
    program_name = os.path.basename(sys.argv[0])
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pfile', type=str, default="default_params.py",
                        help = "Parameter file to override defaults see 'default_params.py'")
    parser.add_argument('--index', type=int, default= -1,
                        help = "The which of the --count processes this one is")
    parser.add_argument('--count', type=int, default= -1,
                        help="How many parallel versions of this program there are")
    args = parser.parse_args()

    try:
        #Check that param file exists
        if '/' not in args.pfile:
            cwd_str = os.getcwd()
            if os.path.isfile(f"{cwd_str}/xms/fitting/params/{args.pfile}"):
                f_path = f"{cwd_str}/xms/fitting/params/{args.pfile}"
            elif os.path.isfile(f"{cwd_str}/xms/fitting/{args.pfile}"):
                f_path = f"{cwd_str}/{args.pfile}"
            else:
                print("Unable to find parameter file")
                print("File must be loadable as a module (.py). It must be in the project directory or the full path must be specified.")
                exit()
        else:
            if os.path.isfile(f"{args.pfile}"):
                f_path = f"{args.pfile}"
            else:
                print("Unable to find parameter file")
                print("File must be loadable as a module (.py). It must be in the project directory or the full path must be specified.")
                exit()

        # Note that only the first instance of this program should do the copy/manipulation of the file
        # The logic to handle that is dealt with inside regulate_parameter_file
        try:
            params, dmp = regulate_parameter_file(f_path, return_out_path=True)
        except Exception as e:
            print(f"regulate_parameter_file failed with exception:\n{e}")
            exit()
    
        # Set-up Logging
        sys.stdout = Logger(f"{dmp}/logging/parametric_fit_function_index_{args.index}_count_{args.count}.log")

        # Here we actually call the fitting function
        parametric_fit(output_directory = params.output_directory,
                        data_folders = params.data_folders,
                        probes = params.probes,
                        val_probes = params.val_probes,
                        test_probes = params.test_probes,
                        data_split = params.data_split,
                        analytes_names = params.analytes_names,
                        n_records_per_probe = params.n_records_per_probe,
                        norm = params.norm,
                        trim = params.trim,
                        patience = params.patience,
                        min_lr = params.min_lr, 
                        epochs = params.epochs,
                        repetitions_unique = params.unique_repetitions,
                        model_params = params.model_params,
                        dataset_dir = params.dataset_directory) # Model Parameters

    except KeyboardInterrupt:
        print('*** INTERRUPT ***')
        return 0

    except CLIError as e:
        indent = len(program_name) * ' '
        sys.stdout.write(program_name + ': ' + e.msg + '\n')
        sys.stdout.write(indent + '  for help use --help\n')
        return 2

    except Exception as e:
        indent = len(program_name) * ' '
        sys.stdout.write(program_name + ': ' + repr(e) + '\n')
        traceback.print_exc()
        sys.stdout.write(indent + '  for help use --help\n')
        return 2

if __name__ == '__main__':
    sys.exit(cli_func())
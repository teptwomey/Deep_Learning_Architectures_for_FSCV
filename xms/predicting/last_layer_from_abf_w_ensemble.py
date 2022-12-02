#!/usr/bin/env python3

"""
last_layer_from_abf_w_ensmeble.py

Thomas Twomey
Montague Lab
Fralin Biomedical Institute at VTC
06/16/2021

Modified: 
11/20/2022

Hi there,

This file is adapted from predict_from_abf_w_ensemble.py. Instead of running 
predictions, it saves output of second to last layer of the model.

"""

# Inputs
import sys
import os
from pathlib import Path
import datetime
import traceback
import argparse
from glob import glob
import numpy as np
import pandas as pd
from tensorflow import keras
from einops.layers.keras import Reduce

project_folder = os.path.abspath(__file__)
project_folder = project_folder[:project_folder[:project_folder.rfind('/')].rfind('/')+1]
sys.path.append(project_folder)

# Local Inputs
from utils.model_metadata import ModelMetadata
from utils.utils import rmse
from data_loader.data_loader import FilePairDataLoader
from abf2np import load_abf

from classifiers import eeg_transformer_tfrecord

__version__ = 1
__date__ = '2021-06-16'
__updated__ = '2022-11-20'
__verbose__ = 1

# Error handling

###############################################################################
# Utility Functions
###############################################################################

class CLIError(Exception):
    """Generic exception to raise and log different fatal errors."""

    def __init__(self, msg):
        super(CLIError).__init__(type(self))
        self.msg = f'error: {msg}'

    def __str__(self):
        return self.msg

    def __unicode__(self):
        return self.msg

def __log(message, test=None):
    global __verbose__

    # default to logging when verbose
    if test is None:
        test = (__verbose__ > 0)

    if test:
        print(message, flush=True)

class Logger(object):
    """
    Logger that redirects std_out to enable print to be dumped to a file and std_out.
    TODO: Better version. Add redirection of stderr.

    Taken from:
    https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
    """

    def __init__(self, output_file):
        self.terminal = sys.stdout
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

###############################################################################
# CLI Parsing Functions
###############################################################################


def __check_output_arg(output):
    """
    Ensures that the output path is a directory.
    If file returns false.
    Creates directory if it does not exist.
    """

    if not os.path.isdir(output):
        if os.path.isfile(output):
            raise (
                CLIError('Output must be a directory')
            )
        else:
            # Make a new directory
            os.makedirs(output, exist_ok=True)
    return True

###############################################################################
# Data Ingestion Functions
###############################################################################

def find_ttl(input_file, channel='TTL_In', random_burst=False):
    """
    Loads the ttl channel of the abf file

    Then searches the ttl data for the rising edge of the ttl signal

    parameters:
        input_file : str
            The file to load the TTLs from.
        channel : int or str (Default: 'TTL_In)
            If int, the index of the channel to load the TTL from.
            If str, the name of the channel to load the TTL from
        random_burst : Boolean (Default: False)
            If the .abf file is being recorded in the 97Hz random burst mode.

    returns:
        pairs : list of tuples
            A list of tuples with the sweep number, and index of the point in the sweep
    """

    if not random_burst:
        ttl_data = load_abf(input_file, [channel], offset=0, num_samples=10000)
    else:
        ttl_data = load_abf(input_file, [channel], offset=0, num_samples=1032)
    return [(i[0], np.argmax(np.diff(ttl_data[i]))) for i in np.argwhere(np.amax(np.diff(ttl_data), axis=1)>1)]

def _overflow_correction(in_arr):
    """
    Corrects overflow in standard case using the diff method.

    parameters:
        in_arr : np.ndarray
            A numpy array with sweeps as the rows

    returns:
        Modifided in_arr
    """

    def in_func(arr):
        """
        Corrects overflow in a single row.
        """

        idxs = np.where(np.abs(np.diff(arr)) > 3000)[0]

        out_arr = np.copy(arr)

        if len(idxs) % 2 != 0:
            raise ValueError("There is an odd number of idxs")

        for i in range(0, len(idxs), 2):
            out_arr[idxs[i]+1:idxs[i+1]+1] = 2000

        return out_arr

    out = np.apply_along_axis(in_func, 1, in_arr)

    return out

def load_input(input_file, channel, random_burst=False):
    """
    Loads the input from an abf, csv, or npy file

    parameters:
        input_file : str
            The path to a csv or abf file
        channel : int
            The channel of the abf file to load, ignored for csv
        random_burst : boolean (Default: False)
            If the data being loaded is random_burst
    """

    __log(f'###\n{input}\n###')
    
    if not os.path.isfile(input_file):
        __log("Input specified is not a file")
        return 1
    
    input_data = None

    __log('Attempting to load data')
    if ".abf" in input_file:
        if not random_burst:
            # triangle waveform case
            input_data = load_abf(input_file, [channel])
        else:
            # Random Burst
            print("Random Burst")
            input_data = load_abf(input_file, [channel], offset=15, num_samples=1000, correct_overflow=True)
    elif ".csv" in input_file:
        input_data = np.loadtxt(open(input_file, "rb"), delimiter=",")
    elif ".npy" in input_file:
        input_data = np.load(input_file)

    else:
        __log("Invalid input type.")
        return 1
    
    __log('Data Loaded')
    __log(f'Input Data has shape: {input_data.shape}')

    __log('Currently no validity checking done an input data')
    
    input_data = _overflow_correction(input_data)
    __log('Overflow correction performed on input data')

    return input_data

def compile_abf_input_dict(input_files, channel, random_burst=False):
    """
    Builds a dictionary with the abf files and the names.

    parameters:
        input_files : list of str
            The list of abf files to load
        channel : int
            What channel of the abf file to load.

    returns:
        input_dict : dict
            A dictionary with {file_name : data_loaded_from_file}
    """

    input_dict = {}
    for input_file in input_files:
        in_file_name = Path(input_file).stem
        input_dict[in_file_name] = (load_input(input_file, channel, random_burst), None)
    return input_dict


def load_probes(folder, probes, trim_cap=2500, analyte_names=['DA','5HT','pH','NE']):
    """
    Loads the probes from folder(s) in the file pair ".mat" format

    parameters:
        folder : list of str
            The folder(s) to load 
        probes : list of str
            The list of probes to load
        trim_cap : int (Default: 2500)
            Sweeps with label that are about this cap will be removed.
        analytes_name : list of str (Default: ['DA','5HT','pH','NE'])

    returns:
        data : dict
            A dictionary with {probe : (x_data, y_data)}
    """

    dl = FilePairDataLoader()
    data = dl.load(folders=folder, probes=probes, analyte_names = analyte_names)
    if trim_cap is not None and trim_cap > 0:
        data = dl.trim(data, trim_cap)
    return data

def create_input_dict(input_files, folder, probes, diff, trim_cap, analyte_names, sample_indices, channel, random_burst=False):
    """
    Handle precedent of arguments
    """

    # Dataloader object to subsample or diff
    dl = FilePairDataLoader()

    if input_files is not None:
        __log("Attempting to load input from --input argument")
        if not isinstance(input_files, list):
            input_files = [input_files]
        input_dict = compile_abf_input_dict(input_files, channel, random_burst)
    elif folder is not None and probes is not None:
        __log("Attempting to load probes with folder/probes arguments")
        input_dict = load_probes(folder, probes, trim_cap, analyte_names)
    if sample_indices is not None:
        input_dict = dl.subsample_dict(input_dict, sample_indices)
    if diff:
        input_dict = dl.dict_diff(input_dict)
    input_dict = dl.reshape_dict(input_dict)

    return input_dict

###############################################################################
# Prediction helper functions
###############################################################################

def eval_prediction(pred, true, analyte_names=['DA','5HT','pH','NE']):
    """
    Calculates the rmse of the predictions for the different analytes
    """

    error = rmse(true, pred)
    df = pd.DataFrame([error], columns= [f"{a}_RMSE" for a in analyte_names])
    return df


def save_predictions_csv(y_pred, pred_file):
    """
    Saves the prediction (numpy array) as a csv file
    """

    np.savetxt(pred_file, y_pred, delimiter=',')
    __log(f'Saved predictions to {pred_file}')

def save_predictions_npy(y_pred, pred_file):
    """
    Saves the prediction (numpy array) as a csv file
    """

    np.save(pred_file, y_pred)
    __log(f'Saved predictions to {pred_file}')

def save_predictions(y_pred, pred_file):
    """
    Wraps save_predictions_csv and potentially others
    Decides what function to call based on file extension
    """

    file_type = os.path.splitext(pred_file)[1][1:]
    if file_type == 'csv':
        save_predictions_csv(y_pred, pred_file)
    if file_type == 'npy':
        save_predictions_npy(y_pred, pred_file)
    else:
        raise NotImplementedError(f"That output file type ({file_type}) has not been implemented")

def make_model_pred_folder(prediction_folder, model_path, model_label=None):
    """
    Uses model label to generate a new folder to place the predictions

    parameters:
        prediction_folder : str
            The 
        model_path : str
            The location of the model, used to make a label if none provided
        model_label : str (Default: None)
            A label for the model

    returns
        The str path of the folder where the actual predictions are stored
    """

    if model_label is None or len(model_label)==0:
        model_label =os.path.basename(os.path.normpath(model_path))
        __log(f'No model_label provided using generated: {model_label}')
    model_prediction_folder = prediction_folder + model_label + '_prediction/'
    if not (os.path.exists(model_prediction_folder)):
        os.makedirs(model_prediction_folder, exist_ok=True)
    return model_prediction_folder

def make_snippet_folder(model_pred_folder, sub_dir=""):
    """
    Makes a snippet plots folder inside the model_pred_folder

    parameters:
        model_pred_folder : str
            Base path for the model_prediction
            
    returns: 
        snippet_folder : str
            path to new folder for snippet plots
    """

    snippet_folder = f"{model_pred_folder}plots/snippets/{sub_dir}/"
    if not (os.path.exists(snippet_folder)):
        os.makedirs(snippet_folder, exist_ok=True)
    return snippet_folder

def make_data_snippet_folder(model_pred_folder, sub_dir=""):
    """
    Makes a DATA snippet folder inside the model_pred_folder

    parameters:
        model_pred_folder : str
            Base path for the model_prediction

    returns: 
        snippet_folder : str
            path to new folder for data snippets
    """

    snippet_folder = f"{model_pred_folder}data/snippets/{sub_dir}/"
    if not (os.path.exists(snippet_folder)):
        os.makedirs(snippet_folder, exist_ok=True)
    return snippet_folder

###############################################################################
# Model Loading Functions
###############################################################################

def create_custom_objects(metadata):
    """
    Loads the loss_function and metrics from the metadata object into a list.
    That list is used when creating the model.

    parameters:
        metadata : ModelMetadata object
            The metadata object loaded from the same location as the model
    
    returns:
        custom_object : List
            List of functions taken from the ModelMetadata object
    """

    custom_objects = {}
    custom_object_list = []
    if metadata.loss_function is not None:
        custom_object_list.append(metadata.loss_function)
    for func in custom_object_list + metadata.metrics:
        custom_objects[func.__name__] = func
    return custom_objects

def load_model(model_dir):
    """
    Loads the model from the model dir.
    Looks for best_model.hdf5 and metadata.pkl

    """
    # Find the model path
    __log(f"Model dir: {model_dir}")
    models = glob(model_dir + 'best_model.hdf5')

    if len(models) < 1:
        raise FileNotFoundError(f"No model found in: {model_dir}")

    model_path = models[0]

    # Find the metadata path
    metadatas = glob(model_dir + 'metadata.pkl')

    if len(metadatas) < 1:
        raise FileNotFoundError(f"No model metadata found in: {model_dir}")

    metadata_path = metadatas[0]

    __log(f'Loading model from: {model_path}')
    __log(f'Loading metadata from: {metadata_path}')

    # Load the metadata
    model_metadata = ModelMetadata()
    model_metadata.load(metadata_path)

    # The function cannot be dependent on an external variable
    custom_objects = create_custom_objects(model_metadata)

    # Actually Loading the model
    if "eeg_transformer" in model_metadata.name:
        custom_objects["ClassToken"] = eeg_transformer_tfrecord.ClassToken
        custom_objects["AddPositionEmbs"] = eeg_transformer_tfrecord.AddPositionEmbs
        custom_objects["Reduce"] = Reduce
    model = keras.models.load_model(model_path, custom_objects=custom_objects)

    new_model = keras.Model(inputs=model.input, outputs=model.layers[-2].output)

    return new_model, model_metadata

###############################################################################
# The main functions
###############################################################################

def predict_from_abf_w_ensemble(input, probe_folder, probes, prediction_folder, models, diff,
        ensemble, channel, analytes, trim, graph, overwrite, sample_indices, snippet, 
        random_burst=False, pred_std=False):
    """
    The top level function for this file. (predict_from_abf_w_ensemble.py)

    Parameters:
        input : str
        probe_folder : list of str or str
        probes : list of str
        prediction_folder : str
        models : list of str
        diff : boolean
        ensemble : boolean
        channel : int
        analytes : list of str
        trim : int
        graph : boolean
            If graphs should be generated.
        overwrite : boolean
            If predictions should overwrite existing predictions.
        sample_indices : list of int
            If the subsampling the sweep, what sweeps to use.
            None to not subsample.
        snippet : boolean
            If snippets should be generated. 
            Data snippets will always be generated.
            Snippet plots will be generated if graph is true.
    """

    start_time = datetime.datetime.now()
    print(f"predict_from_abf_w_ensemble() called at:\n{start_time}")

    # =====================================================================
    # Load the input
    # =====================================================================

    input_dict = create_input_dict(input, probe_folder, probes, diff, trim, analytes, sample_indices, channel, random_burst)

    # # We want to save the arrays so we can go look at them
    # helper_folder = make_model_pred_folder(prediction_folder, None, "ensemble")
    # for in_key in input_dict.keys():
    #     input_data = input_dict[in_key][0]
    #     np.save(f"{helper_folder}{in_key}_data.npy", input_data)

    # =====================================================================
    # Load the model(s) make the predictions ...
    # =====================================================================

    # Keep track of the predictions
    pred_dict = {}
    # Nested dictionary with outer key as probe inner key as analyte 

    # Iterate through the models provided in the CLI
    for model_index, model_dir in enumerate(models):

        # Load the model and the model_metadata
        model, model_metadata = load_model(model_dir)

        # Determine what analytes this model predicts
        predicted_analytes = model_metadata.analytes

        # Create folder to save results if not ensembling.
        # If ensembling the output from each individual model is NOT saved.
        if not ensemble:
            # create folder to save predictions
            model_prediction_folder = make_model_pred_folder(prediction_folder, model_dir, f"{model_metadata.name}")

        # Make the predictions
        # Inside the loop 
        # In_key is a probe or specific test of a probe
        for in_key in input_dict.keys():
            input_data = input_dict[in_key][0]

            # predict concentration using this model (makes predictions in a normal space)
            y_pred = model.predict(input_data, batch_size=64)
            # Revert out of normalized space to real space
            # y_pred = np.apply_along_axis(model_metadata.unnormalization_function, axis=1, arr=y_pred) 

            if ensemble:
                # For each of the analytes that was predicted
                print("Ensemble not supported")

            if not ensemble:
                # If not ensembling save the predictions
                pred_file = f"{model_prediction_folder}{in_key}_last_layer.npy"
                save_predictions(y_pred, pred_file)

                # If graph
                if graph:
                    pass
    
    __log(list(pred_dict.keys()))

    # Ensemble the results
    if ensemble:

        print("Ensemble not supported")

    print(f"predict_from_abf_w_ensemble() finished at:\n{datetime.datetime.now()}")
    print(f"Duration:\n{datetime.datetime.now()-start_time}")

def cli_func(_argv=None):

    if _argv is None:
        pass
    else:
        sys.argv.extend(_argv)

    # For Error reporting
    program_name = os.path.basename(sys.argv[0])
    program_version = 'v%s' % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
    program_shortdesc = __import__('__main__').__doc__.split('\n')[1]
    program_license = '''%s
  Created by Tom Twomey on %s. Derived from works of Jason White.
  Copyright 2021 FBRI. All rights reserved.
USAGE
''' % (program_shortdesc, str(__date__))


    try:
        # Command Line Arguments Parsing
        # Set up argument parser
        parser = argparse.ArgumentParser()
        parser.add_argument('-i','--input', type=str, nargs='+', dest='input', default=None, 
                            help='Input abf(or csv) file to create predictions from. Takes precedent over folder/probe arguments')
        parser.add_argument('-f','--folder', type=str, nargs='+', dest='folder',
                            help = "Alternative to --input option. Where to look for the data if using --probes argument")
        parser.add_argument('-p','--probes', type=str, nargs='+', dest='probes',
                            help = "Alternative to --input option. Used with --folder, specifying probes to load eg. CF082 CF083 ..")
        parser.add_argument('-o','--output', type=str, nargs='?', dest='output', default=None, 
                            help='Output directory to store predictions etc.')
        parser.add_argument('-m','--model', type=str, nargs='+', dest='models', default=None, 
                            help='Model(s) to load to use to make predictions.')
        parser.add_argument('-w','--weights', type=float, nargs='+', dest='weights', default=None, 
                            help='Currently Ignored. Equal weighting is used.')
        parser.add_argument('-d','--diff', dest='diff', action='store_true',
                            help='Should a diff be taken on the input? Defaults to false.')
        parser.add_argument('-e','--ensemble', dest='ensemble', action='store_true',
                            help='Ensemble the results of various with the weights provided.')
        parser.add_argument('-c','--channel', dest='channel', default='0',
                            help='Specify what channel of abf to load. Defaults to 0')
        parser.add_argument('-a','--analytes', type=str, nargs='+', dest='analytes', default=None, 
                            help='Analytes loaded for comparison')
        parser.add_argument('-t','--trim', type=int, dest='trim', default=None, 
                            help='Trim test set to below this number(int)')
        parser.add_argument('-g','--graph', dest='graph', action='store_true',
                            help='Create summary graph outputs')
        parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                            help='overwrite existing output file(s)')
        parser.add_argument('--sample', type=int, nargs='+', dest='sample_indices', default=None, 
                            help='Indicies to sample from sweep.')
        parser.add_argument('--snippet', dest='snippet', action='store_true',
                            help='Create snippets')
        parser.add_argument('--random_burst', dest='random_burst', action='store_true', default=False,
                            help='Specify if using random_burst. Default: False')
        parser.add_argument('--pred_std', dest='pred_std', action='store_true', default=False,
                            help='Specify to save the std of prediction between members of the ensemble. Default: False')


        # Process arguments
        args = parser.parse_args()
        prediction_folder = args.output
        analyte_names = args.analytes
        overwrite = args.overwrite

        # =====================================================================
        # Do some error checking on CL input
        # =====================================================================

        if prediction_folder is None:
            __log("Prediction folder needs to be specified")
            return 1

        if not __check_output_arg(prediction_folder):
            return 1

        if analyte_names is None:
            analyte_names = ['DA', '5HT', 'pH', 'NE']

        if overwrite:
            __log('Overwrite mode on')

        # Set-up Logging to file
        os.makedirs(f"{args.output}/logs/", exist_ok=True)
        sys.stdout = Logger(f"{args.output}/logs/predict_from_abf_w_ensemble__{str(datetime.datetime.now()).replace(' ','__').replace(':','_')[:-7]}.log")

        # =====================================================================
        # Load the input and models and then make the predictions
        # =====================================================================

        predict_from_abf_w_ensemble(args.input,
                                    args.folder, 
                                    args.probes, 
                                    args.output, 
                                    args.models,
                                    args.diff, 
                                    args.ensemble, 
                                    args.channel, 
                                    analyte_names, 
                                    args.trim, 
                                    args.graph, 
                                    args.overwrite, 
                                    args.sample_indices, 
                                    args.snippet,
                                    args.random_burst,
                                    args.pred_std)

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

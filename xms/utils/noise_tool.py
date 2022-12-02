"""
noise_tool.py

Thomas Twomey
Montague Lab
Fralin Biomedical Institute at VTC
11/16/2021

Hi there.

This is where we dev and debug a quick noise tool to get a noise value for a given section of an abf file or a numpy array.

It looks at the flat part of the sweep
"""

# Imports
import numpy as np
import sys
import os
import argparse
from pathlib import Path
from glob import glob
import pandas as pd

project_folder = os.path.abspath(__file__)
project_folder = project_folder[:project_folder[:project_folder.rfind('/')].rfind('/')+1]
sys.path.append(project_folder)

# Local imports
from predicting.abf2np import load_abf


# Functions

def linear_approximation(data, percentile=95):
    """
    A naive implementation that creates a linear approximation of the points in the sweeps for all the points in the data.
    It then calculates the different between the line of best fit and the points in the sweeps, takes the max of the 
    difference for each sweep and returns a given percentile of those maxes.

    parameters:
        data : np.ndarray
            The array with the data (From the flat part)
        percentile:
            The percentile of per sweep max deviation from the linear approximation

    returns:
        The value as a float

    """
    # The x points are just the indicies
    x = np.tile(np.arange(data.shape[1]),(data.shape[0],1))
    y = data
    coef = np.polyfit(x.flatten(), y.flatten(), 1)
    poly1d_fn = np.poly1d(coef)
    line = poly1d_fn(x[0])
    diffs = np.apply_along_axis(lambda x: x-line, 1, y)
    pos_diffs = np.abs(diffs)
    maxs = np.max(pos_diffs, axis=1)
    return np.percentile(maxs,percentile)

def load_two_60hz(file_name, channel_list=[0]):
    """
    Loads the two final sections of the sweep that would contain two 60hz noise waves (one in each section)
    The idea is to capture two waveforms.

    parameters:
        file_name : str
            The file to load
    """
    data = load_abf(file_name, channel_select=channel_list, offset=10000-1666*2, num_samples=1666*2)
    print(file_name)
    print(data.shape)
    return data

def parse_folder_name(folder_path):
    """
    Parse the string file name to create a dictionary with the file attributes.

    args:
        folder_name : str
            The folder name to be parsed
    returns:
        List of dictionaries corresponding to the probe/forcing function combination 
    """

    if folder_path[-1] == "/":
        folder_path = folder_path[:-1]

    # Truncate folder name if needed
    folder_name = str(Path(folder_path).stem)
    
    # Split the folder_name by some string
    components = folder_name.split('__')

    num_components = len(components)
    if num_components < 5:
        raise ValueError(f"Unable to parse {folder_name} expected 5 components but found {num_components}")
    
    # Check if there are more than one probe
    probe_component = components[4]
    probes = []
    for probe in probe_component.split("#"):
        if len(probe) < 2:
            # If empty string skip
            continue
        probes.append(probe)
        
    out_list = []

    for idx, probe in enumerate(probes):
        probe_dict = {
            "date" : components[0],
            "rig" : components[1],
            "schedule" : components[2],
            "protocol" : components[3],
            "probe" : probe,
            "probe_idx" : idx,
            "modifier" : components[5] if num_components > 5 else "",
            "stem" : folder_name,
            "idx_valid" : lambda x: True
        }
        
        out_list.append(probe_dict)
    
    return out_list

def examine_folder_abf(folder_path_to_be_globed, output_folder):
    """
    Finds all the abf files in a folder, loads them, calculates the noise

    parameters:
        folder_path_to_be_globed : str
            The file path to be globbed

    output:
        Saves a csv with the results
    """
    abf_list = glob(folder_path_to_be_globed)

    abf_list.sort()

    noise_list = []
    probe_dataset_list = []

    for abf_file in abf_list:
        folder_dict = parse_folder_name(str(Path(abf_file).parent))
        for d in folder_dict:
            noise = linear_approximation(load_two_60hz(abf_file, [d["probe_idx"]*2]))
            noise_list.append(noise)
            probe_dataset_list.append(f"{d['probe']}_{d['schedule']}_{abf_file[-8:-4]}")
    
    file_dict = {"Probe Dataset": probe_dataset_list, "Noise":noise_list}

    df = pd.DataFrame(file_dict)
    df.to_csv(f"{output_folder}noise.csv", index=False)

###############################################################################
# CLI input
###############################################################################

def cli_func():
    """
    CLI Wrapper of the noise tool

    args:
        --path, -p : list of strings
        --output : The output folder
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, required=True, dest='path',
                        help = "The path to expand to load the abf files")
    parser.add_argument('--output', '-o', type=str,
                        help = "The output folder")
    args = parser.parse_args()

    examine_folder_abf(args.path, args.output)

if __name__ == '__main__':
    sys.exit(cli_func())
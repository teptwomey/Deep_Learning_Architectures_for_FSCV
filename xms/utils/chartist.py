"""
chartist.py

Thomas Twomey
Montague Lab
Fralin Biomedical Institute at VTC
02/04/2022

Modified: 
04/05/2022

Hi there,

This is where most of the charting functions live. Unfortunately most of these
function were developed organically as the need arose so they not as uniform as
one would like. 

Most should work being exported as png's or pdf's.

Happy charting.

(The term chartist is used here in jest and is refering to the flawed 
investment strategy.)
"""

import numpy as np
import pandas as pd
from scipy import stats
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools

def _filename_helper(filename, new_string):
    """
    Helper function to append a sting between existing name and extension.

    Parameters:
        filename : string
        new_sring : string

    Returns:
        Modified filename: string
    """

    name, ext = os.path.splitext(filename)
    return "{name}_{new_str}{ext}".format(name=name, new_str=new_string, ext=ext)

def connected_pairs(y_true, y_pred, filename, analytes_name = ['DA', '5HT'], model_description="", unique=False):
    """
    Function to plot sets of connected pairs representing the true and predicted value for a given sweep. 
    Wraps connected_pairs_2d and connected_pairs_3d to handle 2d, 3d, and 4d input.

    Parameters:
        y_true : np.ndarray
            The true values of the labels
        y_pred : np.ndarray
            The predicted values of the labels
        filename : String
            The Absolute or relative path to save the output graph(s)
        analytes_name : List
            The names of the analytes in order corresponding to the y_true/y_pred.
            Not passed if 4d
            Default: ['DA', '5HT']
        model_description : String
            A descriptive string to add to the title of the graphs
            Default: ""
        unique : Boolean
            If only a single point per unique concentrations should be graphed
            Default: False

    Exeception:
        Note that it will catch most exceptions to prevent failure of parent script
    """

    try:
        if y_true.shape[1] == 4:
            # Remove Ph
            y_true = np.delete(y_true, 2, 1)
            y_pred = np.delete(y_pred, 2, 1)
            return connected_pairs_3d(y_true, y_pred, filename, model_description = model_description, unique = unique)
        if y_true.shape[1] == 3:
            return connected_pairs_3d(y_true, y_pred, filename, analytes_name, model_description, unique)
        if y_true.shape[1] == 2:
            return connected_pairs_2d(y_true, y_pred, filename, analytes_name, model_description, unique)
    except Exception as e:
        print(f"######\nException caught when trying to plot connected_pairs\n######")

def connected_pairs_2d(y_true, y_pred, filename, analytes_name = ['DA', '5HT'], model_description="", unique=False, indicies=[0,1]):
    """
    Function to plot sets of connected pairs representing the true and predicted value for a given sweep in 2d. 
    Considers the indices provided for x and y axis.

    Parameters:
        y_true : np.ndarray
            The true values of the labels
        y_pred : np.ndarray
            The predicted values of the labels
        filename : String
            The Absolute or relative path to save the output graph(s)
        analytes_name : List
            The names of the analytes in order corresponding to the y_true/y_pred.
            Default: ['DA', '5HT']
        model_description : String
            A descriptive string to add to the title of the graphs
            Default: ""
        unique : Boolean
            If only a single point per unique concentrations should be graphed
            Default: False
        indicies : List
            A list of length 2 that specifies the indicies of y_true/y_pred that are used for the x and y axis.
            Default: [0,1]
    """

    # If unique, only select one pair from each known label
    if unique:
        _, unique_indicies = np.unique(y_true, return_index=True, axis=0)
        y_true = y_true[unique_indicies]
        y_pred = y_pred[unique_indicies]
        
    x_idx = indicies[0]
    y_idx = indicies[1]

    xx = np.vstack([y_true[:,x_idx],y_pred[:,x_idx]])
    yy = np.vstack([y_true[:,y_idx],y_pred[:,y_idx]])

    fig = plt.figure(figsize=(8,8))

    plt.scatter(y_true[:,x_idx], y_true[:,y_idx], color="red", s=2, alpha=0.9, zorder = 3.5, label = f"True Labels")
    plt.scatter(y_pred[:,x_idx], y_pred[:,y_idx], color="blue", s=2, alpha=0.2, zorder = 1.5, label = f"Predicted Labels")
    plt.plot(xx, yy, '-y', linewidth = 0.25, alpha = 0.1, zorder = 2.5)

    plt.xlabel(f"{analytes_name[x_idx]} concentration nMol")
    plt.ylabel(f"{analytes_name[y_idx]} concentration nMol")
    plt.legend(loc="upper right")
    plt.title(f"{model_description} Real vs Predicted {analytes_name[x_idx]}, and {analytes_name[y_idx]}")
    plt.grid(True)
    
    fig.savefig(filename, dpi=400, bbox_inches='tight', transparent=False)
    plt.close(fig)

def connected_pairs_3d(y_true, y_pred, filename, analytes_names=['DA', '5HT', 'NE'], model_description="", unique=False):
    """
    Function to plot sets of connected pairs representing the true and predicted value for a given sweep, but in 3d.
    Ooooo, Ahhh

    Parameters:
        y_true : np.ndarray
            The true values of the labels
        y_pred : np.ndarray
            The predicted values of the labels
        filename : String
            The Absolute or relative path to save the output graph(s)
        analytes_name : List
            The names of the analytes in order corresponding to the y_true/y_pred.
            Will be replaced with default if length is less than 3.
            Default: ['DA', '5HT', 'NE']
        model_description : String
            A descriptive string to add to the title of the graphs
            Default: ""
        unique : Boolean
            If only a single point per unique concentrations should be graphed
            Default: False
    """

    if len(analytes_names) <3:
        analytes_names=['DA', '5HT', 'NE']

    # If unique, only select one pair from each known label
    if unique:
        _, unique_indicies = np.unique(y_true, return_index=True, axis=0)
        y_true = y_true[unique_indicies]
        y_pred = y_pred[unique_indicies]

    if y_true.shape[1] != 3:
        return
    
    xx = np.vstack([y_true[:,0],y_pred[:,0]])
    yy = np.vstack([y_true[:,1],y_pred[:,1]])
    zz = np.vstack([y_true[:,2],y_pred[:,2]])

    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(projection='3d')
    #ax = Axes3D(fig)

    for x, y, z in zip(xx.T, yy.T, zz.T):
        ax.plot3D(x,y,z, '-y', linewidth = 0.25, alpha=0.2, zorder = 2.5)
    ax.scatter3D(y_true[:,0], y_true[:,1], y_true[:,2], color="red", s=2, alpha=0.9, zorder = 3.5, label = f"True Labels")
    ax.scatter3D(y_pred[:,0], y_pred[:,1], y_pred[:,2], color="blue", s=2, alpha=0.3, zorder = 1.5, label = f"Predicted Labels")

    ax.set_xlabel(f"{analytes_names[0]} concentration nMol")
    ax.set_ylabel(f"{analytes_names[1]} concentration nMol")
    ax.set_zlabel(f"{analytes_names[2]} concentration nMol")

    ax.text2D(0.05, 0.95, f"{model_description} Real vs Predicted {analytes_names[0]}, {analytes_names[1]}, and {analytes_names[2]}", transform=ax.transAxes)

    fig.savefig(filename, dpi=400, bbox_inches='tight', transparent=False)
    plt.close(fig)

    combinations = list(itertools.combinations([i for i in range(len(analytes_names))],2))
    for idx_a, idx_b in combinations:
        connected_pairs_2d(y_true, y_pred, _filename_helper(filename,f"_2d_{analytes_names[idx_a]}_{analytes_names[idx_b]}"), 
                                                            analytes_names, 
                                                            model_description=f"{model_description} {analytes_names[idx_a]}, {analytes_names[idx_b]}",
                                                            indicies=[idx_a, idx_b])
    
def repetitions_connected_pairs(y_true, y_pred, filename, analytes_name = ['DA', '5HT'], model_description="", mean=True):
    """
    Function to plot sets of connected pairs representing the true and predicted value for a given sweep. 
    Wraps connected_pairs_2d and connected_pairs_3d to handle 2d, 3d, and 4d input.

    Parameters:
        y_true : np.ndarray
            The true values of the labels
        y_pred : np.ndarray
            The predicted values of the labels
        filename : String
            The Absolute or relative path to save the output graph(s)
        analytes_name : List
            The names of the analytes in order corresponding to the y_true/y_pred.
            Not passed if 4d
            Default: ['DA', '5HT']
        model_description : String
            A descriptive string to add to the title of the graphs
            Default: ""
        mean : Boolean
            If only the mean per unique concentrations should be graphed
            Default: True

    Exeception:
        Note that it will catch most exceptions to prevent failure of parent script
    """

    try:
        if y_true.shape[1] == 4:
            # Remove Ph
            y_true = np.delete(y_true, 2, 1)
            y_pred = [np.delete(r_y_pred, 2, 1) for r_y_pred in y_pred]
            return repetitions_connected_pairs_3d(y_true, y_pred, filename, model_description = model_description, mean=mean)
        if y_true.shape[1] == 3:
            return repetitions_connected_pairs_3d(y_true, y_pred, filename, analytes_name, model_description, mean=mean)
        if y_true.shape[1] == 2:
            return repetitions_connected_pairs_2d(y_true, y_pred, filename, analytes_name, model_description, mean=mean)
    except Exception as e:
        print(f"######\nException caught when trying to plot connected_pairs\n{e.message}\n######")

def repetitions_connected_pairs_2d(y_true, y_pred, filename, analytes_names=['DA', '5HT', 'NE'], model_description="", mean=True, indicies=[0,1]):
    """

    UNTESTED!!!!

    TODO TEST, Also this is too ugly

    Function to plot the connected pair plot for each repetition to show the variation between models.
    Defaults to plot a mean prediction per unique label per model.

    Parameters:
        y_true : np.ndarray
            The true values of the labels from each of the R repetitions
        y_pred : np.ndarray
            The predicted values of the labels from "edge" data
        filename : String
            The Absolute or relative path to save the output graph(s)
        analytes_name : List
            The names of the analytes in order corresponding to the y_true/y_pred.
            Will be replaced with default if length is less than 3.
            Default: ['DA', '5HT', 'NE']
        model_description : String
            A descriptive string to add to the title of the graphs
            Default: ""
        mean : boolean
            If the mean for each unqiue label should be used instead of all points
            Default: True
        indicies : List
            A list of length 2 that specifies the indicies of y_true/y_pred that are used for the x and y axis.
            Default: [0,1]
        
    """
    if y_true.shape[1] != 3:
        return

    same_label_idxs = []
    _, unique_indicies = np.unique(y_true, return_index=True, axis=0)
    for unique_idx in unique_indicies:
        same_label_idxs.append(np.where((y_true == y_true[unique_idx,:]).all(axis=1))[0])

    x_idx = indicies[0]
    y_idx = indicies[1]

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot()

    for r, pred_rep in enumerate(y_pred):
        pred_list = []
        true_list = []
        for unique_label_idxs in same_label_idxs:
            if mean:
                    mean_pred = np.mean(pred_rep[unique_label_idxs], axis=0)
                    pred_list.append(mean_pred)
                    true_list.append(y_true[unique_label_idxs[0]])
            else:
                pred_list.append(pred_rep[unique_label_idxs])
                true_list.append(y_true[unique_label_idxs])
        preds = np.vstack(pred_list)
        trues = np.vstack(true_list)

        xx = np.vstack([trues[:,x_idx],preds[:,x_idx]])
        yy = np.vstack([trues[:,y_idx],preds[:,y_idx]])

        ax.plot(xx, yy, '-y', linewidth = 0.25, alpha=0.5, zorder = 2.5)
        ax.scatter(preds[:,x_idx], preds[:,y_idx], s=2, alpha=0.8, zorder = 1.5, label = f"Predicted Labels R{r}")

        if r==0:
            ax.scatter(trues[:,x_idx], trues[:,y_idx], color="red", s=2, alpha=0.9, zorder = 3.5, label = f"True Labels")
    
    ax.set_xlabel(f"{analytes_names[x_idx]} concentration nMol")
    ax.set_ylabel(f"{analytes_names[y_idx]} concentration nMol")
    
    fig.savefig(filename, dpi=400, bbox_inches='tight', transparent=False)
    plt.close(fig)

def repetitions_connected_pairs_3d(y_true, y_pred, filename, analytes_names=['DA', '5HT', 'NE'], model_description="", mean=True):
    """

    UNTESTED!!!!

    TODO TEST, Also this is too ugly

    Function to plot the connected pair plot for each repetition to show the variation between models in 3D.
    Defaults to plot a mean prediction per unique label per model.

    Calls repetitions_connect_pairs_2d for all valid combinations

    Parameters:
        y_true : np.ndarray
            The true values of the labels from each of the R repetitions
        y_pred : np.ndarray
            The predicted values of the labels from "edge" data
        filename : String
            The Absolute or relative path to save the output graph(s)
        analytes_name : List
            The names of the analytes in order corresponding to the y_true/y_pred.
            Will be replaced with default if length is less than 3.
            Default: ['DA', '5HT', 'NE']
        model_description : String
            A descriptive string to add to the title of the graphs
            Default: ""
        mean : boolean
            If the mean for each unqiue label should be used instead of all points
            Default: True
        
    """
    
    if y_true.shape[1] != 3:
        return

    same_label_idxs = []
    _, unique_indicies = np.unique(y_true, return_index=True, axis=0)
    for unique_idx in unique_indicies:
        same_label_idxs.append(np.where((y_true == y_true[unique_idx,:]).all(axis=1))[0])

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')

    for r, pred_rep in enumerate(y_pred):
        pred_list = []
        true_list = []
        for unique_label_idxs in same_label_idxs:
            if mean:
                    mean_pred = np.mean(pred_rep[unique_label_idxs], axis=0)
                    pred_list.append(mean_pred)
                    true_list.append(y_true[unique_label_idxs[0]])
            else:
                pred_list.append(pred_rep[unique_label_idxs])
                true_list.append(y_true[unique_label_idxs])
        preds = np.vstack(pred_list)
        trues = np.vstack(true_list)

        xx = np.vstack([trues[:,0],preds[:,0]])
        yy = np.vstack([trues[:,1],preds[:,1]])
        zz = np.vstack([trues[:,2],preds[:,2]])

        for x, y, z in zip(xx.T, yy.T, zz.T):
            ax.plot3D(x,y,z, '-y', linewidth = 0.25, alpha=0.5, zorder = 2.5)
        ax.scatter3D(preds[:,0], preds[:,1], preds[:,2], s=2, alpha=0.8, zorder = 1.5, label = f"Predicted Labels R{r}")

        if r==0:
            ax.scatter3D(trues[:,0], trues[:,1], trues[:,2], color="red", s=2, alpha=0.9, zorder = 3.5, label = f"True Labels")
    

    ax.text2D(0.05, 0.95, f"{model_description} Repetition Comparison {analytes_names[0]}, {analytes_names[1]}, and {analytes_names[2]}", transform=ax.transAxes)

    ax.set_xlabel(f"{analytes_names[0]} concentration nMol")
    ax.set_ylabel(f"{analytes_names[1]} concentration nMol")
    ax.set_zlabel(f"{analytes_names[2]} concentration nMol")
    
    fig.savefig(filename, dpi=400, bbox_inches='tight', transparent=False)
    plt.close(fig)

    combinations = list(itertools.combinations([i for i in range(len(analytes_names))],2))
    for idx_a, idx_b in combinations:
        repetitions_connected_pairs_2d(y_true, y_pred, _filename_helper(filename,f"_2d_{analytes_names[idx_a]}_{analytes_names[idx_b]}"), 
                                                            analytes_names, 
                                                            model_description=f"{model_description} {analytes_names[idx_a]}, {analytes_names[idx_b]}",
                                                            indicies=[idx_a, idx_b])

def connected_triplets(y_true, edge_pred, all_pred, filename, analytes_name = ['DA', '5HT'], model_description=""):
    """
    Function to plot sets of connected triplets representing the true and predicted value from two models for a given sweep.
    These can be quite busy.

    Parameters:
        y_true : np.ndarray
            The true values of the labels
        edge_pred : np.ndarray
            The predicted values of the labels from "edge" data
        all_pred : np.ndarray
            The predicted values of the labels from "all" data
        filename : String
            The Absolute or relative path to save the output graph(s)
        analytes_name : List
            The names of the analytes in order corresponding to the y_true/y_pred.
            Will be replaced with default if length is less than 3.
            Default: ['DA', '5HT']
        model_description : String
            A descriptive string to add to the title of the graphs
            Default: ""
    """
    xx_a = np.vstack([y_true[:,0],edge_pred[:,0]])
    yy_a = np.vstack([y_true[:,1],edge_pred[:,1]])

    xx_b = np.vstack([y_true[:,0],all_pred[:,0]])
    yy_b = np.vstack([y_true[:,1],all_pred[:,1]])

    fig = plt.figure(figsize=(8,8))

    plt.scatter(y_true[:,0], y_true[:,1], color="red", s=2, alpha=0.9, zorder = 3.5, label = "True Labels")
    plt.scatter(edge_pred[:,0], edge_pred[:,1], color="blue", s=2, alpha=0.2, zorder = 1.5, label = "Edge Model")
    plt.scatter(all_pred[:,0], all_pred[:,1], color="green", s=2, alpha=0.2, zorder = 1.5, label = "All Model")
    plt.plot(xx_a, yy_a, '-c', linewidth = 0.25, alpha = 0.1, zorder = 2.5)
    plt.plot(xx_b, yy_b, '-y', linewidth = 0.25, alpha = 0.1, zorder = 2.5)

    plt.xlabel(analytes_name[0])
    plt.ylabel(analytes_name[1])
    plt.legend(loc="upper right")
    plt.title(f"{model_description} Real vs Predicted {analytes_name[0]}, and {analytes_name[1]}")
    plt.grid(True)

    fig.savefig(filename, dpi=400, bbox_inches='tight', transparent=False)
    plt.close(fig)

def analyte_true_pred(y_true, y_pred, analyte_idx, filename, analytes_names=['DA', '5HT', 'NE'], model_description=""):
    """
    Function to plot the true vs predicted values of analyte: analytes_names[analytes_idx]
    The true value is the x axis prediction is the y axis.

    Parameters:
        y_true : np.ndarray
            The true values of the labels
        y_pred : np.ndarray
            The predicted values of the labels
        analyte_idx : int
            The analyte that is being graphed
        filename : String
            The Absolute or relative path to save the output graph(s)
        analytes_name : List
            The names of the analytes in order corresponding to the y_true/y_pred.
            Will be replaced with default if length is less than 3.
            Default: ['DA', '5HT', 'NE']
        model_description : String
            A descriptive string to add to the title of the graphs
            Default: ""
    """
    fig = plt.figure(figsize=(8,8))
    plt.scatter(y_true[:, analyte_idx], y_pred[:, analyte_idx])
    plt.xlabel(f"True Concentration {analytes_names[analyte_idx]}")
    plt.ylabel(f"Predicted Concentration {analytes_names[analyte_idx]}")
    plt.title(f"{model_description} Real vs Predicted {analytes_names[analyte_idx]}")
    fig.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close(fig)

def analyte_pred_comp(y_true, y_pred, true_idx, pred_idx, filename, exlude_nonzero=False):
    fig = plt.figure(figsize=(8,8))
    if exlude_nonzero:
        temp = y_true[:, true_idx]
        valid_idx = np.where(temp == 0)
        x_vals = temp[valid_idx]
        y_vals = y_pred[:,pred_idx][valid_idx]
    else:
        x_vals = y_true[:,pred_idx]
        y_vals = y_pred[:,pred_idx]
    plt.scatter(x_vals, y_vals)
    fig.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close(fig)


def analytes_comp_arr(
        y_true, 
        y_pred, 
        filename, 
        analytes_names=['DA', '5HT', 'pH', 'NE'], 
        exclude_nonzero=True
    ):
    """
    Function to plot the true value of analyte_x vs predicted value of 
    analyte_y for all combinations of analytes. For the case in which analyte_x
    is not analte_y the points of interest are constrained such that true value
    of analyte_y is 0 (if exlude_nonzero is True).

    Parameters:
        y_true : np.ndarray
            The true values of the labels
        y_pred : np.ndarray
            The predicted values of the labels
        filename : String
            The Absolute or relative path to save the output graph(s)
        analytes_name : List
            The names of the analytes in order corresponding to the y_true/y_pred.
            Will be replaced with default if length is less than 3.
            Default: ['DA', '5HT', 'pH', 'NE']
        exclude_nonzero : Boolean
            If nonzero values should be exluded for cross analyte comparision
            Default: True
    """

    num_analytes = len(analytes_names)
    size_ph = num_analytes
    if 'pH' in analytes_names:
        size_ph = num_analytes -1
    fig, axs = plt.subplots(size_ph, size_ph, tight_layout=True, figsize=(12, 8))
    for it in range(num_analytes):
        if analytes_names[it] == 'pH':
             continue
        for ip in range(num_analytes):
            if analytes_names[ip] == 'pH':
                 continue
            yt, yp = y_true[:,it], y_pred[:,ip]
            if it != ip and exclude_nonzero:
                ptz = y_true[:,ip] == 0.0
                yt, yp = yt[ptz], yp[ptz]
            itx = it-1 if it>=(num_analytes-1) else it
            ipx = ip-1 if ip>=(num_analytes-1) else ip
            axs[itx, ipx].scatter(yt, yp, s=1)
            axs[itx, ipx].set_xlabel('True %s (nmol/L)'%analytes_names[it])
            axs[itx, ipx].set_ylabel('Predicted %s (nmol/L)'%analytes_names[ip])
            #axs[itx, ipx].set_ylim(-1000, 3000)
    #fig.suptitle(f"Analyte Correlations")
    fig.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close(fig)

def analytes_comp_arr_all(
        y_true, 
        y_pred, 
        filename, 
        analytes_names=['DA', '5HT', 'pH', 'NE'], 
        exclude_nonzero=True, 
        exclude_pH=False, 
        use_mean=False, 
        sd_error_bars=False, 
        use_residuals=False, 
        n_bins=None
    ):
    """
    Function to plot the true value of analyte_x vs predicted value of 
    analyte_y for all combinations of analytes. For the case in which analyte_x
    is not analte_y the points of interest are constrained such that true value
    of analyte_y is 0 (if exlude_nonzero is True).

    Parameters:
        y_true : np.ndarray
            The true values of the labels
        y_pred : np.ndarray
            The predicted values of the labels
        filename : String
            The Absolute or relative path to save the output graph(s)
        analytes_name : List
            The names of the analytes in order corresponding to the y_true/y_pred.
            Will be replaced with default if length is less than 3.
            Default: ['DA', '5HT', 'pH', 'NE']
        exclude_nonzero : Boolean
            If nonzero values should be exluded for cross analyte comparision
            Default: True
        use_mean : Boolean
            If true, the plot will use means and error bars
            Defualt: False
        sd_error_bars : Boolean
            If true, the error bars will be standard deviation, else will be standard error of the mean SEM.
            Inconsequntial if 
            Default: False
        n_bins : int
            If set bins the data into n_bins and calculates error bars for those points.
            Overrides use_mean and sd_error_bars
    """

    num_analytes = len(analytes_names)
    n_plots = num_analytes-1 if exclude_pH and 'pH' in analytes_names else num_analytes
    fig, axs = plt.subplots(n_plots, n_plots, tight_layout=True, figsize=(12, 8), squeeze=False)
    for it in range(num_analytes):
        if analytes_names[it] == 'pH' and exclude_pH:
                 continue
        for ip in range(num_analytes):
            if analytes_names[ip] == 'pH' and exclude_pH:
                 continue
            
            # yr used to calculate the residual
            yt, yp, yr = y_true[:,it], y_pred[:,ip], y_true[:,ip]
            
            # Exclude the non-zero points
            if it != ip and exclude_nonzero:
                if analytes_names[ip] == 'pH':
                    ptz = np.isclose(y_true[:,ip], 7.4, atol = 0.05)
                else:
                    ptz = y_true[:,ip] == 0.0
                yt, yp, yr = yt[ptz], yp[ptz], yr[ptz]

            if use_residuals and it != ip:
                # Calculate residuals
                # off diagonals
                y_res = yp - yr
                yp = y_res

            itx = it-1 if it>(n_plots-1) else it
            ipx = ip-1 if ip>(n_plots-1) else ip

            if n_bins is not None:
                
                # calculate the bins
                max = np.amax(y_true)
                increment = max / n_bins

                bin_indicies = []
                for i in range(n_bins):
                    bin_indicies.append(np.where((yt>i*increment)&(yt<=(i+1)*increment)))

                bins = [(yt[idxs], yp[idxs]) for idxs in bin_indicies]
                t_means = np.array([np.mean(yt) for yt, _ in bins])
                p_means = np.array([np.mean(yp) for _, yp in bins])
                x_sem = np.array([stats.sem(yt) for yt, _ in bins])
                y_sem = np.array([stats.sem(yp) for _, yp in bins])

                axs[itx, ipx].errorbar(t_means, p_means, yerr=y_sem, xerr=x_sem, fmt='o', ms=1, elinewidth=0.5, capsize=2)

            # Calc means and error bars
            elif use_mean:
                # caculate the mean for each x value
                y_true_unique = np.unique(yt)
                #TODO: Finish this cleanly
                y_pred_avg = [np.mean(yp[np.where(yt==ytu)]) for ytu in y_true_unique]

                if sd_error_bars:
                    # cacluate the error bars in STD
                    y_pred_std = [np.std(yp[np.where(yt==ytu)]) for ytu in y_true_unique]
                    axs[itx, ipx].errorbar(y_true_unique, y_pred_avg, yerr=y_pred_std, fmt='o', ms=1, elinewidth=0.5, capsize=2)
                else:
                    # Standard Error (SEM) error bars
                    y_pred_sem = [stats.sem(yp[np.where(yt==ytu)]) for ytu in y_true_unique]
                    axs[itx, ipx].errorbar(y_true_unique, y_pred_avg, yerr=y_pred_sem, fmt='o', ms=1, elinewidth=0.5, capsize=2)
                #axs[itx, ipx].scatter(y_true_unique, y_pred_avg, s=1)
            else:
                axs[itx, ipx].scatter(yt, yp, s=1)
            
            # Set x label accordingly for pH
            if analytes_names[it] == 'pH': # x?
                axs[itx, ipx].set_xlabel('True %s '%analytes_names[it])
            else:
                axs[itx, ipx].set_xlabel('True %s (nmol/L)'%analytes_names[it])
            
            if use_residuals:
                res_string = " residual"
            else:
                res_string = ""
            # Set y label accordingly for pH
            if analytes_names[ip] == 'pH':
                axs[itx, ipx].set_ylabel(f'Predicted {analytes_names[ip]}{res_string}')
                axs[itx, ipx].set_ylim(6.75, 8)
            else:
                axs[itx, ipx].set_ylabel(f'Predicted {analytes_names[ip]}{res_string} (nmol/L)')
                axs[itx, ipx].set_ylim(-1000, 3000)
    #fig.suptitle(f"Analyte Correlations")
    fig.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close(fig)

def time_series_model_comp(
        pred_arr, 
        filename, 
        analyte_names=['DA', '5HT', 'pH', 'NE'], 
        truth=None, 
        range=None
    ):
    """
    Displays the predictions as a time series.
    """

    for idx, analyte in enumerate(analyte_names):
        fig = plt.figure(figsize=(18,8))
        for r, pred in enumerate(pred_arr):
            if range is None:
                plt.plot(pred[:,idx], lw=0.3, label=f"Model Repetition {r}")
            else:
                plt.plot(pred[:,idx][range[0]:range[1]], lw=0.6, label=f"Model Repetition {r}")
        
        if truth is not None:
            if range is None:
                plt.plot(truth[:,idx], lw=2, label="True Values")
            else:
                plt.plot(truth[:,idx][range[0]:range[1]], lw=2, label="True Values")

        plt.ylabel(f"Prediced Concentration {analyte_names[idx]}")
        plt.xlabel(f"Time")
        plt.legend(loc="upper right")
        plt.title(f"Models' Predictions {analyte_names[idx]}")
        fig.savefig(_filename_helper(filename,analyte_names[idx]), dpi=400, bbox_inches='tight')
        plt.close(fig)

def moving_average(a, n=3):
    """
    Helper function for rolling/moving averages

    parameters:
        a : np nd array
            The array to have rolling/moving averages about
        n : int
            The number of points before and after to load

    returns:
        A smaller array shifted
    """

    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def make_data_snippets(
        y_pred, 
        pairs, 
        filename, 
        start_offset=55, 
        end_offset=85, 
        analytes=['DA', '5HT', 'pH', 'NE'], 
        overwrite=False
    ): 
    """
    Produces data snippets around TTL events. 
    Is this a graphing util? Maybe?
    I am thinking of it as helper function that allows us to make graphs after the fact.

    parameters:
        y_pred : np nd array
            The predictions of the analytes in ['DA', '5HT', 'pH', 'NE'] order
        paris : list of tuples
            Each tuple contains:
                The sweep number that the TTL is in
                The index of the point in the sweep that contains the rising edge of the ttl
        filename : str 
            What to save data as
        start_offset : int (Default: 65)
            How many sweeps (10 Hz) to load before the TTL
            Should be larger than the plots to be made.
        end_offset : int (Default: 115)
            How many sweeps (10 Hz) to load after the TTL
            Should be larger than the plots to be made.
        analytes : list of str (Default: ['DA', '5HT', 'pH', 'NE'])
            The analytes that are predicted, in order.
        overwrite : Boolean (Default: False)
            If an existing file should be overwritten.

    output:
        Saves a CSV file for each TTL
    """

    manipulated_data = y_pred

    for ttl_idx, pair in enumerate(pairs):
        idx, loc = pair

        if idx < start_offset+1:
            continue

        #################################################################
        # Non-Averaged Data
        #################################################################

        # We are going to work in units of 1e-5 s (The polling rate)
        pred_snippet = manipulated_data[idx-start_offset-1:idx+end_offset+1]

        # Setting the start of the window to zero
        pred_y = pred_snippet
        center_of_triange_offset = 159+500 # Use the center of the triangle as the time where the measurement took place
        pred_x = np.linspace(center_of_triange_offset, center_of_triange_offset+(len(pred_y)*10000), len(pred_y), endpoint=False) 
        zero = start_offset*10000+loc
        # Final x values converted back into seconds
        adjusted_x = (pred_x-zero)/1e5

        # Check to see what is closer
        # find the location with the smallest abs
        real_x = adjusted_x[np.argmin(np.abs(adjusted_x))-start_offset : np.argmin(np.abs(adjusted_x))+end_offset+1]
        quantized_x = np.linspace(-start_offset/10.0, (-start_offset/10.0)+((len(real_x)-1)*0.1), len(real_x))
        shifted_pred_y = pred_y[np.argmin(np.abs(adjusted_x))-start_offset : np.argmin(np.abs(adjusted_x))+end_offset+1]

        snippet_dict = {"True Time" : real_x, "Approximate Time" : quantized_x} 

        for idx, analyte in enumerate(analytes):
            snippet_dict[analyte] = shifted_pred_y[:,idx]

        for key in snippet_dict.keys():
            print(f"Key: {key}, Shape: {snippet_dict[key].shape}")

        # Check if the file already exists
        data_snippet_file_path = f"{filename}_TTL_{ttl_idx}.csv"
        if not os.path.isfile(data_snippet_file_path) or overwrite:
            pd.DataFrame(snippet_dict).to_csv(data_snippet_file_path)
        else:
            print(f"File already exists: {data_snippet_file_path}")

def make_data_snippets_larb(
        y_pred, 
        pairs, 
        filename, 
        start_offset=550, 
        end_offset=850, 
        analytes=['DA', '5HT', 'pH', 'NE'], 
        overwrite=True
    ): 
    """
    Produces data snippets around TTL events. 
    Is this a graphing util? Maybe?
    I am thinking of it as helper function that allows us to make graphs after the fact.

    parameters:
        y_pred : np nd array
            The predictions of the analytes in ['DA', '5HT', 'pH', 'NE'] order
        paris : list of tuples
            Each tuple contains:
                The sweep number that the TTL is in
                The index of the point in the sweep that contains the rising edge of the ttl
        filename : str 
            What to save data as
        start_offset : int (Default: 65)
            How many sweeps (10 Hz) to load before the TTL
            Should be larger than the plots to be made.
        end_offset : int (Default: 115)
            How many sweeps (10 Hz) to load after the TTL
            Should be larger than the plots to be made.
        analytes : list of str (Default: ['DA', '5HT', 'pH', 'NE'])
            The analytes that are predicted, in order.
        overwrite : Boolean (Default: False)
            If an existing file should be overwritten.s

    output:
        Saves a CSV file for each TTL

    """

    manipulated_data = y_pred

    for ttl_idx, pair in enumerate(pairs):
        idx, loc = pair

        if idx < start_offset+1:
            continue

        #################################################################
        # Non-Averaged Data
        #################################################################

        pred_snippet = manipulated_data[idx-start_offset-1:idx+end_offset+1]

        # Setting the start of the window to zero
        pred_y = pred_snippet
        center_of_triange_offset = 15+500 # Use the center of the triangle as the time where the measurement took place
        pred_x = np.linspace(center_of_triange_offset, center_of_triange_offset+(len(pred_y)*1032), len(pred_y), endpoint=False) 
        zero = start_offset*1032+loc
        # Final x values converted back into seconds
        adjusted_x = (pred_x-zero)/1e5

        # Check to see what is closer
        # find the location with the smallest abs
        real_x = adjusted_x[np.argmin(np.abs(adjusted_x))-start_offset : np.argmin(np.abs(adjusted_x))+end_offset+1]
        quantized_x = np.linspace(-start_offset*1032/100000, (-start_offset*1032/100000)+((len(real_x)-1)*1032/100000), len(real_x))
        shifted_pred_y = pred_y[np.argmin(np.abs(adjusted_x))-start_offset : np.argmin(np.abs(adjusted_x))+end_offset+1]

        snippet_dict = {"True Time" : real_x, "Approximate Time" : quantized_x} 

        for idx, analyte in enumerate(analytes):
            snippet_dict[analyte] = shifted_pred_y[:,idx]

        for key in snippet_dict.keys():
            print(f"Key: {key}, Shape: {snippet_dict[key].shape}")

        # Check if the file already exists
        data_snippet_file_path = f"{filename}_TTL_{ttl_idx}.csv"
        if not os.path.isfile(data_snippet_file_path) or overwrite:
            pd.DataFrame(snippet_dict).to_csv(data_snippet_file_path)
        else:
            print(f"File already exists: {data_snippet_file_path}")
        

def make_data_plot_snippets(
        named_file_dict, 
        out_filename, 
        rolling_window=0.5, 
        start_offset=50, 
        end_offset=100, 
        z_score=True, 
        analytes=['DA', '5HT', 'NE'], 
        tick_spacing=1, 
        ttl_line_clr="blue", 
        y_min=-5, 
        y_max=15
    ):
    """ 
    Produces snippet plots from data snippets

    But the different thing about this one is that it takes a named_file_dict.
    What is a named_file_dict? I don't know.

    TODO Finish
    """
    
    # Make a figure
    fig = plt.figure(figsize=(12, 8), dpi=400)
    ax = fig.add_subplot(111)

    for name in named_file_dict.keys():
        f_name = named_file_dict[name][0]
        color = named_file_dict[name][1]
        snippet_df = pd.read_csv(f_name)

        for analyte in analytes:
            # We use "full" here to denote that the data snippet may be larger 
            # than the snippet we want to plot.
            full_analyte = np.array(snippet_df[analyte])
            full_TIME = np.array(snippet_df['True Time'])

            # This pulls out Just the acutal snippet
            s = (-0.1*start_offset)
            e = (0.1*end_offset)
            snippet_analyte = full_analyte[np.where(abs(s+e-2*full_TIME)<(e-s))]
            snippet_TIME = full_TIME[np.where(abs(s+e-2*full_TIME)<(e-s))]

            print(np.where(snippet_TIME<0))

            # The part of snippet to base the z-score on
            z_ref = snippet_analyte[np.where(snippet_TIME<0)]

            # Project the whole thinig to Z-space
            z_full_analyte = stats.zmap(full_analyte, z_ref, axis=0)

            if rolling_window > 0: # New rolling window in terms of seconds
                # Need to calcualte the size of the window based on the length of time
                window_size = int(rolling_window / (snippet_TIME[1] - snippet_TIME[0]))
                rw_z_full_analyte = moving_average(z_full_analyte, window_size)
                indices = np.where(abs(s+e-2*full_TIME)<(e-s))
                print(indices)
                shift_indices = (np.array([i-int((window_size-1)/2) for i in indices[0]]),)
                print(shift_indices)
                shift_rw_z_snippet_analyte = rw_z_full_analyte[shift_indices]
                to_use_analyte = shift_rw_z_snippet_analyte
            else:
                z_snippet_analyte = z_full_analyte[np.where(abs(s+e-2*full_TIME)<(e-s))]
                to_use_analyte = z_snippet_analyte

            # TODO combine name and analyte inteligently
            ax.plot(snippet_TIME, to_use_analyte, label=name, lw=3, color=color)

        # Plot the vertical line
        ax.axvline(x=0, c=ttl_line_clr, linestyle='--', lw=3)        

        ax.set_xlabel(f"Time (s)")
        ax.set_ylabel(f"{'NT' if len(analytes)>1 else analytes[0]} [Z]")
        ax.set_xticks(np.arange(-start_offset/10, end_offset/10+1, tick_spacing))
        ax.set_xlim((1.05 * (-start_offset/10)), (end_offset/10*1.05))

        if y_min is not None and y_max is not None:
            ax.set_ylim((y_min,y_max))

        if len(analytes)>1 or len(named_file_dict)>1:
            ax.legend(loc="upper right")
        #ax.set_title(f"DA Predictions")

        fig.savefig(out_filename, dpi=400, bbox_inches='tight')
        plt.close(fig)

    raise NotImplementedError

def make_snippets(
        y_pred, 
        pairs, 
        filename, 
        start_offset=50, 
        end_offset=100, 
        z_score=True, 
        rolling_window=5, 
        analytes=['DA', '5HT', 'NE'], 
        predicted_analytes=['DA', '5HT', 'pH', 'NE'], 
        tick_spacing=1, 
        scatter=False, 
        title_str=""
    ): 
    """
    Produces snippet plots around TTL events

    parameters:
        y_pred : np nd array
            The predictions of the analytes in ['DA', '5HT', 'pH', 'NE'] order
        paris : list of tuples
            Each tuple contains:
                The sweep number that the TTL is in
                The index of the point in the sweep that contains the rising edge of the ttl
        filename : str 
            What to save the images as. Appended with the TTL number
        start_offset : int (Default: 30)
            How many sweeps (10 Hz) to load before the TTL
        end_offset : int (Default: 60)
            How many sweeps (10 Hz) to load after the TTl
        z_score : boolean (Default: True)
            If true a z_score will be taken based on the part of the snippet before 
        rolling_window : int (Default: 3)
            If greater than a rolling window of that length will be used to calcualte a mean. The mean will be shifted to align with the original predictions.
        analytes : list (Default: ['DA', '5HT', 'NE'])
            The list of analytes that will be included in the snippet
        tick_spacing : int
            The spacing between the ticks on the x axis
        scatter : boolean
            If the a scatter plot should be made instead of regular plots

    output:
        Creates a plot for each TTL in each File

    """

    manipulated_data = y_pred

    for ttl_idx, pair in enumerate(pairs):
        idx, loc = pair

        fig = plt.figure(figsize=(12, 8), dpi=400)
        ax = fig.add_subplot(111)

        #################################################################
        # Non-Averaged Data
        #################################################################

        # We are going to work in units of 1e-5 s (The polling rate)
        pred_snippet = manipulated_data[idx-start_offset:idx+end_offset]
        
        if z_score:
            # Use the window (len = start_offset) before the TTL to determine the projection to z-space
            z_ref = manipulated_data[idx-start_offset:idx]
            # Project into z-space
            snippet_manipulated_data = stats.zmap(manipulated_data, z_ref, axis=0)

            # Update the snippet
            pred_snippet = snippet_manipulated_data[idx-start_offset:idx+end_offset]
        else:
            snippet_manipulated_data = manipulated_data

        # Caculate the rolling window with the snippet_manipuated_data
        if rolling_window > 1:
            rm_data = [moving_average(snippet_manipulated_data[:,i], rolling_window) for i in range(snippet_manipulated_data.shape[1])]
            rm_data = np.hstack([arr.reshape((arr.shape[0],1)) for arr in rm_data])

        # Setting the start of the window to zero
        pred_y = pred_snippet
        center_of_triange_offset = 159+500 # Use the center of the triangle as the time where the measurement took place
        pred_x = np.linspace(center_of_triange_offset, center_of_triange_offset+(len(pred_y)*10000), len(pred_y), endpoint=False) 
        zero = start_offset*10000+loc
        # Final x values converted back into seconds
        adjusted_x = (pred_x-zero)/1e5

        #################################################################
        # Rolling-Mean Data
        #################################################################
        if rolling_window>1:
            rm_snippet = rm_data[idx-start_offset:idx+end_offset]
            rm_adjusted_x = adjusted_x + ((rolling_window-1)*0.05)
    
        ax.axvline(x=0, c='black')
        for idx, analyte in enumerate(predicted_analytes):
            if analyte not in analytes:
                continue
            if scatter:
                ax.scatter(adjusted_x, pred_snippet[:,idx], label=analyte)
            else:
                if rolling_window > 1:
                    ax.plot(rm_adjusted_x[:len(rm_snippet[:,idx])], rm_snippet[:,idx], label=f"{analyte} Rolling Mean (window: {rolling_window})")
                else:
                    ax.plot(adjusted_x, pred_snippet[:,idx], label=analyte)
        ax.set_xlabel(f"Time in Seconds Relative to TTL")
        if z_score:
            ax.set_ylabel(f"Z-Scored Predicted Value")
        else:
            ax.set_ylabel(f"Predicted Value (nmol/L)")
        ax.set_xticks(np.arange(-start_offset/10, end_offset/10+1, tick_spacing))
        ax.set_xlim((1.05 * (-start_offset/10)), (end_offset/10*1.05))
        # if z_score:
        #     ax.set_ylim((-5,5))
        ax.legend(loc="upper left")
        ax.set_title(f"{title_str}Snippet of TTL {ttl_idx}")

        fig.savefig(_filename_helper(filename,str(ttl_idx)), dpi=400, bbox_inches='tight')
        plt.close(fig)

def make_seth_snippets(
        y_pred, 
        pairs, 
        filename, 
        start_offset=50, 
        end_offset=100, 
        z_score=True, 
        rolling_window=5, 
        analytes=['DA', '5HT', 'NE'], 
        predicted_analytes=['DA', '5HT', 'pH', 'NE'], 
        tick_spacing=1, 
        scatter=False, 
        title_str="", 
        ttl_line_clr="blue", 
        overwrite=True, 
        z_method="before", 
        zlim=(-3,13), 
        ylim=(None, None)
    ): 
    """
    Produces snippet plots around TTL events

    This time we are using parameters that Seth prefers.

    parameters:
        y_pred : np nd array
            The predictions of the analytes in ['DA', '5HT', 'pH', 'NE'] order
        paris : list of tuples
            Each tuple contains:
                The sweep number that the TTL is in
                The index of the point in the sweep that contains the rising edge of the ttl
        filename : str 
            What to save the images as. Appended with the TTL number
        start_offset : int (Default: 30)
            How many sweeps (10 Hz) to load before the TTL
        end_offset : int (Default: 60)
            How many sweeps (10 Hz) to load after the TTl
        z_score : boolean (Default: True)
            If true a z_score will be taken based on the part of the snippet before 
        rolling_window : int (Default: 3)
            If greater than a rolling window of that length will be used to calcualte a mean. The mean will be shifted to align with the original predictions.
        analytes : list (Default: ['DA', '5HT', 'NE'])
            The list of analytes that will be included in the snippet
        tick_spacing : int
            The spacing between the ticks on the x axis
        scatter : boolean
            If the a scatter plot should be made instead of regular plots
        title_str : String (Default: "")
        ttl_line_clr : string (Default: "blue")
            The string color to be used for the veritcle line in the snippets
        overwrite : boolean (Default: True)
        z_method : string (Default: "before")
            The method of z-scoring, ignored if z-score is false.
            Methods:
                before : use the section from start_offset to zero 
                whole : use the whole snippet, start_offset to end_offset
        zlim : tuple of int (Default: (-3,13))
        ylim : typle of int (Default: (None,None)
    """

    manipulated_data = y_pred

    for ttl_idx, pair in enumerate(pairs):
        idx, loc = pair

        fig = plt.figure(figsize=(12, 8), dpi=400)
        ax = fig.add_subplot(111)

        #################################################################
        # Non-Averaged Data
        #################################################################

        # We are going to work in units of 1e-5 s (The polling rate)
        pred_snippet = manipulated_data[idx-start_offset:idx+end_offset]
        
        if z_score:
            if z_method == "before":
                # Use the window (len = start_offset) before the TTL to determine the projection to z-space
                z_ref = manipulated_data[idx-start_offset:idx]
            elif z_method == "whole":
                z_ref = manipulated_data[idx-start_offset:idx+end_offset]
            else:
                raise ValueError("Accepted Values for z_method: 'before' and 'whole'")
            # Project into z-space
            snippet_manipulated_data = stats.zmap(manipulated_data, z_ref, axis=0)

            # Update the snippet
            pred_snippet = snippet_manipulated_data[idx-start_offset:idx+end_offset]
        else:
            snippet_manipulated_data = manipulated_data

        # Caculate the rolling window with the snippet_manipuated_data
        if rolling_window > 1:
            rm_data = [moving_average(snippet_manipulated_data[:,i], rolling_window) for i in range(snippet_manipulated_data.shape[1])]
            rm_data = np.hstack([arr.reshape((arr.shape[0],1)) for arr in rm_data])

        # Setting the start of the window to zero
        pred_y = pred_snippet
        center_of_triange_offset = 159+500 # Use the center of the triangle as the time where the measurement took place
        pred_x = np.linspace(center_of_triange_offset, center_of_triange_offset+(len(pred_y)*10000), len(pred_y), endpoint=False) 
        zero = start_offset*10000+loc
        # Final x values converted back into seconds
        adjusted_x = (pred_x-zero)/1e5

        #################################################################
        # Rolling-Mean Data
        #################################################################
        if rolling_window>1:
            rm_snippet = rm_data[idx-start_offset:idx+end_offset]
            rm_adjusted_x = adjusted_x + ((rolling_window-1)*0.05)

            # Modify it
            if not z_score:
                for idx, analyte in enumerate(predicted_analytes):
                    rm_snippet[:,idx] = rm_snippet[:,idx] - rm_snippet[start_offset-2,idx]
    
        ax.axvline(x=0, c=ttl_line_clr, linestyle='--', linewidth=3.0) # 'red' or 'blue' depending on light
        for idx, analyte in enumerate(predicted_analytes):
            if analyte not in analytes:
                continue
            if scatter:
                ax.scatter(adjusted_x, pred_snippet[:,idx], label=analyte)
            else:
                a_c = {"DA":'k', "5HT":'m', "NE":'c'}[analyte]
                if rolling_window > 1:
                    ax.plot(rm_adjusted_x[:len(rm_snippet[:,idx])], rm_snippet[:,idx], color=a_c, linewidth=3.0, label=f"{analyte}") # Rolling Mean (window: {rolling_window})")
                else:
                    ax.plot(adjusted_x, pred_snippet[:,idx], color=a_c, label=analyte, linewidth=3.0)
        ax.set_xlabel(f"Time (s)")
        if z_score:
            ax.set_ylabel(f"NT [z]")
        else:
            ax.set_ylabel(f"Predicted Value (nmol/L)")
        ax.set_xticks(np.arange(-start_offset/10, end_offset/10+1, tick_spacing))
        ax.set_xlim((1.05 * (-start_offset/10)), (end_offset/10*1.05))

        if z_score and zlim is not None:
            ax.set_ylim((zlim[0],zlim[1]))

        if not z_score and ylim[0] is not None:
            ax.set_ylim(ylim[0], ylim[1])

        ax.legend(loc="upper left")
        # ax.set_title(f"{title_str}TTL {ttl_idx}")

        fig_file_name = _filename_helper(filename,str(ttl_idx))

        if not os.path.isfile(fig_file_name) or overwrite:
            fig.savefig(fig_file_name, dpi=400, bbox_inches='tight')
        else:
            print(f"File already exists: {fig_file_name}")

        plt.close(fig)

def averaged_snippets_in_file(
        y_pred, 
        pairs, 
        filename, 
        start_offset=50, 
        end_offset=100, 
        z_score=True, 
        rolling_window=5, 
        analytes=['DA', '5HT', 'NE'], 
        tick_spacing=1, 
        scatter=False, 
        title_str=""
    ):
    """
    A function that makes snippets averaged over all the TTL's over the file.
    """

    mean_window_x_list =[]
    mean_window_y_list_DA =[]
    mean_window_y_list_5HT =[]
    mean_window_y_list_NE =[]

    manipulated_data = y_pred

    for ttl_idx, pair in enumerate(pairs):
        idx, loc = pair
        
        if idx < start_offset+1:
            continue

        #################################################################
        # Non-Averaged Data
        #################################################################

        # We are going to work in units of 1e-5 s (The polling rate)
        pred_snippet = manipulated_data[idx-start_offset:idx+end_offset]
        
        if z_score:
            # Use the window (len = start_offset) before the TTL to determine the projection to z-space
            z_ref = manipulated_data[idx-start_offset:idx]
            # Project into z-space
            snippet_manipulated_data = stats.zmap(manipulated_data, z_ref, axis=0)

            # Update the snippet
            pred_snippet = snippet_manipulated_data[idx-start_offset:idx+end_offset]
        else:
            snippet_manipulated_data = manipulated_data

        # Caculate the rolling window with the snippet_manipuated_data
        # if rolling_window > 1:
        #     rm_data = [moving_average(snippet_manipulated_data[:,i], rolling_window) for i in range(4)]
        #     rm_data = np.hstack([arr.reshape((arr.shape[0],1)) for arr in rm_data])

        # Setting the start of the window to zero
        pred_y = pred_snippet
        center_of_triange_offset = 159+500 # Use the center of the triangle as the time where the measurement took place
        pred_x = np.linspace(center_of_triange_offset, center_of_triange_offset+(len(pred_y)*10000), len(pred_y)) 
        zero = start_offset*10000+loc
        # Final x values converted back into seconds
        adjusted_x = (pred_x-zero)/1e5

        #################################################################
        # Rolling-Mean Data
        #################################################################
        # if rolling_window > 1:
        # rm_snippet = rm_data[idx-start_offset:idx+end_offset]
        # rm_adjusted_x = adjusted_x + ((rolling_window-1)*0.05)
    
        for idx, analyte in enumerate(['DA', '5HT', 'pH', 'NE']):
            # if analyte not in analytes:
            #     continue
           
            if idx == 0:
                mean_window_x_list.append(adjusted_x)
                mean_window_y_list_DA.append(pred_snippet[:,idx])
            elif idx == 1:
                mean_window_y_list_5HT.append(pred_snippet[:,idx])
            elif idx == 3:
                mean_window_y_list_NE.append(pred_snippet[:,idx])

    average_x = np.mean(np.vstack(mean_window_x_list), axis=0)
    average_y_DA = np.mean(np.vstack(mean_window_y_list_DA), axis=0)
    average_y_5HT = np.mean(np.vstack(mean_window_y_list_5HT), axis=0)
    average_y_NE = np.mean(np.vstack(mean_window_y_list_NE), axis=0)

    fig = plt.figure(figsize=(12, 8), dpi=400)
    ax = fig.add_subplot(111)

    for avg, l in [(average_y_DA,"DA"), (average_y_5HT,"5HT"), (average_y_NE,"NE")]:
        if l not in analytes:
            continue
        if rolling_window>1:
            l = f"{l} Rolling Mean (window: {rolling_window})"
            rm_avg = moving_average(avg, rolling_window)
            if (rolling_window-1) % 2 == 0:
                ax.plot(average_x[(rolling_window-1)//2:-(rolling_window-1)//2], rm_avg, label=l)
            else:
                ax.plot(average_x[(rolling_window-1)//2+1:-(rolling_window-1)//2], rm_avg, label=l)
        else:
            ax.plot(average_x, avg, label=l)

    ax.axvline(x=0, c='black')
    ax.set_xlabel(f"Time in Seconds Relative to TTL")
    if z_score:
        ax.set_ylabel(f"Z-Scored Predicted Value")
    else:
        ax.set_ylabel(f"Predicted Value (nmol/L)")
    ax.set_xticks(np.arange(-start_offset/10, end_offset/10+1, tick_spacing))
    ax.set_xlim((1.05 * (-start_offset/10)), (end_offset/10*1.05))
    if z_score:
        ax.set_ylim((-5,5))
    ax.legend(loc="upper left")
    ax.set_title(f"{title_str} Average of {len(mean_window_x_list)} Snippets")
    fig.savefig(_filename_helper(filename,"mean"), dpi=400, bbox_inches='tight')
    plt.close(fig)

def plot_mean_data_snippets(
        data_snippet_files, 
        fig_file_name, 
        analyte_names=['DA', '5HT', 'pH', 'NE'], 
        smooth_window=5, 
        ttl_line_clr="blue", 
        overwrite=False
    ):
    """
    For a given list of files:
        load
        z-score them (over the whole snippet)
        
    Combine z-scored snippets
    Apply smoothing?
    plot

    parameters:
        data_snippet_files : list of str
    """

    if os.path.isfile(fig_file_name) and not overwrite:
        print(f"File already exists: {fig_file_name}")
        return

    data_arr = []
    time_arr = []

    for file in data_snippet_files:
        # Load the data
        df = pd.read_csv(file)
        data = df[analyte_names].to_numpy()
        time = df['Approximate Time'].to_numpy()

        z_data = stats.zscore(data, axis=0)

        data_arr.append(z_data)
        time_arr.append(time)

    # Choose the time array that is the longest
    time =  max(time_arr, key=lambda a:len(a))

    # Get the mean for all the analytes
    stack_data = np.dstack(data_arr)
    mean_snippet = np.nanmean(stack_data, axis=2)

    if smooth_window > 1:
        mean_snippet = np.apply_along_axis(moving_average, 0, mean_snippet, smooth_window)

        time = moving_average(time, smooth_window)
    
    # Make a figure
    fig = plt.figure(figsize=(12, 8), dpi=400)
    ax = fig.add_subplot(111)

    for idx, analyte in enumerate(analyte_names):
        a_c = {"DA":'k', "5HT":'m', "NE":'c', "pH":'y'}[analyte]
        ax.plot(time, mean_snippet[:,idx], label=analyte, lw=3, c=a_c)

    # Plot the vertical line
    ax.axvline(x=0, c=ttl_line_clr, linestyle='--', lw=3) 

    ax.set_xlabel(f"Time (s)")
    ax.set_ylabel(f"{'NT' if len(analyte_names)>1 else analyte_names[0]} [Z]")
    
    if len(analyte_names)>1:
        ax.legend(loc="upper right")

    fig.savefig(fig_file_name, dpi=400, bbox_inches='tight')

    plt.close(fig)
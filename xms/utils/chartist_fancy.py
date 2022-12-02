"""
chartist_fancy.py

Thomas Twomey
Montague Lab
Fralin Biomedical Institute at VTC
02/04/2022

Modified: 
04/05/2022

Hi there,

This is where the "fancy" charting functions live. Unfortunately most of these
function were developed organically as the need arose so they not as uniform as
one would like. The distinction between chartist_fancy.py and chartist.py is 
not as clear as it should be. 

Most should work being exported as png's or pdf's.

Happy charting.

"""

import numpy as np
from scipy import stats
import pandas as pd
import os
import sys
import matplotlib
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Patch
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from sklearn.metrics import r2_score
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

def split_bins(y_true, n_bins = None):
    """
    If n_bins is not provided, tries to find a good number of bins that have realativly equal sizes.

    Returns the a list of indices coressponding to y_true for each of the bins
    """

    if n_bins is None:
        raise NotImplementedError("Please provide a number of bins")
    
    max = np.amax(y_true)
    increment = max / n_bins

    indicies = []
    for i in range(n_bins):
        indicies.append(np.where((y_true>i*increment)&(y_true<=(i+1)*increment)))
    #print(indicies)
    return indicies

def analyte_true_pred_bin(
        y_true, 
        y_pred, 
        filename, 
        analytes_names=['DA', '5HT', 'pH', 'NE'], 
        model_description="", 
        n_bins = 40, 
        best_fit = True, 
        pearson_r=False, 
        r2=False, 
        slope=False, 
        ms=1, 
        error_bars=True,
        text_size = 18,
        xy = False
    ):
    """
    Function to plot the true vs predicted values of analyte: analytes_names[analytes_idx]
    The true value is the x axis prediction is the y axis.

    Arguments:
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

    # Keep track of all the plots
    figs = []

    max_limits = (sys.float_info.max, sys.float_info.min, sys.float_info.max, sys.float_info.min)
    
    for analyte_idx, analyte in enumerate(analytes_names):
        if analyte == 'pH':
            continue
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        y_t = y_true[:,analyte_idx]
        # Figures out the indices for the bins
        bin_indicies = split_bins(y_t, n_bins)

        # Seth colors
        a_c = {"DA":'C0', "5HT":'C1', "NE":'C3'}

        for in_idx, in_analyte in enumerate(analytes_names):
            if in_analyte == 'pH':
                # Disregard pH becase the scale is no good
                continue

            if analyte_idx != in_idx:
                # If not the analyte of interest
                # calculate the residuals
                y_res = y_pred[:, in_idx] - y_true[:, in_idx]
                y_show = y_res
            else:
                # If it is the analyte of interest
                # use raw prediction
                y_show = y_pred[:, in_idx]

            bins = [(y_t[idxs], y_show[idxs]) for idxs in bin_indicies]
            t_means = np.array([np.mean(yt) for yt, _ in bins])
            p_means = np.array([np.mean(yp) for _, yp in bins])
            x_sem = np.array([stats.sem(yt) for yt, _ in bins])
            y_sem = np.array([stats.sem(yp) for _, yp in bins])
            # x_std = np.array([np.std(yt) for yt, _ in bins])
            # y_std = np.array([np.std(yp) for _, yp in bins])
            # print(y_std)

            p_val = p_means

            #ax.scatter(t_means, p_means[:,in_idx], s=1, )
            if error_bars:
                #ax.errorbar(t_means, p_val, yerr=y_sem, xerr=x_sem, fmt='o', ms=ms, elinewidth=0.5, capsize=2, label=in_analyte, color=f"C{in_idx}")
                m, b = np.polyfit(t_means, p_val, 1)
                ax.errorbar(t_means, p_val, yerr=y_sem, xerr=x_sem, fmt='o', ms=ms, elinewidth=0.5, capsize=2, label=f"{in_analyte} -- y = {b:.1f} + {m:.3f}x", color=a_c[in_analyte])
                ax.plot(t_means, np.poly1d(np.polyfit(t_means, p_val, 1))(t_means), c=a_c[in_analyte])

            else:
                # ax.scatter(t_means, p_val, s=ms**2, label=in_analyte, color=f"C{in_idx}")
                ax.scatter(t_means, p_val, s=ms**2, label=in_analyte, color=a_c[in_analyte])

            if analyte_idx == in_idx and best_fit:
                
                #r, _ = np.polyfit(t_means, p_val, 1)
                
                txt = ""
                if pearson_r or r2 or slope:
                    if slope:
                        m, b = np.polyfit(t_means, p_val, 1)
                        txt+=f"y = {b:.1f} + {m:.3f}x"
                    if pearson_r:
                        r, _ = stats.pearsonr(t_means, p_val)
                        txt+=f"r = {round(r, 4):03}"
                    if r2:
                        r = r2_score(t_means, p_val)
                        if len(txt)>0:
                            txt+="\n"
                        txt+=f"R2 = {round(r, 4):03}"

                    # at = AnchoredText(txt, loc=(2000.0,1500.0))#"center right")
                    # at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
                    # ax.add_artist(at)

                    # these are matplotlib.patch.Patch properties
                    # props = dict(boxstyle='round', facecolor='none')

                    # # place a text box in upper left in axes coords
                    # ax.text(0.73, 0.6, txt, transform=ax.transAxes, fontsize=10,
                    #         verticalalignment='center', bbox=props)
                
                ax.plot(t_means, np.poly1d(np.polyfit(t_means, p_val, 1))(t_means), c='k', label=txt)

        if xy:
            # Plot the x==y
            ax.plot([i for i in range(2501)], [i for i in range(2501)], ":", c="r", lw=3)

        ax.set_xlabel(f"True Concentration {analyte} [nM]", fontsize=text_size, fontweight='bold')
        ax.set_ylabel(f"Predicted for {analyte},\nError for {', '.join([a for a in analytes_names if a !=analyte and a != 'pH'])} [nM]", fontsize=text_size, fontweight='bold')
        ax.set_title(f"{model_description} True vs Predicted {analyte}", fontsize=text_size, fontweight='bold')
        ax.legend(loc='upper left', prop={'size': 14})

        if analyte == 'DA' or analyte =='5HT' or analyte=='NE':
            # Enforce same limits for these analytes
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            max_limits = (min(xmin, max_limits[0]), max(xmax, max_limits[1]), min(ymin, max_limits[2]), max(ymax, max_limits[3]))

        figs.append((fig, _filename_helper(filename, str(analyte)), analyte))

    for fig, name, analyte in figs:
        if 'DA' == analyte or "5HT" == analyte or 'NE' == analyte:
            fig.axes[0].set_xlim(max_limits[0], max_limits[1])
            fig.axes[0].set_ylim(max_limits[2], max_limits[3])
        fig.savefig(name, dpi=400, bbox_inches='tight')
        plt.close(fig)

def simple_true_pred(
        y_true, 
        y_pred, 
        filename, 
        analytes_names=['DA', '5HT', 'pH', 'NE'], 
        model_description="", 
        best_fit=True, 
        pearson_r=False, 
        r2=False, 
        denote_mixtures=True, 
        ommit_pH=True, 
        save_data=False, 
        slope=False, 
        ms=1, 
        error_bars=True,
        text_size = 16,
        all_fit=False,
        xy=False
    ):
    """ 
    Plots the true and pred data.
    
    Arguments:
        compiled_true_y : np.ndarray
            The true values of the labels
        compiled_pred_y : np.ndarray
            The predicted values of the labels
        filename : String
            What to save the graph as.
        analytes_name : List
            The names of the analytes in order corresponding to the y_true/y_pred.
            Will be replaced with default if length is less than 3.
            Default: ['DA', '5HT', 'pH', 'NE']
        model_description : String
            A descriptive string to add to the title of the graphs
            Default: None
        best_fit : boolean
            Default: True
        pearson_r : boolean
            Default: False
        ommit_pH : boolean
            If pH should be excluded from the plots
            Default: True
        save_data : boolean
            Default: False
        slope : boolean
            Default: False
        ms : int
            The marker size for the plot
            Default: 1
        error_bars : boolean
            If error bars should be plotted
            Default: True
    """

    # Keep track of all the plots
    figs = []

    max_limits = (sys.float_info.max, sys.float_info.min, sys.float_info.max, sys.float_info.min) # x_min, x_max, y_min, y_max

    # Make plots
    for analyte_idx, analyte in enumerate(analytes_names):
        if analyte == 'pH' and ommit_pH:
            continue
        df_dict = {}
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        print(y_true.shape)
        if len(y_true.shape)<=1:
            y_true = y_true.reshape((y_true.shape[0],1))
        if len(y_pred.shape)<=1:
            y_pred = y_pred.reshape((y_pred.shape[0],1))

        y_t = y_true[:,analyte_idx]

        for in_idx, in_analyte in enumerate(analytes_names):
            if in_analyte == 'pH' and ommit_pH:
                continue

            if analyte_idx != in_idx:
                # calculate the residuals
                y_res = y_pred[:, in_idx] - y_true[:, in_idx]
                y_show = y_res

                # Find the values that represent mixtures
                mixture_values = np.unique(y_t[np.where(y_true[:, in_idx])])
            else:
                # use raw prediction
                y_show = y_pred[:, in_idx]
                mixture_values = []

                # Find the mixtures in the main analyte
                # Go through the other analytes
                for other_idx, other_analyte in enumerate(analytes_names):
                    if other_analyte == 'pH' or other_analyte == in_analyte:
                        continue
                    # Add the unique values to the list
                    mixture_values.append(np.unique(y_t[np.where(y_true[:, other_idx])]))
                # Combine the numpy arrays
                mixture_values = np.concatenate(mixture_values) if len(mixture_values)>0 else None

            # Calculate the average and Standard Deviation of the Mean for the unique values
            y_true_unique = np.unique(y_t)
            y_pred_avg = [np.mean(y_show[np.where(y_t==ytu)]) for ytu in y_true_unique]
            y_pred_sem = [stats.sem(y_show[np.where(y_t==ytu)]) for ytu in y_true_unique]

            # Make a dataframe to export
            residual_str = "Residual " if analyte_idx != in_idx else ""
            df_dict[f"{analyte} True"] = y_true_unique
            df_dict[f"{in_analyte} {residual_str}Mean Pred"] = y_pred_avg
            df_dict[f"{in_analyte} {residual_str}SEM"] = y_pred_sem

            if mixture_values is not None and denote_mixtures:
                mixtures_x = []
                mixtures_y = []
                # Find the points with that are mixtures
                for t, p in zip(y_true_unique, y_pred_avg):
                    #print(t)
                    if t in mixture_values and t != 0:
                        mixtures_x.append(t)
                        mixtures_y.append(p)


                # Plot squares around the mixtures
                ax.scatter(mixtures_x, mixtures_y, marker='s', s=80, alpha=0.3, c=f"C9", label="Mixtures")#, linewidths=1, alpha=0.1)

            # Plot the points
            if error_bars:
                ax.errorbar(y_true_unique, y_pred_avg, yerr=y_pred_sem, fmt='o', ms=ms, elinewidth=0.5, capsize=2, label=in_analyte, color=f"C{in_idx}")
                #ax.errorbar(t_means, p_val, yerr=y_sem, xerr=x_sem, fmt='o', ms=ms, elinewidth=0.5, capsize=2, label=in_analyte, color=f"C{in_idx}")
            else:
                ax.scatter(y_true_unique, y_pred_avg, s=ms**2, label=in_analyte, color=f"C{in_idx}")
            
            if not all_fit:
                if analyte_idx == in_idx and best_fit:
                    ax.plot(y_true_unique, np.poly1d(np.polyfit(y_true_unique, y_pred_avg, 1))(y_true_unique), color="black") # TODO change color to be based on ananlyte
                    #r, _ = np.polyfit(y_true_unique, y_pred_avg, 1)
                    if pearson_r or r2 or slope:
                        txt = ""
                        if slope:
                            m, b = np.polyfit(y_true_unique, y_pred_avg, 1)
                            txt+=f"y = {b:.1f} + {m:.3f}x"
                        if pearson_r:
                            r, _ = stats.pearsonr(y_true_unique, y_pred_avg)
                            #ax.text(2200, 1700, f"r = {round(r, 4):03}")
                            txt+=f"r = {round(r, 4):03}"
                        if r2:
                            r = r2_score(y_true_unique, y_pred_avg)
                            #ax.text(2200, 1500, f"R2 = {round(r, 4):03}")
                            if len(txt)>0:
                                txt+="\n"
                            txt+=f"R2 = {round(r, 4):03}"

                        at = AnchoredText(txt, loc="center right", prop=dict(fontweight="bold"))
                        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
                        ax.add_artist(at)
            else:
                if best_fit:
                    ax.plot(y_true_unique, np.poly1d(np.polyfit(y_true_unique, y_pred_avg, 1))(y_true_unique), color="black", lw=3) # TODO change color to be based on ananlyte
                    #r, _ = np.polyfit(y_true_unique, y_pred_avg, 1)
                    if pearson_r or r2 or slope:
                        txt = ""
                        if slope:
                            m, b = np.polyfit(y_true_unique, y_pred_avg, 1)
                            txt+=f"y = {b:.1f} + {m:.3f}x"
                        if pearson_r:
                            r, _ = stats.pearsonr(y_true_unique, y_pred_avg)
                            #ax.text(2200, 1700, f"r = {round(r, 4):03}")
                            txt+=f"r = {round(r, 4):03}"
                        if r2:
                            r = r2_score(y_true_unique, y_pred_avg)
                            #ax.text(2200, 1500, f"R2 = {round(r, 4):03}")
                            if len(txt)>0:
                                txt+="\n"
                            txt+=f"R2 = {round(r, 4):03}"

                        if analyte_idx == in_idx:
                            at = AnchoredText(txt, loc="center right", prop=dict(fontweight="bold"))
                        else:
                            at = AnchoredText(txt, loc="lower right", prop=dict(fontweight="bold"))
                        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
                        ax.add_artist(at)

        if xy:
            # Plot the x==y
            ax.plot([i for i in range(2501)], [i for i in range(2501)], ":", c="r", lw=3)

        ax.set_xlabel(f"True Concentration {analyte} (nM)", fontsize=text_size, fontweight='bold')
        ax.set_ylabel(f"Predicted for {analyte}, Error for {', '.join([a for a in analytes_names if a !=analyte and a != 'pH'])} (nM)", fontsize=text_size, fontweight='bold')
        ax.set_title(f"{model_description} True vs Predicted {analyte}", fontsize=text_size, fontweight='bold')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        if analyte == 'DA' or analyte =='5HT' or analyte=='NE' or analyte=='NA':
            # Enforce same limits for these analytes
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            max_limits = (min(xmin, max_limits[0]), max(xmax, max_limits[1]), min(min(ymin, max_limits[2]), 0), max(ymax, max_limits[3]))

        # The data used to make the plot
        df = pd.DataFrame(df_dict)

        figs.append((fig, _filename_helper(filename, str(analyte)), analyte, df))

    for fig, name, analyte, data_frame in figs:
        if 'DA' == analyte or "5HT" == analyte or 'NE' == analyte or 'ACh' == analyte or analyte=='NA':
            fig.axes[0].set_xlim(max_limits[0], max_limits[1])
            fig.axes[0].set_ylim(max_limits[2], max_limits[3])
        elif 'pH' == analyte:
            #fig.axes[0].set_xlim(max_limits[0], max_limits[1])
            fig.axes[0].set_ylim(6.5, 8)
        fig.savefig(name, dpi=400, bbox_inches='tight')
        if save_data:
            name = name[:-3]+"csv"
            data_frame.to_csv(name)
        plt.close(fig)
    

def simple_true_error(
        y_true, 
        y_pred, 
        filename, 
        analytes_names=['DA', '5HT', 'pH', 'NE'], 
        model_description="", 
        best_fit=True, 
        pearson_r=False, 
        r2=False, 
        denote_mixtures=True, 
        ommit_pH=True, 
        abs=False
    ):
    """
    Modification of simple_true_pred that splots the error of the predictions instead of the predicted values. 

    Plots the true and pred data.
    
    Arguments:
        y_true : np.ndarray
            The true values of the labels
        y_pred : np.ndarray
            The predicted values of the labels
        filename : String
            What to save the graph as.
        analytes_name : List
            The names of the analytes in order corresponding to the y_true/y_pred.
            Will be replaced with default if length is less than 3.
            Default: ['DA', '5HT', 'pH', 'NE']
        model_description : String
            A descriptive string to add to the title of the graphs
            Default: None
        best_fit : boolean
            Default: True
        pearson_r : boolean
            Default: False
        r2 : boolean
            Default: False
        denote_mixtures : boolean
            Default: True
        ommit_pH : boolean
            If pH should be excluded from the plots
            Default: True
        abs : boolean
            Default: True
    """
    
    # Keep track of all the plots
    figs = []

    max_limits = (sys.float_info.max, sys.float_info.min, sys.float_info.max, sys.float_info.min) # x_min, x_max, y_min, y_max

    # Make plots
    for analyte_idx, analyte in enumerate(analytes_names):
        if analyte == 'pH' and ommit_pH:
            continue
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        print(y_true.shape)
        if len(y_true.shape)<=1:
            y_true = y_true.reshape((y_true.shape[0],1))
        if len(y_pred.shape)<=1:
            y_pred = y_pred.reshape((y_pred.shape[0],1))

        y_t = y_true[:,analyte_idx]

        for in_idx, in_analyte in enumerate(analytes_names):
            if in_analyte == 'pH' and ommit_pH:
                continue

            if analyte_idx != in_idx:
                continue # only show the error for the analyte of interest
                # calculate the residuals
                y_res = y_pred[:, in_idx] - y_true[:, in_idx]
                y_show = y_res

                # Find the values that represent mixtures
                mixture_values = np.unique(y_t[np.where(y_true[:, in_idx])])
            else:
                # use residuals
                y_show = y_pred[:, in_idx] - y_true[:, in_idx]

                if abs:
                    y_show = np.abs(y_show)
                mixture_values = []

                # Find the mixtures in the main analyte
                # Go through the other analytes
                for other_idx, other_analyte in enumerate(analytes_names):
                    if other_analyte == 'pH' or other_analyte == in_analyte:
                        continue
                    # Add the unique values to the list
                    mixture_values.append(np.unique(y_t[np.where(y_true[:, other_idx])]))
                # Combine the numpy arrays
                mixture_values = np.concatenate(mixture_values) if len(mixture_values)>0 else None

            # Calculate the average and Standard Deviation of the Mean for the unique values
            y_true_unique = np.unique(y_t)
            y_pred_avg = [np.mean(y_show[np.where(y_t==ytu)]) for ytu in y_true_unique]
            y_pred_sem = [stats.sem(y_show[np.where(y_t==ytu)]) for ytu in y_true_unique]

            # Plot the points
            ax.errorbar(y_true_unique, y_pred_avg, yerr=y_pred_sem, fmt='o', ms=1, elinewidth=0.5, capsize=2, label=in_analyte, color=f"C{in_idx}")

            if mixture_values is not None and denote_mixtures:
                mixtures_x = []
                mixtures_y = []
                # Find the points with that are mixtures
                for t, p in zip(y_true_unique, y_pred_avg):
                    #print(t)
                    if t in mixture_values and t != 0:
                        mixtures_x.append(t)
                        mixtures_y.append(p)


                # Plot squares around the mixtures
                ax.scatter(mixtures_x, mixtures_y, marker='s', s=80, alpha=0.3, c=f"C9", label="Mixtures")#, linewidths=1, alpha=0.1)

            if analyte_idx == in_idx and best_fit:
                ax.plot(y_true_unique, np.poly1d(np.polyfit(y_true_unique, y_pred_avg, 1))(y_true_unique), color="black") # TODO change color to be based on ananlyte
                #r, _ = np.polyfit(y_true_unique, y_pred_avg, 1)
                if pearson_r or r2:
                    txt = ""
                    if pearson_r:
                        r, _ = stats.pearsonr(y_true_unique, y_pred_avg)
                        #ax.text(2200, 1700, f"r = {round(r, 4):03}")
                        txt+=f"r = {round(r, 4):03}"
                    if r2:
                        r = r2_score(y_true_unique, y_pred_avg)
                        #ax.text(2200, 1500, f"R2 = {round(r, 4):03}")
                        if len(txt)>0:
                            txt+="\n"
                        txt+=f"R2 = {round(r, 4):03}"

                    at = AnchoredText(txt, loc="center right")
                    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
                    ax.add_artist(at)

        ax.set_xlabel(f"True Concentration {analyte}")
        ax.set_ylabel(f"Error in Predicted Concentration for {analyte}")
        ax.set_title(f"{model_description} Real vs Error of Predicted {analyte}")

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        if analyte == 'DA' or analyte =='5HT' or analyte=='NE':
            # Enforce same limits for these analytes
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            max_limits = (min(xmin, max_limits[0]), max(xmax, max_limits[1]), min(ymin, max_limits[2]), max(ymax, max_limits[3]))

        figs.append((fig, _filename_helper(filename, str(analyte)), analyte))
        

    for fig, name, analyte in figs:
        if 'DA' == analyte or "5HT" == analyte or 'NE' == analyte:
            fig.axes[0].set_xlim(max_limits[0], max_limits[1])
            fig.axes[0].set_ylim(max_limits[2], max_limits[3])
        fig.savefig(name, dpi=400, bbox_inches='tight')
        plt.close(fig)

def hist_approx(data, bucket_size = 50, max_val = 3000):
    """
    Divides the data into buckets

    Arguments:
        data : An array of data.
        bucket_size : int
        max_val : int

    Returns:
        List of counts in buckets
    """

    int_div_data = data // bucket_size
    counts = [np.count_nonzero(int_div_data == i) for i in range(max_val//bucket_size+1)]
    counts = [c/len(data) for c in counts]
    return [0] + counts

def hist_approx_pos_neg(data, bucket_size = 50):
    """
    Divides the data into buckets
    This one returns the labels too

    Arguments:
        data : An array of data.
        bucket_size : Int
    """

    int_div_data = data // bucket_size

    min_val = int(np.min(int_div_data)) -1
    max_val = int(np.max(int_div_data)) + 1

    counts = [np.count_nonzero(int_div_data == i) for i in range(min_val, max_val)]
    counts = [0] + [c/len(data) for c in counts] + [0]
    labels = [(min_val-1)*bucket_size] + [i*bucket_size for i in range(min_val, max_val)] + [(max_val+1)*bucket_size]
    return labels, counts

def sequential_hist(
        compiled_true_y, 
        compiled_pred_y, 
        filename, 
        analytes_names=['DA', '5HT', 'pH', 'NE'], 
        title=None, 
        bucket_size=50, 
        ommit_pH=True
    ):
    """
    For every unique true value, plots a histogram of the distribution of the predicted values

    Depricated. Please use sequential_hist_2

    Arguments:
        compiled_true_y : np.ndarray
            The true values of the labels
        compiled_pred_y : np.ndarray
            The predicted values of the labels
        filename : String
            What to save the graph as.
        analytes_name : List
            The names of the analytes in order corresponding to the y_true/y_pred.
            Will be replaced with default if length is less than 3.
            Default: ['DA', '5HT', 'pH', 'NE']
        title : String
            A descriptive string to add to the title of the graphs
            Default: None
        bucket_size : int
            The size of the buckets for the histograms
            Default: 50
        ommit_pH : boolean
            If pH should be excluded from the plots
            Default: True
    """

    for analyte_idx, analyte in enumerate(analytes_names):
        if analyte == 'pH' and ommit_pH:
            continue

        true_y = compiled_true_y[:,analyte_idx]
        pred_y = compiled_pred_y[:,analyte_idx]

        fig = plt.figure(figsize=(10,6), dpi=200)
        ax = fig.gca(projection='3d')

        max_val = int(max(np.max(pred_y), np.max(true_y))) + bucket_size

        xs = [0]+[i*bucket_size for i in range(max_val//bucket_size+1)]
        #print(xs)
        verts = []
        zs, zs_idx = np.unique(true_y,return_index=True)


        for z, z_idx in zip(zs, zs_idx):
            ys = hist_approx(pred_y[np.where(true_y==z)], bucket_size, max_val)
            #print([1 if s > 0 else 0 for s in ys])
            verts.append(list(zip(xs, ys)))

        poly = PolyCollection(verts)
        poly.set_alpha(0.7)
        ax.add_collection3d(poly, zs=zs, zdir='x')

        ax.set_xlabel('True Concentration (nmol/L)')
        ax.set_xlim3d(0, ((np.max(true_y)//bucket_size)+1)*bucket_size)
        ax.set_ylabel('Predicted Concentration (nmol/L)')
        ax.set_ylim3d(((np.min(pred_y)//bucket_size)+1)*bucket_size, ((np.max(pred_y)//bucket_size)+1)*bucket_size)
        # ax.set_zlabel('Z')
        ax.set_zlim3d(0, 0.4)
        #ax.set_axis_off()
        ax.grid(False)
        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        #ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent

        ax.w_zaxis.line.set_lw(0.)
        ax.set_zticks([])

        ax.view_init(30, -105)

        if title is not None:
            ax.set_title(f"{analyte} {title}")

        fig.savefig(_filename_helper(filename, analyte), bbox_inches='tight')
        plt.close(fig)

def sequential_hist2(
        compiled_true_y, 
        compiled_pred_y, 
        filename, 
        analytes_names=['DA', '5HT', 'pH', 'NE'], 
        title=None, 
        bucket_size=50, 
        ommit_pH=True, 
        pred_min = None, 
        height_divider = 5,
    ):
    """
    Yet another plot of the true vs predicted values for each of the analytes in analytes_names.
    The true value is the x axis prediction is the y axis.
    Creates bins of the true values and then creates histograms of those true values

    Arguments:
        compiled_true_y : np.ndarray
            The true values of the labels
        compiled_pred_y : np.ndarray
            The predicted values of the labels
        filename : String
            What to save the graph as.
        analytes_name : List
            The names of the analytes in order corresponding to the y_true/y_pred.
            Will be replaced with default if length is less than 3.
            Default: ['DA', '5HT', 'pH', 'NE']
        title : String
            A descriptive string to add to the title of the graphs
            Default: ""
        bucket_size : int
            The size of the buckets for the histograms
            Default: 50
        ommit_pH : boolean
            If pH should be excluded from the plots
            Default: True
        pred_min : int
            The minimum predicted value for the plot. If None, will scale to fit all points
            Default : None
        height_divider : int
            How to scale the residual histograms for side analytes
            Default : 5
    """
    
    figs = []
    max_limits = (sys.float_info.max, sys.float_info.min, sys.float_info.max, sys.float_info.min)

    for analyte_idx, analyte in enumerate(analytes_names):
        if analyte == 'pH' and ommit_pH:
            continue

        true_y = compiled_true_y[:,analyte_idx]
        #pred_y = compiled_pred_y[:,analyte_idx]

        fig = plt.figure(figsize=(10,6), dpi=200)
        ax = fig.gca(projection='3d')

        legend_elements = []

        for in_idx, in_analyte in enumerate(analytes_names):
            if in_analyte == 'pH':
                # Disregard pH becase the scale is no good
                continue
            # The divider keeps the analyte of interest bigger
            divider = 1
            if analyte_idx != in_idx:
                # If not the analyte of interest
                # calculate the residuals
                y_res = compiled_pred_y[:, in_idx] - compiled_true_y[:, in_idx]
                y_show = y_res

                # For the error ananlytes we divied the qunatities by 5
                divider = height_divider
            else:
                # If it is the analyte of interest
                # use raw prediction
                y_show = compiled_pred_y[:, in_idx]

            verts = []
            zs = np.unique(true_y,return_index=False)

            for z in zs:
                
                xs, ys = hist_approx_pos_neg(y_show[np.where(true_y==z)], bucket_size)
                #print([1 if s > 0 else 0 for s in ys])
                verts.append(list(zip(xs, [y/divider for y in ys])))

            poly = PolyCollection(verts)
            poly.set_alpha(0.65) # We want to be able to see through some of it
            poly.set_facecolor(f'C{in_idx}') # The analytes will have the same color across graphs
            ax.add_collection3d(poly, zs=zs, zdir='x')

            # Add it to the legend
            divider_str = f" 1/{divider} Height" if divider > 1 else ""
            residual_str = f"Residual " if analyte_idx != in_idx else ""
            legend_elements.append(Patch(facecolor=f'C{in_idx}', label=f'{in_analyte} {residual_str}Histogram{divider_str}'))

        ax.set_xlabel(f"True Concentration {analyte}")
        ax.set_xlim3d(0, ((np.max(true_y)//bucket_size)+1)*bucket_size)
        ax.set_ylabel(f"Predicted Concentration") # for {analyte},\nError in Prediction for {', '.join([a for a in analytes_names if a !=analyte and a != 'pH'])}")
        if pred_min is None:
            ax.set_ylim3d(min(0, (np.min(y_show)//bucket_size+1)*bucket_size), ((np.max(y_show)//bucket_size)+1)*bucket_size)
        else:
            ax.set_ylim3d(pred_min, ((np.max(y_show)//bucket_size)+1)*bucket_size)
        # ax.set_zlabel('Z')
        ax.set_zlim3d(0, 0.4)
        #ax.set_axis_off()
        ax.grid(False)
        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        #ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent

        ax.w_zaxis.line.set_lw(0.)
        ax.set_zticks([])

        ax.view_init(30, -105)

        if title is not None:
            ax.set_title(f"{analyte} {title}")

        # Add the legend
        ax.legend(handles=legend_elements)

        if analyte == 'DA' or analyte =='5HT' or analyte=='NE':
            # Find the maximum of the various analytes
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            max_limits = (min(xmin, max_limits[0]), max(xmax, max_limits[1]), min(ymin, max_limits[2]), max(ymax, max_limits[3]))

        figs.append((fig, _filename_helper(filename, str(analyte)), analyte))

    # Enforce same limits for these analytes
    for fig, name, analyte in figs:
        if 'DA' == analyte or "5HT" == analyte or 'NE' == analyte:
            fig.axes[0].set_xlim(max_limits[0], max_limits[1])
            fig.axes[0].set_ylim(max_limits[2], max_limits[3])
        fig.savefig(name, dpi=400, bbox_inches='tight')
        plt.close(fig)

def sequential_hist2_true_error(
        compiled_true_y, 
        compiled_pred_y, 
        filename, 
        analytes_names=['DA', '5HT', 'pH', 'NE'], 
        title=None, 
        bucket_size=50, 
        ommit_pH=True, 
        pred_min = None, 
        height_divider = 1, 
        abs=False
    ):
    """
    Yet another plot of the true vs predicted values for each of the analytes in analytes_names.
    The true value is the x axis, prediction error is the y axis.
    Creates bins of the true values and then creates histograms of those true values

    Arguments:
        compiled_true_y : np.ndarray
            The true values of the labels
        compiled_pred_y : np.ndarray
            The predicted values of the labels
        filename : int
            The analyte that is being graphed
        analytes_name : List
            The names of the analytes in order corresponding to the y_true/y_pred.
            Will be replaced with default if length is less than 3.
            Default: ['DA', '5HT', 'pH', 'NE']
        title : String
            A descriptive string to add to the title of the graphs
            Default: ""
        bucket_size : int
            The size of the buckets for the histograms
            Default: 50
        ommit_pH : boolean
            If pH should be excluded from the plots
            Default: True
        pred_min : int
            The minimum predicted value for the plot. If None, will scale to fit all points
            Default : None
        height_divider : int
            How to scale the residual histograms for side analytes
            Default : 5
        abs : boolean
            If the absolute values of the error should be taken
    """
    
    figs = []
    max_limits = (sys.float_info.max, sys.float_info.min, sys.float_info.max, sys.float_info.min)

    for analyte_idx, analyte in enumerate(analytes_names):
        if analyte == 'pH' and ommit_pH:
            continue

        true_y = compiled_true_y[:,analyte_idx]
        #pred_y = compiled_pred_y[:,analyte_idx]

        fig = plt.figure(figsize=(10,6), dpi=200)
        ax = fig.gca(projection='3d')

        legend_elements = []

        for in_idx, in_analyte in enumerate(analytes_names):
            if in_analyte == 'pH':
                # Disregard pH becase the scale is no good
                continue
            # The divider keeps the analyte of interest bigger
            divider = 1
            if analyte_idx != in_idx:
                continue
                # If not the analyte of interest
                # calculate the residuals
                y_res = compiled_pred_y[:, in_idx] - compiled_true_y[:, in_idx]
                y_show = y_res

                # For the error ananlytes we divied the qunatities by 5
                divider = height_divider
            else:
                # If it is the analyte of interest
                # use the residuals
                y_show = compiled_pred_y[:, in_idx] - compiled_true_y[:, in_idx]

                if abs:
                    y_show = np.abs(y_show)

            verts = []
            zs = np.unique(true_y,return_index=False)

            for z in zs:
                
                xs, ys = hist_approx_pos_neg(y_show[np.where(true_y==z)], bucket_size)
                #print([1 if s > 0 else 0 for s in ys])
                verts.append(list(zip(xs, [y/divider for y in ys])))

            poly = PolyCollection(verts)
            poly.set_alpha(0.65) # We want to be able to see through some of it
            poly.set_facecolor(f'C{in_idx}') # The analytes will have the same color across graphs
            ax.add_collection3d(poly, zs=zs, zdir='x')

            # Add it to the legend
            divider_str = f" 1/{divider} Height" if divider > 1 else ""
            residual_str = f"Residual " if analyte_idx != in_idx else ""
            legend_elements.append(Patch(facecolor=f'C{in_idx}', label=f'{in_analyte} {residual_str}Histogram{divider_str}'))

        ax.set_xlabel(f"True Concentration {analyte}")
        ax.set_xlim3d(0, ((np.max(true_y)//bucket_size)+1)*bucket_size)
        ax.set_ylabel(f"Predicted Concentration Error") # for {analyte},\nError in Prediction for {', '.join([a for a in analytes_names if a !=analyte and a != 'pH'])}")
        if pred_min is None:
            ax.set_ylim3d(min(0, (np.min(y_show)//bucket_size+1)*bucket_size), ((np.max(y_show)//bucket_size)+1)*bucket_size)
        else:
            ax.set_ylim3d(pred_min, ((np.max(y_show)//bucket_size)+1)*bucket_size)
        # ax.set_zlabel('Z')
        ax.set_zlim3d(0, 0.4)
        #ax.set_axis_off()
        ax.grid(False)
        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        #ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent

        ax.w_zaxis.line.set_lw(0.)
        ax.set_zticks([])

        ax.view_init(30, -105)

        if title is not None:
            ax.set_title(f"{analyte} {title}")

        # Add the legend
        ax.legend(handles=legend_elements)

        if analyte == 'DA' or analyte =='5HT' or analyte=='NE':
            # Find the maximum of the various analytes
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            max_limits = (min(xmin, max_limits[0]), max(xmax, max_limits[1]), min(ymin, max_limits[2]), max(ymax, max_limits[3]))

        figs.append((fig, _filename_helper(filename, str(analyte)), analyte))

    # Enforce same limits for these analytes
    for fig, name, analyte in figs:
        if 'DA' == analyte or "5HT" == analyte or 'NE' == analyte:
            fig.axes[0].set_xlim(max_limits[0], max_limits[1])
            fig.axes[0].set_ylim(max_limits[2], max_limits[3])
        fig.savefig(name, dpi=400, bbox_inches='tight')
        plt.close(fig)

def sequential_hist_bin(
        compiled_true_y, 
        compiled_pred_y, 
        filename, 
        analytes_names=['DA', '5HT', 'pH', 'NE'], 
        title=None, 
        bucket_size=50, 
        n_bins=40, 
        ommit_pH=True, 
        pred_min = None
    ):
    """
    Yet another plot of the true vs predicted values for each of the analytes in analytes_names.
    The true value is the x axis prediction is the y axis.
    Creates bins of the true values and then creates histograms of those true values

    Arguments:
        compiled_true_y : np.ndarray
            The true values of the labels
        compiled_pred_y : np.ndarray
            The predicted values of the labels
        filename : int
            The analyte that is being graphed
        analytes_name : List
            The names of the analytes in order corresponding to the y_true/y_pred.
            Will be replaced with default if length is less than 3.
            Default: ['DA', '5HT', 'pH', 'NE']
        title : String
            A descriptive string to add to the title of the graphs
            Default: ""
        bucket_size : int
            The size of the buckets for the histograms
            Default: 50
        n_bins : int
            The number of bins to group the true values into. Each bin becomes a histogram.
            Default: 40
        ommit_pH : boolean
            If pH should be excluded from the plots
            Default: True
        pred_min : int
            The minimum predicted value for the plot. If None, will scale to fit all points
            Default : None
    """

    # Keep track of all the plots
    figs = []

    max_limits = (sys.float_info.max, sys.float_info.min, sys.float_info.max, sys.float_info.min)
    
    for analyte_idx, analyte in enumerate(analytes_names):
        if analyte == 'pH' and ommit_pH:
            continue
        
        # Load the true labels for the analyte of interest
        y_t = compiled_true_y[:,analyte_idx]

        # Figures out the indices for the bins
        bin_indicies = split_bins(y_t, n_bins)

        fig = plt.figure(figsize=(10,6), dpi=200)
        ax = fig.gca(projection='3d')

        legend_elements = []

        for in_idx, in_analyte in enumerate(analytes_names):
            if in_analyte == 'pH':
                # Disregard pH becase the scale is no good
                continue
            # The divider keeps the analyte of interest bigger
            divider = 1
            if analyte_idx != in_idx:
                # If not the analyte of interest
                # calculate the residuals
                y_res = compiled_pred_y[:, in_idx] - compiled_true_y[:, in_idx]
                y_show = y_res

                # For the error ananlytes we divied the qunatities by 5
                divider = 5
            else:
                # If it is the analyte of interest
                # use raw prediction
                y_show = compiled_pred_y[:, in_idx]

            max_val = int(max(np.max(y_show), np.max(y_t))) + bucket_size

            bins = [(y_t[idxs], y_show[idxs]) for idxs in bin_indicies]
            t_means = np.array([np.mean(yt) for yt, _ in bins]) # use this
            p_vals = [yp for _, yp in bins]

            verts = []
            for t, p in zip(t_means, p_vals):
                
                xs, ys = hist_approx_pos_neg(p, bucket_size)
                #print([1 if s > 0 else 0 for s in ys])
                verts.append(list(zip(xs, [y/divider for y in ys])))

            poly = PolyCollection(verts)
            poly.set_alpha(0.65) # We want to be able to see through some of it
            poly.set_facecolor(f'C{in_idx}') # The analytes will have the same color across graphs
            ax.add_collection3d(poly, zs=t_means, zdir='x')

            # Add it to the legend
            divider_str = f"Residual Histogram 1/{divider} Height" if divider > 1 else "Histogram"
            legend_elements.append(Patch(facecolor=f'C{in_idx}', label=f'{in_analyte} {divider_str}'))

        ax.set_xlabel(f"True Concentration {analyte}")
        ax.set_xlim3d(0, ((np.max(t_means)//bucket_size)+1)*bucket_size)
        ax.set_ylabel(f"Predicted Concentration") # for {analyte},\nError in Prediction for {', '.join([a for a in analytes_names if a !=analyte and a != 'pH'])}")
        if pred_min is None:
            ax.set_ylim3d(min(0, (np.min(y_show)//bucket_size+1)*bucket_size), ((np.max(y_show)//bucket_size)+1)*bucket_size)
        else:
            ax.set_ylim3d(pred_min, ((np.max(y_show)//bucket_size)+1)*bucket_size)
        ax.set_zlim3d(0, 0.4)

        # Don't show the grid
        ax.grid(False)

        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # make the grid lines transparent
        ax.w_zaxis.line.set_lw(0.)
        ax.set_zticks([])

        # Determines the "Camera" position
        ax.view_init(30, -105)

        if title is not None:
            ax.set_title(f"{analyte} {title}")

        # Add the legend
        ax.legend(handles=legend_elements)

        if analyte == 'DA' or analyte =='5HT' or analyte=='NE':
            # Find the maximum of the various analytes
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            max_limits = (min(xmin, max_limits[0]), max(xmax, max_limits[1]), min(ymin, max_limits[2]), max(ymax, max_limits[3]))

        figs.append((fig, _filename_helper(filename, str(analyte)), analyte))

    # Enforce same limits for these analytes
    for fig, name, analyte in figs:
        if 'DA' == analyte or "5HT" == analyte or 'NE' == analyte:
            fig.axes[0].set_xlim(max_limits[0], max_limits[1])
            fig.axes[0].set_ylim(max_limits[2], max_limits[3])
        fig.savefig(name, dpi=400, bbox_inches='tight')
        plt.close(fig)
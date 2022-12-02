"""
nose_removal.py

Thomas Twomey
Montague Lab
Fralin Biomedical Institute at VTC
08/12/2022

Hi there,

This file is for the noise removal algorithms.

"""

# imports
import numpy as np
import scipy as sp
import scipy.ndimage

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

def remove_signal(noise_section, deg=6):
    """
    Function to remove the "signal" or larger trend from a noise section.

    For each sweep, independently fits a polynomial to the rolling mean (window=1667)
    and removes that polynomial.

    parameters:
        noise_section : np.ndarray
            The array with the noise. Should exclude the "spike". 
            Developed with last 7000 points of sweep.
        deg : int (Default: 6)
            The degree of the polynomial to fit.
    
    returns:
        The isolated noise in array. 833 Points removed from start and end of each sweep.
    """

    def rs(sns, degree = 6):
        """
        Removes the signal in a single noise section (sns)

        Isolated in a function to be applied along the axis
        """
        
        # Use a moving window average of length 1/60

        x = np.arange(len(sns))
        ma = moving_average(sns, 1667)
        mx = x[833:-833]

        poly = np.polyfit(mx, ma, deg=degree)

        noise = sns[833:-833] - np.poly1d(poly)(mx)

        return noise

    just_noise = np.apply_along_axis(lambda x: rs(x, deg), axis=1, arr=noise_section)

    return just_noise

def nns_spikes(whole_sweeps, noise_section_start=3000):
    """
    Naive Noise Subtraction (nns) spikes

    Given the whole_sweep, returns the spike section with the noise removed.
    The noise is removed for each sweep by isolating the noise from the "flat part" of the 
    sweep and subtracting that from spike.

    Intended for spikes or large perturbations that occur at 60 Hz 

    parameters:
        whole_sweeps : np.ndarray
            The sweeps that contain the spike and the noise
        noise_section_start : int (Default 3000)
            Where to start the section that is used to isolate the noise.

    returns:
        The spikes from each sweep with the noise removed.
    """

    spikes = whole_sweeps[:,159:1159]
    noise_sections = whole_sweeps[:, noise_section_start:]

    # 5195 = (3 * 10000 / 6) + 159
    just_noise = remove_signal(noise_sections)[:, 5159-noise_section_start-833:5159-noise_section_start+1000-833]
    just_signal = spikes - just_noise

    return just_signal

def nns_sweeps(whole_sweeps, noise_section_start=3000):
    """
    Naive Noise Subtraction (nns) sweeps

    Given the whole_sweep, returns the whole sweep with the noise removed.
    The noise is removed for each sweep by isolating the noise from the "flat part" of the 
    sweep and tiling it and subtracting it from the whole dirty sweep.

    Intended for spikes or large perturbations that occur at 60 Hz 

    parameters:
        whole_sweeps : np.ndarray
            The sweeps that contain the spike and the noise
        noise_section_start : int (Default 3000)
            Where to start the section that is used to isolate the noise.

    returns:
        Each sweep with the noise removed.
    """
    
    #spikes = whole_sweeps[:,159:1159]
    noise_sections = whole_sweeps[:, noise_section_start:]

    # 5000 = (3 * 10000 / 6)
    just_noise = remove_signal(noise_sections)[:, 5000-noise_section_start-833:5000-noise_section_start+1667-833]

    just_signal = whole_sweeps - np.tile(just_noise, 6)[:,:10000]

    return just_signal

def bg_sub_gaussian_filt_2d(consecutive_sweeps, within_sweep_sz=15, across_sweep_sz=3):
    """
    Simple gaussian 2d filter split at the middle of the spike
    """

    base = consecutive_sweeps[0]

    var = consecutive_sweeps-base

    filtered_var = sp.ndimage.filters.gaussian_filter(var, [within_sweep_sz, across_sweep_sz])

    cleaned_sweeps = filtered_var + base

    return cleaned_sweeps
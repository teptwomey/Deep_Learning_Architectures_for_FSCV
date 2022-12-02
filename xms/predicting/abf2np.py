"""
@author: Thomas Twomey, Montague Lab, FBRI at VTC
@contact: twomey@vtc.vt.edu

Loader for abf files to numpy arrays

"""

import pyabf
import numpy as np

def _overflow_correction(in_arr):
    """
    Corrects overflow in standard case.
    """

    def in_func(arr):
        idxs = np.where(np.abs(np.diff(arr)) > 3000)[0]
        #print(idxs)

        out_arr = np.copy(arr)

        if len(idxs) % 2 != 0:
            raise ValueError("There is an odd number of idxs")

        for i in range(0, len(idxs), 2):
            out_arr[idxs[i]+1:idxs[i+1]+1] = 2000

        return out_arr

    out = np.apply_along_axis(in_func, 1, in_arr)

    return out

def load_abf(input_file, channel_select=None, offset=159, num_samples=1000, correct_overflow=True):
    """
    Loads the input_file and outputs a numpy array
    @param input_file: input file path for abf file
    @param channel_select: List of channels to load, defaults to all
    """

    abf = pyabf.ABF(input_file)

    # TODO this is directly from Jason's code
    if channel_select is None:
        channel_select = []

    # see if channel_select contains adcNames or channelNumbers
    try:
        [int(x) for x in channel_select]
    except ValueError:
        use_channel_numbers = False
    else:
        use_channel_numbers = True
    
    # next, determine the list of channels to convert
    if len(channel_select) == 0:
        # simple: use all channels
        channels_to_convert = abf.channelList
    elif use_channel_numbers:
        # also simple: use supplied numeric list
        channels_to_convert = [int(i) for i in channel_select]
    else:
        # less-simple: find numeric indices in adcNames
        try:
            channels_to_convert = [abf.adcNames.index(s) for s in channel_select]
        except ValueError as err:
            if not err.args:
                err.args = ('',)
            err.args = err.args + (f' {abf.adcNames}',)
            raise
    
    # Create the numpy array
    sweep_samples = abf.sweepPointCount
    sweep_count = abf.sweepCount
    abf_data = np.zeros(shape=(sweep_count, sweep_samples))#, len(channels_to_convert)))

    # Fill the numpy array

    for c in channels_to_convert:
                    abf_data[:, :] = abf.data[c].reshape(
                        sweep_count,
                        sweep_samples)
    
    # Select the values of interest
    abf_data = abf_data[:, offset:offset+num_samples]
    abf_data = abf_data

    # Correct overflow if specified.
    if correct_overflow:
        abf_data = _overflow_correction(abf_data)

    # return the numpy array after it has been filled
    return abf_data

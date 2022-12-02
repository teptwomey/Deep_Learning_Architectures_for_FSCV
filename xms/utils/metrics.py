"""
metrics.py

Thomas Twomey
Montague Lab
Fralin Biomedical Institute at VTC
08/03/2021

A collection of metric factories.

The idea here is to create functions that can be serialized (using dill) and loaded with no 
dependencies.

"""
import tensorflow as tf
import numpy as np

dtype = tf.float32

def tf_pmse_factory(analytes, analytes_stds, reduce_mean=False):
    """
    Function factory for projected mean square error

    args:
        analytes : list
            List of analytes that a funciton will be generated for
        analytes_stds : np.ndarry
            Corresponding array of the stds of the analytes
        reduce_mean : Boolean (Default: False)
            Creates a mean across analytes function

    returns:
        list of functions
    """

    # Simple argument validity checks
    #if isinstance(analytes, list):
    #    if(not isinstance(analytes_stds, list) or len(analytes) != len(analytes_stds)):
    #        raise TypeError("Incompatible types in tf_pmse_factory arguments.")
    output = []
    
    def idx_fact(idx):
        def tf_pmse(y_true, y_pred):
            '''projected mse: projects the concentrations back to nM and compute the mse using TensorFlow functions so that it can be used as metric.
            Arguments:
                y_true : np.ndarray
                    true concentrations
                y_pred : np.ndarray
                    predicted concentrations
            Returns:
                mse : np.ndarray
            '''
            std = tf.convert_to_tensor(analytes_stds, dtype=dtype)
            var = std**2

            d = tf.reduce_mean((y_true - y_pred)**2, axis=0)
            dp = tf.math.multiply(d,var)
            
            if idx == -1:
                return tf.sqrt(tf.reduce_mean(dp))
            else:
                return tf.sqrt(dp[idx])
        return tf_pmse

    if reduce_mean:
        output.append(idx_fact(-1))
        return output

    for idx, analyte in enumerate(analytes):
        func = idx_fact(idx)
        func.__name__ = f'{func.__name__}_{analyte}'
        output.append(func)
    
    return output

def tf_rmse(y_true, y_pred):
    """
    rmse: RMSE of concentrations using TensorFlow functions so that it can be used as metric.
    Note that this is not projected

    args:
        y_true : np.ndarray
            true concentrations
        y_pred : np.ndarray
            predicted concentrations
    returns:
        mse : np.ndarray
    """
    return tf.sqrt(tf.reduce_mean((y_true - y_pred)**2))

def zscore_factory(analytes_mean, analytes_std):
    """
    Function factory for zscore projection

    args:
        analytes_mean : np.ndarray
            array of the means for each analyte
        analytes_std : np.ndarry
            Corresponding array of the stds of the analytes

    returns:
        functions
    """

    def zscore(x, inverse=False):
        '''Z-score the data.
        Arguments:
            x : np.ndarray
                data to be z-scores
        Returns:
            x : np.ndarray
        '''    
            
        mean = analytes_mean
        std = analytes_std

        if inverse:
            x = (x*std)+mean
        else:
            x = (x-mean)/std
        
        return x
    
    return zscore

def inverse_zscore_factory(analytes_mean, analytes_std):
    """
    Function factory for zscore projection

    args:
        analytes_mean : np.ndarray
            array of the means for each analyte
        analytes_std : np.ndarry
            Corresponding array of the stds of the analytes

    returns:
        functions
    """

    def zscore(x, inverse=True):
        '''Z-score the data.
        Arguments:
            x : np.ndarray
                data to be z-scores
        Returns:
            x : np.ndarray
        '''    
            
        mean = analytes_mean
        std = analytes_std

        if inverse:
            x = (x*std)+mean
        else:
            x = (x-mean)/std
        
        return x
    
    return zscore



def shifted_zscore_factory(analytes_mean, analytes_std):
    """
    Function factory for shifted-zscore projection

    Shifts up 10 standard deviations.

    args:
        analytes_mean : np.ndarray
            array of the means for each analyte
        analytes_std : np.ndarry
            Corresponding array of the stds of the analytes

    returns:
        functions
    """

    def shifted_zscore(x, inverse=False):
        '''Z-score where the mean is shifted by 10 standard deviations in order to avoud negative values.
        Arguments:
            x : np.ndarray
                data to be z-scores
        Returns:
            x : np.ndarray
        '''
            
        mean = analytes_mean - 10 * analytes_std
        std = analytes_std

        if inverse:
            x = (x*std)+mean
        else:
            x = (x-mean)/std
        
        return x
    
    return shifted_zscore

def inverse_shifted_zscore_factory(analytes_mean, analytes_std):
    """
    Function factory for inverse-shifted-zscore projection

    Shifts up 10 standard deviations.

    args:
        analytes_mean : np.ndarray
            array of the means for each analyte
        analytes_std : np.ndarry
            Corresponding array of the stds of the analytes

    returns:
        functions
    """

    def shifted_zscore(x, inverse=True):
        '''Z-score where the mean is shifted by 10 standard deviations in order to avoud negative values.
        Arguments:
            x : np.ndarray
                data to be z-scores
        Returns:
            x : np.ndarray
        '''
            
        mean = analytes_mean - 10 * analytes_std
        std = analytes_std

        if inverse:
            x = (x*std)+mean
        else:
            x = (x-mean)/std
        
        return x
    
    return shifted_zscore

def divide_by_mean_factory(analytes_mean):
    """
    Function factory for divide-by-mean projection

    args:
        analytes_mean : np.ndarray
            array of the means for each analyte

    returns:
        function
    """

    def divide_by_mean(x):
        return x/analytes_mean
    
    return divide_by_mean

def multiply_by_mean_factory(analytes_mean):
    """
    Function factory for multiply-by-mean projection

    args:
        analytes_mean : np.ndarray
            array of the means for each analyte

    returns:
        function
    """

    def multiply_by_mean(x):
        return x*analytes_mean
    
    return multiply_by_mean

def tf_mean_pmse_factory(analytes, analytes_means):
    """
    Function factory for projected mean square error 
    Projects from divided by mean space

    args:
        analytes : list
            List of analytes that a function will be generated for
        analytes_means : np.ndarry
            Corresponding array of the means of the analytes

    returns:
        list of functions
    """
    
    output = []
    
    def idx_fact(idx):
        def tf_mean_pmse(y_true, y_pred):
            '''projected mse: projects the concentrations back to nM and compute the mse using TensorFlow functions so that it can be used as metric.
            Arguments:
                y_true : np.ndarray
                    true concentrations
                y_pred : np.ndarray
                    predicted concentrations
            Returns:
                mse : np.ndarray
            '''
            means = tf.convert_to_tensor(analytes_means, dtype=dtype)

            p_y_true = tf.math.multiply(y_true, means)
            p_y_pred = tf.math.multiply(y_pred, means)

            dp = tf.reduce_mean((p_y_true - p_y_pred)**2, axis=0)
            
            if idx == -1:
                return tf.sqrt(tf.reduce_mean(dp))
            else:
                return tf.sqrt(dp[idx])
        return tf_mean_pmse

    for idx, analyte in enumerate(analytes):
        func = idx_fact(idx)
        func.__name__ = f'{func.__name__}_{analyte}'
        output.append(func)
    
    return output


def log_factory(analytes):
    """
    Wrapper for natural log to maintain pattern

    Limited utility
    """

    def nat_log(x):
        return np.log(x+1)

    return nat_log

def exp_factory(analytes):
    """
    Wrapper for natural log to maintain pattern

    Limited utility
    """

    def nat_exp(x):
        return np.exp(x)-1

    return nat_exp

def tf_exp_pmse_factory(analytes):
    """
    Function factory for projected mean square error 
    Projects log space

    args:
        analytes : list
            List of analytes that a funciton will be generated for

    returns:
        list of functions
    """

    output = []

    def idx_fact(idx):
        def tf_exp_pmse(y_true, y_pred):
            """
            """
            p_y_true = tf.math.exp(y_true)-1
            p_y_pred = tf.math.exp(y_pred)-1

            dp = tf.reduce_mean((p_y_true - p_y_pred)**2, axis=0)

            return tf.sqrt(dp[idx])
        
        return tf_exp_pmse

    for idx, analyte in enumerate(analytes):
        func = idx_fact(idx)
        func.__name__ = f'{func.__name__}_{analyte}'
        output.append(func)
    
    return output
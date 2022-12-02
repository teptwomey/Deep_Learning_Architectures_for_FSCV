"""
data_loader.py

Thomas Twomey
Montague Lab
Fralin Biomedical Institute at VTC
10/13/2022

Contains the DataLoader class
"""

# Imports
import numpy as np
from glob import glob
from itertools import compress
from scipy.io import loadmat
from scipy import stats


class DataLoader():
    """
    "Interface" for a DataLoader Class

    A Dataloader is used to load data from "original" source.
    """

    def load(self, **kwargs):
        """
        Loads data and returns it as a dictionary of numpy arrays
        key = probe, value = (x_data, y_data) or (x_data) if no labels
        """

        raise NotImplementedError

    def dict_diff(self, data_dict):
        """
        Takes a dict and returns the modified dict with a diff taken on the sweeps

        Arguments:
            data_dict : dict
                The data to be diff-ed
        """

        for key in data_dict.keys():
            # Diff the x value set if needed
            sweeps = np.diff(data_dict[key][0])
            labels = data_dict[key][1]
            # Update the dict with the altered values
            data_dict[key] = (sweeps, labels)
        return data_dict

    def dict_zscore_by_probe(self, data_dict):
        """
        Performs a z_score on the sweeps for each probe in the data_dict

        Arguments:
            data_dict : dict
                The data to be z-scored

        Returns:
            Modified data_dict
        """

        for probe in data_dict.keys():
            sweeps = data_dict[probe][0]
            labels = data_dict[probe][1]

            print(f"Probe: {probe}")
            print(f"Shape: {sweeps.shape}")

            normed_sweeps = stats.zscore(sweeps, axis=0)
            data_dict[probe] = (normed_sweeps, labels)
        return data_dict

    def gen_gaussian_noisy_dict(self, data_dict, std=1.6, seed=42):
        """
        For a given data_dict, creates a duplicate with independent gaussian noise added.
        
        Arguments:
            data_dict : dict
                The "clean" data_dict to add noise to.
            std : float (Default: 1.6)
                The std to use to set the spread of the gaussian noise
            seed : int (Default: 42)
                The seed is used once to initialize a generator which is then used for all the probes.
                The seed does not apply to all of the probes independently. 

        Returns:
            Modified data_dict
        """

        out_data_dict = {}

        rng = np.random.default_rng(seed)

        for probe in data_dict.keys():

            sweeps = data_dict[probe][0]
            labels = data_dict[probe][1]

            noisy_sweeps = sweeps + rng.normal(loc=0, scale=std, size=sweeps.shape)
            out_data_dict[probe] = (noisy_sweeps, labels)
        
        return out_data_dict

    def mix_data_dicts(self, a, b, ratio_a=0.5, seed=42):
        """
        Mixes two data_dicts for each overlapping probe_concentration. 

        Arguments:
            a : data_dict
            b : data_dict
            ratio_a : float (Default: 0.5)
                What ratio of the output should be from data_dict a
            seed : int (Default: 42)
                The seed used to instantiate the random number generator.
                The same generator is used for each probe-concentration to 
                select the ordering of the indices selected
        
        Returns:
            Modified data_dict
        """

        mix_data_dict = {}

        rng = np.random.default_rng(seed)

        a_keys = a.keys()
        b_keys = b.keys()

        a_only_keys = [k for k in a_keys if k not in b_keys]
        b_only_keys = [k for k in b_keys if k not in a_keys]
        a_b_keys = [k for k in a_keys if k in b_keys]

        for probe in a_only_keys:
            mix_data_dict[probe] = a[probe]

        for probe in b_only_keys:
            mix_data_dict[probe] = b[probe]

        for probe in a_b_keys:
            a_sweeps = a[probe][0]
            a_labels = a[probe][1]

            b_sweeps = b[probe][0]
            b_labels = b[probe][1]

            if not np.array_equal(a_labels, b_labels):
                raise NotImplementedError("For now labels must match for each probe.")

            # Find all the unique labels
            _, unique_idicies = np.unique(a_labels, return_index=True, axis=0)

            # Create a new array to fill
            combined_sweeps = np.zeros(a_sweeps.shape, dtype=np.float32)

            for ui in unique_idicies:
                # Find the indicies where the label is equal to our unique index
                indicies = np.where((a_labels == a_labels[ui,:]).all(axis=1))[0]

                # Shuffle the array
                indicies = rng.permutation(indicies)

                a_indicies = indicies[:int(len(indicies)*ratio_a)]
                b_indicies = indicies[int(len(indicies)*ratio_a):]

                combined_sweeps[a_indicies] = a_sweeps[a_indicies]
                combined_sweeps[b_indicies] = b_sweeps[b_indicies]

            mix_data_dict[probe] = (combined_sweeps, a_labels)
            
        return(mix_data_dict)

    def dict_scale_by_probe(self, data_dict):
        """
        For each probe in the data dict, scale the magnitude of the sweeps, by the mean(abs(current))

        Arguments:
            data_dict : dict

        Returns:
            Modified data_dict
        """

        for probe in data_dict.keys():
            sweeps = data_dict[probe][0]
            labels = data_dict[probe][1]

            normed_sweeps = sweeps / np.mean(np.abs(sweeps))
            data_dict[probe] = (normed_sweeps, labels)
        return data_dict

    def dict_scale_by_sweep(self, data_dict, method="mean"):
        """
        For each probe in the data dict, scale the magnitude of the sweeps, by the mean(abs(current))

        Arguments:
            data_dict : dict
            method : String (Default: "mean")
                Implemented methods:
                    mean
                    median
                    max

        Returns:
            Modified data_dict

        """

        for probe in data_dict.keys():
            sweeps = data_dict[probe][0]
            labels = data_dict[probe][1]

            if method.lower() == "mean":
                divider = np.mean(sweeps, axis=1)
            elif method.lower() == "median":
                divider = np.median(sweeps, axis=1)
            elif method.lower() == "max":
                divider = np.max(sweeps, axis=1)
            else:
                raise ValueError(f"Method: {method} not recognised. Implemented methods are: 'mean', 'median', 'max'.")
            normed_sweeps = sweeps / divider.reshape(divider.shape[0],1)
            data_dict[probe] = (normed_sweeps, labels)
        return data_dict

    def reshape_dict(self, data_dict):
        """
        Reshapes the sweep in the dict to make compliant with models

        Adds a dimension.

        Arguments:
            data_dict : dict
    
        Returns:
            Modified data_dict
        """

        for key in data_dict.keys():
            sweeps = data_dict[key][0]
            labels = data_dict[key][1]
            sweeps = sweeps.reshape((sweeps.shape[0], sweeps.shape[1], 1))
            # Update the dict with the altered values
            data_dict[key] = (sweeps, labels)
        return data_dict

    def subsample_dict(self, data_dict, indicies):
        """
        Samples the dict to have the given indices.

        In this case we are sampling within the sweep.
        A sweep is the normally 1000 points of current measurements.
        The indicies are of the points in the sweep.

        Arguments:
            data_dict : dict
            indicies : list of int
                The labels to subsample
        
        Returns:
            Modified data_dict.
        """

        # Data prep loop
        for key in data_dict.keys():
            # Get the sweeps
            sweeps = data_dict[key][0]

            # Check if the key has already been subsampled
            if sweeps.shape[1] > len(indicies):
                sweeps = sweeps[:,indicies]

            labels = data_dict[key][1]
            # Update the dict with the altered values
            data_dict[key] = (sweeps, labels)
        return data_dict

    def drop_mixtures(self, data):
        """
        Function that takes the data as a dictionary and removes all sweeps with mixture labels

        Arguments:
            data : data_dict

        Returns:
            Modified data_dict
        """
        for probe in data.keys():
            sweeps, labels = data[probe]
            edge_indices = np.where(np.sum(np.where(labels == 0, 1, 0),axis=1)>1)[0]
            print(labels.shape)
            print()
            print(labels[edge_indices].shape)
            data[probe] = (sweeps[edge_indices], labels[edge_indices])
        return data


    def trim(self, data, limit):
        """
        Function that takes the data as a dictionary and removes all sweeps with label > limit

        Arguments:
            data : data_dict
            limit : int
                Removes all sweeps with label > limit
        
        Returns:
            Modified data_dict
        """

        for probe in data.keys():
            sweeps, labels = data[probe]
            print(f"Sweeps Shape: {sweeps.shape}")
            print(f"Labels Shape: {labels.shape}")
            good_indicies = np.where(np.all(labels<=limit, axis=1))
            data[probe] = (sweeps[good_indicies], labels[good_indicies])
        return data

    def trim_low(self, data, limit):
        """
        Function that takes the data as a dictionary and removes all sweeps with label < limit

        Arguments:
            data : data_dict
            limit : int
                Removes all sweeps with label < limit

        Returns:
            Modified data_dict
        """
        
        for probe in data.keys():
            sweeps, labels = data[probe]
            print(f"Sweeps Shape: {sweeps.shape}")
            print(f"Labels Shape: {labels.shape}")
            good_indicies = np.where(np.any(labels>=limit, axis=1))
            data[probe] = (sweeps[good_indicies], labels[good_indicies])
        return data


class FilePairDataLoader(DataLoader):
    """
    DataLoader for the "standard" case in which there is a pair of file 
    "voltammograms.mat" and "labels.mat" that contain the probe name in a 
    folder or set of folders.

    Derived from Leonardo's get_multiple_data_v2
    
    """

    @staticmethod
    def long_form(abv):
        """
        Wraps a dictionary that returns the name (long form) of an analyte for standard abbreviations (short forms)

        Returns None on miss

        Arguments:
            abv : str
                string abbreviation

        Returns:
            
        """
        abv_dict = {
            'da':'Dopamine',
            '5ht':'Serotonin',
            'ph':'pH',
            'ne':'Norepinephrine',
            'hiaa':'HydroxyindoleaceticAcid',
            '5hiaa':'HydroxyindoleaceticAcid',
            'flx': 'Fluoxetine',
            'kyna':'KynurenicAcid',
            'kya':'KynurenicAcid',
            'ach':'Acetylcholine',
            'ch':"Choline",
            'glu':'Glutamate'
        }

        return abv_dict.get(abv.lower(), None)

    def extended_load_files(self, vfiles, n_records_per_label_per_probe=-1, analyte_names=['DA', '5HT', 'pH', 'NE']):
        """
        Load data from each matlab file in vfiles parameter. Each file should be named 'voltammograms.mat', and they 
        should contain a struct with a 'voltammograms' field. This field is a cell where each entry is a TxS matrix, 
        where T is the number of samples (time) and S is the the number of sweeps (number of voltammograms). This 
        function also expects that each folder containing one 'voltammogram.mat' file also contains a 'labels.mat' file,
        which in turn contains a struct with a 'labels' field. This field should be a cell where each entry is a SxA 
        matrix, where S is the number of sweeps and A is the number of analytes.

        Additionally the "extended" refers to the ability to search the 'chemicals' row(?) of the .mat file and determine which indices to load

        Arguments:
            vfiles : list
                List of voltammogram files to load. Each entry should contain the full path of the voltammogram.mat file.
            n_records_per_label_per_probe : int
                Maximum umber of sweeps for each label for each probe. Defaults to -1 for no limit.
            analyte_names : list
                The analytes to load, by name, should be provided.
                Default: ['DA', '5HT', 'pH', 'NE']

        Returns:
            x : np.ndarray
        """  

        def safe_index(lst, thing):
            """
            Finds the index of thing in lst

            Catches ValueError internally and returns -1
            """
            try:
                return lst.index(thing)
            except ValueError:
                return -1

        # These are the default
        ANALYTES_IDX = [0, 1, 2, 3]
        DEFAULT_ANALYTES = ['Dopamine', 'Serotonin', 'pH', 'Norepinephrine', 'HydroxyindoleaceticAcid', 
                         'KynurenicAcid', 'Acetylcholine', 'Glutamate']
        npdt = np.float32

        yv = []
        yl = []
        for vf in vfiles:

            if vf.find('.mat') > -1:
                mat_dict = loadmat(vf)
                # Check to find voltammograms_# 
                voltammogram_key = [k for k in list(mat_dict.keys()) if "voltammograms" in k][0]
                bv = np.array(mat_dict[voltammogram_key])
                label_mat_dict = loadmat(vf.replace('voltammograms.mat', 'labels.mat'))
                bl = np.array(label_mat_dict['labels'])
                if 'chemicals' in label_mat_dict:
                    blc = [e[0] for e in label_mat_dict['chemicals'][0]]
                    inner_analytes_idx = [safe_index(blc, self.long_form(chem)) for chem in analyte_names]

                else:
                    # Note that this is imperfect
                    # Consider the out of order case, or the case in which there are not 4 analytes
                    print("Be-Careful this makes assumptions")
                    blc = [e for e in DEFAULT_ANALYTES]
                    inner_analytes_idx = [safe_index(blc, self.long_form(chem)) for chem in analyte_names]
            else:
                raise ValueError(f'Unknown file type: {vf}')

            # iterate through matlab cell contents
            for (xv, xl) in zip(bv,bl):
                v = xv[0].T # voltammetry
                # Add a zero column at the end if there is a need to pull from it
                # print(f"inner_analytes_idx: {inner_analytes_idx}")
                # print(xl)
                if -1 in inner_analytes_idx:
                    xl = [np.append(xl[0],np.zeros((xl[0].shape[0],1)), axis=1)]
                l = xl[0][:,inner_analytes_idx] # labels (concentrations)
                v = v.astype(npdt)
                l = l.astype(npdt)

                yv.append(v)
                yl.append(l)

        x = np.vstack(yv)
        y = np.vstack(yl)


        _, ulidx = np.unique(y, return_index=True, axis=0)
    
        # get n unique records per label (tuple), or all data (-1)
        if n_records_per_label_per_probe > -1:
            # just to make sure there is only once concentration tuple in this record
            u_x = []
            u_y = []

            # Make a random number generator
            rng = np.random.default_rng(seed=42)

            for idx in ulidx:
                one_label_idxs = np.where((y == y[idx,:]).all(axis=1))[0]
                # Randomly shuffle
                rng.shuffle(one_label_idxs)

                u_x.append(x[one_label_idxs[:n_records_per_label_per_probe], :])
                u_y.append(y[one_label_idxs[:n_records_per_label_per_probe], :])
            x = np.concatenate(u_x)
            y = np.concatenate(u_y)
        
        return (x,y)

    def load_files(self, vfiles, n_records_per_label_per_probe=-1):
        """
        Load data from each matlab file in vfiles parameter. Each file should be named 'voltammograms.mat', and they 
        should contain a struct with a 'voltammograms' field. This field is a cell where each entry is a TxS matrix, 
        where T is the number of samples (time) and S is the the number of sweeps (number of voltammograms). This 
        function also expects that each folder containing one 'voltammogram.mat' file also contains a 'labels.mat' file,
        which in turn contains a struct with a 'labels' field. This field should be a cell where each entry is a SxA 
        matrix, where S is the number of sweeps and A is the number of analytes.

        Arguments:
            vfiles : list
                List of voltammogram files to load. Each entry should contain the full path of the voltammogram.mat file.
            n_records_per_label_per_probe : int
                Maximum umber of sweeps for each label for each probe. Defaults to -1 for no limit.

        Returns:
            x : np.ndarray
        """  
        ANALYTES_IDX = [0, 1, 2, 3]
        npdt = np.float32

        yv = []
        yl = []
        for vf in vfiles:
        
            if vf.find('.mat') > -1:
                bv = np.array(loadmat(vf)['voltammograms'])
                bl = np.array(loadmat(vf.replace('voltammograms.mat', 'labels.mat'))['labels'])
            else:
                raise ValueError(f'Unknown file type: {vf}')

            # iterate through matlab cell contents
            for (xv, xl) in zip(bv,bl):
                v = xv[0].T # voltammetry
                l = xl[0][:,ANALYTES_IDX] # labels (concentrations)
                v = v.astype(npdt)
                l = l.astype(npdt)

                yv.append(v)
                yl.append(l)

        x = np.vstack(yv)
        y = np.vstack(yl)


        _, ulidx = np.unique(y, return_index=True, axis=0)
    
        # get n unique records per label (tuple), or all data (-1)
        if n_records_per_label_per_probe > -1:
            # just to make sure there is only once concentration tuple in this record
            u_x = []
            u_y = []

            # Make a random number generator
            rng = np.random.default_rng(seed=42)

            for idx in ulidx:
                one_label_idxs = np.where((y == y[idx,:]).all(axis=1))[0]
                # Randomly shuffle
                rng.shuffle(one_label_idxs)

                u_x.append(x[one_label_idxs[:n_records_per_label_per_probe], :])
                u_y.append(y[one_label_idxs[:n_records_per_label_per_probe], :])
            x = np.concatenate(u_x)
            y = np.concatenate(u_y)
        
        return (x,y)
        
    def load(self, **kwargs):
        """
        Generic Loading function argument parser

        kwargs:
            folders : str
                path to folder(s) with voltammogram.mat and labels.mat files
                Follow unix shell expansion rules
            probes : list
                List of string names of probes. 
                Used for the name of files and the dictionary that returns the data
            n_records_per_label_per_probe : int
                (Optional)
                Maximum number of records to load per unique label per probe
        """

        # Basic type checking
        if not "folders" in kwargs and not "probes" in kwargs:
            raise TypeError(f"Attempted to load data from folders probes pair but one or both is not passed")

        # check if n_records_per_label_per_probe is provided
        if "n_records_per_label_per_probe" in kwargs:
            n_records_per_label_per_probe = kwargs["n_records_per_label_per_probe"]
            print(f"Number of records per label per probe: {n_records_per_label_per_probe}")
        else:
            n_records_per_label_per_probe = -1

        # check if analyte_names is provided
        if "analyte_names" in kwargs:
            analyte_names = kwargs["analyte_names"]
        elif "analytes_name" in kwargs:
            analyte_names = kwargs["analytes_name"]
        elif "analytes_names" in kwargs:
            analyte_names = kwargs["analytes_names"]
        else:
            analyte_names = None

        folders = kwargs["folders"]
        probes = kwargs["probes"]
        file='voltammograms.mat'

        voltammograms = []
        print(folders)
        if isinstance(folders, list):
            print('loading multiple folders')
            for folder in folders:
                voltammograms += glob(folder + file)
        else:
            print('loading one folder')
            voltammograms = glob(folders + file)

        print('number of voltammograms files %d'%len(voltammograms))
        
        probes_data = {}
        for probe in probes:
            #Find all the files that have file name in their path 
            probe_files = list(compress(voltammograms, np.array([x.find(probe+'/') > -1  or x.find(probe+'__') > -1 for x in voltammograms])))
            print(f"Attempting to load {probe}")
            if len(probe_files) < 1:
                raise ValueError(f"No files found for probe: {probe}")
            if analyte_names is None:
                probes_data[probe] = self.load_files(probe_files, n_records_per_label_per_probe)
                print("Called the old loading function because 'analyte_names' was not provided as a parameter")
            else:
                probes_data[probe] = self.extended_load_files(probe_files, n_records_per_label_per_probe, analyte_names)

        return probes_data

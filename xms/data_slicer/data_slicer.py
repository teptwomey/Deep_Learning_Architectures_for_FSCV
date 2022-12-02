"""
data_slicer.py

Thomas Twomey
Montague Lab
Fralin Biomedical Institute at VTC
04/08/2022

Classes to split data into train, validation, and test data sets. 

"""


# Imports
import numpy as np

class DataSlicer():
    """
    Template for class that does data manipulation of already loaded data

    """

    def slice(self, data, **kwargs):
        """
        Note that data is the new standard of a {probe: data} dictionary

        Return structure is lose to allow for various functionality    
        """
        raise NotImplementedError

class TrainValTestDataSlicer(DataSlicer):
    """
    Does the "Defaults" splitting into any combination of train, val, and test set 
    """

    def pool_probes(self, data, probes):
        """
        For a given list of probes combine their x and y arrays.
        Returns an (x,y) tuple

        args:
            data : dict
                dictionary in the form {probe_name:(x,y)}
                must contain all probes
            probes : list
                List of probes to pool
        """
        if data is None:
            raise ValueError("A dictionary of probes must be provided.")
        if probes is None or len(probes)==0:
            return (None, None)
        pooled_x = np.vstack([data[probe][0] for probe in probes])
        pooled_y = np.vstack([data[probe][1] for probe in probes])
        return (pooled_x, pooled_y)

    def unique_concentration_split(self, data, val_ratio, val_set_index=0):
        """
        splits a dataset based on unique concentrations
        
        args:
            data: tuple
            val_ratio: float
            val_set_index: int (default 0)
                Index of validation set, used to select a unique (discrete) validation set.
                Will wrap back around if val_set_index > 1/val_ratio -1

        """
        if isinstance(data, tuple) and len(data) == 2:
            x_data, y_data = data
        else:
            raise TypeError("unique_concentration_split looks for a tuple with length 2.")
        
        # Wrap the index back around
        val_set_index = val_set_index % int(1/val_ratio)

        _, ulidx = np.unique(y_data, return_index=True, axis=0)
        idxs = np.random.RandomState(seed=42).permutation(ulidx.shape[0])
        
        u_x1 = []
        u_x2 = []
        u_y1 = []
        u_y2 = []
        for (iidx, idx) in enumerate(ulidx[idxs]):
            one_label_idxs = np.where((y_data == y_data[idx,:]).all(axis=1))[0]
            
            if len(ulidx)>1/val_ratio:
                if iidx/len(ulidx) >= 1-(val_ratio*(val_set_index+1)) and iidx/len(ulidx)<1-(val_ratio*val_set_index): # Added equal to allow in arr[9] if len(arr)=10
                    # Validation
                    u_x2.append(x_data[one_label_idxs, :])
                    u_y2.append(y_data[one_label_idxs, :])
                else:
                    # Training
                    u_x1.append(x_data[one_label_idxs, :])
                    u_y1.append(y_data[one_label_idxs, :])
            else:
                if iidx == val_set_index%int(1/val_ratio):
                    # Validation
                    u_x2.append(x_data[one_label_idxs, :])
                    u_y2.append(y_data[one_label_idxs, :])
                else:
                    # Training
                    u_x1.append(x_data[one_label_idxs, :])
                    u_y1.append(y_data[one_label_idxs, :])

        x_1, y_1 = np.concatenate(u_x1), np.concatenate(u_y1)
        x_2, y_2 = np.concatenate(u_x2), np.concatenate(u_y2)
        
        return ((x_1,y_1),(x_2,y_2))


    @staticmethod
    def intersect2d(A, B):
        """
        Finds the 2d intersection of A and B

        args:
            A: The np nd array
            B: The np nd array
        """
        _, ncols = A.shape
        dtype={'names':['f{}'.format(i) for i in range(ncols)],
            'formats':ncols * [A.dtype]}

        C = np.intersect1d(A.view(dtype), B.view(dtype))

        # This last bit is optional if you're okay with "C" being a structured array...
        C = C.view(A.dtype).reshape(-1, ncols)
        return C

    @staticmethod
    def set_diff_2d(A, B, unique=False):
        """
        Finds the diff of two 2d arrays

        args:
            A: The np nd array
            B: The np nd array
        """
        _, ncols = A.shape
        dtype={'names':['f{}'.format(i) for i in range(ncols)],
            'formats':ncols * [A.dtype]}

        C = np.setdiff1d(A.view(dtype), B.view(dtype), assume_unique=unique)

        # This last bit is optional if you're okay with "C" being a structured array...
        C = C.view(A.dtype).reshape(-1, ncols)
        return C

    def unique_mixture_allocate(self, data, exclude_ratio, num_folds, index=0, external=None):
        """
        For a set of data (x_data, y_data) and a exclude ratio create:
        -train set (or train and val) that includes
            - all the edge data, 
            - any labels designated in external
            - (1-exclude_ratio) of the mixture points excluding any labels designated in external
        -test set (or val) that includes
            - exclude ratio of mixture points excluding and labels designated in external

        args:
            data : 
            exclude_ratio :
            num_folds :
            index : int (Default: 0)
            external : list (Default: None)
        """
        if isinstance(data, tuple) and len(data) == 2:
            x_data, y_data = data
        else:
            raise TypeError("unique_concentration_split looks for a tuple with length 2.")

        # Find the unique points
        unique_y_data = np.unique(y_data, axis=0)

        # Constrict to the area points
        unique_y_area_data = unique_y_data[np.sum(np.where(unique_y_data == 0, 1, 0),axis=1)<1]

        # Determine what is not in the chunk
        y_edge_indicies = np.where(np.sum(np.where(y_data == 0, 1, 0),axis=1)>0)[0]
        non_chunk_indicies = y_edge_indicies

        # Remove the external points
        if external is not None:
            unique_external = np.unique(external)
            chunk_labels = TrainValTestDataSlicer.set_diff_2d(unique_y_area_data, unique_external, True)

             # I can't find a better way of doing this so I will use a list comprehension
            # I know it is too slow
            y_external_indicies = [y_data[(y_data == l).all(axis=1)] for l in external]

            def contains_row(arr, row):
                return np.equal(row, arr).all(axis=1).any()

            y_external_indicies_2 = np.where(np.apply_along_axis(contains_row, axis=1, arr=y_data))
            print("External_indicies methods equal:")
            print(y_external_indicies == y_external_indicies_2)

            non_chunk_indicies = np.concatenate([non_chunk_indicies, y_edge_indicies])
        else:
            chunk_labels = unique_y_area_data

        # Shuffle the indicies
        idxs = np.random.RandomState(seed=42).permutation(chunk_labels.shape[0])
        
        u_x1 = []
        u_x2 = []
        u_y1 = []
        u_y2 = []

        start_idx = ((len(chunk_labels)/num_folds) * index) % len(chunk_labels)
        end_idx = (start_idx + (exclude_ratio * len(chunk_labels))) % len(chunk_labels)

        print("Chunk Labels")
        with np.printoptions(threshold=np.inf):
            print(chunk_labels)
        #print(chunk_labels)
        print(f"Len of chunk labels: {len(chunk_labels)}")
        print(f"Start index: {start_idx}")
        print(f"End index: {end_idx}")

        # For each label in chunk labels decide if it goes in train or val
        for (iidx, label) in enumerate(chunk_labels[idxs]):
            one_label_idxs = np.where((y_data == label).all(axis=1))[0]
            
            if end_idx > start_idx:
                # Contiguous section
                if iidx >= start_idx and iidx < end_idx:
                    # Validation
                    u_x2.append(x_data[one_label_idxs, :])
                    u_y2.append(y_data[one_label_idxs, :])
                else:
                    # Training
                    u_x1.append(x_data[one_label_idxs, :])
                    u_y1.append(y_data[one_label_idxs, :])
            else:
                # Two discrete sections
                if iidx >= start_idx or iidx < end_idx:
                    # Validation
                    u_x2.append(x_data[one_label_idxs, :])
                    u_y2.append(y_data[one_label_idxs, :])
                else:
                    # Training
                    u_x1.append(x_data[one_label_idxs, :])
                    u_y1.append(y_data[one_label_idxs, :])

        # Training
        if exclude_ratio< 1:
            x_1, y_1 = np.concatenate(u_x1), np.concatenate(u_y1)
        # Validation
        x_2, y_2 = np.concatenate(u_x2), np.concatenate(u_y2)
        
        # Add back edge and external points
        if exclude_ratio < 1:
            x_1 = np.vstack([x_1, x_data[non_chunk_indicies]])
            y_1 = np.vstack([y_1, y_data[non_chunk_indicies]])
        else:
            x_1 = x_data[non_chunk_indicies]
            y_1 = y_data[non_chunk_indicies]
        
        
        # Shuffle the training data
        idxs = np.random.RandomState(seed=42).permutation(x_1.shape[0])
        x_1 = x_1[idxs]
        y_1 = y_1[idxs]

        # Output for testing
        print("Length of test set labels:")
        print(y_2.shape[0])

        # Output for testing
        print("Length of train set labels:")
        print(y_1.shape[0])

        return ((x_1,y_1),(x_2,y_2))

    def train_test_by_probe(self, data, train_probes, test_probes):
        """
        Makes a train and test sets based on probe labels

        args:
            data : data_dict
            train_probes : list
                list of probes for train set
            test_probes : list
                list of probe for test set
        """
        return (self.pool_probes(data, train_probes), self.pool_probes(data, test_probes))

    def train_val_test_by_probe(self, data, train_probes, val_probes, test_probes):
        """
        Makes a train, val, and test sets based on probe labels.

        Just pools data from probes, no within probe data split.

        args:
            data : data_dict
            train_probes : list
                list of probes for train set
            val_probes : list
                list of probes for val set
            test_probes : list
                list of probe for test set
        """
        return (self.pool_probes(data, train_probes), self.pool_probes(data, val_probes), self.pool_probes(data, test_probes))


    def train_val_test_by_probe_stratified(self, data, train_val_probes, test_probes, val_ratio, val_set_index=0):
        """
        Splits into train_val and test by probe labels.
        Then does a concentration based split per probe in the train_val set

        args:
            data : data_dict
            train_val_probes : list
            test_probes : list
            val_ratio : float
            val_set_index : int (Default: 0)
        """
        test_set = self.pool_probes(data, test_probes)
        x_train = []
        y_train = []
        x_val = []
        y_val = []
        for probe in train_val_probes:
            ((xt, yt), (xv, yv)) = self.unique_concentration_split(data[probe], val_ratio, val_set_index)
            x_train.append(xt)
            y_train.append(yt)
            x_val.append(xv)
            y_val.append(yv)
        x_train = np.vstack(x_train)
        y_train = np.vstack(y_train)
        x_val = np.vstack(x_val)
        y_val = np.vstack(y_val)
        return ((x_train, y_train), (x_val, y_val), test_set)

    def k_fold_single_probe(self, data, probe, val_ratio, num_folds, val_set_index=0, fold_index=0):
        """
        Splits the data from a single probe into a train_val and test set based on num_fold and fold_index for k-fold CV.
        Then splits the train_val set into train and val sets based on val_ratio and fold_index

        args:
            data : data_dict
            probe : String
                The probe that has the test running on it
            val_ratio : float
                The ratio of probes that are used in the validation, not test set.
            num_folds : int
                The number of folds to make.
            val_set_index : int (Default: 0)
                The validation set to use
            fold_index : int (Default: 0)
                The fold to use as the test set.

        """

        ((x_train_val, y_train_val), (x_test, y_test)) = self.unique_concentration_split(data[probe], (1.0/num_folds), fold_index)
        ((x_train, y_train), (x_val, y_val)) = self.unique_concentration_split((x_train_val, y_train_val), val_ratio, val_set_index)
        return ((x_train, y_train), (x_val, y_val), (x_test, y_test))

    def mixture_k_fold_single_probe(self, data, probe, val_ratio, exclude_ratio, num_folds, val_set_index=0, fold_index=0):
        """
        Splits the data from a single probe into a train_val and test set based on num_fold and fold_index for k-fold CV.
        Only split the fixtures into folds. Always puts edge data into train

        Then splits the train_val set into train and val sets based on val_ratio and fold_index

        args:
            data : data_dict
            probe : String
                The probe that has the test running on it
            val_ratio : float
                The ratio of the data that is used for the validation, not test set.
                Note that the this split is done after the test data is removed.
            exclude_ratio : float
                The ratio of the data that is used for the test set.
            num_folds : int
                The number of folds to make.
            val_set_index : int (Default: 0)
                The validation set to use
            fold_index : int (Default: 0)
                The fold to use as the test set.

        """

        ((x_train_val, y_train_val), (x_test, y_test)) = self.unique_mixture_allocate(data[probe], exclude_ratio, num_folds, fold_index)
        ((x_train, y_train), (x_val, y_val)) = self.unique_concentration_split((x_train_val, y_train_val), val_ratio, val_set_index)
        return ((x_train, y_train), (x_val, y_val), (x_test, y_test))

    def slice(self, data, **kwargs):
        """
        The arbitor of functions
        """
        if ('train_probes' in kwargs):
            if 'test_probe' in kwargs:
                if isinstance(kwargs['test_probe'], list):
                    test_probes = kwargs['test_probe']
                else:
                    test_probes = [kwargs['test_probe']]
            elif 'test_probes' in kwargs:
                test_probes = kwargs['test_probes']
            else:
                raise NameError(f"Incorrect arguments for study strap slice. Missing test_probe(s)")
        else:
                raise NameError("Incorrect arguments for study strap slice. Missing train_probes")

        if 'val_ratio' in kwargs:
            return self.train_val_test_by_probe_stratified(data, kwargs['train_probes'], test_probes, kwargs['val_ratio'])
        else:
            return self.train_test_by_probe(data, kwargs['train_probes'], test_probes)


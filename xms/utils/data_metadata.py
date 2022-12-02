"""
data_metadata.py Generates a metadata file for a dataset

@author:    Thomas Twomey
            Montague Lab
            Fralin Biomedical Research Institute at VTC

@contact:   twomey@vtc.vt.edu

"""

""" 
Usage:

Model Creation
===============================================================================
dm = DataMetadata(
    name = "Name"
    metrics = list of functions
    normalization_function = function to convert y values to model space
    unnormalization_function = function to convert model output to real space
    analytes = list of Analytes in model
    means = list of means of the analytes, used for normalization and metrics
    stds = list stds of the analytes, used for normalization and metrics
    data_folder = directory where training data came from
    probe_list = list of probes used in training
    test_probes = Probes in the test dataset
    val_probes = Probes in the val dataset if not val_split is None
)

dm.update(
    parameter = new_val
)

dm.validate() # Does some (not complete) type and validity checking

dm.save(model_directory+"metadata.pkl") # Saves as binary (dill format)
# Should be saved in same directory as dataset to enable easy loading

Model Loading
===============================================================================
dm = DataMetadata()
dm.load(filename)

access via
dm.parameter

Description
===============================================================================
Creates a file to hold metadata for the testing and use of the models.
- Name (label)
- Metrics (stored as list)
- Normalization and Un-normalization (revert) functions
- Analytes being modeled
- Mean and Std of each analyte
- Data_folder where training data was sourced
- All Probes (stored as list) 
- Val Probes (stored as list) 
- Test Probes (stored as list) 
- Val Ratio 
"""

# Imports
from dill import dump, load
from pathlib import Path


class DataMetadata:
    """
    The Metadata Handler
    """
    def __init__(self, name = None, metrics = None, normalization_function = None, unnormalization_function = None, 
                analytes = None, means = None, stds = None, data_folders = None, all_probes=None, test_probes=None, 
                val_probes=None, val_ratio=None):

        self.name = name
        self.metrics = metrics
        self.normalization_function = normalization_function
        self.unnormalization_function = unnormalization_function
        self.analytes = analytes
        self.means = means
        self.stds = stds
        self.data_folders = data_folders
        self.all_probes = all_probes
        self.test_probes = test_probes
        self.val_probes = val_probes
        self.val_ratio = val_ratio
        

    def validate(self):
        """
        Validates
        """
        for (key, value) in self.__dict__.items():
            # TODO: more validation
            if value is not None and key in ["loss_function", "normalization_function", "unnormalization_function"]:
                # Call the functions to force them to be built
                try:
                    value() #TODO I dont think will actually force the function to be built
                except Exception:
                    continue
        return True

    def save_txt_file(self, file_name):
        out = ""
        for (key, value) in self.__dict__.items():
            out = f"{out}{key}: {value}\n"
        p = Path(file_name)

        if p.suffix != '.txt':
            file_name = f"{p.parent}/metadata.txt"
        
        with open(file_name, 'w') as file:
            file.write(out)

    def load(self, file_name):
        """
        Load the data from a file
        """
        print(f"Loading data from {file_name}")
        with open(file_name, 'rb') as file:
            self.__dict__ = load(file)

    def save(self, file_name):
        """
        Save the data to a file
        """
        self.validate()
        with open(file_name, 'wb') as file:
            dump(self.__dict__, file)
        self.save_txt_file(file_name)



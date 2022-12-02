"""
model_metadata.py Generates a metadata file for models

@author:    Thomas Twomey
            Montague Lab
            Fralin Biomedical Research Institute at VTC

@contact:   twomey@vtc.vt.edu

"""

""" 
Usage:

Model Creation
===============================================================================
mm = ModelMetadata(
    name = "Name"
    metrics = list of functions
    loss_function = function if non-standard
    normalization_function = function to convert y values to model space
    unnormalization_function = function to convert model output to real space
    analytes = list of Analytes in model
    means = list of means of the analytes, used for normalization and metrics
    stds = list stds of the analytes, used for normalization and metrics
    data_folder = directory where training data came from
    probe_list = list of probes used in training
)

mm.update(
    parameter = new_val
)

mm.validate() # Does some (not complete) type and validity checking

mm.save(model_directory+"metadata.pkl") # Saves as binary (dill format)
# Should be saved in same directory as model to enable easy loading

Model Loading
===============================================================================
mm = ModelMetadata()
mm.load(filename)

access via
mm.parameter

Description
===============================================================================
Creates a file to hold metadata for the testing and use of the models.
- Name (label)
- Metrics (stored as list)
- Loss function (only specified if not standard)
- Normalization and Un-normalization (revert) functions
- Analytes being modeled
- Mean and Std of each analyte
- Datafolder where training data was sourced
- Probes (stored as list) 
"""
# Imports
from dill import dump, load
from pathlib import Path


class ModelMetadata:
    """
    The Metadata Handler
    """
    def __init__(self, name = None, metrics = None, loss_function = None, 
                normalization_function = None, unnormalization_function = None, 
                analytes = None, means = None, stds = None, data_folder = None, 
                probe_list = None, patience = None, epochs = None,
                batch_size = None, all_probes=None, test_probes=None, val_probes=None):

        self.name = name
        self.metrics = metrics
        self.loss_function = loss_function
        self.normalization_function = normalization_function
        self.unnormalization_function = unnormalization_function
        self.analytes = analytes
        self.means = means
        self.stds = stds
        self.data_folder = data_folder
        self.probe_list = probe_list
        self.patience = patience
        self.epochs = epochs
        self.batch_size = batch_size
        self.all_probes = all_probes
        self.test_probes = test_probes
        self.val_probes = val_probes
        

    def update(self, **kwargs):
        """
        Update existing value by keyword
        """
        for (key, value) in kwargs.items():
            if key in self.__dict__:
                self.__dict__[key] = value
            else:
                raise ValueError(f"Invalid key: {key}")

    def validate(self):
        """
        Validates
        """
        for (key, value) in self.__dict__.items():
            if value is None:
                if key == "loss_function":
                    print("Loss function omitted, must be using default loss")
                    continue
                if key == "probe_list":
                    print("Probe list omitted. Not recommended.")
                    continue
                if key == "metrics":
                    print("No Metrics reported. Model will not load if custom metrics were used.")
                    continue
                if key in ["patience", "epochs", "batch_size"]:
                    print(f"Hyperparameter {key} omitted")
                    continue
                if key in ["all_probes", "test_probes", "val_probes"]:
                    print(f"{key} Not required for backwards compatibility. Strongly recommended")
                    continue
                print("Value of " +str(key) + " is needed. Please add.")

                return False

            if value is not None and key in ["loss_function", "normalization_function", "unnormalization_function"]:
                # Call the functions to force them to be built
                try:
                    value() #I dont think will actually force the function to be built
                except Exception:
                    continue
                continue
        if not len(self.analytes) == len(self.stds) == len(self.means):
            print("Mismatch in number of analytes, means, and stds provided.")
            return False
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



# Deep Learning Architectures for FSCV, a Comparison

See paper: 

A companion repository.

## Directories

### classifiers
- This is where the actual deep learning architectures are implemented. 
- All models have the same interface.
### data_loader
- This handles a few approached to data loading from files to numpy arrays. 
- The class FilePairDataLoader is implemented for our internal pipeline and as an example.
### data_recorder
- Wraps the interaction with TFRecord files.
### data_slicer
- This is where datasets are cut up into training, validation and training sets.
- The class TrainValTestDataSlicer is included as it handles the standard usecases.
### fitting
- This is where all the parts are put together for the training/fitting of the model.
- parametric_fit_function.py is the current version.
- params -> folder to store/load model parameter files
- scripts -> folder to store/load SLURM/Bash scripts that call parametric_fit_function.py with a certain parameter file.
### predicting
- This is where all the parts are put together for predicting and evaluation of a model
- predict_from_abf_w_ensemble.py is the file of interest.
- scripts -> folder to store/load SLURM/Bash scripts that call predict_from_abf_w_ensemble.py
### utils
- Model and charting utilities for the other components.
### figures_and_tables
- Data compilation and plotting.

## Architectures

### InceptionTime
- Derivative of InceptionTime as created by hfawaz https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py
- From paper: https://arxiv.org/abs/1909.04939

### Multi-layer Perception (MLP)
- Just a simple fully connected network

### EEG-Transformer
- Derivative of the EEG-Transformer as created by eeyhsong https://github.com/eeyhsong/EEG-Transformer/
- From paper: https://arxiv.org/pdf/2106.11170

### SSVEPformer
- From paper: https://arxiv.org/abs/2210.04172v1

### Fully Convolutional Network (FCN)
- Derivative of FCN implementation from https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline/blob/master/FCN.py
- From paper: https://arxiv.org/abs/1611.06455

## Parameters

Inside fitting/params/76PE_10-F-CV/ are sub-directories with folders for each model.
"""
data_recorder.py

Thomas Twomey
Montague Lab
Fralin Biomedical Research Institute at VTC
01/24/2022

Helper file to wrap our interaction with TensorFlow Record objects

Hi there,

The goal here is to wrap the interaction with the creation and reading of 
TensorFlow Record objects. Contains a few parsing functions and loading 
functions for the TFRecords. This function is useful for dataset are 
sufficiently large such that loading them directly is not feasible or fast 
enough.

Based on the following tutorials: 

https://keras.io/examples/keras_recipes/creating_tfrecords/

https://www.tensorflow.org/tutorials/load_data/tfrecord

https://keras.io/examples/keras_recipes/tfrecord/

"""

# Imports
from glob import glob
from pathlib import Path
import tensorflow as tf
import numpy as np
import json

###############################################################################
# Handy helper functions
###############################################################################

def _bytes_feature(value):
    """
    Returns a bytes_list from a string / byte.
    """
    #print(f"Executing Eagerly: {tf.executing_eagerly()}")
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    #return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def _float_feature(value):
    """
    Returns a float_list from a float / double.
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """
    Returns an int64_list from a bool / enum / int / uint.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
    """
    Wraps the tensorflow.io.serialize_tensor function
    """
    
    return tf.io.serialize_tensor(array)


###############################################################################
# np.array -> TFRecord
###############################################################################

def _parse_one_dimension_arr(in_arr, label_arr):
    """
    """
    data = {
        'dim0' : _int64_feature(in_arr.shape[0]),
        'in_arr' : _bytes_feature(serialize_array(in_arr)),
        'n_analytes' : _int64_feature(label_arr.shape[0]),
        'label' : _bytes_feature(serialize_array(label_arr))
    }
    #create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out

def _parse_two_dimension_arr(in_arr, label_arr):
    """
    """

    # print(f"_parse_two_dimension_arr in_arr.shape[0]: {in_arr.shape[0]}")
    # print(f"_parse_two_dimension_arr in_arr.shape[1]: {in_arr.shape[1]}")


    data = {
        'dim0' : _int64_feature(in_arr.shape[0]),
        'dim1' : _int64_feature(in_arr.shape[1]),
        'in_arr' : _bytes_feature(serialize_array(in_arr)),
        'n_analytes' : _int64_feature(label_arr.shape[0]),
        'label' : _bytes_feature(serialize_array(label_arr))
    }
    #create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out

def _parse_three_dimension_arr(in_arr, label_arr):
    """
    """
    data = {
        'dim0' : _int64_feature(in_arr.shape[0]),
        'dim1' : _int64_feature(in_arr.shape[1]),
        'dim2' : _int64_feature(in_arr.shape[2]),
        'in_arr' : _bytes_feature(serialize_array(in_arr)),
        'n_analytes' : _int64_feature(label_arr.shape[0]),
        'label' : _bytes_feature(serialize_array(label_arr))
    }
    #create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out

def _parse_four_dimension_arr(in_arr, label_arr):
    """
    """
    data = {
        'dim0' : _int64_feature(in_arr.shape[0]),
        'dim1' : _int64_feature(in_arr.shape[1]),
        'dim2' : _int64_feature(in_arr.shape[2]),
        'dim3' : _int64_feature(in_arr.shape[3]),
        'in_arr' : _bytes_feature(serialize_array(in_arr)),
        'n_analytes' : _int64_feature(label_arr.shape[0]),
        'label' : _bytes_feature(serialize_array(label_arr))
    }
    #create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out

def _parse_n_dimension_arr(in_arr, label_arr):
    """
    """

    # Determine the number of dimensions

    dim = len(in_arr.shape)

    fcn_list = [_parse_one_dimension_arr, _parse_two_dimension_arr, _parse_three_dimension_arr, _parse_four_dimension_arr]

    return fcn_list[dim-1](in_arr, label_arr)

###############################################################################
# TFRecord -> np.array
###############################################################################

def _parse_four_dimension_tfr_element(element):
    """
    """
    data = {
        'dim0' : tf.io.FixedLenFeature([], tf.int64),
        'dim1' : tf.io.FixedLenFeature([], tf.int64),
        'dim2' : tf.io.FixedLenFeature([], tf.int64),
        'dim3' : tf.io.FixedLenFeature([], tf.int64),
        'in_arr' : tf.io.FixedLenFeature([], tf.string),
        'n_analytes' : tf.io.FixedLenFeature([], tf.int64),
        'label' : tf.io.FixedLenFeature([], tf.string)
    }

    content = tf.io.parse_single_example(element, data)

    label = content['label']
    in_arr = content['in_arr']

    #get our 'feature'
    f_in_arr = tf.io.parse_tensor(in_arr, out_type=tf.float32)
    f_in_arr = tf.reshape(f_in_arr, shape=[content['dim0'],content['dim1'],content['dim2'],content['dim3']])

    # Reshape the label
    f_label = tf.io.parse_tensor(label, out_type=tf.float32)
    f_label = tf.reshape(f_label, shape=(content['n_analytes']))

    return (f_in_arr, f_label)

def _parse_three_dimension_tfr_element(element):
    """
    """
    data = {
        'dim0' : tf.io.FixedLenFeature([], tf.int64),
        'dim1' : tf.io.FixedLenFeature([], tf.int64),
        'dim2' : tf.io.FixedLenFeature([], tf.int64),
        'in_arr' : tf.io.FixedLenFeature([], tf.string),
        'n_analytes' : tf.io.FixedLenFeature([], tf.int64),
        'label' : tf.io.FixedLenFeature([], tf.string)
    }

    content = tf.io.parse_single_example(element, data)

    label = content['label']
    in_arr = content['in_arr']

    #get our 'feature'
    f_in_arr = tf.io.parse_tensor(in_arr, out_type=tf.float32)
    f_in_arr = tf.reshape(f_in_arr, shape=[content['dim0'],content['dim1'],content['dim2']])

    # Reshape the label
    f_label = tf.io.parse_tensor(label, out_type=tf.float32)
    f_label = tf.reshape(f_label, shape=(content['n_analytes']))

    return (f_in_arr, f_label)

def _parse_two_dimension_tfr_element(element):
    """
    """
    data = {
        'dim0' : tf.io.FixedLenFeature([], tf.int64),
        'dim1' : tf.io.FixedLenFeature([], tf.int64),
        'in_arr' : tf.io.FixedLenFeature([], tf.string),
        'n_analytes' : tf.io.FixedLenFeature([], tf.int64),
        'label' : tf.io.FixedLenFeature([], tf.string)
    }

    content = tf.io.parse_single_example(element, data)

    label = content['label']
    in_arr = content['in_arr']

    print(f"Dim0: {content['dim0']}")
    print(f"Dim1: {content['dim1']}")

    #get our 'feature'
    f_in_arr = tf.io.parse_tensor(in_arr, out_type=tf.float32)
    #f_in_arr = tf.reshape(f_in_arr, shape=(999,1))#content['dim0'],content['dim1']])

    # Reshape the label
    f_label = tf.io.parse_tensor(label, out_type=tf.float32)
    #f_label = tf.reshape(f_label, shape=[(4)])#content['n_analytes']))

    return (f_in_arr, f_label)

def _parse_one_dimension_tfr_element(element):
    """
    """
    data = {
        'dim0' : tf.io.FixedLenFeature([], tf.int64),
        'in_arr' : tf.io.FixedLenFeature([], tf.string),
        'n_analytes' : tf.io.FixedLenFeature([], tf.int64),
        'label' : tf.io.FixedLenFeature([], tf.string)
    }

    content = tf.io.parse_single_example(element, data)

    label = content['label']
    in_arr = content['in_arr']

    #get our 'feature'
    f_in_arr = tf.io.parse_tensor(in_arr, out_type=tf.float32)
    print(f_in_arr)
    f_in_arr = tf.reshape(f_in_arr, shape=[content['dim0']])

    # Reshape the label
    f_label = tf.io.parse_tensor(label, out_type=tf.float32)
    f_label = tf.reshape(f_label, shape=[content['n_analytes']])

    return (f_in_arr, f_label)

def _parse_fcns_from_shape(input_shape, output_shape):
    """
    Given the shape of a feature (not whole array) for the input and output,
    returns the functions.

    parameters:
        input_shape : tuple
            Shape of a single feature in the input array.
        output_shape : tuple
            Shape of a single label array.
            Currently Unused.
    """

    in_dim = len(input_shape)

    in_fcn_list = [
        _parse_one_dimension_arr, 
        _parse_two_dimension_arr, 
        _parse_three_dimension_arr, 
        _parse_four_dimension_arr
    ]

    out_fcn_list = [
        _parse_one_dimension_tfr_element, 
        _parse_two_dimension_tfr_element, 
        _parse_three_dimension_tfr_element, 
        _parse_four_dimension_tfr_element
    ]

    print(f"parse_fcns_from_shape, in_func: {in_fcn_list[in_dim-2].__name__}")
    print(f"parse_fcns_from_shape, out_func: {out_fcn_list[in_dim-2].__name__}")

    return in_fcn_list[in_dim-1], out_fcn_list[in_dim-1]

def _parse_fcns_from_arr(in_arr):
    """
    From the input array, returns the functions that parse to and from TFRecords
    """
    # Determine the number of dimensions
    feature_in_shape = in_arr[0].shape

    return _parse_fcns_from_shape(feature_in_shape, None)


###############################################################################
# Writing the TFRecord 
###############################################################################

def write_examples_tfr(out_dir, in_arr, labels, filename, n_nodes, in_parse_fcn=None, extendable=False):
    """
    The function will:
    Calculate the number of shards. From the guide:
        "Ideally, you should shard the data to ~10*N files, as long as ~(size)/(10*N) is 10 MB+ (and ideally 100 MB+)"
    
    Ideally we will also denote the shape in the of the features to that we know how to load the data.

    parameters:
        out_dir : string or path-like
            Where to save the shards
        in_arr : np.ndarray
            The input array to split up into shards as a TFRecord
        labels : np.ndarray
            The label array to split up into shards as a TFRecord
        filename : string (Default: 'shard')
            What to call the shards
        n_nodes : int (Default: 2)
            How many nodes are going to be using the TFRecord
        in_parse_fcn : fcn (Default: None)
        extendable : Boolean (Default: False)
    """

    # Calculate shard count:
    size = in_arr.nbytes
    shard_count = n_nodes * 10
    if size // shard_count < 1e8:
        shard_count = size // 1.1e8
    shard_count = max(int(shard_count),1)
    print(f"Shard Count: {shard_count}")

    # Infer parse_fcn if not provided
    if in_parse_fcn is None:
        print(f"Inferring TFRecord parsing functions from arrays.")
        in_parse_fcn, _ = _parse_fcns_from_shape(in_arr[0].shape, labels[0].shape)

    # Keep track of where to load from the inputs
    current_row = 0

    for i in range(shard_count):
        if not extendable:
            shard_file_path = f"{out_dir}/{filename}_{i:03}_{shard_count:03}.tfrecords"
        else:
            # The 'XXX' is to denote that there is no fixed number of records
            shard_file_path = f"{out_dir}/{filename}_{i:03}_XXX.tfrecords"

        # Calculate how many examples to put in this file
        n_observations = in_arr.shape[0] // shard_count + (1 if i < in_arr.shape[0] % shard_count else 0)
        n_observations = int(n_observations)

        print(f"Attempting to save {n_observations} examples to {shard_file_path}")

        with tf.io.TFRecordWriter(shard_file_path) as writer:
            for i in range(n_observations):
                example = in_parse_fcn(in_arr[current_row], labels[current_row])
                #print(f"## LOOK ## in_arr[current_row].shape: {in_arr[current_row].shape}")
                writer.write(example.SerializeToString())
                current_row+=1

def _save_spec_file(out_dir, input_shape, output_shape, file_stem="record_spec"):
    """
    Saves a specification file to inform the loading of the TFRecord
    """
    spec_dict = {
        "input_shape" : input_shape,
        "output_shape" : output_shape,
    }
    
    file_path = f"{out_dir}/{file_stem}.json"

    with open(file_path, 'w') as f:
        json.dump(spec_dict, f)

def create_shard_dir(out_dir, in_arr, labels, filename="shard", n_nodes=2, in_parse_fcn=None, extendable=False):
    """
    This function will:
    - Create the dir that will hold the shards
    - Create the shards (or some of them)
    - Denote the shape of the items in the shards (some other file)

    parameters:
        out_dir : string or path-like
            Where to save the shards
        in_arr : np.ndarray
            The input array to split up into shards as a TFRecord
        labels : np.ndarray
            The label array to split up into shards as a TFRecord
        filename : string (Default: 'shard')
            What to call the shards
        n_nodes : int (Default: 2)
            How many nodes are going to be using the TFRecord
        in_parse_fcn : fcn (Default: None)
        extendable : Boolean (Default: False)

    """

    # Create the directory
    dir = Path(out_dir)    
    if not dir.exists():
        dir.mkdir()
    elif not dir.is_dir():
        raise FileExistsError(f"out_dir exists and is not a directory.")

    # Create the shards
    write_examples_tfr(out_dir, in_arr, labels, filename, n_nodes, in_parse_fcn, extendable)

    # Save the Spec File
    _save_spec_file(out_dir, in_arr[0].shape, labels[0].shape)

def append_shard_dir(out_dir, in_arr, labels):
    """
    This function will:
    - Ensure that dir exists
    - Check to see that the existing array shape matches the new shapes
    - Infer the filename, shard_size from existing shards
    - Add new shards

    parameters:
        out_dir : string or path-like
            Where to save the shards
        in_arr : np.ndarray
            The input array to split up into shards as a TFRecord
        labels : np.ndarray
            The label array to split up into shards as a TFRecord
    """

    # Ensure the dir exists
    dir = Path(out_dir)    
    if not dir.exists():
        raise FileNotFoundError(f"out_dir: {out_dir} does not exist. Please use create_shard_dir.")
    elif not dir.is_dir():
        raise FileExistsError(f"out_dir: {out_dir} exists and is not a directory.")

    # Check to see that the existing array shape matches the new shape
    input_shape, output_shape = _load_spec_file(out_dir)

    if input_shape != list(in_arr[0].shape) or output_shape != list(labels[0].shape):
        raise ValueError("Shape of features in existing shards do not match new features.")
    print(f"Inferring TFRecord parsing functions from arrays.")
    
    in_parse_fcn, _ = _parse_fcns_from_shape(input_shape, output_shape)

    # Infer the filename, shard_size from existing shards
    sample_glob = glob(f"{out_dir}/*.tfrecords")

    if len(sample_glob) < 1:
        raise FileNotFoundError(f"No existing shards found in {out_dir} Please use create_shard_dir.")
    
    # Determine the max index that already exists
    max_index = max([int(Path(s).stem.split('_')[1]) for s in sample_glob])

    sample_file = sample_glob[0]

    filename_stem = Path(sample_file).stem.split('_')[0]

    # Determine the number of new shards to create
    # Size in bytes
    shard_size = Path(sample_file).stat().st_size
    new_shard_count = (in_arr.nbytes + labels.nbytes) // shard_size
    if ((in_arr.nbytes + labels.nbytes) / shard_size) - new_shard_count > 0.3:
        new_shard_count += 1


    # Create the shards

    # Keep track of where to load from the inputs
    current_row = 0

    for i in range(max_index+1, max_index+new_shard_count+1):
        shard_file_path = f"{out_dir}/{filename_stem}_{i:03}_XXX.tfrecords"

        # Calculate how many examples to put in this file
        n_observations = in_arr.shape[0] // new_shard_count + (1 if i < in_arr.shape[0] % new_shard_count else 0)
        n_observations = int(n_observations)

        print(f"Attempting to save {n_observations} examples to {shard_file_path}")

        with tf.io.TFRecordWriter(shard_file_path) as writer:
            for i in range(n_observations):
                example = in_parse_fcn(in_arr[current_row], labels[current_row])
                writer.write(example.SerializeToString())
                current_row+=1
    
def create_or_append_shard_dir(out_dir, in_arr, labels, filename="shard", n_nodes=2, in_parse_fcn=None, extendable=True):
    """
    Function to arbitrate which shard function to create.
    
    """

    print(f"inside: create_or_append_shard_dir")
    print(f"outdir: {out_dir}")
    print(f"in_arr.shape: {in_arr.shape}")
    print(f"label.shape: {labels.shape}")

    if not _spec_file_exists(out_dir):
        create_shard_dir(out_dir, in_arr, labels, filename, n_nodes, in_parse_fcn, extendable)
    else:
        append_shard_dir(out_dir, in_arr, labels)

###############################################################################
# Reading from the TFRecord 
###############################################################################

def _spec_file_exists(in_dir, file_stem=None):
    """
    Returns if a specification file exists.
    """
    if file_stem is None or len(file_stem) < 1:
        file_glob = glob(f"{in_dir}/*.json")
        if len(file_glob) < 1:
            return False
    else:
        file_glob = glob(f"{in_dir}/{file_stem}.json")
        if len(file_glob) < 1:
            return False

    return True

def _load_spec_file(in_dir, file_stem=None):
    """
    Loads a specification file to inform the loading of the TFRecord
    If file_stem is not specified pulls first json from dir.

    parameters:
        int_dir : string or path-like
            Where to load the shards from
    """
    if file_stem is None or len(file_stem) < 1:
        file_glob = glob(f"{in_dir}/*.json")
        if len(file_glob) < 1:
            raise FileNotFoundError(f"No spec file found in {in_dir}")
    else:
        file_glob = glob(f"{in_dir}/{file_stem}.json")
        if len(file_glob) < 1:
            raise FileNotFoundError(f"Spec file {in_dir}/{file_stem}.json Does Not Exist.")

    file_name = file_glob[0]

    with open(file_name, 'r') as f:
        spec_dict = json.load(f)
    
    return spec_dict["input_shape"], spec_dict["output_shape"]

def reconstruct_dataset(file_list, out_parse_fcn=None):
    """
    Recreate the arrays from a list of tfrecord files

    parameters:
        file_list : list of strings
            The various files to load and incorporate into the tfrecord.
    """
    
    # Check file_list
    if not isinstance(file_list, list):
        file_list = [file_list]

    if len(file_list) < 1:
        raise ValueError("No files were passed to be loaded.")

    # Check if out_parse_fcn is provided
    if out_parse_fcn is None:
        # Parsing function is not provided thus we need to infer data types 
        # from the directory the files live in. Assume that all files are in
        # in the save directory or share the same formatting.
        parent_dir = Path(file_list[0]).parent
        input_shape, output_shape = _load_spec_file(parent_dir)
        _, out_parse_fcn = _parse_fcns_from_shape(input_shape, output_shape)


    #create the dataset
    dataset = tf.data.TFRecordDataset(file_list)

    #pass every single feature through our mapping function
    dataset = dataset.map(
        out_parse_fcn,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # for sample in dataset.take(10):
    #     print(sample[0].numpy().shape)
    #     print(sample[1].numpy().shape)

    dataset = dataset.shuffle(1024)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.batch(64)
    return dataset

def load_dataset_from_dir(in_dir, out_parse_fcn=None):
    """
    Recreate the arrays from all the tfrecord files in a directory

    parameters:
        in_dir : np.ndarray
        out_parse_fcn : function (Default: None)
    """

    # Ensure the dir exists
    dir = Path(in_dir)    
    if not dir.exists():
        raise FileNotFoundError(f"out_dir: {in_dir} does not exist.")
    elif not dir.is_dir():
        raise FileExistsError(f"out_dir: {in_dir} exists and is not a directory.")

    files = glob(f"{in_dir}/*.tfrecords")

    if len(files) < 1:
        raise FileNotFoundError(f"No .tfrecords files found in {in_dir}")

    return reconstruct_dataset(files)
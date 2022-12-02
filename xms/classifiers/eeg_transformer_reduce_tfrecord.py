"""
ssvep_former_tfrecord.py

Thomas Twomey
Montague Lab
Fralin Biomedical Research Institute at VTC
10/19/2022

Derivative of a eeg_transformer as described in:
https://arxiv.org/pdf/2106.11170.pdf
and 
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9845479

Code derived from repo:
https://github.com/eeyhsong/EEG-Transformer/

Note: this version uses a mean reduction layer instead of a token extraction layer.

"""

import numpy as np
import time
import pandas as pd
from datetime import datetime
from utils.utils import save_logs, plot_epochs_metric, calculate_metrics, save_test_duration, rmse
import tensorflow as tf
from einops.layers.keras import Reduce

ANALYTES_NAMES = ['DA', '5HT', 'pH', 'NE']


"""
ClassToken Class taken from:
https://github.com/faustomorales/vit-keras/blob/master/vit_keras/layers.py
Apache License 2.0
"""
@tf.keras.utils.register_keras_serializable()
class ClassToken(tf.keras.layers.Layer):
    """Append a class token to an input layer."""

    def build(self, input_shape):
        cls_init = tf.zeros_initializer()
        self.hidden_size = input_shape[-1]
        self.cls = tf.Variable(
            name="cls",
            initial_value=cls_init(shape=(1, 1, self.hidden_size), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        return tf.concat([cls_broadcasted, inputs], 1)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



"""
AddPositionEmbs Class taken from:
https://github.com/faustomorales/vit-keras/blob/master/vit_keras/layers.py
Apache License 2.0

Modified for single dimension input
"""
@tf.keras.utils.register_keras_serializable()
class AddPositionEmbs(tf.keras.layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def build(self, input_shape):
        self.pe = tf.Variable(
            name="pos_embedding",
            initial_value=tf.random_normal_initializer(stddev=0.06)(
                shape=(1, input_shape[1], input_shape[2])
            ),
            dtype="float32",
            trainable=True,
        )

    def call(self, inputs):
        return inputs + tf.cast(self.pe, dtype=inputs.dtype)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Regression_Model:

    def __init__(self, output_directory, input_shape, output_shape, model_params, verbose=False, build=True, batch_size=64,
                nb_epochs=100, metrics=None, loss=None, pre_model=None, normalize_y=(lambda x: x, lambda x: x), patience=50,
                analytes_names=ANALYTES_NAMES, min_lr = 0.00001):

        # Training params 
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.metrics = metrics
        self.loss = loss
        self.pre_model = pre_model
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.output_directory = output_directory

        # Data params
        self.analytes_names = analytes_names
        self.normalize_data = normalize_y[0]
        self.revert_data = normalize_y[1]
        self.input_shape = input_shape
        self.output_shape = output_shape

        # Ensure that model_params is correctly formed
        if not "model_type" in model_params or not model_params["model_type"].lower() == "eeg_transformer_reduce":
            raise ValueError("model_params has incorrect 'model_type' value.")
        for key in ["dropout", "patch_size", "dropout", "head_size", "num_heads", "num_layers"]:
            if key not in model_params:
                raise ValueError(f"model_params does not have needed key: {key}")

        # Model params
        self.hidden_size = model_params["hidden_size"]
        self.patch_size = model_params["patch_size"]
        self.dropout = model_params["dropout"]
        self.head_size = model_params["head_size"]
        self.num_heads = model_params["num_heads"]
        self.num_layers = model_params["num_layers"]
        self.mlp_dim = model_params["mlp_dim"]

        if build == True:
            self.model = self.build_model(self.input_shape, self.output_shape, pre_model=self.pre_model)
            self.model.save(self.output_directory + 'model_init.hdf5')


    def _save_logs(self, hist, duration,
                  lr=True, plot_test_acc=True):
        """
        Internal function that saves various csv and can create plots
        """
        hist_df = pd.DataFrame(hist.history)
        hist_df.to_csv(self.output_directory + 'history.csv', index=False)

        if plot_test_acc:
            print('using val_loss to find best metrics')
            index_best_model = hist_df['val_loss'].idxmin()
        else:
            print('using loss to find best metrics')
            index_best_model = hist_df['loss'].idxmin()

        row_best_model = hist_df.loc[index_best_model]

        df_best_model = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                                     columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_learning_rate', 'best_model_nb_epoch'])

        df_best_model['best_model_train_loss'] = row_best_model['loss']
        if plot_test_acc:
            df_best_model['best_model_val_loss'] = row_best_model['val_loss']

        if lr == True:
            df_best_model['best_model_learning_rate'] = row_best_model['lr']
        df_best_model['best_model_nb_epoch'] = index_best_model

        df_best_model.to_csv(self.output_directory + 'df_best_model.csv', index=False)

        if plot_test_acc:
            # plot losses
            plot_epochs_metric(hist, self.output_directory + 'epochs_loss.png')

            for (iname, name) in enumerate(self.analytes_names):
                #plot_epochs_metric(hist, self.output_directory + f'epochs_{name}.png', metric=f'tf_pmse_{name}')
                try:
                    plot_epochs_metric(hist, self.output_directory + f'epochs_{name}.png', metric=[m.__name__ for m in self.metrics if name in m.__name__][0])
                except Exception as e:
                    print(e)
                    continue

    

    def patch_embedding(self, input):
        """
        Contains:
            Convolution
            ClassToken Layer
            Add Position Embedding layer
        """
        x = input

        # Convolutional Block
        # vit only has 1 convolution layer
        # eeg_tranformer has 2 convolution layers
        x = tf.keras.layers.Conv1D(filters=self.hidden_size,
                                kernel_size=self.patch_size, # 27? 999 = 37 * 27 or 9?
                                strides=self.patch_size,
                                padding="valid",
                                name="embedding",)(x)
        # Class token
        x = ClassToken(name="class_token")(x)

        x = AddPositionEmbs(name="position_embedding")(x)

        return x

    def transformer_block(self, input):
        """
        """

        x = input

        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.MultiHeadAttention(key_dim=self.head_size, num_heads=self.num_heads)(x, x) # x,x for self-attention
        x = tf.keras.layers.Dropout(self.dropout)(x)
        
        # residual connection
        x = tf.keras.layers.Add()([input, x])

        y = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # MLP block
        y = tf.keras.layers.Dense(self.mlp_dim)(y)
        y = tf.keras.layers.Activation(activation='gelu')(y)
        y = tf.keras.layers.Dropout(self.dropout)(y)
        y = tf.keras.layers.Dense(self.hidden_size)(y) # may want to change the parameters here
        y = tf.keras.layers.Dropout(self.dropout)(y)
        
        # residual connection
        out = tf.keras.layers.Add()([x, y])

        return out



    def build_model(self, input_shape, output_shape, pre_model=None): 
        """
        Builds the model with inception modules.

        params:
            input_shape : array-like?
            output_shape : array-like?
            pre_model : keras model
                Trained model to load weights from. Defaults to None
                Should be exact same shape as new model
        """

        input_layer = tf.keras.layers.Input(input_shape)

        x = self.patch_embedding(input_layer)

        for i in range(self.num_layers):
            x = self.transformer_block(x)

        # x = tf.keras.layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(x)

        x = Reduce('b n e -> b e', reduction='mean')(x)

        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        output_layer = tf.keras.layers.Dense(output_shape, activation='softplus')(x)
        
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        print(model.summary())

        # exit()

        if not pre_model is None:
            print('loading previous weights (L-1 layers)...')
            for i in range(len(model.layers)-1):
                model.layers[i].set_weights(pre_model.layers[i].get_weights())
        else:
            print('starting model from scratch...')

        # Handle parameters or their absence
        if self.metrics is None:
            metrics = []
        else:
            metrics = self.metrics
        
        if self.loss is None:
            loss = 'mse' 
        else:
            loss = self.loss

        print(f'Compiling with loss {loss}, Adam (patience: {self.patience} for val_loss, verbose=1) and metrics: ', [m.__name__ for m in metrics])

        # Actually does the compilation of the model
        model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(), metrics=metrics)

        # Callback that reduces learning rate on plateau of val_loss
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=self.patience, min_lr=self.min_lr, verbose=1)

        # Callback that keeps a copy of the best model by val_loss saved
        file_path = self.output_directory + 'best_model.hdf5'
        model_checkpoint_val = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)

        # Callback that keeps a copy of the best model by loss saved
        file_path = self.output_directory + 'best_train_model.hdf5'
        model_checkpoint_train = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

        # Callback that same a copy of the model every 25 epochs
        file_path = self.output_directory + "model_epoch{epoch:08d}.hdf5"
        model_checkpoint_n_epoch = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, period=25)

        # # TensorBoard profiler
        # logs = f"{self.output_directory}/tfbp_logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        # tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
        #                                          histogram_freq = 1,
        #                                          profile_batch = '500,520')

        # The callbacks are stored as part of the object to be used in the fit function (not saved)
        self.callbacks = [reduce_lr, model_checkpoint_train, model_checkpoint_val, model_checkpoint_n_epoch] #, tboard_callback]

        

        return model

    def fit_ds(self, ds_train, ds_val, plot_test_acc=False, end_epoch=None, start_epoch=0):
        """
        A wrapper for the keras fit function for TFRECORD dataset

        Assumes that the normalization of the y values has already been done.
        """

        mini_batch_size = self.batch_size
        
        print(f'mini batch size: {mini_batch_size}')

        # Record start time for performance validation purposes
        start_time = time.time()

        # Set the final epoch
        epochs = self.nb_epochs if end_epoch is None else end_epoch

        print(f"ds_train: {ds_train}")

        # Call the keras fit function
        hist = self.model.fit(ds_train, epochs=epochs, initial_epoch=start_epoch,
                              verbose=self.verbose, validation_data=ds_val, callbacks=self.callbacks)

        # Calculate duration
        duration = time.time() - start_time

        self._save_logs(hist, duration, plot_test_acc=plot_test_acc)

        # Save the final model and minimal data about it
        self.model.save(self.output_directory + 'last_model.hdf5')
        df_last_model = pd.DataFrame({"epochs" : [epochs]})
        df_last_model.to_csv(self.output_directory+'df_last_model.csv')


        df_metrics = self.eval_ds(ds_val)

        # Clean up
        tf.keras.backend.clear_session()

        return df_metrics

    def eval_ds(self, ds_test):
        start_time = time.time()
        model = self.get_best_model()
        metrics = model.evaluate(ds_test, verbose=0)
        
        test_duration = time.time() - start_time

        print(dict(zip(model.metrics_names, metrics)))

        return pd.DataFrame(dict(zip(model.metrics_names, [[m] for m in metrics])))
        
    def predict_ds(self, ds_test, return_df_metrics=True, project_y=True, both = False):
        """
        
        """
        start_time = time.time()
        model = self.get_best_model()
        y_pred = model.predict(ds_test)
        # Load predictions in modified space
        if both:
            return (y_pred, self.eval_ds(ds_test))
        if return_df_metrics:
            df_metrics = self.eval_ds(ds_test)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred
    
    def load_best_model(self):
        """
        Calls get_best_model() and sets self.model to output
        """
        self.model = self.get_best_model()

    def get_best_model(self):
        """
        Loads the weights from best_model.hdf5 and determines the metrics from object properties.
        Then returns the model.
        """
        model_path = self.output_directory + 'best_model.hdf5'
        custom_objects = {"ClassToken" : ClassToken, "AddPositionEmbs": AddPositionEmbs}
        if not self.metrics is None:
            for metric in self.metrics:
                custom_objects[metric.__name__] = metric
        if not self.loss is None:
            custom_objects[self.loss.__name__] = self.loss
        print(custom_objects)
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)  

    def load_last_model(self):
        """
        Loads the weights from last_model.hdf5 and determines the metrics from object properties.
        Then returns the model.
        """
        model_path = self.output_directory + 'last_model.hdf5'
        custom_objects = {"ClassToken" : ClassToken, "AddPositionEmbs": AddPositionEmbs}
        if not self.metrics is None:
            for metric in self.metrics:
                custom_objects[metric.__name__] = metric
        if not self.loss is None:
            custom_objects[self.loss.__name__] = self.loss
        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects) 

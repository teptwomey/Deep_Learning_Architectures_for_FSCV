"""
fb-ssvep_former_tfrecord.py

Thomas Twomey
Montague Lab
Fralin Biomedical Research Institute at VTC
10/12/2022

Derivative of a SSVEPformer as described in:
https://arxiv.org/abs/2210.04172v1

"""

import numpy as np
from scipy.signal import butter, sosfilt, sosfreqz
import time
import pandas as pd
from datetime import datetime
from utils.utils import save_logs, plot_epochs_metric, calculate_metrics, save_test_duration, rmse
import tensorflow as tf

ANALYTES_NAMES = ['DA', '5HT', 'pH', 'NE']

# class Filter_bank:
class Linear(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

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
        if not "model_type" in model_params or not model_params["model_type"].lower() == "fb_ssvep_former":
            raise ValueError("model_params has incorrect 'model_type' value.")
        for key in ["output_mult", "dropout"]:
            if key not in model_params:
                raise ValueError(f"model_params does not have needed key: {key}")

        # Model params
        self.output_mult = model_params["output_mult"]
        self.dropout = model_params["dropout"]
        self.fb_n_filts = model_params["fb_n_filts"]
        self.fb_filt_step = model_params["fb_filt_step"]
        self.fs = model_params["fs"]
        self.fb_order = model_params["order"]

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

    def chnl_comb(self, input):
        """
        
        """
        x = input
        x = tf.keras.layers.Conv1D(2,1, padding='same')(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Activation(activation='gelu')(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)

        return x

    def conv_attn(self, input):
        """
        """
        x = input
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Conv1D(2,31, padding='same')(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Activation(activation='gelu')(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)

        # residual connection
        x = tf.keras.layers.Add()([input, x])

        return x

    def chnl_mlp(self, input, n_channels=2):
        """
        """
        x = input
        x = tf.keras.layers.LayerNormalization()(x)
        linear_layer = tf.keras.layers.Dense(self.input_shape[0]*2)
        
        # Apply this layer to each channel of x
        channel_outputs = []
        for i in range(n_channels):
            # Slicing the ith channel:
            c = tf.keras.layers.Lambda(lambda v: v[:, i])(x)
            c = linear_layer(c)
            c = tf.keras.layers.Reshape([-1,1])(c)
            channel_outputs.append(c)

        x = tf.keras.layers.Concatenate()(channel_outputs)

        x = tf.keras.layers.Activation(activation='gelu')(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)

        # residual connection
        x = tf.keras.layers.Add()([input, x])

        return x

    def mlp_head(self, input, n_outputs=4, output_mult=60):
        """
        """
        x = input
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)

        x = tf.keras.layers.Dense(output_mult * n_outputs)(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Activation(activation='gelu')(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)

        return x
        
    def ssvep(self, input):
        """
        """

        input_fft = tf.signal.fft(tf.cast(input, dtype=tf.complex64))

        real = tf.math.real(input_fft)
        imag = tf.math.imag(input_fft)

        concat_fft = tf.keras.layers.concatenate([real,imag], axis=1)

        x = self.chnl_comb(concat_fft)

        # SSVEPformer encoder
        # sub-encoder
        x = self.conv_attn(x)
        x = self.chnl_mlp(x)
        # # sub-encoder
        x = self.conv_attn(x)
        x = self.chnl_mlp(x)

        x = self.mlp_head(x, n_outputs=self.output_shape, output_mult=self.output_mult)

        output_layer = tf.keras.layers.Dense(self.output_shape, activation='softplus')(x)

        return output_layer
        
    def filter_bank(self, input):
        """
        """

        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            if low == 0:
                sos = butter(order, high, analog=False, btype='low', output='sos')
            else:
                sos = butter(order, [low, high], analog=False, btype='band', output='sos')
            return sos

        # @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
        def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
            sos = butter_bandpass(lowcut, highcut, fs, order=order)
            y = sosfilt(sos, data)
            return y

        def get_butter_bandpass_filter_func(lowcut, highcut, fs, order=5):
            sos = butter_bandpass(lowcut, highcut, fs, order=order)

            # @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
            def func(data):
                y = sosfilt(sos, data)
                return y

            return func

        
        sub_model_results = []

        # Filter banks
        for n in range(self.fb_n_filts):

            low_cut = self.fb_filt_step*n
            high_cut = self.fs//2 - 1

            # filtered_input = butter_bandpass_filter(input, low_cut, high_cut, self.fs, self.fb_order)

            filt_func = get_butter_bandpass_filter_func(low_cut, high_cut, self.fs, self.fb_order)
            
            filtered_input = tf.py_function(filt_func, [input], tf.float32)

            # filtered_input = tf.reshape(filtered_input, input.shape)

            sub_model_results.append(self.ssvep(filtered_input))
        
        # Stack the outputs
        x = tf.keras.layers.concatenate(sub_model_results)

        # Go to one channel
        x = tf.keras.layers.Conv1D(1,1, padding='same')(x)

        return x


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

        # Filter Bank

        output_layer = self.filter_bank(input_layer)
        
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

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

        # TensorBoard profiler
        # logs = f"{self.output_directory}/tfbp_logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        # tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
        #                                          histogram_freq = 1,
        #                                          profile_batch = '500,520')

        # The callbacks are stored as part of the object to be used in the fit function (not saved)
        self.callbacks = [reduce_lr, model_checkpoint_train, model_checkpoint_val, model_checkpoint_n_epoch] #, tboard_callback]

        print(model.summary())

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
        custom_objects = {}
        if not self.metrics is None:
            for metric in self.metrics:
                custom_objects[metric.__name__] = metric
        if not self.loss is None:
            custom_objects[self.loss.__name__] = self.loss
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)  

    def load_last_model(self):
        """
        Loads the weights from last_model.hdf5 and determines the metrics from object properties.
        Then returns the model.
        """
        model_path = self.output_directory + 'last_model.hdf5'
        custom_objects = {}
        if not self.metrics is None:
            for metric in self.metrics:
                custom_objects[metric.__name__] = metric
        if not self.loss is None:
            custom_objects[self.loss.__name__] = self.loss
        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects) 

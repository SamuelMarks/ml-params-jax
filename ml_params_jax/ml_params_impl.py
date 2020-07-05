""" Implementation of ml_params API """

# Mostly based off https://github.com/google/jax/blob/6aa8f24/examples/spmd_mnist_classifier_fromscratch.py
import time
from collections import deque
from os import path
from sys import stdout
from typing import Tuple

import numpy as np
import tensorflow as tf
from jax import tree_map
from ml_params.base import BaseTrainer
from ml_prepare.exectors import build_tfds_dataset

from ml_params_jax import get_logger
from ml_params_jax.datasets import load_data_from_jax_tfds_or_ml_prepare
from ml_params_jax.spmd_mnist_classifier_fromscratch import accuracy

logger = get_logger('.'.join((path.basename(path.dirname(__file__)),
                              path.basename(__file__).rpartition('.')[0])))


class JAXTrainer(BaseTrainer):
    """ Implementation of ml_params BaseTrainer for TensorFlow """

    data = None  # type: (None or Tuple[tf.data.Dataset, tf.data.Dataset] )
    model = None  # contains the model, e.g., a `tl.Serial`
    num_batches = None  # type: None or int
    num_devices = None  # type: None or int
    train_images = None
    train_labels = None
    test_images = None
    test_labels = None

    def __init__(self):
        super(JAXTrainer, self).__init__()

    def load_data(self, dataset_name, data_loader=load_data_from_jax_tfds_or_ml_prepare,
                  data_type='infer', output_type=None, K=None,
                  **data_loader_kwargs):
        """
        Load the data for your ML pipeline. Will be fed into `train`.

        :param dataset_name: name of dataset
        :type dataset_name: ```str```

        :param data_loader: function that returns the expected data type.
         Defaults to TensorFlow Datasets and ml_prepare combined one.
        :type data_loader: ```(*args, **kwargs) -> tf.data.Datasets or Any```

        :param data_type: incoming data type, defaults to 'infer'
        :type data_type: ```str```

        :param output_type: outgoing data_type, defaults to no conversion
        :type output_type: ```None or 'numpy'```

        :param K: backend engine, e.g., `np` or `tf`
        :type K: ```None or np or tf or Any```

        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```**data_loader_kwargs```

        :return: Dataset splits (by default, your train and test)
        :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
        """
        self.data = super(JAXTrainer, self).load_data(
            dataset_name=dataset_name,
            data_loader=data_loader,
            data_type=data_type,
            output_type=output_type,
            **data_loader_kwargs
        )
        if len(self.data) == 7:
            (self.data, self.num_batches, self.num_devices,
             self.train_images, self.train_labels,
             self.test_images, self.test_labels) = self.data

    def train(self,
              callbacks,
              epochs,
              loss,
              metrics=(('accuracy', accuracy),),
              metric_emit_freq=lambda step: True,
              optimizer=None,
              save_directory=None,
              output_type='infer',
              writer=stdout,
              layer_sizes=(784, 1024, 1024, 10),
              param_scale=0.1,
              step_size=0.001,
              batch_size=128,
              *args, **kwargs):
        """
        Run the training loop for your ML pipeline.

        :param callbacks: Collection of callables that are run inside the training loop
        :type callbacks: ```None or List[Callable] or Tuple[Callable]```

        :param epochs: number of epochs (must be greater than 0)
        :type epochs: ```int```

        :param loss: Loss function, can be a string (depending on the framework) or an instance of a class
        :type loss: ```str or Callable or Any```

        :param metrics: Collection of metrics to monitor, e.g., accuracy, f1
        :type metrics: ```None or List[Callable or str] or Tuple[Callable or str]```

        :param metric_emit_freq: Frequency of metric emission, e.g., `lambda: epochs % 10 == 0`, defaults to every epoch
        :type metric_emit_freq: ```None or (*args, **kwargs) -> bool```

        :param optimizer: Optimizer, can be a string (depending on the framework) or an instance of a class
        :type callbacks: ```str or Callable or Any```

        :param save_directory: Directory to save output in, e.g., weights in h5 files. If None, don't save.
        :type save_directory: ```None or str```

        :param output_type: `if save_directory is not None` then save in this format, e.g., 'h5'.
        :type output_type: ```str```

        :param writer: Writer for all output, could be a TensorBoard instance, a file handler like stdout or stderr
        :type writer: ```stdout or Any```

        :param layer_sizes:
        :type layer_sizes: ```(int,int,int,int)```

        :param param_scale:
        :type param_scale: ```float```

        :param step_size:
        :type step_size: ```float```

        :param batch_size:
        :type batch_size: ```int```

        :param \*args:
        :param \**kwargs:
        :return:
        """
        super(JAXTrainer, self).train(callbacks=callbacks,
                                      epochs=epochs,
                                      loss=loss,
                                      metrics=metrics,
                                      metric_emit_freq=metric_emit_freq,
                                      optimizer=optimizer,
                                      save_directory=save_directory,
                                      output_type='infer',
                                      writer=writer,
                                      *args, **kwargs)

        assert self.data is not None
        assert self.num_batches is not None
        assert self.num_devices is not None
        assert self.train_images is not None
        assert self.train_labels is not None
        assert self.test_images is not None
        assert self.test_labels is not None

        batches = self.data  # TODO: Maybe deepcopy?

        replicated_params, spmd_update = self.model(step_size, param_scale, layer_sizes, self.num_devices)

        for epoch in range(epochs):
            start_time = time.time()
            for _ in range(self.num_batches):
                replicated_params = spmd_update(replicated_params, next(batches))
            epoch_time = time.time() - start_time

            # We evaluate using the jitted `accuracy` function (not using pmap) by
            # grabbing just one of the replicated parameter values.
            params = tree_map(lambda x: x[0], replicated_params)

            if metric_emit_freq(epoch):
                def output_metric(met_name_met_fun):
                    met_name, met_fun = met_name_met_fun
                    train_met = met_fun(params, (self.train_images, self.train_labels))
                    test_met = met_fun(params, (self.test_images, self.test_labels))
                    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
                    print("Training set {}\t{}".format(met_name, train_met))
                    print("Test set {}\t\t{}".format(met_name, test_met))
                    return met_name, (train_met, test_met)

                deque(map(output_metric, metrics), maxlen=0)


del Tuple, build_tfds_dataset, get_logger

__all__ = ['JAXTrainer']

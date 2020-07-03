""" Implementation of ml_params API """

# Mostly based off https://github.com/google/jax/blob/6aa8f24/examples/spmd_mnist_classifier_fromscratch.py
import time
from os import path
from typing import Tuple

import numpy as np
import numpy.random as npr
import tensorflow as tf
from jax import tree_map
from jax.lib import xla_bridge
from ml_params.base import BaseTrainer
from ml_prepare.datasets import datasets2classes
from ml_prepare.exectors import build_tfds_dataset

import ml_params_jax.datasets
from ml_params_jax import get_logger
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

    def load_data(self, dataset_name, data_loader=None,
                  data_loader_kwargs=None, data_type='infer',
                  output_type=None, K=None):
        """
        Load the data for your ML pipeline. Will be fed into `train`.

        :param dataset_name: name of dataset
        :type dataset_name: ```str```

        :param data_loader: function that returns the expected data type.
         Defaults to TensorFlow Datasets and ml_prepare combined one.
        :type data_loader: ```None or (*args, **kwargs) -> tf.data.Datasets or Any```

        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```None or dict```

        :param data_type: incoming data type, defaults to 'infer'
        :type data_type: ```str```

        :param output_type: outgoing data_type, defaults to no conversion
        :type output_type: ```None or 'numpy'```

        :param K: backend engine, e.g., `np` or `tf`
        :type K: ```None or np or tf or Any```

        :return: Dataset splits (by default, your train and test)
        :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
        """
        self.data = super(JAXTrainer, self).load_data(
            dataset_name=dataset_name,
            data_loader=data_loader or self.load_data_from_jax_tfds_or_ml_prepare,
            data_loader_kwargs=data_loader_kwargs,
            data_type=data_type,
            output_type=output_type
        )

    def load_data_from_jax_tfds_or_ml_prepare(self, dataset_name, jax_datasets_dir=None,
                                              batch_size=128, data_loader_kwargs=None):
        """
        Acquire from the official TFDS model zoo through JAX wrapper, or the ophthalmology focussed ml-prepare library

        :param dataset_name: name of dataset
        :type dataset_name: ```str```

        :param jax_datasets_dir: directory to look for models in. Default is ~/jax_datasets.
        :type jax_datasets_dir: ```None or str```

        :param batch_size:
        :type batch_size: ```int```

        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```None or dict```

        :return: Train and tests dataset splits
        :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
        """
        data_loader_kwargs.update({
            'dataset_name': dataset_name,
            'tfds_dir': jax_datasets_dir,

        })
        if 'scale' not in data_loader_kwargs:
            data_loader_kwargs['scale'] = 255

        if dataset_name in datasets2classes:
            ds_builder = build_tfds_dataset(**data_loader_kwargs)

            if hasattr(ds_builder, 'download_and_prepare_kwargs'):
                download_and_prepare_kwargs = getattr(ds_builder, 'download_and_prepare_kwargs')
                delattr(ds_builder, 'download_and_prepare_kwargs')
            else:
                download_and_prepare_kwargs = None

            return BaseTrainer.common_dataset_handler(
                ds_builder=ds_builder,
                download_and_prepare_kwargs=download_and_prepare_kwargs,
                scale=None, K=None, as_numpy=False
            )
        else:
            ml_params_jax.datasets._DATA = jax_datasets_dir
            self.train_images, self.train_labels, self.test_images, self.test_labels = getattr(ml_params_jax.datasets,
                                                                                               dataset_name)()
            num_train = self.train_images.shape[0]
            num_complete_batches, leftover = divmod(num_train, batch_size)
            self.num_batches = num_complete_batches + bool(leftover)

            # For this manual SPMD example, we get the number of devices (e.g. GPUs or
            # TPU cores) that we're using, and use it to reshape data minibatches.
            self.num_devices = xla_bridge.device_count()

            def data_stream():
                rng = npr.RandomState(0)
                while True:
                    perm = rng.permutation(num_train)
                    for i in range(self.num_batches):
                        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                        images, labels = self.train_images[batch_idx], self.train_labels[batch_idx]
                        # For this SPMD example, we reshape the data batch dimension into two
                        # batch dimensions, one of which is mapped over parallel devices.
                        batch_size_per_device, ragged = divmod(images.shape[0], self.num_devices)
                        if ragged:
                            msg = "batch size must be divisible by device count, got {} and {}."
                            raise ValueError(msg.format(batch_size, self.num_devices))
                        shape_prefix = (self.num_devices, batch_size_per_device)
                        images = images.reshape(shape_prefix + images.shape[1:])
                        labels = labels.reshape(shape_prefix + labels.shape[1:])
                        yield images, labels

            return data_stream()

    def train(self, epochs, model_func=None, layer_sizes=(784, 1024, 1024, 10),
              param_scale=0.1, step_size=0.001, batch_size=128,
              *args, **kwargs):
        super(JAXTrainer, self).train(epochs=epochs, *args, **kwargs)

        assert self.data is not None
        assert self.num_batches is not None
        assert self.num_devices is not None
        assert self.train_images is not None
        assert self.train_labels is not None
        assert self.test_images is not None
        assert self.test_labels is not None

        batches = self.data  # TODO: Maybe deepcopy?

        replicated_params, spmd_update = model_func(step_size, param_scale, layer_sizes, self.num_devices)

        for epoch in range(epochs):
            start_time = time.time()
            for _ in range(self.num_batches):
                replicated_params = spmd_update(replicated_params, next(batches))
            epoch_time = time.time() - start_time

            # We evaluate using the jitted `accuracy` function (not using pmap) by
            # grabbing just one of the replicated parameter values.
            params = tree_map(lambda x: x[0], replicated_params)
            train_acc = accuracy(params, (self.train_images, self.train_labels))
            test_acc = accuracy(params, (self.test_images, self.test_labels))
            print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
            print("Training set accuracy\t{}".format(train_acc))
            print("Test set accuracy\t\t{}".format(test_acc))


del Tuple, build_tfds_dataset, get_logger

__all__ = ['JAXTrainer']

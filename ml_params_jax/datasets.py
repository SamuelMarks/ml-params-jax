import numpy.random as npr
from jax.lib import xla_bridge
from ml_params.datasets import load_data_from_ml_prepare
from ml_prepare.datasets import datasets2classes

import ml_params_jax.stolen.datasets


def load_data_from_jax_tfds_or_ml_prepare(dataset_name, tfds_dir=None,
                                          K=None, as_numpy=False, batch_size=128, **data_loader_kwargs):
    """
    Acquire from the official TFDS model zoo through JAX wrapper, or the ophthalmology focussed ml-prepare library

    :param dataset_name: name of dataset
    :type dataset_name: ```str```

    :param tfds_dir: directory to look for models in. Default is ~/tensorflow_datasets.
    :type tfds_dir: ```None or str```

    :param K: backend engine, e.g., `np` or `tf`
    :type K: ```None or np or tf or Any```

    :param as_numpy: Convert to numpy ndarrays
    :type as_numpy: ```bool```

    :param data_loader_kwargs: pass this as arguments to data_loader function
    :type data_loader_kwargs: ```**data_loader_kwargs```

    :return: Train and tests dataset splits
    :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
    """

    data_loader_kwargs.update({
        'dataset_name': dataset_name,
        'tfds_dir': tfds_dir,

    })
    if 'scale' not in data_loader_kwargs:
        data_loader_kwargs['scale'] = 255

    if dataset_name in datasets2classes:
        return load_data_from_ml_prepare(dataset_name=dataset_name,
                                         tfds_dir=tfds_dir,
                                         **data_loader_kwargs)
    else:
        ml_params_jax.stolen.datasets._DATA = tfds_dir
        train_images, train_labels, test_images, test_labels = getattr(ml_params_jax.stolen.datasets,
                                                                       dataset_name)()
        num_train = train_images.shape[0]
        num_complete_batches, leftover = divmod(num_train, batch_size)
        num_batches = num_complete_batches + bool(leftover)

        # For this manual SPMD example, we get the number of devices (e.g. GPUs or
        # TPU cores) that we're using, and use it to reshape data minibatches.
        num_devices = xla_bridge.device_count()

        def data_stream():
            rng = npr.RandomState(0)
            while True:
                perm = rng.permutation(num_train)
                for i in range(num_batches):
                    batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                    images, labels = train_images[batch_idx], train_labels[batch_idx]
                    # For this SPMD example, we reshape the data batch dimension into two
                    # batch dimensions, one of which is mapped over parallel devices.
                    batch_size_per_device, ragged = divmod(images.shape[0], num_devices)
                    if ragged:
                        msg = "batch size must be divisible by device count, got {} and {}."
                        raise ValueError(msg.format(batch_size, num_devices))
                    shape_prefix = (num_devices, batch_size_per_device)
                    images = images.reshape(shape_prefix + images.shape[1:])
                    labels = labels.reshape(shape_prefix + labels.shape[1:])
                    yield images, labels

        return data_stream(), num_batches, num_devices, train_images, train_labels, test_images, test_labels

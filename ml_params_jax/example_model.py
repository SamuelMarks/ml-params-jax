import numpy as np
import numpy.random as npr

from jax import jit, grad, pmap, partial
from jax.scipy.special import logsumexp
from jax.lib import xla_bridge
from jax.tree_util import tree_map
from jax import lax

from ml_params_jax.spmd_mnist_classifier_fromscratch import init_random_params, loss


def get_model(step_size, param_scale, layer_sizes, num_devices):
    @partial(pmap, axis_name='batch')
    def spmd_update(params, batch):
        grads = grad(loss)(params, batch)
        # We compute the total gradients, summing across the device-mapped axis,
        # using the `lax.psum` SPMD primitive, which does a fast all-reduce-sum.
        grads = [(lax.psum(dw, 'batch'), lax.psum(db, 'batch')) for dw, db in grads]
        return [(w - step_size * dw, b - step_size * db)
                for (w, b), (dw, db) in zip(params, grads)]

    # We replicate the parameters so that the constituent arrays have a leading
    # dimension of size equal to the number of devices we're pmapping over.
    init_params = init_random_params(param_scale, layer_sizes)
    replicate_array = lambda x: np.broadcast_to(x, (num_devices,) + x.shape)
    replicated_params = tree_map(replicate_array, init_params)
    return replicated_params, spmd_update


__all__ = ['get_model']

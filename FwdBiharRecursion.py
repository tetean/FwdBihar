"""
Author: NeterOster (neteroster@gmail.com)
Date: 2024/10/23
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ===================== Basic Functions =====================

@jax.jit
def d_tanh(x):
    return 1 - jnp.tanh(x) ** 2


@jax.jit
def dd_tanh(x):
    tanh_x = jnp.tanh(x)

    return -2 * tanh_x * (1 - tanh_x ** 2)


@jax.jit
def ddd_tanh(x):
    sech_x = 1 / jnp.cosh(x)

    return 4 * sech_x ** 2 - 6 * sech_x ** 4


@jax.jit
def dddd_tanh(x):
    sech_x = 1 / jnp.cosh(x)

    return jnp.tanh(x) * (-8 * sech_x ** 2 + 24 * sech_x ** 4)


@jax.jit
def lap_ldot(A, x, jac, lap):
    return A @ x, A @ jac, lap @ A.T


@jax.jit
def lap_rdot(x, A, jac, lap):
    return lap_ldot(A.T, x, jac, lap)


@jax.jit
def lap_tanh(x, jac, lap):
    return jnp.tanh(x), d_tanh(x)[:, None] * jac, lap * d_tanh(x) + jnp.sum(jac * jac, axis=1) * dd_tanh(x)


@jax.jit
def lap_add(x, v, jac, lap):
    return x + v, jac, lap


@jax.jit
def lap_addv(x, y, jac_x, lap_x, jac_y, lap_y):
    return x + y, jac_x + jac_y, lap_x + lap_y


@jax.jit
def lap_mul(x, y, jac_x, lap_x, jac_y, lap_y):
    return x * y, jac_x * y[:, None] + x[:, None] * jac_y, lap_x * y + lap_y * x + 2 * jnp.sum(jac_x * jac_y, axis=1)


##########################

@jax.jit
def bihar_dot(A, x, jac, lap, lap_jac, bihar, hessian):
    new_lap, new_lap_jac, new_bihar = lap_rdot(lap, A.T, lap_jac, bihar)

    new_hessian = jnp.tensordot(A, hessian, axes=1)

    return A @ x, A @ jac, new_lap, new_lap_jac, new_bihar, new_hessian


@jax.jit
def bihar_tanh(x, jac, lap, lap_jac, bihar, hessian):
    d_tanh_val = d_tanh(x)
    dd_tanh_val = dd_tanh(x)
    ddd_tanh_val = ddd_tanh(x)
    dddd_tanh_val = dddd_tanh(x)
    jjsum_val = jnp.sum(jac * jac, axis=1)

    new_lap, new_lap_jac, new_bihar = lap_mul(lap, d_tanh_val, lap_jac, bihar, dd_tanh_val[:, None] * jac,
                                              dd_tanh_val * lap + jjsum_val * ddd_tanh_val)

    dd_tanh_jac = ddd_tanh_val[:, None] * jac
    dd_tanh_lap = ddd_tanh_val * lap + jjsum_val * dddd_tanh_val

    jjsum_jac = 2 * jnp.einsum('bi,bij->bj', jac, hessian)
    jjsum_lap = 2 * jnp.sum(hessian ** 2, axis=(1, 2)) + 2 * jnp.sum(jac * lap_jac, axis=1)

    s, s_jac, s_lap = lap_mul(jjsum_val, dd_tanh_val, jjsum_jac, jjsum_lap, dd_tanh_jac, dd_tanh_lap)

    new_lap, new_lap_jac, new_bihar = lap_addv(new_lap, s, new_lap_jac, new_bihar, s_jac, s_lap)

    new_hessian = jnp.einsum('bi,bj->bij', jac, jac) * dd_tanh_val[:, None, None] \
                  + hessian * d_tanh_val[:, None, None]

    return jnp.tanh(x), d_tanh_val[:, None] * jac, new_lap, new_lap_jac, new_bihar, new_hessian


@jax.jit
def bihar_add(x, v, jac, lap, lap_jac, bihar, hessian):
    return x + v, jac, lap, lap_jac, bihar, hessian


# =================== END Basic Functions ===================

def get_bihar_func(func):
    laplacian_fn = lambda x: jnp.trace(jax.hessian(func)(x), axis1=-1, axis2=-2)[0]
    laplacian_grad_fn = lambda x: jax.grad(laplacian_fn)(x)  # for debug only
    bihar_fn = lambda x: jnp.trace(jax.hessian(laplacian_fn)(x), axis1=-1, axis2=-2)

    return bihar_fn


def get_hessian(func):
    return jax.hessian(func)


@jax.jit
def MLP(params, x):
    for layer in params[:-1]:
        W = layer['W']
        b = layer['B']
        x = jnp.tanh(jnp.dot(W, x) + b)

    return jnp.dot(params[-1][0], x) + params[-1][1]


def init_params(layers):
    keys = jax.random.split(jax.random.PRNGKey(0), len(layers) - 1)
    params = []
    for key, n_in, n_out in zip(keys, layers[:-1], layers[1:]):
        lb, ub = -(1 / jnp.sqrt(n_in)), (1 / jnp.sqrt(n_in))
        W = lb + (ub - lb) * jax.random.uniform(key, shape=(n_out, n_in))
        B = jax.random.uniform(key, shape=(n_out,))
        params.append((W, B))
    return params


def forward_biharmonic(params, x):
    jac = jnp.eye(x.shape[0])
    lap = jnp.zeros_like(x)
    lap_jac = jnp.zeros_like(jac)
    bihar = jnp.zeros_like(lap)
    hess = jnp.zeros((x.shape[0], x.shape[0], x.shape[0]))

    for layer in params[:-1]:
        W = layer['W']
        b = layer['B']
        z, jac_z, lap_z, lap_jac_z, bihar_z, hess_z = bihar_dot(W, x, jac, lap, lap_jac, bihar, hess)
        t, jac_t, lap_t, lap_jac_t, bihar_t, hess_t = bihar_add(z, b, jac_z, lap_z, lap_jac_z, bihar_z, hess_z)
        a, jac_a, lap_a, lap_jac_a, bihar_a, hess_a = bihar_tanh(t, jac_t, lap_t, lap_jac_t, bihar_t, hess_t)

        x, jac, lap, lap_jac, bihar, hess = a, jac_a, lap_a, lap_jac_a, bihar_a, hess_a

    W = params[-1]['W']
    b = params[-1]['B']

    z, jac_z, lap_z, lap_jac_z, bihar_z, hess_z = bihar_dot(W, x, jac, lap, lap_jac, bihar, hess)
    a, jac_a, lap_a, lap_jac_a, bihar_a, hess_a = bihar_add(z, b, jac_z, lap_z, lap_jac_z, bihar_z, hess_z)

    x, jac, lap, lap_jac, bihar, hess = a, jac_a, lap_a, lap_jac_a, bihar_a, hess_a

    return bihar.squeeze()


# forward_biharmonic_vmap = jax.jit(jax.vmap(forward_biharmonic, in_axes=(None, 0)))
# bihar_fn = get_bihar_func(lambda x: MLP(params, x))
# bihar_fn_vmap = jax.jit(jax.vmap(bihar_fn))
#
# params = init_params([32, 32, 32, 1])
# x = jax.random.uniform(jax.random.PRNGKey(0), shape=(500, 32))
#
# import timeit
#
# xx = bihar_fn_vmap(x)  # warm up
#
# timer = timeit.Timer(lambda: bihar_fn_vmap(x))
#
# n, t = timer.autorange()
#
# print("JAX Autodiff Bihar: ")
# print(t / n)
#
# yy = forward_biharmonic_vmap(params, x)  # warm up
#
# timer = timeit.Timer(lambda: forward_biharmonic_vmap(params, x))
#
# n, t = timer.autorange()
#
# print("Forward Bihar: ")
# print(t / n)
#
# print("Correctness: ")
# print(jnp.allclose(xx, yy))
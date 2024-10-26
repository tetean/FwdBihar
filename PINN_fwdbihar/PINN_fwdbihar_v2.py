'''
@ Author: tetean
@ Create time: 2024/10/24 10:30
@ Info:
'''
import jax
import jax.numpy as jnp
from jax import jit, vmap
import optax
from util import TestModel
import time
import FwdBiharRecursion
from util.Conf import write_res, load_config, merge_dicts
import os, pickle


@jax.jit
def MLP(params, x):
    """多层感知机函数"""
    for layer in params[:-1]:
        x = jnp.tanh(jnp.dot(layer['W'], x) + layer['B'])
    return jnp.dot(params[-1]['W'], x) + params[-1]['B']

def init_params(layers):
    """初始化网络参数"""
    keys = jax.random.split(jax.random.PRNGKey(0), len(layers) - 1)
    params = []
    for key, n_in, n_out in zip(keys, layers[:-1], layers[1:]):
        lb, ub = -(1 / jnp.sqrt(n_in)), (1 / jnp.sqrt(n_in))
        W = lb + (ub - lb) * jax.random.uniform(key, shape=(n_out, n_in))
        B = jax.random.uniform(key, shape=(n_out,))
        params.append({'W': W, 'B': B})
    return params


def get_bihar_func(func, params):
    @jax.jit
    def bihar(data):
        value = func(params, data)
        return value
    return bihar


def generate_data(num_interior, num_boundary, d):
    """生成训练或测试数据"""
    interior_points = jax.random.uniform(jax.random.PRNGKey(0), (num_interior, d), minval=-1, maxval=1)

    boundary_points = []
    for i in range(d):
        for val in [-1, 1]:
            points = jax.random.uniform(jax.random.PRNGKey(i), (num_boundary // (2 * d), d), minval=-1, maxval=1)
            points = points.at[:, i].set(val)
            boundary_points.append(points)

    boundary_points = jnp.concatenate(boundary_points, axis=0)
    return interior_points, boundary_points

@jit
def f(x, d):
    """源项函数"""
    return -1 / d ** 2 * jnp.sin(jnp.sum(x, axis=-1) / d)

@jit
def h(x, d):
    """边界条件函数"""
    return (jnp.sum(x, axis=-1) / d) ** 2 + jnp.sin(jnp.sum(x, axis=-1) / d)

@jit
def simple_layer(x, A, b):
    return jnp.tanh(A @ x + b)



@jit
def loss_fn(params, interior_points, boundary_points, d):
    """损失函数"""

    def bihar_u_net(x):
        return FwdBiharRecursion.forward_biharmonic(params, x)

    def u_net(x):
        return MLP(params, x)

    vmap_bihar = bihar_u_net
    vmap_bihar = jit(vmap(vmap_bihar))

    vmap_u_net = jit(vmap(u_net))
    # t2 = time.time()
    interior_loss = jnp.mean((
                                -vmap_bihar(interior_points).reshape(-1)
                              - f(interior_points, d)
                             ) ** 2)
    jax.block_until_ready(interior_loss)
    # d2 = time.time() - t2
    # print(f'd2: {d2}')
    boundary_loss = jnp.mean((vmap_u_net(boundary_points).reshape(-1)
                              - h(boundary_points, d)
                              ) ** 2)
    return interior_loss + boundary_loss


def train(layers, num_epochs, learning_rate, interior_points, boundary_points, d):
    """训练PINN"""
    params = init_params(layers)
    optimizer = optax.adam(learning_rate)  # 使用 optax 的 adam
    opt_state = optimizer.init(params)

    loss_stk = []



    for epoch in range(num_epochs):
        # 计算损失和梯度
        # t1 = time.time()
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, interior_points, boundary_points, d))(params)

        # jax.block_until_ready(grads)
        # d1 = time.time() - t1
        # print(f'd1: {d1}')
        # print(f'grads: {grads}')
        # 更新参数
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        loss_stk.append(loss)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    return params, loss_stk

# 文件管理
conf = load_config()
res_dir = './result/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
total_files = len([file for file in os.listdir(res_dir)])
hibar_res_dir = os.path.join(res_dir, f'{total_files + 1}_fwdbihar')
os.makedirs(hibar_res_dir)
res_file = os.path.join(hibar_res_dir, 'result.yaml')

for exp_name, exp in conf.items():
    layers = exp['layers']  # 输入层和输出层维度
    num_epochs = exp['num_epochs']
    learning_rate = exp['learning_rate']

    # 生成训练数据
    num_interior = exp['num_interior']
    num_boundary = exp['num_boundary']
    test_num_interior = exp['test_num_interior']
    test_num_boundary = exp['test_num_boundary']
    d = layers[0]  # 维度参数
    interior_points, boundary_points = generate_data(num_interior, num_boundary, d)

    # 开始训练
    start_time = time.time()

    trained_params, loss_stk = train(layers, num_epochs, learning_rate, interior_points, boundary_points, d)

    jax.block_until_ready(trained_params)
    duration = time.time() - start_time

    # 误差检验
    # interior_error, boundary_error = TestModel.test_model(trained_params,  d, test_num_interior, test_num_boundary,'relative', 'L2', MLP)

    l2_rel_in, l2_rel_b = TestModel.test_model(
        trained_params,  d, test_num_interior, test_num_boundary,'relative', 'L2', MLP
    )
    l2_abs_in, l2_abs_b = TestModel.test_model(
        trained_params,  d, test_num_interior, test_num_boundary,'absolute', 'L2', MLP
    )
    linf_rel_in, linf_rel_b = TestModel.test_model(
        trained_params,  d, test_num_interior, test_num_boundary,'relative', 'Linf', MLP
    )
    linf_abs_in, linf_abs_b = TestModel.test_model(
        trained_params,  d, test_num_interior, test_num_boundary,'absolute', 'Linf', MLP
    )

    conf[exp_name]['fwdbihar'] = {
            'l2_error':
            {
                'interior': {'relative': l2_rel_in, 'absolute': l2_abs_in},
                'boundary': {'relative': l2_rel_b, 'absolute': l2_abs_b}
            },
            'linf_error':
            {
                'interior': {'relative': linf_rel_in, 'absolute': linf_abs_in},
                'boundary': {'relative': linf_rel_b, 'absolute': linf_abs_b}
            },
            'duration_seconds': duration
        }

    # print(f'Interior Error: {interior_error}')
    # print(f'Boundary Error: {boundary_error}')
    print(f'Duration: {duration}')
    write_res(conf, res_file)
    with open(os.path.join(hibar_res_dir, f'loss_fwdbihar_{exp_name}.pkl'), 'wb') as file:
        pickle.dump(loss_stk, file)

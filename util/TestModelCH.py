'''
@ Author: tetean
@ Create time: 2024/10/31 11:29
@ Info:
'''
import scipy.io
import jax.numpy as jnp
import jax
from joblib import load
import matplotlib.pyplot as plt
import sys
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), '../PINN_fwdbihar/Cahn-Hilliard-Eq'))

# 读取 .mat 文件
__data = scipy.io.loadmat('ch2dn41.mat')

__params = load('params.joblib')

variable = __data['test_data']
true_val = __data['test_u']

# print(variable, true_val)
def __MLP(params, x):
    """多层感知机函数"""
    for layer in params[:-1]:
        x = jnp.tanh(jnp.dot(layer['W'], x) + layer['B'])
    return jnp.dot(params[-1]['W'], x) + params[-1]['B']


def calculate_error(error_type, norm_type, predicted_values, true_values):
    """
    计算误差

    参数:
    params: 网络参数
    test_points: 测试点
    d: 维度
    error_type: 误差类型 ('relative' 或 'absolute')
    norm_type: 范数类型 ('L2' 或 'Linf')

    返回:
    计算得到的误差
    """
    if norm_type == 'L2':
        if error_type == 'relative':
            error = jnp.linalg.norm(true_values - predicted_values) / jnp.linalg.norm(true_values)
        else:  # absolute
            error = jnp.linalg.norm(true_values - predicted_values)
    elif norm_type == 'Linf':
        if error_type == 'relative':
            error = jnp.max(jnp.abs(true_values - predicted_values) / jnp.abs(true_values))
        else:  # absolute
            error = jnp.max(jnp.abs(true_values - predicted_values))
    else:
        raise ValueError("Unsupported norm type. Choose 'L2' or 'Linf'.")

    return float(error)

def test_model(params, MLP):
    vmap_MLP = jax.jit(jax.vmap(MLP, in_axes=(None, 0)))

    u_pred = vmap_MLP(params, variable).reshape(-1)
    u_true = true_val.reshape(-1)

    error = {}
    error['relative_L2'] = calculate_error('relative', 'L2', u_pred, u_true)
    error['absolute_L2'] = calculate_error('absolute', 'L2', u_pred, u_true)
    error['relative_Linf'] = calculate_error('relative', 'Linf', u_pred, u_true)
    error['absolute_Linf'] = calculate_error('absolute', 'Linf', u_pred, u_true)

    print(f'relative_L2: ', error['relative_L2'],
          f'absolute_L2: ',error['absolute_L2'],
          f'relative_Linf: ',error['relative_Linf'],
          f'absolute_Linf: ',error['absolute_Linf'])

    # # 画出结果
    # t_test = 0.05
    # u_pred_plot = vmap_MLP(params, )
    # plt.figure(figsize=(8, 6))
    # plt.contourf(variable[:, 0].reshape(-1), variable[:, 1].reshape(-1), t_test * jnp.ones(X_test.size), levels=50, cmap='viridis')
    # plt.colorbar()
    # plt.title(f'Solution at t = {t_test}')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

test_model(__params, __MLP)
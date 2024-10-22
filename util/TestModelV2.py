'''
@ Author: tetean
@ Time: 2024/10/22 14:29
@ Info: 
'''
import torch
def u(x, d):
    """精确解函数"""
    s = torch.sum(x, dim=-1) / d
    return (s) ** 2 + torch.sin(s)


def generate_data(num_interior, num_boundary, d):
    """生成训练或测试数据"""
    interior_points = torch.empty(num_interior, d).uniform_(-1, 1)

    boundary_points = []
    for i in range(d):
        for val in [-1, 1]:
            points = torch.empty(num_boundary // (2 * d), d).uniform_(-1, 1)
            points[:, i] = val
            boundary_points.append(points)

    boundary_points = torch.cat(boundary_points, dim=0)
    return interior_points, boundary_points

def calculate_error(params, test_points, d, error_type, norm_type, model):
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

    true_values = u(test_points, d)

    vmap_u_net = jax.jit(jax.vmap(model, in_axes=(None, 0)))

    predicted_values = vmap_u_net(params, test_points).reshape(-1)

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



def test_model(trained_params, d, test_interior, test_boundary, error_type, norm_type, model):
    """
    测试模型

    参数:
    trained_params: 训练后的网络参数
    d: 维度
    test_interior: 测试用内部点数量
    test_boundary: 测试用边界点数量
    error_type: 误差类型
    norm_type: 范数类型

    返回:
    interior_error: 内部点误差
    boundary_error: 边界点误差
    """
    test_in, test_b = generate_data(test_interior, test_boundary, d)

    interior_error = calculate_error(trained_params, test_in, d, error_type, norm_type, model)
    boundary_error = calculate_error(trained_params, test_b, d, error_type, norm_type, model)

    return interior_error, boundary_error
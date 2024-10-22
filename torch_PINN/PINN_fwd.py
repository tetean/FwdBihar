import torch
import torch.optim as optim
import time

from sympy import hessian
from torch.autograd import functional as F
import torch.nn as nn

import FwdBihar
import jax.numpy as jnp
import jax

def init_params(layers):
    """初始化网络参数"""
    params = []
    for n_in, n_out in zip(layers[:-1], layers[1:]):
        W = nn.Parameter(torch.empty(n_out, n_in).to(device).uniform_(-1 / (n_in ** 0.5), 1 / (n_in ** 0.5))).to(device)
        B = nn.Parameter(torch.empty(n_out).to(device).uniform_(-1, 1)).to(device)
        params.append({'W': W, 'B': B})  # 确保参数需要梯度并移动到设备
    # print(params)
    return params



def MLP(params, x):
    """多层感知机函数，支持多批次输入"""
    for layer in params[:-1]:
        x = torch.tanh(torch.matmul(x, layer['W'].T) + layer['B'])
    return torch.matmul(x, params[-1]['W'].T) + params[-1]['B']


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
    # 启用梯度跟踪
    # interior_points.requires_grad = True
    # boundary_points.requires_grad = True
    return interior_points, boundary_points


def f(x, d):
    """源项函数"""
    return -1 / d ** 2 * torch.sin(torch.sum(x, dim=-1) / d)


def h(x, d):
    """边界条件函数"""
    return (torch.sum(x, dim=-1) / d) ** 2 + torch.sin(torch.sum(x, dim=-1) / d)


def hessian_fn(params, x):
    """计算 Hessian 矩阵，输出形状为 (batch_size, output_size, input_dim, input_dim)"""
    batch_size = x.shape[0]
    output_size = MLP(params, x).shape[1]
    input_dim = x.shape[1]

    # 创建一个用于存放 Hessian 矩阵的张量
    hessian_matrices = torch.zeros(batch_size, output_size, input_dim, input_dim)

    for i in range(batch_size):
        def output_fn(x_i):
            return MLP(params, x_i.unsqueeze(0)).squeeze(0)

        # 对每个样本 x[i] 计算 Hessian
        hessian_matrices[i] = F.hessian(output_fn, x[i])

    return hessian_matrices


def fourth_derivative(params, x):
    """
    计算 y = MLP(params, x) 对 x 的四阶导数矩阵，
    输出形状为 (batch_size, output_size, input_dim, input_dim, input_dim, input_dim)
    """
    # 确保 x 具有需要计算梯度的功能
    x.requires_grad_(True)

    # 第一步：计算 y
    y = MLP(params, x)  # y 的形状为 (batch_size, output_size)
    batch_size, output_size = y.shape
    input_dim = x.shape[1]

    # 初始化存储四阶导数的张量
    fourth_derivs = torch.zeros(batch_size, output_size, input_dim, input_dim, input_dim, input_dim, dtype=x.dtype)

    # 逐批次和逐输出元素计算四阶导数
    for b in range(batch_size):
        for o in range(output_size):
            # 计算 y[b, o] 对 x 的一阶导数
            grad_1 = torch.autograd.grad(y[b, o], x, create_graph=True, retain_graph=True)[0]

            for i in range(input_dim):
                # 计算一阶导数 grad_1[:, i] 对 x 的二阶导数
                grad_2 = torch.autograd.grad(grad_1[b, i], x, create_graph=True, retain_graph=True)[0]

                for j in range(input_dim):
                    # 计算二阶导数 grad_2[:, j] 对 x 的三阶导数
                    grad_3 = torch.autograd.grad(grad_2[b, j], x, create_graph=True, retain_graph=True)[0]

                    for k in range(input_dim):
                        # 计算三阶导数 grad_3[:, k] 对 x 的四阶导数
                        grad_4 = torch.autograd.grad(grad_3[b, k], x, create_graph=True, retain_graph=True)[0]

                        # 保存四阶导数值
                        fourth_derivs[b, o, i, j, k, :] = grad_4[b]

    return fourth_derivs


def get_bihar(x):
    # x.shape = (batch_size, out_dim, in_dim, in_dim, in_dim, in_dim)
    # y.shape = (batch_size, out_dim, in_dim, in_dim)

    # 使用高级索引来直接选择相应元素
    # 选择 x[:, :, :, i, :, i]，其中 i 对应维度 k
    indices = torch.arange(x.shape[2])  # 创建一个 [0, 1, ..., in_dim-1] 的索引向量
    y = x[:, :, indices, indices, :, indices]  # 使用这个索引从 x 中提取对应的元素

    # 在最后两个维度上求和
    y = torch.sum(y, dim=(2, 3))

    return y


def loss_fn(params, interior_points, boundary_points, d):
    """损失函数"""

    d1 = time.time()

    # 将 PyTorch 张量转换为 NumPy 数组
    numpy_array = interior_points.numpy()

    # 将 NumPy 数组转换为 JAX 数组
    interior_points = jnp.array(numpy_array)

    for layers in params:
        layers['W'] = layers['W'].numpy()
        layers['W'] = jnp.array(layers['W'])
        layers['B'] = layers['B'].numpy()
        layers['B'] = jnp.array(layers['B'])
        print(layers['W'])

    def u_net(x):
        return FwdBihar.MLP(params, x)

    vmap_u_net = jax.jit(jax.vmap(u_net))
    bihar_u = vmap_u_net(interior_points)

    d1 = time.time() - d1
    print(f'bihar time: {d1}')


    bihar_u = get_bihar(bihar_u).to(device)
    # print(bihar_u.shape)
    # print(f(interior_points, d).shape)
    # print(interior_points.shape)
    interior_loss = torch.mean((bihar_u - f(interior_points, d)) ** 2)

    boundary_loss = torch.mean((MLP(params, boundary_points) - h(boundary_points, d)) ** 2)

    return interior_loss + boundary_loss


def train(layers, num_epochs, learning_rate, interior_points, boundary_points, d, batch_size):
    """训练PINN，支持多批次输入"""
    params = init_params(layers)
    # for layer in params:
    #     layer['W'] = layer['W'].to(device)
    #     layer['B'] = layer['B'].to(device)

    optimizer = optim.Adam([{'params': [layer['W'], layer['B']]} for layer in params], lr=learning_rate)

    start_time = time.time()
    for epoch in range(num_epochs):


        # Shuffle the data
        perm = torch.randperm(interior_points.size(0))
        interior_points = interior_points[perm]
        boundary_points = boundary_points[perm % boundary_points.size(0)]

        for i in range(0, interior_points.size(0), batch_size):
            # Prepare batch
            interior_batch = interior_points[i:i + batch_size]
            boundary_batch = boundary_points[i:i + batch_size]

            optimizer.zero_grad()

            d2 = time.time()
            loss = loss_fn(params, interior_batch, boundary_batch, d)

            d2 = time.time() - d2
            print(f'd2: {d2}')

            d3 = time.time()
            loss.backward()
            d3 = time.time() - d3
            print(f'd3: {d3}')
            optimizer.step()


        if epoch % 1 == 0:
            duration = time.time() - start_time
            start_time = time.time()
            print(f'Epoch {epoch}, Loss: {loss.item()}, Duration: {duration:.4f} seconds')

    return params

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
print(f'[info] Now, training on {device}')
# 参数设置
layers = [2, 8, 8, 1]  # 输入层和输出层维度
num_epochs = 2000
learning_rate = 0.001
batch_size = 32  # 设置批次大小

# 生成训练数据
num_interior = 1000
num_boundary = 1000
d = layers[0]  # 维度参数
interior_points, boundary_points = generate_data(num_interior, num_boundary, d)


# 将数据移动到GPU
interior_points = interior_points.to(device)
boundary_points = boundary_points.to(device)

# 开始训练
start_time = time.time()
trained_params = train(layers, num_epochs, learning_rate, interior_points, boundary_points, d, batch_size)

# 计算训练时长
duration = time.time() - start_time

# 误差检验
# interior_error, boundary_error = TestModel.test_model(trained_params, d, test_num_interior, test_num_boundary, 'relative', 'L2', MLP)
#
# print(f'Interior Error: {interior_error}')
# print(f'Boundary Error: {boundary_error}')
# print(f'Duration: {duration}')
# # 帮我定义一个函数，使用torch求y对x的四阶导数矩阵，
# def fourth_derivative(params, x):
#     """计算四阶导数矩阵，输出形状为 (batch_size, output_size, input_dim, input_dim, input_dim, input_dim)"""
#     y = MLP(params, x)
#     batch_size = x.shape[0]
#     output_size = y.shape[1]
#     input_dim = x.shape[1]
# def MLP(params, x):
#     """多层感知机函数，支持多批次输入"""
#     for layer in params[:-1]:
#         x = torch.tanh(torch.matmul(x, layer['W'].T) + layer['B'])
#     return torch.matmul(x, params[-1]['W'].T) + params[-1]['B']
'''
@ Author: tetean
@ Create time: 2024/10/27 22:09
@ Info:
'''
import jax
import jax.numpy as jnp
import jax.random as random
import optax
import matplotlib.pyplot as plt

import os
from joblib import dump
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"




# 设置超参数
T = 1  # 最大时间
num_internal_points = 10000  # 内部点数
num_boundary_points = 800  # 边界点数
num_initial_points = 10000  # 初始点数
epsilon = 0.05  # 小参数
boundary_loss_weight = 1000  # 边界损失权重

# 初始条件
# def u0(x, y):
#     return jnp.tanh(1 / jnp.sqrt(2 * epsilon) *
#                      jnp.minimum(jnp.sqrt((x + 0.3)**2 + y**2) - 0.3,
#                                  jnp.sqrt((x - 0.3)**2 + y**2) - 0.25))

def u0(x, y):
    r = 0.4
    R1 = jnp.sqrt((x - 0.7 * r) ** 2 + y ** 2)
    R2 = jnp.sqrt((x + 0.7 * r) ** 2 + y ** 2)
    return jnp.maximum(jnp.tanh((r - R1) / (2 * epsilon)),
                       jnp.tanh((r - R2) / (2 * epsilon)))

# MLP定义
@jax.jit
def MLP(params, x):
    """多层感知机函数"""
    for layer in params[:-1]:
        x = jnp.tanh(jnp.dot(layer['W'], x) + layer['B'])
    return jnp.dot(params[-1]['W'], x) + params[-1]['B']

def init_params(layers):
    """初始化网络参数"""
    keys = random.split(random.PRNGKey(0), len(layers) - 1)
    params = []
    for key, n_in, n_out in zip(keys, layers[:-1], layers[1:]):
        lb, ub = -(1 / jnp.sqrt(n_in)), (1 / jnp.sqrt(n_in))
        W = lb + (ub - lb) * random.uniform(key, shape=(n_out, n_in))
        B = random.uniform(key, shape=(n_out,))
        params.append({'W': W, 'B': B})
    return params

def get_bihar_func(func):
    @jax.jit
    def condense_B(x):
        # x的形状是(m, n, n, n, n)
        # 取出x[:, j, j, k, k]，并将结果存储到y
        y = x[:, jnp.arange(x.shape[1]), jnp.arange(x.shape[1]), :, :]
        y = y[:, :, jnp.arange(x.shape[1]), jnp.arange(x.shape[1])]
        return y

    def bihar(data):
        B = jax.hessian(jax.hessian(func))(data)
        B = condense_B(B)
        B = B[:, :-1, :-1]
        return jnp.sum(B, axis=(1, 2))

    return bihar


def get_lap_func(func):
    def lap(data):
        hess = jax.hessian(func)(data)
        return jnp.trace(hess, axis1=-1, axis2=-2)
    return lap

def get_u_func(func):
    def u(data):
        return func(data)
    return u


def get_g_func(func):
    # func: MLP
    # data: x
    @jax.jit
    def g(data):
        hess = jax.hessian(func)(data)
        lap_u = jnp.trace(hess, axis1=-1, axis2=-2)
        u = func(data)
        return -epsilon ** 2 * lap_u + u ** 3 - u
    return g

# 边界损失和初始损失计算
def boundary_initial_loss(params, boundary_points, initial_points):
    # # Neumann 边界条件，边界损失
    # grad_u = jax.jacobian(MLP, argnums=1)
    # vmap_grad_u = jax.jit(jax.vmap(grad_u, in_axes=(None, 0)))
    # grad_u = vmap_grad_u(params, boundary_points)[:, :, :2]  # 只保留 x 和 y 方向的导数
    #
    # boundary_loss = jnp.mean(jnp.square(grad_u))  # 假设边界法向量为标准单位向量，直接计算梯度平方和

    grad_u = jax.jacobian(MLP, argnums=1)
    vmap_grad_u = jax.jit(jax.vmap(grad_u, in_axes=(None, 0)))
    grad_u = vmap_grad_u(params, boundary_points)[:, :, :2]

    # 定义单位法向量
    nx = jnp.where(boundary_points[:, 0] == -1, -1, jnp.where(boundary_points[:, 0] == 1, 1, 0))
    ny = jnp.where(boundary_points[:, 1] == -1, -1, jnp.where(boundary_points[:, 1] == 1, 1, 0))
    normal_vector = jnp.stack([nx, ny], axis=-1)

    # 计算法向导数：grad_u 和 normal_vector 的点积
    normal_grad_u = jnp.sum(grad_u * normal_vector[:, None, :], axis=-1)

    # 边界损失
    boundary_loss = jnp.mean(jnp.square(normal_grad_u))


    g = get_g_func(lambda x: MLP(params, x))

    grad_g = jax.jacobian(g)
    vmap_grad_g = jax.jit(jax.vmap(grad_g))
    grad_g = vmap_grad_g(boundary_points)[:, :, 2]

    normal_grad_g = jnp.sum(grad_g * normal_vector[:, None, :], axis=-1)

    boundary_loss += jnp.mean(jnp.square(normal_grad_g))

    # # 定义 u 在 x 和 y 方向的二阶导数
    # hess_u = jax.hessian(MLP, argnums=1)
    # vmap_hess_u = jax.jit(jax.vmap(hess_u, in_axes=(None, 0)))
    # hess_u = vmap_hess_u(params, boundary_points)
    #
    # # 提取二阶导数
    # u_xx = hess_u[:, 0, 0]  # 对 x 的二阶导
    # u_yy = hess_u[:, 1, 1]  # 对 y 的二阶导
    # lap_u = u_xx + u_yy  # 拉普拉斯
    #
    # # 计算 g(x, y) = -ε²Δu + u³ - u
    # u_values = jax.vmap(lambda x: MLP(params, x))(boundary_points)
    # g_boundary = -epsilon ** 2 * lap_u + u_values ** 3 - u_values
    #
    # # 计算 g 在 x 和 y 方向的梯度
    # grad_g = jax.jacobian(lambda x: -epsilon ** 2 * lap_u + MLP(params, x) ** 3 - MLP(params, x))
    # vmap_grad_g = jax.jit(jax.vmap(grad_g))
    # grad_g_boundary = vmap_grad_g(boundary_points)[:, :, :2]  # 只保留 x 和 y 的梯度
    #
    # # 假设法向量为单位向量，计算 grad_g 在法向方向的平方和
    # boundary_loss += jnp.mean(jnp.square(grad_g_boundary))



    # 初始损失
    u_init_pred = jax.vmap(MLP, in_axes=(None, 0))(params, initial_points)
    u_init_true = u0(initial_points[:, 0], initial_points[:, 1])
    initial_loss = jnp.mean(jnp.square(u_init_pred - u_init_true))

    return boundary_loss, initial_loss


# 定义损失函数
def loss_fn(params, internal_points, boundary_points, initial_points):
    # 计算偏导数
    hess_u = jax.hessian(MLP, argnums=1)
    vmap_hess_u = jax.jit(jax.vmap(hess_u, in_axes=(None, 0)))

    hess_u = vmap_hess_u(params, internal_points)  # (points_num, 1, d + 1, d + 1)
    u_xx = hess_u[:, :, 0, 0].reshape(-1)
    u_yy = hess_u[:, :, 1, 1].reshape(-1)
    lap_u = u_xx + u_yy

    def u3_net(x):
        return MLP(params, x) ** 3

    hess_u3 = jax.hessian(u3_net)
    vmap_hess_u3 = jax.jit(jax.vmap(hess_u3))
    hess_u3 = vmap_hess_u3(internal_points)
    u3_xx = hess_u3[:, :, 0, 0].reshape(-1)
    u3_yy = hess_u3[:, :, 1, 1].reshape(-1)
    lap_u3 = u3_xx + u3_yy

    bihar_u = get_bihar_func(lambda x: MLP(params, x))
    vmap_bihar_u = jax.jit(jax.vmap(bihar_u))
    bihar_u = vmap_bihar_u(internal_points).reshape(-1)

    # batch_size = 36  # 分批大小
    # num_batches = internal_points.shape[0] // batch_size
    #
    # bihar_u = []
    # for i in range(num_batches):
    #     batch_points = internal_points[i * batch_size: (i + 1) * batch_size]
    #     bihar_u_batch = vmap_bihar_u(batch_points).reshape(-1)
    #     bihar_u.append(bihar_u_batch)


    jac = jax.jacobian(MLP, argnums=1)
    vmap_jac = jax.jit(jax.vmap(jac, in_axes=(None, 0)))
    jac = vmap_jac(params, internal_points)  # (points_num, 1, d + 1)
    u_t = jac[:, :, 2].reshape(-1)
    # u_x = jac[:, :, 0].reshape(-1)
    # u_y = jac[:, :, 1].reshape(-1)

    # 计算损失
    pde_loss = jnp.mean(jnp.square(u_t + epsilon**2 * bihar_u - lap_u3 + lap_u))

    # # 获取边界点
    # boundary_loss = jnp.mean()  # 边界条件，假设为0
    #
    # # 获取初始点
    # initial_loss = jnp.mean()

    boundary_loss, initial_loss = boundary_initial_loss(params, boundary_points, initial_points)

    # print(pde_loss)
    return pde_loss + boundary_loss_weight * boundary_loss + initial_loss

# 生成训练点
def generate_points():
    # 内部点
    x_internal = random.uniform(random.PRNGKey(0), shape=(num_internal_points,)) * 2 - 1
    y_internal = random.uniform(random.PRNGKey(1), shape=(num_internal_points,)) * 2 - 1
    t_internal = random.uniform(random.PRNGKey(2), shape=(num_internal_points,)) * T  # t在[0, T]内
    internal_points = jnp.column_stack((x_internal, y_internal, t_internal))  # 形状为 (10000, 3)

    # 边界点
    num_boundary_per_side = num_boundary_points // 4  # 每个边界的点数

    x_boundary = jnp.concatenate([
        jnp.full((num_boundary_per_side,), -1),  # 左边界
        jnp.full((num_boundary_per_side,), 1),  # 右边界
        random.uniform(random.PRNGKey(3), shape=(num_boundary_per_side * 2,)) * 2 - 1  # 上下边界
    ])

    y_boundary = jnp.concatenate([
        random.uniform(random.PRNGKey(4), shape=(num_boundary_per_side * 2,)) * 2 - 1,
        jnp.full((num_boundary_per_side,), -1),  # 下
        jnp.full((num_boundary_per_side,), 1)  # 上
    ])

    t_boundary = random.uniform(random.PRNGKey(6), shape=(num_boundary_points,)) * T  # t在[0, T]内
    boundary_points = jnp.column_stack((x_boundary, y_boundary, t_boundary))  # 形状为 (num_boundary_points, 3)

    # 初始点
    x_initial = random.uniform(random.PRNGKey(7), shape=(num_initial_points,)) * 2 - 1
    y_initial = random.uniform(random.PRNGKey(8), shape=(num_initial_points,)) * 2 - 1
    t_initial = jnp.zeros_like(x_initial)  # 初始点的时间为0
    initial_points = jnp.column_stack((x_initial, y_initial, t_initial))  # 形状为 (num_initial_points, 3)

    return internal_points, boundary_points, initial_points


# 创建模型和优化器
layers = [3] + [24] * 4 + [1]  # 输入层、4个隐藏层（每层40个神经元）、输出层
params = init_params(layers)
optimizer = optax.adam(learning_rate=0.001)
opt_state = optimizer.init(params)

# 生成训练点
internal_points, boundary_points, initial_points = generate_points()

# 训练循环
for epoch in range(10000):
    loss, grads = jax.value_and_grad(loss_fn)(params, internal_points, boundary_points, initial_points)

    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    # if epoch % 1000 == 0:
    print(f"Epoch: {epoch}, Loss: {loss}")

# 保存参数


dump(params, 'params.joblib')




# # 加载参数
# loaded_params = load('params.joblib')

# 可视化结果
x_test = jnp.linspace(-1, 1, 100)
y_test = jnp.linspace(-1, 1, 100)
t_test = 0.05  # 测试时间点
X_test, Y_test = jnp.meshgrid(x_test, y_test)

u_pred = MLP(params, jnp.concatenate([X_test.flatten(), Y_test.flatten(), t_test * jnp.ones(X_test.size)], axis=-1))

# 画出结果
plt.figure(figsize=(8, 6))
plt.contourf(X_test, Y_test, u_pred.reshape(X_test.shape), levels=50, cmap='viridis')
plt.colorbar()
plt.title(f'Solution at t = {t_test}')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

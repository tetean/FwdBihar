'''
@Author: tetean
@Time: 2024/10/21 11:06 AM
@Info:
'''
'''
@Author: tetean
@Time: 2024/10/19 8:12 PM
@Info:
'''
import jax
import jax.numpy as jnp
from jax.lax import add
from jax.lax import stop_gradient as sg
from jax import tree_util
__STK_MAX = 1 << 5


# @tree_util.register_pytree_node_class
class FBihar:
    """
    Hessian 类，用于存储向量及其雅可比和 Hessian
    """
    def __init__(self, x=None, jac=None, hess=None, trd=None, bihar=None):
        self.x = x
        self.jac = jac
        self.hess = hess
        self.trd = trd
        self.bihar = bihar

    # def tree_flatten(self):
    #     """
    #     将类实例拆分为子元素和辅助数据。
    #     返回一个元组（子元素的元组, 辅助数据）
    #     """
    #     children = (self.x, self.jac, self.hess, self.trd, self.bihar)
    #     aux_data = None  # 辅助数据
    #     return children, aux_data
    #
    # @classmethod
    # def tree_unflatten(cls, aux_data, children):
    #     """
    #     从子元素和辅助数据中重建类实例。
    #     """
    #     x, jac, hess, trd, bihar = children
    #     return cls(x, jac, hess, trd, bihar)

@jax.jit
def dtanh(x):
    """计算双曲正切函数的导数"""
    return 1 - jnp.tanh(x) ** 2

@jax.jit
def condense_B(x):
    # 假设x的形状是(m, n, n, n, n)
    # 取出x[:, j, j, k, k]，并将结果存储到y
    y = x[:, jnp.arange(x.shape[1]), jnp.arange(x.shape[1]), :, :]
    y = y[:, :, jnp.arange(x.shape[1]), jnp.arange(x.shape[1])]
    return y

@jax.jit
def condense_T(x):
    # x的形状是(m, n, n, n)
    # 使用切片操作直接提取需要的元素
    y = x[:, jnp.arange(x.shape[1]), jnp.arange(x.shape[1]), :]
    return y
@jax.jit
def condense_H(x):
    # x的形状是(m, n, n)
    # 使用切片操作直接提取需要的元素
    y = x[:, jnp.arange(x.shape[1]), jnp.arange(x.shape[1])]
    return y

@jax.jit
def fwd_hess(JF_u, Ju_x, HF_u, Hu_x):
    # term1 = jnp.einsum('ik,kjl->ijl', JF_u, Hu_x)
    # term2 = jnp.einsum('kj,ikm,ml->ijl', Ju_x, HF_u, Ju_x)
    return add(
        jnp.einsum('ik,kjl->ijl', JF_u, Hu_x),
        jnp.einsum('kj,ikm,ml->ijl', Ju_x, HF_u, Ju_x)
    )
    # return term1 + term2



@jax.jit
def fwd_trd(JF_u, Ju_x, HF_u, Hu_x, TF_u, Tu_x):
    # term1 = jnp.einsum('hj,j123->h123', JF_u, Tu_x)
    # term2 = jnp.einsum('hjk,j12,k3->h123', HF_u, Hu_x, Ju_x) + jnp.einsum('hjk,j13,k2->h123', HF_u, Hu_x, Ju_x) \
    #         + jnp.einsum('hjk,j23,k1->h123', HF_u, Hu_x, Ju_x)
    # term3 = jnp.einsum('hjkl,j1,k2,l3->h123', TF_u, Ju_x, Ju_x, Ju_x)

    return add(jnp.einsum('hj,j123->h123', JF_u, Tu_x),
               add(jnp.einsum('hjk,j12,k3->h123', HF_u, Hu_x, Ju_x),
                   add(jnp.einsum('hjk,j13,k2->h123', HF_u, Hu_x, Ju_x),
                       add(jnp.einsum('hjk,j23,k1->h123', HF_u, Hu_x, Ju_x),
                           jnp.einsum('hjkl,j1,k2,l3->h123', TF_u, Ju_x, Ju_x, Ju_x)))))

    # TF_x = term1 + term2 + term3
    # return TF_x

@jax.jit
def fwd_bihar(JF_u, Ju_x, HF_u, Hu_x, TF_u, Tu_x, BF_u, Bu_x_cond):


    Tu_x_cond = condense_T(Tu_x)
    Hu_x_cond = condense_H(Hu_x)

    # term1 = jnp.einsum('hj,j13->h13', JF_u, Bu_x_cond)
    # term2 = 2 * (jnp.einsum('hjk,j13,k3->h13', HF_u, Tu_x_cond, Ju_x) + jnp.einsum('hjk,j31,k1->h13', HF_u, Tu_x_cond,
    #                                                                                Ju_x))
    # term3 = jnp.einsum('hjk,j1,k3->h13', HF_u, Hu_x_cond, Hu_x_cond) + 2 * jnp.einsum('hjk,j13,k13->h13', HF_u, Hu_x,
    #                                                                                   Hu_x)
    # term4 = jnp.einsum('hjkl,j1,k3,l3->h13', TF_u, Hu_x_cond, Ju_x, Ju_x) \
    #         + 4 * jnp.einsum('hjkl,j13,k1,l3->h13', TF_u, Hu_x, Ju_x, Ju_x) \
    #         + jnp.einsum('hjkl,j3,k1,l1->h13', TF_u, Hu_x_cond, Ju_x, Ju_x)
    # term5 = jnp.einsum('hjklm,j1,k1,l3,m3->h13', BF_u, Ju_x, Ju_x, Ju_x, Ju_x)
    # BF_x = term1 + term2 + term3 + term4 + term5
    BF_x = add(jnp.einsum('hj,j13->h13', JF_u, Bu_x_cond),
        add(2 * jnp.einsum('hjk,j13,k3->h13', HF_u, Tu_x_cond, Ju_x),
        add(2 * jnp.einsum('hjk,j31,k1->h13', HF_u, Tu_x_cond,Ju_x),
        add(jnp.einsum('hjk,j1,k3->h13', HF_u, Hu_x_cond, Hu_x_cond),
        add(2 * jnp.einsum('hjk,j13,k13->h13', HF_u, Hu_x, Hu_x),
        add(jnp.einsum('hjkl,j1,k3,l3->h13', TF_u, Hu_x_cond, Ju_x, Ju_x),
        add(4 * jnp.einsum('hjkl,j13,k1,l3->h13', TF_u, Hu_x, Ju_x, Ju_x),
        add(jnp.einsum('hjkl,j3,k1,l1->h13', TF_u, Hu_x_cond, Ju_x, Ju_x),
        jnp.einsum('hjklm,j1,k1,l3,m3->h13', BF_u, Ju_x, Ju_x, Ju_x, Ju_x)))))))))
    return BF_x

# @jax.jit
# def fwd_bihar(JF_u, Ju_x, HF_u, Hu_x, TF_u, Tu_x, BF_u, Bu_x_cond):
#
#     out_dim = JF_u.shape[0]
#     in_dim = Ju_x.shape[1]
#     mid_dim = JF_u.shape[1]
#     Tu_x_cond = condense_T(Tu_x)
#     Hu_x_cond = condense_H(Hu_x)
#     term1 = jnp.zeros((out_dim, in_dim, in_dim))  # 初始化结果数组
#
#     for h in range(out_dim):
#         for j in range(in_dim):
#             term1 = term1.at[h, :, :].set(term1[h, :, :] + JF_u[h, j] * Bu_x_cond[j, :, :])
#
#     term2 = jnp.zeros((out_dim, in_dim, in_dim))  # 初始化结果数组
#
#     # 计算第一个einsum部分
#     # for h in range(out_dim):
#     #     for j in range(in_dim):
#     #         for k in range(in_dim):
#     #             term2 = term2.at[h, :, :].set(term2[h, :, :] + HF_u[h, j, k] * Tu_x_cond[j, :, :] @ Ju_x[k, :])
#
#     # 计算第二个einsum部分
#     # for h in range(out_dim):
#     #     for j in range(in_dim):
#     #         for k in range(in_dim):
#     #             term2 = term2.at[h, :, :].set(term2[h, :, :] + HF_u[h, j, k] * Tu_x_cond[j, :, :] @ Ju_x[:, k])
#
#     # 乘以2
#     term2 = 2 * term2
#     return term1 + term2

def simple_layer(x, A, b):
    return jnp.tanh(A @ x + b)

hess_f = jax.hessian(simple_layer)
trd_f = jax.jacobian(hess_f)
bihar_f = jax.hessian(hess_f)

@jax.jit
def MLP(params, x, rec=False):
    idx = -1
    top = 1 if not rec else __STK_MAX
    info = [None] * top

    # def simple_layer(x, A, b):
    #     return jnp.tanh(A @ x + b)
    #
    # hess_f = jax.hessian(simple_layer)
    # trd_f = jax.jacobian(hess_f)
    # bihar_f = jax.hessian(hess_f)
    # trd_f = jax.jacobian(hess_f)
    # bihar_f = jax.hessian(hess_f)

    for layer in params[:-1]:
        W = layer['W']
        b = layer['B']
        x1 = jnp.dot(W, x)
        x2 = jnp.tanh(x1)
        if idx < 0:
            # 第一层的雅可比和 Hessian
            jac = jnp.diag(dtanh(x1)) @ W
            # hess = tanh_hessian(W, x, b)
            hess = hess_f(x, W, b)
            # u = W @ x + b
            # hess = fwd_hess(jnp.diag(dtanh(u)), W, jnp.zeros((b.shape[0], u.shape[0], u.shape[0])), jnp.zeros((u.shape[0], x.shape[0], x.shape[0])))
            # hess = fwd_hess(JF_u, Ju_x, HF_u, Hu_x)

            trd = trd_f(x, W, b)

            bihar = bihar_f(x, W, b)
            bihar = condense_B(bihar)
        else:
            # 计算后续层的雅可比和 Hessian

            JF_u = jnp.diag(dtanh(x1)) @ W
            jac = JF_u @ info[idx % top].jac

            # HF_u = tanh_hessian(W, x, b)
            # hess = fwd_hess(info[-1].hess, jnp.diag(dtanh(x1)) @ W, info[-1].jac, HF_u, [W.shape[0], in_dim])
            HF_u = hess_f(x, W, b)
            hess = fwd_hess(JF_u, info[idx % top].jac, HF_u, info[idx % top].hess)

            TF_u = trd_f(x, W, b)
            trd = fwd_trd(JF_u, info[idx % top].jac, HF_u, info[idx % top].hess, TF_u, info[idx % top].trd)

            BF_u = bihar_f(x, W, b)
            bihar = fwd_bihar(JF_u, info[idx % top].jac, HF_u, info[idx % top].hess, TF_u, info[idx % top].trd, BF_u, info[idx % top].bihar)
        x = x2
        idx += 1
        info[idx % top] = FBihar(x2, jac, hess, trd, bihar)

    W = params[-1]['W']
    b = params[-1]['B']


    if idx < 0:
        jac = W
        hess = jnp.zeros((W.shape[0], x.shape[0], x.shape[0]))
        trd = jnp.zeros((W.shape[0], x.shape[0], x.shape[0], x.shape[0]))
        bihar = jnp.zeros((W.shape[0], x.shape[0], x.shape[0]))
    else:

        JF_u = W
        jac =  None if not rec else JF_u @ info[-1].jac

        HF_u = jnp.zeros((W.shape[0], x.shape[0], x.shape[0]))
        # hess = final_hess(W, info[-1].hess, [W.shape[0], in_dim])
        hess = None if not rec else fwd_hess(JF_u, info[idx % top].jac, HF_u, info[idx % top].hess)

        TF_u = jnp.zeros((W.shape[0], x.shape[0], x.shape[0], x.shape[0]))
        trd = None if not rec else fwd_trd(JF_u, info[idx % top].jac, HF_u, info[idx % top].hess, TF_u, info[idx % top].trd)

        BF_u = jnp.zeros((W.shape[0], x.shape[0], x.shape[0], x.shape[0], x.shape[0]))
        bihar = fwd_bihar(JF_u, info[idx % top].jac, HF_u, info[idx % top].hess, TF_u, info[idx % top].trd, BF_u, info[idx % top].bihar)
    x = None if not rec else jnp.dot(W, x) + b

    bihar = jnp.sum(bihar, axis=(1, 2))
    idx += 1
    info[idx % top] = FBihar(x, jac, hess, trd, bihar)
    return bihar


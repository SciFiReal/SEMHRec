# coding: utf-8
# @email  : enoche.chow@gmail.com

"""
Utility functions
##########################
"""
import os

import numpy as np
import torch
import importlib
import datetime
import random

from scipy.sparse import diags


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y-%H-%M-%S')

    return cur


def get_model(model_name):
    r"""Automatically select model class based on model name
    Args:
        model_name (str): model name
    Returns:
        Recommender: model class
    """
    model_file_name = model_name.lower()
    module_path = '.'.join(['models', model_file_name])
    if importlib.util.find_spec(module_path, __name__):
        model_module = importlib.import_module(module_path, __name__)

    model_class = getattr(model_module, model_name)
    return model_class


def get_trainer():
    return getattr(importlib.import_module('common.trainer'), 'Trainer')


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 增加
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)


def early_stopping(value, best, cur_step, max_step, bigger=True):
    r""" validation-based early stopping

    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better

    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value > best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value < best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric) + ': ' + '%.04f' % value + '    '
    return result_str


############ LATTICE Utilities #########

def build_knn_neighbourhood(adj, topk):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix


def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm


def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim

def get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization='none'):
    from torch_scatter import scatter_add
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    if normalization == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif normalization == 'rw':
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight

def get_dense_laplacian(adj, normalization='none'):
    if normalization == 'sym':
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    elif normalization == 'rw':
        rowsum = torch.sum(adj, -1)
        d_inv = torch.pow(rowsum, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diagflat(d_inv)
        L_norm = torch.mm(d_mat_inv, adj)
    elif normalization == 'none':
        L_norm = adj
    return L_norm

def build_knn_normalized_graph(adj, topk, is_sparse, norm_type):
    device = adj.device
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    if is_sparse:
        tuple_list = [[row, int(col)] for row in range(len(knn_ind)) for col in knn_ind[row]]
        row = [i[0] for i in tuple_list]
        col = [i[1] for i in tuple_list]
        i = torch.LongTensor([row, col]).to(device)
        v = knn_val.flatten()
        edge_index, edge_weight = get_sparse_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
        return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)
    else:
        weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)

# ----------------------------新添-------------------------------
# -------多模态超图的构建:构造模态的KNN图，矩阵值设为1，表示模态相似，不关心模态的相似度，不需要做归一化处理


def build_knn_normalized_hyper_graph(adj, topk):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    adj = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)  # 产生一个和adj完全相同dtype的张量（float32）
    adj[adj > 0] = 1.
    return adj

# 构造多模态超图并做归一化处理：稀疏张量->密集张量->稀疏张量

def build_Hypergraph_normalized(Hypergraph, normalization):
    # torch.sparse.mm（也叫 torch.sparse.spmm）只支持稀疏矩阵 × 密集矩阵  = 密集张量矩阵
    Hypergraph_mul = torch.sparse.mm(Hypergraph, Hypergraph.to_dense().T)

    # 做归一化
    if normalization == 'sym_pytorch':
        # -----PyTorch 代码实现的是对称归一化的邻接矩阵（或拉普拉斯矩阵）
        rowsum = torch.sum(Hypergraph_mul, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        Hypergraph_mul = torch.mm(torch.mm(d_mat_inv_sqrt, Hypergraph_mul), d_mat_inv_sqrt)
    Hypergraph_mul = Hypergraph_mul.to_sparse()

    # 稀疏张量
    return Hypergraph_mul


# 用户-用户的超图（稀疏张量->稀疏张量）

def get_U2U_mat(R, norm_type='rw'):
    # # ================= 新增：物品度归一化 =================
    # # 1. 计算物品度数(每个物品被交互的用户数)
    # item_degrees = np.array(R.sum(axis=0)).ravel()  # shape (item_num,)
    #
    # # 2. 避免除零和数值不稳定
    # item_degrees = np.maximum(item_degrees, 1)  # 最小值为1
    # inv_sqrt_degrees = 1.0 / np.sqrt(item_degrees)  # 更稳健的IDF类归一化
    #
    # # 3. 创建物品度对角归一化矩阵
    # D_e_inv = diags(inv_sqrt_degrees)
    #
    # # 4. 应用物品度归一化：R_norm = R × D_e^{-1/2}
    # R = R.dot(D_e_inv)  # 稀疏矩阵高效乘法
    # # ================= 新增：物品度归一化 =================

    # 1) 先做稀疏共现：用户-用户共现矩阵。结果仍是稀疏格式 CSR （更高效）
    U2U = R.dot(R.T).tocsr()
    # #保留共同交互 > k 件物品的用户集合  增加过滤条件
    # k = 5  # 假设我们设定 k = 5
    # U2U.data[U2U.data <= k] = 0  # 过滤掉那些共交互物品数 <= k 的用户对
    # 2) 去掉自连接 diagonal
    U2U.setdiag(0)
    U2U.eliminate_zeros()  # 删除为零的元素
    # 3) 随机游走归一化（行归一化）：U2U_norm = D^{-1} * U2U
    if norm_type == 'rw':
        row_sum = np.array(U2U.sum(axis=1)).ravel()  # shape (user_num,)
        # 避免除零
        row_sum[row_sum == 0] = 1.0
        D_inv = diags(1.0 / row_sum)                 # 构造对角稀疏矩阵
        U2U = D_inv.dot(U2U)                         # 依然是 csr_matrix
    else:
        raise NotImplementedError(f"norm_type={norm_type} 未实现")
    # 4) 转回稀疏张量
    U2U = U2U.tocoo()
    coords = np.vstack((U2U.row, U2U.col))
    i = torch.from_numpy(coords).long()
    v = torch.from_numpy(U2U.data).float()
    shape = (U2U.shape[0], U2U.shape[1])
    return torch.sparse_coo_tensor(i, v, size=shape)



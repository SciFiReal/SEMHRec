# coding: utf-8
# @email: y463213402@gmail.com
r"""
MGCN
################################################
Reference:
    https://github.com/demonph10/MGCN
    ACM MM'2023: [Multi-View Graph Convolutional Network for Multimedia Recommendation]
    https://arxiv.org/abs/2308.03588
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.abstract_recommender import GeneralRecommender
from src.utils.utils import (build_sim, compute_normalized_laplacian, build_knn_neighbourhood,
                             build_knn_normalized_graph, build_knn_normalized_hyper_graph,
                             build_Hypergraph_normalized, get_U2U_mat)


class MGCN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MGCN, self).__init__(config, dataset)
        self.sparse = True
        self.cl_loss = config['cl_loss']
        self.na_loss = config['na_loss']
        self.align_loss = config['align_loss']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.Item_layers = config['Item_layers']
        self.User_layers = config['User_layers']

        # 加载稀疏交互矩阵coo
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # 构造模态数据集的邻接矩阵文件路径
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.knn_k, self.sparse))

        # 多模态和用户的超图邻接矩阵文件路径
        multimodal_adj_file = os.path.join(dataset_path, 'multimodal_adj_{}_{}.pt'.format(self.knn_k, self.sparse))
        U2U_adj_file = os.path.join(dataset_path, 'U2U_adj_{}.pt'.format(self.sparse))
        # I2I_adj_file = os.path.join(dataset_path, 'I2I_adj.pt')

        # 构造用户-用户超图邻接矩阵
        if os.path.exists(U2U_adj_file):
            interaction_matrix_U2U_hyper = torch.load(U2U_adj_file)
        else:
            interaction_matrix_U2U_hyper = get_U2U_mat(self.interaction_matrix)  # 稀疏张量
            torch.save(interaction_matrix_U2U_hyper, U2U_adj_file)
        self.interaction_matrix_U2U_hyper = interaction_matrix_U2U_hyper.cuda()

        # 构造模态数据集的超图邻接矩阵和文件路径
        if (self.v_feat is not None) and (self.t_feat is not None):
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)

            if os.path.exists(multimodal_adj_file):
                Hypergraph_mul = torch.load(multimodal_adj_file)
            else:

                # 拼接2个模态
                image_adj_hyper = build_sim(self.image_embedding.weight.detach())  # 计算机模态间的相似度
                image_adj_hyper = build_knn_normalized_hyper_graph(image_adj_hyper, topk=self.knn_k)  # 构造模态KNN相似图

                text_adj_hyper = build_sim(self.text_embedding.weight.detach())
                text_adj_hyper = build_knn_normalized_hyper_graph(text_adj_hyper, topk=self.knn_k)

                Hypergraph = torch.cat((image_adj_hyper, text_adj_hyper), dim=1)  # 将2个模态的邻接矩阵按列拼接
                Hypergraph = Hypergraph.to_sparse()  # 转为稀疏张量
                Hypergraph_mul = build_Hypergraph_normalized(Hypergraph, 'sym_pytorch')

                torch.save(Hypergraph_mul, multimodal_adj_file)
            self.item_multimodal_hypergraph = Hypergraph_mul.cuda()  # 默认张量在cpu上，.cuda()，张量移到gpu上

        # 构建用户-项目的交互矩阵和归一化邻接矩阵
        self.norm_adj = self.get_adj_mat()
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())  # 计算模态间的相似度，得到相似度矩阵
                image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse,
                                                       norm_type='sym')  # 构造模态的KNN矩阵
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda()

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        self.softmax = nn.Softmax(dim=-1)

        self.query_common = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        self.gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_t = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.user_gate = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 2, bias=False)  # 输出两个 logit
        )

        self.tau = 0.5

    def pre_epoch_processing(self):
        pass

    # 获取归一化的用户-项目邻接矩阵
    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            # norm_adj = adj.dot(d_mat_inv)
            # print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, adj, train=False):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)

        # 1. Behavior-Guided Purifier
        # 引入ID特征影响门控值
        gate_input_image = torch.cat([self.item_id_embedding.weight, image_feats], dim=1)
        image_item_embeds = self.item_id_embedding.weight * self.gate_v(gate_input_image)

        gate_input_text = torch.cat([self.item_id_embedding.weight, text_feats], dim=1)
        text_item_embeds = self.item_id_embedding.weight * self.gate_t(gate_input_text)

        # image_item_embeds = image_feats
        # text_item_embeds = text_feats

        # 3. User-Item View  得到u_ui_emb, i_ui_emb
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        u_ui_emb, i_ui_emb = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)

        # 3. Item-Item View
        # 得到用户和项目的模态嵌入表示

        # 获取每个用户的交互次数（转换为稠密张量并调整形状）
        user_interaction_counts = torch.sparse.sum(self.R, dim=1).to_dense().unsqueeze(1)

        if self.sparse:
            for i in range(self.n_layers):
                image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
        else:
            for i in range(self.n_layers):
                image_item_embeds = torch.mm(self.image_original_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_user_embeds = image_user_embeds / (user_interaction_counts + 1e-8)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)

        if self.sparse:
            for i in range(self.n_layers):
                text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        else:
            for i in range(self.n_layers):
                text_item_embeds = torch.mm(self.text_original_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_user_embeds = text_user_embeds / (user_interaction_counts + 1e-8)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        user_pref_vec = torch.cat([image_user_embeds, text_user_embeds], dim=1)  # (n_users , 2d)
        user_logits = self.user_gate(user_pref_vec)  # (n_users , 2)
        user_weight = self.softmax(user_logits)  # softmax 按行做 → 每行和为 1
        item_logits = torch.cat([self.query_common(image_item_embeds),  # (n_items , 1)
                                 self.query_common(text_item_embeds)], dim=-1)  # (n_items , 2)
        item_weight = self.softmax(item_logits)
        weight_common = torch.cat([user_weight, item_weight], dim=0)  # (n_users+n_items , 2)
        common_embeds = (weight_common[:, 0].unsqueeze(1) * image_embeds +
                         weight_common[:, 1].unsqueeze(1) * text_embeds)
        uu_emb, ii_emb = torch.split(common_embeds, [self.n_users, self.n_items], dim=0)

        # common_embeds = image_embeds + text_embeds
        # uu_emb, ii_emb = torch.split(common_embeds, [self.n_users, self.n_items], dim=0)

        # 3. 模态超图和用户超图
        # 3.1 特征融合
        # 最终的项目表示，传统协同过滤的物品特征 + 标准化超图卷积的物品特征
        i_ui_emb_i = ii_emb   # ii_emb  #  item_embeds + ii_emb  # 是否需要归一化呢
        # 最终的用户表示，传统协同过滤的用户特征 + 标准化超图卷积的用户特征
        u_ui_emb_u = uu_emb   # uu_emb  # user_embeds + uu_emb

        # i_ui_emb_i = self.dropout(i_ui_emb_i)
        # u_ui_emb_u = self.dropout(u_ui_emb_u)

        # 3.2 超图卷积
        # i2i项目超图
        item_embeds_new = i_ui_emb_i
        for i in range(self.Item_layers):
            item_embeds_new = torch.sparse.mm(self.item_multimodal_hypergraph, item_embeds_new)

        # u2u超图卷积
        user_embeds_new = u_ui_emb_u
        for i in range(self.User_layers):
            user_embeds_new = torch.sparse.mm(self.interaction_matrix_U2U_hyper, user_embeds_new)

        # 3.3 最终表示
        # 最终的项目表示，传统协同过滤的物品特征 + 标准化超图卷积的物品特征
        i_ui_emb_i2i_end = i_ui_emb + F.normalize(item_embeds_new, p=2, dim=1)   # 是否需要归一化呢
        # i_ui_emb_i2i_end = i_ui_emb + item_embeds_new  # 是否需要归一化呢
        # 最终的用户表示，传统协同过滤的用户特征 + 标准化超图卷积的用户特征
        u_ui_emb_u2u_end = u_ui_emb + F.normalize(user_embeds_new, p=2, dim=1)
        # u_ui_emb_u2u_end = u_ui_emb + user_embeds_new


        if train:
            return u_ui_emb_u2u_end, i_ui_emb_i2i_end, u_ui_emb, i_ui_emb, user_embeds_new, item_embeds_new, image_item_embeds, text_item_embeds
        return u_ui_emb_u2u_end, i_ui_emb_i2i_end

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    # 获取邻居用户的损失
    def get_neighbor_aggregate_loss(self, embedding1, embedding2, tau):
        embedding1 = torch.nn.functional.normalize(embedding1)
        embedding2 = torch.nn.functional.normalize(embedding2)

        pos_score = (embedding1 * embedding2).sum(dim=-1)
        pos_score = torch.exp(pos_score / tau)

        total_score = torch.matmul(embedding1, embedding2.transpose(0, 1)) + torch.matmul(embedding1,
                                                                                          embedding1.transpose(0, 1))
        total_score = torch.exp(total_score / tau).sum(dim=1)

        na_loss = -torch.log(pos_score / total_score + 10e-6)
        return torch.mean(na_loss)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_ui_emb, i_ui_emb, uu_emb, ii_emb, uu_emb_uu, ii_emb_ii, image_item_embeds, text_item_embeds = self.forward(
            self.norm_adj, train=True)

        u_g_embeddings = u_ui_emb[users]
        pos_i_g_embeddings = i_ui_emb[pos_items]
        neg_i_g_embeddings = i_ui_emb[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)
        # 1. ID和模态对齐:统一的 InfoNCE 对齐方案
        # # # ID-视觉模态对齐
        # id_emb_norm = F.normalize(self.item_id_embedding.weight, dim=1)
        # image_item_norm = F.normalize(image_item_embeds, dim=1)
        # # ID-文本模态对齐
        # text_item_norm = F.normalize(text_item_embeds, dim=1)
        #
        # # 统一的 InfoNCE 对齐方案
        # loss_id_image = self.InfoNCE(id_emb_norm, image_item_norm, 0.2)  # ID - 图像
        # loss_id_text = self.InfoNCE(id_emb_norm, text_item_norm, 0.2)  # ID - 文本

        # 总损失（可加权）
        # ----直接对齐，不归一化
        loss_id_image = self.InfoNCE(self.item_id_embedding.weight[pos_items], image_item_embeds[pos_items], 0.2)  # ID - 图像
        loss_id_text = self.InfoNCE(self.item_id_embedding.weight[pos_items], text_item_embeds[pos_items], 0.2)  # ID - 文本
        align_loss = (loss_id_image + loss_id_text) * self.align_loss

        # 邻居损失函数
        na_loss = self.get_neighbor_aggregate_loss(u_g_embeddings, pos_i_g_embeddings, 0.2) * self.na_loss  # 5.0

        # 对比损失函数
        cl_loss = (self.InfoNCE(ii_emb_ii[pos_items], ii_emb[pos_items], 0.2) + self.InfoNCE(
            uu_emb_uu[users], uu_emb[users], 0.2)) * self.cl_loss

        return batch_mf_loss + batch_emb_loss + batch_reg_loss + cl_loss + na_loss + align_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
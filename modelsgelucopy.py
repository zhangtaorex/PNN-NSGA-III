import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Union
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value):
    """注意力机制的实现, 输入分别是query, key, value, mask: 掩码张量,
       dropout是nn.Dropout层的实例化对象, 默认为None
    """
    # 在函数中, 首先取query的最后一维的大小, 一般情况下就等同于我们的词嵌入维度, 命名为d_k
    d_k = query.size(-1)
    # print("d_k:",d_k) #64
 
    # 按照注意力公式, 将query与key的转置相乘, 这里面key是将最后两个维度进行转置,
    # 再除以缩放系数根号下d_k, 这种计算方法也称为缩放点积注意力计算.
    # 得到注意力得分张量scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print("scores.shape",scores.shape) #torch.Size([2, 8, 4, 4])

 
    # print("scores.shape:", scores.shape) #torch.Size([2, 4, 4])
    # print("scores:",scores)
    # tensor([[[1.4670e+04,  -1.0000e+09, -1.0000e+09,  -1.0000e+09],
    #          [-7.1877e+02, 1.3407e+04,  -1.0000e+09,  -1.0000e+09],
    #          [-7.7895e+02, 1.1335e+03,  1.3097e+04,   -1.0000e+09],
    #          [-1.8603e+02, -5.8361e+02, 1.7998e+02,    1.1442e+04]],
    #
    #        [[1.1710e+04,  -1.0000e+09, -1.0000e+09,   -1.0000e+09],
    #         [4.9352e+02,  1.3066e+04,  -1.0000e+09,   -1.0000e+09],
    #         [-7.1906e+02, 6.3984e+02,   1.3662e+04,   -1.0000e+09],
    #         [6.2098e+02,  3.5394e+02,  -5.2597e+02,   1.3532e+04]]],
    #        grad_fn= < MaskedFillBackward0 >)
 
    # 对scores的最后一维进行softmax操作, 使用F.softmax方法, 第一个参数是softmax对象, 第二个是目标维度.
    # 这样获得最终的注意力张量
    p_attn = F.softmax(scores, dim = -1)
    # 最后, 根据公式将p_attn与value张量相乘获得最终的query注意力表示, 同时返回注意力张量
    return torch.matmul(p_attn, value), p_attn
class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        """在类的初始化时, 会传入三个参数，head代表头数，embedding_dim代表词嵌入的维度，
           dropout代表进行dropout操作时置0比率，默认是0.1."""
        super(MultiHeadedAttention, self).__init__()
 
        # 在函数中，首先使用了一个测试中常用的assert语句，判断h是否能被d_model整除，
        # 这是因为我们之后要给每个头分配等量的词特征.也就是embedding_dim/head个.
        assert embedding_dim % head == 0
        # 得到每个头获得的分割词向量维度d_k
        self.d_k = embedding_dim // head
        # 传入头数h
        self.head = head
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
 
        # self.attn为None，它代表最后得到的注意力张量，现在还没有结果所以为None.
        self.attn = None

    def forward(self, query, key, value, mask=None):
        """前向逻辑函数, 它的输入参数有四个，前三个就是注意力机制需要的Q, K, V，
           最后一个是注意力机制中可能需要的mask掩码张量，默认是None. """
        batch_size = query.size(0)
        query, key, value = \
           [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
            for model, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        return self.linears[-1](x)

# class SelfAttention(nn.Module):
#     def __init__(self,dim_input, dim_q, dim_v):
#         '''
#         参数说明：
#         dim_input: 输入数据x中每一个样本的向量维度
#         dim_q:     Q矩阵的列向维度, 在运算时dim_q要和dim_k保持一致;
#                    因为需要进行: K^T*Q运算, 结果为：[dim_input, dim_input]方阵
#         dim_v:     V矩阵的列项维度,维度数决定了输出数据attention的列向维度
#         '''
#         super(SelfAttention,self).__init__()
        
#         #dim_k = dim_q
#         self.dim_input = dim_input
#         self.dim_q = dim_q
#         self.dim_k = dim_q
#         self.dim_v = dim_v
        
#         #定义线性变换函数
#         self.linear_q = nn.Linear(self.dim_input, self.dim_q, bias=False) 
#         self.linear_k = nn.Linear(self.dim_input, self.dim_k, bias=False)
#         self.linear_v = nn.Linear(self.dim_input, self.dim_v, bias=False)
#         self._norm_fact = 1 / math.sqrt(self.dim_k)

#     def forward(self,x):

#         batch, n, dim_q = x.shape
        
#         q = self.linear_q(x)   # Q: batch_size * seq_len * dim_k
#         k = self.linear_k(x)   # K: batch_size * seq_len * dim_k
#         v = self.linear_v(x)   # V: batch_size * seq_len * dim_v
#         # print(f'x.shape:{x.shape} \n  Q.shape:{q.shape} \n  K.shape: {k.shape} \n  V.shape:{v.shape}')
#         #K^T*Q
#         dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact
#         #归一化获得attention的相关系数：A
#         dist = torch.softmax(dist, dim=-1)
#         # print('attention matrix: ', dist.shape)
#         #socre与v相乘，获得最终的输出
#         att = torch.bmm(dist, v)
#         # print('attention output: ',att.shape)
#         return att

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            `x`: shape (batch_size, max_len, d_model)

        Returns:
            same shape as input x
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size: int, dropout: float):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))
class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        # features = d_model
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class parallel_net(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m,nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
            nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.BatchNorm1d):
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)
    def __init__(self, model_dim, hidden_size_skin, hidden_size_stringer, hidden_size_features) -> None:
        super(parallel_net,self).__init__()
        self.skin_embedding_tabel = nn.Embedding(5,model_dim)
        self.stringer_embedding_tabel = nn.Embedding(5,model_dim)
        self.Bi_lstm_skin = nn.LSTM(input_size = model_dim, hidden_size = hidden_size_skin, batch_first = True,
                                bidirectional = True
                                )
        self.Bi_lstm_stringer = nn.LSTM(input_size = model_dim, hidden_size = hidden_size_stringer, batch_first = True,
                                bidirectional = True
                                )
        self.attention_skin = MultiHeadedAttention(4, 2*hidden_size_skin)
        self.attention_stringer = MultiHeadedAttention(4, 2*hidden_size_stringer)
        self.feedforward_skin = FeedForward(2*hidden_size_skin,hidden_size_skin)
        self.feedforward_stringer = FeedForward(2*hidden_size_stringer, hidden_size_stringer)
        self.sub_skin_1 = SublayerConnection(2*hidden_size_skin, 0.1)
        self.sub_skin_2 = SublayerConnection(2*hidden_size_skin, 0.1)
        self.sub_stringer_1 = SublayerConnection(2*hidden_size_stringer, 0.1)
        self.sub_stringer_2 = SublayerConnection(2*hidden_size_stringer, 0.1)
        self.skin_features_layer = nn.Linear(2*hidden_size_skin, hidden_size_skin)
        self.stringer_features_layer = nn.Linear(2*hidden_size_stringer, hidden_size_stringer)
        self.fc1 = nn.Sequential(nn.Linear(14,32),nn.ReLU(),nn.Linear(32,64),nn.ReLU(),nn.Linear(64,128),nn.ReLU(),
                                 nn.Linear(128,256),nn.ReLU(),nn.Linear(256, 512),nn.ReLU(),
                                 nn.Linear(512 , hidden_size_features))
        self.fc2 = nn.Sequential( 
                                nn.Linear(6*hidden_size_skin+6*hidden_size_stringer+hidden_size_features, 2048),nn.ReLU(),
                                 nn.Linear(2048,1024),nn.ReLU(),
                                 nn.Linear(1024,512),nn.ReLU(),
                                 nn.Linear(512,256),nn.ReLU(),
                                 nn.Linear(256,128), nn.ReLU(),
                                 nn.Linear(128,64),nn.ReLU(),
                                 nn.Linear(64,32),nn.ReLU(),
                                 nn.Linear(32,16),nn.ReLU(),
                                 nn.Linear(16,8),nn.ReLU(),
                                 nn.Linear(8,4),nn.ReLU(),
                                 nn.Linear(4,1)
                                 )
        self.fc3 = nn.Sequential( 
                                nn.Linear(6*hidden_size_skin+6*hidden_size_stringer+hidden_size_features, 2048),nn.ReLU(),
                                 nn.Linear(2048,1024),nn.ReLU(),
                                 nn.Linear(1024,512),nn.ReLU(),
                                 nn.Linear(512,256),nn.ReLU(),
                                 nn.Linear(256,128), nn.ReLU(),
                                 nn.Linear(128,64),nn.ReLU(),
                                 nn.Linear(64,32),nn.ReLU(),
                                 nn.Linear(32,16),nn.ReLU(),
                                 nn.Linear(16,8),nn.ReLU(),
                                 nn.Linear(8,4),nn.ReLU(),
                                 nn.Linear(4,1)
                                 )
        
        for name, parameter in self.Bi_lstm_skin.named_parameters():
        
            torch.nn.init.orthogonal_(parameter.unsqueeze(0))
        for name, parameter in self.Bi_lstm_stringer.named_parameters():
            torch.nn.init.orthogonal_(parameter.unsqueeze(0))

    def forward(self,skin_features, stringer_features, other_features):
        # skin/stringer_feature size->(batch,seq)
        emb_skin = self.skin_embedding_tabel(skin_features)
        emb_stringer = self.stringer_embedding_tabel(stringer_features)
        # embed  skin/stringer_size -> (batch, seq , model_dim)
        skin_fea,(h_skin, o_skin) = self.Bi_lstm_skin(emb_skin)
        stringer_fea,(h_stringer, o_stringer) = self.Bi_lstm_stringer(emb_stringer)
        #bi_lstm  skin/stringer_size ->(batch,seq,2*hidden_size_skin/stringer)
        h_skin_0 = h_skin[0]
        h_skin_1 = h_skin[1]
        o_skin_0 = o_skin[0]
        o_skin_1 = o_skin[1]
        h_stringer_0 = h_stringer[0]
        h_stringer_1 = h_stringer[1]
        o_stringer_0 = o_stringer[0]
        o_stringer_1 = o_stringer[1]
        skin_att = self.attention_skin(skin_fea,skin_fea,skin_fea)
        stringer_att = self.attention_stringer(stringer_fea,stringer_fea,stringer_fea)
        #attention  skin/stringer_size ->(batch,seq,2*hidden_size_skin/stringer)
        sublayer_skin = lambda x: self.attention_skin(skin_fea, skin_fea, skin_fea)
        sublayer_stringer = lambda x: self.attention_stringer(stringer_fea, stringer_fea, stringer_fea)
        skin_att = self.sub_skin_1(skin_fea, sublayer_skin)
        stringer_att = self.sub_stringer_1(stringer_fea, sublayer_stringer)
        skin_att = self.sub_skin_2(skin_att, self.feedforward_skin)
        stringer_att = self.sub_stringer_2(stringer_att, self.feedforward_stringer)
        skin_att = self.skin_features_layer(skin_att)
        stringer_att = self.stringer_features_layer(stringer_att)
        avg_pool_skin = F.adaptive_avg_pool1d(skin_att.permute(0,2,1),1).view(skin_features.size(0),-1)
        max_pool_skin = F.adaptive_max_pool1d(skin_att.permute(0,2,1),1).view(skin_features.size(0),-1)
        avg_pool_stringer = F.adaptive_avg_pool1d(stringer_att.permute(0,2,1),1).view(skin_features.size(0),-1)
        max_pool_stringer = F.adaptive_max_pool1d(stringer_att.permute(0,2,1),1).view(skin_features.size(0),-1)
        #pool skin/stringer_size -> (batch,hidden_size_skin/stringer)
        #concat skin/stringer_size -> (batch, 2*hidden_size_skin/stringer)
        # skin_feas = torch.cat([h_skin_0,h_skin_1,avg_pool_skin,max_pool_skin], dim=1)
        # stringer_feas = torch.cat([h_stringer_0,h_stringer_1,avg_pool_stringer,max_pool_stringer], dim=1)
        # other_feas = self.fc1(other_features)
        skin_feas = torch.cat([h_skin_0,h_skin_1,o_skin_0,o_skin_1,avg_pool_skin,max_pool_skin], dim=1)
        stringer_feas = torch.cat([h_stringer_0,h_stringer_1,o_stringer_0,o_stringer_1,avg_pool_stringer,max_pool_stringer], dim=1)
        # skin_feas = torch.cat([avg_pool_skin,max_pool_skin], dim=1)
        # stringer_feas = torch.cat([avg_pool_stringer,max_pool_stringer], dim=1)
        other_feas = self.fc1(other_features)
        # concat all_features ->(batch, 2*hidden_size_skin+2*hidden_size_stringer+hidden_features)
        all_features = torch.cat([skin_feas, stringer_feas, other_feas], dim=1)
        yasuo = self.fc2(all_features)
        jianqie = self.fc2(all_features)
        pred = torch.cat([yasuo,jianqie], dim=1)
        return pred
    


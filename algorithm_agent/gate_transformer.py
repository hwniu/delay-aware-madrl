import math

import torch
import numpy as np

# 位置编码
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, dim):
        super(PositionalEmbedding, self).__init__()
        self.dim = dim
        inv_freq = 1 / (10000 ** (torch.arange(0.0, dim, 2.0) / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, positions):
        sinusoid_inp = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[:, None, :]

# MLP
class PositionWiseFun(torch.nn.Module):
    def __init__(self, d_input, d_inner, dropout):
        super(PositionWiseFun, self).__init__()

        self.d_input = d_input
        self.d_inner = d_inner
        self.dropout = dropout
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_input, d_inner),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_inner, d_input),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input_):
        ff_out = self.ff(input_)
        return ff_out


# GRU单元
class GatingMechanism(torch.nn.Module):
    def __init__(self, d_input, bg=0.1):
        super(GatingMechanism, self).__init__()
        self.Wr = torch.nn.Linear(d_input, d_input)
        self.Ur = torch.nn.Linear(d_input, d_input)
        self.Wz = torch.nn.Linear(d_input, d_input)
        self.Uz = torch.nn.Linear(d_input, d_input)
        self.Wg = torch.nn.Linear(d_input, d_input)
        self.Ug = torch.nn.Linear(d_input, d_input)
        self.bg = bg

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x, y):
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g

# extra Long 多头注意力机制
class MultiHeadAttentionXL(torch.nn.Module):
    def __init__(self, dim_input, dim_inner, n_heads=4, out_dropout=0.1, attention_out_dropout=0.0):
        super(MultiHeadAttentionXL, self).__init__()
        self.dim_input = dim_input
        self.dim_inner = dim_inner
        self.n_heads = n_heads

        # 为了提高训练效率，value/query vector使用同一个神经网络
        self.key_value_linear = torch.nn.Linear(self.dim_input, self.dim_inner * self.n_heads * 2, bias=False)
        self.query_linear = torch.nn.Linear(self.dim_input, self.dim_inner * self.n_heads, bias=False)

        self.scale = 1 / math.sqrt(self.dim_inner)

        self.attention_out_dropout = torch.nn.Dropout(attention_out_dropout)

        # 位置注意力
        self.position_attention = torch.nn.Linear(self.dim_input, self.dim_inner * self.n_heads, bias=False)

        # output layer
        self.output_layer = torch.nn.Linear(self.dim_inner * self.n_heads, self.dim_input, bias=False)
        self.out_dropout = torch.nn.Dropout(out_dropout)

    def _rel_shift(self, shift_input):
        zero_pad = torch.zeros((shift_input.size(0), 1, *shift_input.size()[2:]), device=shift_input.device,
                               dtype=shift_input.dtype)
        return (
            torch.cat([zero_pad, shift_input], dim=1)
                .view(shift_input.size(1) + 1, shift_input.size(0), *shift_input.size()[2:])[1:]
                .view_as(shift_input)
        )

    def forward(self, multi_head_input, position_emb, memory, u, v, mask=None):
        current_seq = multi_head_input.shape[0]
        pre_seq = memory.shape[0]
        input_with_memory = torch.cat([memory, multi_head_input], dim=0)

        head_key, head_value = torch.chunk(self.key_value_linear(input_with_memory), 2, dim=-1)
        head_query = self.query_linear(multi_head_input)

        assert head_query.shape[1] == head_key.shape[1]

        content_attention = torch.einsum("ibhd,jbhd->ijbh", (
                (head_query.view(current_seq, head_query.shape[1], self.n_heads, self.dim_inner)) + u),
                                         (head_key.view(current_seq + pre_seq, head_query.shape[1], self.n_heads,
                                                        self.dim_inner)))

        head_position = self.position_attention(position_emb)
        position_attention = torch.einsum("ibhd,jhd->ijbh", (
            (head_query.view(current_seq, head_query.shape[1], self.n_heads, self.dim_inner) + v),
            head_position.view(current_seq + pre_seq, self.n_heads, self.dim_inner)
        ))
        position_attention = self._rel_shift(position_attention)

        attention = position_attention + content_attention

        if mask is not None and mask.any().item():
            attention = attention.masked_fill(mask[..., None], -float("inf"))
        attention = torch.softmax(attention * self.scale, dim=1)
        attention = self.attention_out_dropout(attention)

        attention_weight_values = (
            torch.einsum("ijbh,jbhd->ibhd",
                         (attention,
                          head_value.view(current_seq + pre_seq, head_query.shape[1], self.n_heads, self.dim_inner)
                          ),
                         ).contiguous().view(current_seq, head_query.shape[1], self.n_heads * self.dim_inner)

        )

        attention_output = self.out_dropout(self.output_layer(attention_weight_values))
        return attention_output

class StableTransformerEncoderLayerXL(torch.nn.Module):
    def __init__(
            self,
            n_heads,
            dim_input,
            dim_head_inner,
            dim_ff_inner,
            mha_out_dropout,
            gating=True,
            attention_dropout=0.0,
    ):
        super(StableTransformerEncoderLayerXL, self).__init__()

        self.gating = gating
        self.gate1 = GatingMechanism(dim_input)
        self.gate2 = GatingMechanism(dim_input)
        self.mha = MultiHeadAttentionXL(
            dim_input,
            dim_head_inner,
            n_heads=n_heads,
            out_dropout=mha_out_dropout,
            attention_out_dropout=attention_dropout,
        )
        self.ff = PositionWiseFun(dim_input, dim_ff_inner, mha_out_dropout)
        self.norm1 = torch.nn.LayerNorm(dim_input)
        self.norm2 = torch.nn.LayerNorm(dim_input)

    def forward(self, encode_input, pos_emb, u, v, mask=None, memory=None):
        src2 = self.norm1(encode_input)
        src2 = self.mha(src2, pos_emb, memory, u, v, mask=mask)
        src = self.gate1(encode_input, src2) if self.gating else encode_input + src2
        src2 = self.ff(self.norm2(src))
        src = self.gate2(src, src2) if self.gating else src + src2
        return src

class StableTransformerXL(torch.nn.Module):
    def __init__(
            self,
            d_input,
            n_layers,
            n_heads,
            d_head_inner,
            d_ff_inner,
            memory_shape,
            dropout=0.1,
            dropouta=0.0,
    ):
        """

        :param d_input: 编码向量的维度，待定
        :param n_layers: 编码器层的层数
        :param n_heads: 多头注意力机制的头数
        :param d_head_inner: Q-K-V 单头向量的维度
        :param d_ff_inner: MLP隐藏层输出维度
        :param dropout: 编码器输出dropout
        :param dropouta: 注意力机制输出dropout
        """
        super(StableTransformerXL, self).__init__()

        (
            self.n_layers,
            self.n_heads,
            self.d_input,
            self.d_head_inner,
            self.d_ff_inner,
        ) = (n_layers, n_heads, d_input, d_head_inner, d_ff_inner)

        self.pos_embs = PositionalEmbedding(d_input)
        self.drop = torch.nn.Dropout(dropout)
        self.layers = torch.nn.ModuleList(
            [
                StableTransformerEncoderLayerXL(
                    n_heads,
                    d_input,
                    dim_head_inner=d_head_inner,
                    dim_ff_inner=d_ff_inner,
                    mha_out_dropout=dropout,
                    attention_dropout=dropouta,
                )
                for _ in range(n_layers)
            ]
        )

        # u and v are global parameters: maybe changing these to per-head parameters might help performance?
        U_data = torch.Tensor(self.n_heads, self.d_head_inner).cuda().to(dtype=torch.float16)
        V_data = torch.Tensor(self.n_heads, self.d_head_inner).cuda().to(dtype=torch.float16)
        U_data = U_data.cpu()
        V_data = V_data.cpu()
        for i in range(self.n_heads):
            for j in range(self.d_head_inner):
                if np.isinf(U_data[i, j]) or np.isnan(U_data[i, j]):
                    U_data[i, j] = 0
                if np.isinf(V_data[i][j]) or np.isnan(V_data[i, j]):
                    V_data[i, j] = 0
        U_data = U_data.cuda().to(dtype=torch.float16)
        V_data = V_data.cuda().to(dtype=torch.float16)
        self.u, self.v = (
            torch.nn.Parameter(U_data),
            torch.nn.Parameter(V_data),
        )

        self.memory_shape = memory_shape

    def init_memory(self, device=torch.device("cpu")):
        # shape: the initial memory shape
        return [torch.zeros(self.memory_shape, dtype=torch.float).to(device=device, dtype=torch.float16) for _ in range(self.n_layers + 1)]

    def update_memory(self, previous_memory, hidden_states):
        """
        + Arguments
            - previous_memory: List[torch.FloatTensor],
            - hidden_states: List[torch.FloatTensor]
        """
        assert len(hidden_states) == len(previous_memory)
        mem_len, seq_len = previous_memory[0].size(0), hidden_states[0].size(0)
        # mem_len, seq_len = 3, hidden_states[0].size(0)
        # print(mem_len, seq_len)

        with torch.no_grad():
            new_memory = []
            end_idx = mem_len + seq_len
            beg_idx = max(0, end_idx - mem_len)
            for m, h in zip(previous_memory, hidden_states):
                cat = torch.cat([m, h], dim=0)
                new_memory.append(cat[beg_idx:end_idx].detach())
        return new_memory

    def forward(self, inputs, memory=None):
        if memory is None:
            memory = self.init_memory(inputs.device)
        assert len(memory) == len(self.layers) + 1

        cur_seq, bs = inputs.shape[:2]
        prev_seq = memory[0].size(0)

        dec_attn_mask = (
            torch.triu(
                torch.ones((cur_seq, cur_seq + prev_seq)),
                diagonal=1 + prev_seq,
            ).bool()[..., None].to(inputs.device)
        )

        pos_ips = torch.arange(cur_seq + prev_seq - 1, -1, -1.0, dtype=torch.float).to(
            inputs.device
        )
        pos_emb = self.drop(self.pos_embs(pos_ips))
        if self.d_input % 2 != 0:
            pos_emb = pos_emb[:, :, :-1]

        hidden_states = [inputs]
        layer_out = inputs
        for mem, layer in zip(memory, self.layers):
            layer_out = layer(
                layer_out,
                pos_emb.to(dtype=torch.float16),
                self.u,
                self.v,
                mask=dec_attn_mask,
                memory=mem,
            )
            hidden_states.append(layer_out)

        # Memory is treated as a const., don't propagate through it
        # new_memory = [[T x B x d_inner] x 4]
        memory = self.update_memory(memory, hidden_states)
        return {"logits": layer_out, "memory": memory}

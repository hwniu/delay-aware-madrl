import torch
import torch.nn as nn
import torch.nn.functional as func

from algorithm_agent.gate_transformer import StableTransformerXL

class Critic(nn.Module):
    def __init__(self, dim_aug_state, dim_action, n_agent, n_transformer_layers=1, n_attention_heads=2):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_aug_state = dim_aug_state
        self.dim_action = dim_action
        aug_state_dim = self.dim_aug_state * n_agent
        act_dim = self.dim_action * n_agent
        self.computing_memory = None
        self.transformer = StableTransformerXL(d_input=aug_state_dim, n_layers=n_transformer_layers, n_heads=n_attention_heads,
                                               d_head_inner=32, d_ff_inner=64, memory_shape=(6, 1, aug_state_dim))   # 6:max_delay_step
        # 设置神经网络层次
        self.dense_layer1 = nn.Linear(aug_state_dim, 1024)
        self.dense_layer2 = nn.Linear(1024 + act_dim, 512)
        self.dense_layer3 = nn.Linear(512, 300)
        self.dense_layer4 = nn.Linear(300, 1)

    # obs:batch_size * obs_dim
    def forward(self, aug_states, action):
        result = self.transformer(aug_states, self.computing_memory)
        result_state, self.computing_memory = result['logits'], result['memory']
        # Mean Pool layer
        transformer_state = torch.mean(result_state, dim=0)
        values = func.relu(self.dense_layer1(transformer_state))
        combine_values = torch.cat([values, action], dim=1)
        values = func.relu(self.dense_layer2(combine_values))
        values = func.relu(self.dense_layer3(values))
        values = func.relu(self.dense_layer4(values))

        return values

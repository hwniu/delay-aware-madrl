import torch
import torch.nn as nn
import torch.nn.functional as func
from algorithm_agent.gate_transformer import StableTransformerXL

class Actor(nn.Module):
    def __init__(self, aug_state_item_dim, action_dim, n_transformer_layers=1, n_attention_heads=2):
        # print('model.dim_action',dim_action)
        super(Actor, self).__init__()
        self.computing_memory = None
        self.transformer = StableTransformerXL(d_input=aug_state_item_dim, n_layers=n_transformer_layers, n_heads=n_attention_heads,
                                               d_head_inner=32, d_ff_inner=64, memory_shape=(6, 1, aug_state_item_dim))   # 6:max_delay_step
        # 设置网络层次层级
        self.dense_layer1 = nn.Linear(aug_state_item_dim, 500)
        self.dense_layer2 = nn.Linear(500, 128)
        self.dense_layer3 = nn.Linear(128, 128)
        self.dense_layer4 = nn.Linear(128, action_dim)

    def forward(self, aug_state):
        # 根据状态值设置动作
        result = self.transformer(aug_state, self.computing_memory)
        result_state, self.computing_memory = result['logits'], result['memory']
        # Mean Pool layer
        transformer_state = torch.mean(result_state, dim=0)
        action = func.relu(self.dense_layer1(transformer_state))
        action = func.relu(self.dense_layer2(action))
        action = func.relu(self.dense_layer3(action))
        action = func.relu(self.dense_layer4(action))
        return action

# test_actor = Actor(aug_state_item_dim=5, action_dim=2)
# test_aug_state = torch.tensor(data=[i for i in range(10)], dtype=torch.float32).reshape(2, 1, -1)
# test_Result = test_actor(test_aug_state)


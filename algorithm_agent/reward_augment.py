import math
import torch
from environment.env_config import UE_NUM_PER_CYBERTWIN
def reward_augment(state_buffer, reward_discount):
    if len(state_buffer) == 0:
        return torch.Tensor([-270*UE_NUM_PER_CYBERTWIN, -270*UE_NUM_PER_CYBERTWIN, -270*UE_NUM_PER_CYBERTWIN, -270*UE_NUM_PER_CYBERTWIN, -270*UE_NUM_PER_CYBERTWIN])
    else:
        generation_time_list = [item.generation_time for item in state_buffer]
        max_generation_time = max(generation_time_list)
        aug_reward = 0
        for state in state_buffer:
            aug_reward = aug_reward + state.reward * math.pow(reward_discount, max_generation_time - state.generation_time)
        return aug_reward


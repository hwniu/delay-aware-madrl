import torch
import numpy as np
from environment.env_config import MEC_NUM

def state_augment(delay_state, action_buffer, pre_last_action_delay, action_dim, state_dim):
    # state 为类, action_buffer 内部也为类
    delay_total = delay_state.delay + pre_last_action_delay
    if len(action_buffer) == 0:
        delay_total = 1
        action_none = torch.zeros(MEC_NUM, action_dim)
        for i in range(MEC_NUM):
            for j in range(action_dim):
                action_none[i, j] = np.random.uniform(0, 1)
        aug_state = torch.cat([delay_state.state, action_none], dim=1).reshape(MEC_NUM, delay_total, 1, -1)
    elif len(action_buffer) < delay_total:
        delay_total = len(action_buffer)
        aug_state = torch.zeros(MEC_NUM, delay_total, 1, state_dim + action_dim)
        for i in range(delay_total):
            aug_state[:, i, :, :] = torch.cat([delay_state.state, action_buffer[-(delay_total - i)].action], dim=1).reshape(MEC_NUM, 1, -1)
    elif len(action_buffer) >= delay_total:
        aug_state = torch.zeros(MEC_NUM, int(delay_total), 1, state_dim + action_dim)
        for i in range(int(delay_total)):
            aug_state[:, i, :, :] = torch.cat([delay_state.state, action_buffer[-(int(delay_total) - i)].action], dim=1).reshape(MEC_NUM, 1, -1)
    return aug_state

def combine_resource_task_state():
    pass


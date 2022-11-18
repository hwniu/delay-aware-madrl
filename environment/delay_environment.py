import numpy as np
import torch

from environment.undelay_environment import Env
from utils.logger import Logger
from environment.env_config import MEC_NUM
from environment.env_config import UE_NUM_PER_CYBERTWIN
from environment.env_config import AVERAGE_POISSON_DELAY

class DelayEnvironment(object):
    def __init__(self, n_agents=MEC_NUM, average_delay=AVERAGE_POISSON_DELAY):
        self.env = Env()
        self.now = 0  # 记录整个过程中的step

        self.n_agents = n_agents
        self.average_delay = average_delay
        self.last_execute_action_generation_time = 0
        self.sending_state_buffer = []
        self.state_buffer = []
        self.environment_state_instance = Delay_State(generation_time=0, state=None, delay=0, reward=0)

    def add_sending_state_buffer(self, state):
        self.sending_state_buffer.append(state)
        return

    def pop_sending_state_buffer(self, sent_state_buffer):
        # 根据时间戳可以筛选出 state_buffer，不进行该操作时，sending_state_buffer与state_buffer所存储内容无区别
        pass

    def set_time(self, time):
        self.now = time
        self.env.now = time

    def improve_execute_action(self, action_buffer):
        if action_buffer is None:
            generation_state_delay = np.random.poisson(self.average_delay, 1)[0]
            next_state = self.env.get_environment_state()
            reward = torch.Tensor([-270*UE_NUM_PER_CYBERTWIN, -270*UE_NUM_PER_CYBERTWIN, -270*UE_NUM_PER_CYBERTWIN, -270*UE_NUM_PER_CYBERTWIN, -270*UE_NUM_PER_CYBERTWIN])
            self.environment_state_instance = Delay_State(generation_time=self.now, state=next_state,
                                                          delay=generation_state_delay, reward=reward)
            return self.environment_state_instance
        else:
            reward_list = []
            for item in action_buffer:
                reward = self.env.set_action(action=item.action)
                reward_list.append(reward)
            reward_average = sum(reward_list) / len(reward_list)
            next_state = self.env.get_environment_state()
            generation_state_delay = np.random.poisson(self.average_delay, 1)[0]
            self.environment_state_instance = Delay_State(generation_time=self.now, state=next_state, delay=generation_state_delay, reward=reward_average)
            return self.environment_state_instance

    def reset(self):
        pass

class Delay_State(object):
    def __init__(self, generation_time, state, delay, reward):
        self.generation_time = generation_time
        self.state = state
        self.delay = delay
        self.reward = reward

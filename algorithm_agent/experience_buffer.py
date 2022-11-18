import random
from collections import namedtuple

from utils.logger import Logger

Experience = namedtuple('Experience', ('states', 'actions', 'next_states', 'rewards'))
# aug_state 由 real_state 增强得到,
Aug_experience = namedtuple('Aug_experience', ('now', 'real_state', 'real_state_time', 'aug_state', 'actions', 'next_aug_state', 'rewards', 'pre_state', 'pre_state_time'))
# real_time 指的是 增强型状态空间中真实状态的生成时间, 用于匹配其他经验中的pre_state
Action_buffer = namedtuple('Action_buffer', ('action', 'action_time'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Experience(*args)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        # print(len(self.memory),batch_size)
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class AugReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        # Todo
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Aug_experience(*args)
        self.position = int((self.position + 1) % self.capacity)

    def resample(self, batch_size):
        #  Timestamp Aligned
        return random.sample(self.memory, batch_size)

    def timestamp_align(self):
        for i in range(len(self.memory)):
            for j in range(len(self.memory)):
                if self.memory[i].real_state_time == self.memory[j].pre_state_time:
                    self.memory[j].pre_state = self.memory[i].real_state  # 不是实例化

class ActionBuffer(object):
    def __init__(self, max_length=1000):
        self.max_length = max_length
        self.buffer = []

    def push(self, action):
        if len(self.buffer) < self.max_length:
            self.buffer.append(action)
        else:
            Logger.logger.error("-------- The action buffer is full, please clean --------")
            raise

    # delete the useless historical actions
    def pop(self):
        pass
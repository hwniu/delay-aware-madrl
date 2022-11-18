from utils.logger import Logger
import numpy as np
from environment.env_config import ENERGY_COEFFICIENT
from environment.env_config import CLOUD_COMPUTING
from environment.env_config import CLOUD_CACHE
from environment.env_config import AVERAGE_WIERD_RATE
from environment.env_config import MAX_WIRED_RATE

class Cloud(object):
    def __init__(self, cpu_frequency=CLOUD_COMPUTING, cache=CLOUD_CACHE, average_wired_rate=AVERAGE_WIERD_RATE, max_wired_rate=MAX_WIRED_RATE, energy_coefficient=ENERGY_COEFFICIENT):
        # Logger.logger.info('Cloud Parameter: cpu_frequency=%d, cache=%d, average_wire_rate=%d, max_wired_rate=%d,', cpu_frequency, cache, average_wired_rate, max_wired_rate)
        self.energy_coefficient = energy_coefficient
        self.computing = cpu_frequency
        self.caching = cache
        self.ave_wired_rate = average_wired_rate
        self.max_wierd_rate = max_wired_rate

        # initially
        self.available_computing = cpu_frequency
        self.available_caching = cache

        self.execute_buffer = []

    def reset(self):
        self.available_computing = self.computing
        self.available_caching = self.caching
        self.execute_buffer = []

    def generate_wired_rate(self):
        rate = np.random.normal(loc=self.ave_wired_rate, scale=5)
        if rate > self.max_wierd_rate:
            rate = self.max_wierd_rate
        return rate

    def push_execute_buffer(self, sub_task_size, allocated_computing, start_time, communication_time):
        self.execute_buffer.append([sub_task_size, allocated_computing, start_time, communication_time])

    def pop_execute_buffer(self, time):
        pop_index = []
        for i in range(len(self.execute_buffer)):
            if time - self.execute_buffer[i][2] >= self.execute_buffer[i][3] + self.execute_buffer[i][0] / self.execute_buffer[i][1]:
                pop_index.append(i)
                self.available_computing = self.available_computing + self.execute_buffer[i][1]
                self.available_caching = self.available_caching + self.execute_buffer[i][0]
        execute_buffer_temp = []
        for i in range(len(self.execute_buffer)):
            if i not in pop_index:
                execute_buffer_temp.append(self.execute_buffer[i])
        self.execute_buffer = execute_buffer_temp

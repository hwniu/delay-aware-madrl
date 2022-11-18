import random

from utils.logger import Logger
from environment.env_config import UE_COMPUTING
from environment.env_config import UE_CACHE
from environment.env_config import UE_MAX_UPLINK_POWER
from environment.env_config import UE_D2D_POWER
from environment.env_config import UE_D2D_BANDWIDTH
from environment.env_config import MAX_DISTANCE_UE
from environment.env_config import ENERGY_COEFFICIENT

class Ue(object):
    def __init__(self, cpu_frequency=UE_COMPUTING, cache=UE_CACHE, ue_max_uplink_power=UE_MAX_UPLINK_POWER, d2d_power=UE_D2D_POWER, d2d_bandwidth=UE_D2D_BANDWIDTH):
        distance = random.randint(5, MAX_DISTANCE_UE)
        # Logger.logger.info('Ue Parameter: cpu_frequency=%d, cache=%d, ue_max_uplink_power=%lf, d2d_power=%lf, d2d_bandwidth=%d, to_mec_distance=%d', cpu_frequency, cache, ue_max_uplink_power, d2d_power, d2d_bandwidth, distance)
        self.cpu_frequency = cpu_frequency
        self.cache = cache
        self.ue_max_uplink_power = ue_max_uplink_power
        self.d2d_power = d2d_power
        self.d2d_bandwidth = d2d_bandwidth
        self.energy_coefficient = ENERGY_COEFFICIENT

        # initially
        self.available_computing = cpu_frequency
        self.available_caching = cache
        self.ue_uplink_power = 0
        self.uplink_bandwidth = 0

        self.to_mec_distance = distance
        self.execute_buffer = []
        self.receive_buffer = []

    def reset(self):
        self.available_computing = self.cpu_frequency
        self.available_caching = self.cache
        self.ue_uplink_power = 0
        self.uplink_bandwidth = 0
        self.execute_buffer = []

    def push_execute_buffer(self, sub_task_size, allocated_computing, start_time, communication_time):
        self.execute_buffer.append([sub_task_size, allocated_computing, start_time, communication_time])
        # 更新资源信息
        self.available_computing = self.available_computing - allocated_computing
        self.available_caching = self.available_caching - sub_task_size

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



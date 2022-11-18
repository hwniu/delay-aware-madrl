import numpy as np

from utils.logger import Logger
from environment.env_config import ENERGY_COEFFICIENT
from environment.env_config import MEC_ANTENNAS
from environment.env_config import MEC_COMPUTING
from environment.env_config import MEC_CACHE
from environment.env_config import MEC_BETWEEN_BANDWIDTH
from environment.env_config import MEC_BETWEEN_CARRIER_FREQUENCY
from environment.env_config import MEC_POWER
from environment.env_config import UPLINK_BANDWIDTH

class Mec(object):
    def __init__(self, cpu_frequency=MEC_COMPUTING, cache=MEC_CACHE, uplink_bandwidth=UPLINK_BANDWIDTH, mec_mec_bandwidth=MEC_BETWEEN_BANDWIDTH,
                 mec_mec_carrier_frequency=MEC_BETWEEN_CARRIER_FREQUENCY, mec_power=MEC_POWER, n_antennas=MEC_ANTENNAS, energy_coefficient=ENERGY_COEFFICIENT):

        # Logger.logger.info('Mec server Parameter: cpu_frequency=%d, cache=%d, uplink_bandwidth=%d, mec_mec_bandwidth=%d, mec_mec_carrier_frequency=%d, mec_power=%lf, n_antennas=%d', cpu_frequency, cache, uplink_bandwidth, mec_mec_bandwidth, mec_mec_carrier_frequency, mec_power, n_antennas)
        self.computing = cpu_frequency
        self.caching = cache
        self.uplink_bandwidth = uplink_bandwidth
        self.mec_mec_bandwidth = mec_mec_bandwidth
        self.mec_mec_carrier_frequency = mec_mec_carrier_frequency
        self.mec_power = mec_power
        self.n_antennas = n_antennas
        self.energy_coefficient = energy_coefficient

        # initially
        self.available_computing = cpu_frequency
        self.available_caching = cache
        self.available_uplink_bandwidth = uplink_bandwidth
        self.available_mec_mec_bandwidth = mec_mec_bandwidth

        self.execute_buffer = []
        self.uplink_bandwidth_buffer = []
        self.mec_mec_bandwidth_buffer = []

    def reset(self):
        self.available_computing = self.computing
        self.available_caching = self.caching
        self.available_uplink_bandwidth = self.uplink_bandwidth
        self.available_mec_mec_bandwidth = self.mec_mec_bandwidth
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

    def push_uplink_bandwidth(self, allocated_uplink_bandwidth, start_time, uplink_communication_time):
        self.uplink_bandwidth_buffer.append([allocated_uplink_bandwidth, start_time, uplink_communication_time])
        self.available_uplink_bandwidth = self.available_uplink_bandwidth - allocated_uplink_bandwidth

    def pop_uplink_bandwidth(self, time):
        pop_index = []
        for i in range(len(self.uplink_bandwidth_buffer)):
            if time - self.uplink_bandwidth_buffer[i][1] >= self.uplink_bandwidth_buffer[i][2]:
                pop_index.append(i)
                self.available_uplink_bandwidth = self.available_uplink_bandwidth + self.uplink_bandwidth_buffer[i][0]
        uplink_bandwidth_buffer_temp = []
        for i in range(len(self.uplink_bandwidth_buffer)):
            if i not in pop_index:
                uplink_bandwidth_buffer_temp.append(self.uplink_bandwidth_buffer[i])
        self.uplink_bandwidth_buffer = uplink_bandwidth_buffer_temp

    def push_mec_mec_bandwidth(self, allocated_uplink_bandwidth, start_time, mec_mec_communication_time):
        self.mec_mec_bandwidth_buffer.append([allocated_uplink_bandwidth, start_time, mec_mec_communication_time])
        self.available_mec_mec_bandwidth = self.available_mec_mec_bandwidth - allocated_uplink_bandwidth

    def pop_mec_mec_bandwidth_buffer(self, time):
        pop_index = []
        for i in range(len(self.mec_mec_bandwidth_buffer)):
            if time - self.mec_mec_bandwidth_buffer[i][1] >= self.mec_mec_bandwidth_buffer[i][2]:
                pop_index.append(i)
                self.available_mec_mec_bandwidth = self.available_mec_mec_bandwidth + self.mec_mec_bandwidth_buffer[i][0]
        mec_mec_bandwidth_buffer_temp = []
        for i in range(len(self.mec_mec_bandwidth_buffer)):
            if i not in pop_index:
                mec_mec_bandwidth_buffer_temp.append(self.mec_mec_bandwidth_buffer[i])
        self.mec_mec_bandwidth_buffer = mec_mec_bandwidth_buffer_temp
import numpy as np
import random
import math

import torch

from utils.logger import Logger
from environment.env_config import UE_NUM_PER_CYBERTWIN
from environment.env_config import CLOUD_NUM
from environment.env_config import MEC_NUM
from environment.env_config import MEC_ANTENNAS
from environment.env_config import MEC_ADJACENT_MATRIX
from environment.env_config import RICIAN_FACTOR
from environment.env_config import DISTANCE_BETWEEN_MEC
from environment.env_config import MEC_BETWEEN_CARRIER_FREQUENCY
from environment.env_config import MEC_ADJACENT_NUM
from environment.env_config import CYBERTWIN_PER_MEC
from environment.env_config import CYBERTWIN_NUM  # CYBERTWIN_NUM = MEC_NUM
from environment.env_config import NOISE_VARIANCE
from environment.env_config import MEC_CHANNEL_DISCOUNT_FACTOR
from environment.env_config import UE_CHANNEL_DISCOUNT_FACTOR
from environment.env_config import UE_COMPUTING
from environment.env_config import UE_CACHE
from environment.env_config import MEC_COMPUTING
from environment.env_config import MEC_CACHE
from environment.env_config import MEC_BETWEEN_BANDWIDTH
from environment.env_config import UPLINK_BANDWIDTH
from environment.env_config import CLOUD_CACHE
from environment.env_config import CLOUD_COMPUTING
from environment.env_config import MAX_TASK_SIZE
from environment.env_config import MIN_TASK_SIZE
from environment.env_config import MAX_MAX_TASK_TOLERANCE_DELAY
from environment.env_config import MIN_MAX_TASK_TOLERANCE_DELAY
from environment.env_config import TASK_NUM_PER_STEP
from environment.env_config import UE_MAX_UPLINK_POWER
from environment.env_config import UE_D2D_BANDWIDTH
from environment.env_config import UE_D2D_POWER
from environment.env_config import UE_D2D_CARRIER_FREQUENCY
from environment.env_config import TASK_COMPUTING_DENSITY
from environment.env_config import ENERGY_COEFFICIENT
from environment.task import Task

from environment.ue import Ue
from environment.mec import Mec
from environment.cloud import Cloud

c_speed = math.pow(10, 8)

class Env(object):
    def __init__(self):
        Logger.logger.info('Environment Parameter: cloud_num=%d, mec_num=%d, mec_ue_num=%d', CLOUD_NUM, MEC_NUM, UE_NUM_PER_CYBERTWIN)
        self.cloud = Cloud()
        self.n_mecs = [Mec() for i in range(MEC_NUM)]
        self.now = 0

        self.n_ues = [[] for i in range(MEC_NUM)]
        self.tasks = [[] for i in range(MEC_NUM)]
        for i in range(len(self.n_ues)):
            for j in range(UE_NUM_PER_CYBERTWIN):
                self.n_ues[i].append(Ue())
                self.tasks[i].append(Task(time=0))

        self.ue_distance_matrix = np.zeros((MEC_NUM, UE_NUM_PER_CYBERTWIN, UE_NUM_PER_CYBERTWIN))
        self.generate_ue_distance_matrix()

        self.ue_mec_channel_matrix = np.zeros(shape=(MEC_NUM, UE_NUM_PER_CYBERTWIN, 1, MEC_ANTENNAS), dtype=complex)
        self.ue_mec_channel_pre_snr_matrix = np.zeros(shape=(MEC_NUM, UE_NUM_PER_CYBERTWIN))
        self.generate_ue_channel_matrix()
        self.computing_ue_mec_pre_snr()

        self.mec_adjacency_matrix = np.array(MEC_ADJACENT_MATRIX)
        self.mec_adjacency_channel_matrix = np.zeros(shape=(MEC_NUM, MEC_NUM, MEC_ANTENNAS, MEC_ANTENNAS), dtype=complex)  # 该矩阵与相邻矩阵相似, MEC_NUM x MEC_NUM x MEC_ANTENNAS x MEC_ANTENNAS,
        self.mec_adjacency_channel_pre_snr_matrix = np.zeros(shape=(MEC_NUM, MEC_NUM))
        self.generate_mec_adjacency_channel_matrix()
        self.computing_mec_pre_snr()

        # obtain resources information: for a agent, MEC:2x2, cloud:1x2, nearby mec
        self.resources_information = np.zeros((MEC_NUM, UE_NUM_PER_CYBERTWIN + 2 + 1 + MEC_ADJACENT_NUM, 2))
        self.task_information = np.zeros((MEC_NUM, UE_NUM_PER_CYBERTWIN, 2))
        self.task_generation_matrix = np.zeros((MEC_NUM, UE_NUM_PER_CYBERTWIN))
        self.generate_task_matrix()

        self.environment_state = None

    def reset(self, time):
        # 重置环境中所有的元素
        self.cloud.reset()
        for i in range(len(self.n_mecs)):
            self.n_mecs[i].reset()
        for i in range(len(self.n_ues)):
            for j in range(UE_NUM_PER_CYBERTWIN):
                self.n_ues[i][j].reset()
                self.tasks[i][j].reset(time)
        self.get_environment_resources_state()
        self.generate_ue_channel_matrix()
        self.computing_ue_mec_pre_snr()
        self.generate_mec_adjacency_channel_matrix()
        self.computing_mec_pre_snr()
        self.generate_task_matrix()
        resource_state = self.resources_information.reshape((MEC_NUM, 1, -1))
        mec_channel_state = torch.zeros((MEC_NUM, MEC_ADJACENT_NUM))
        for i in range(MEC_NUM):
            temp = 0
            for j in range(MEC_NUM):
                if (self.mec_adjacency_channel_pre_snr_matrix * MEC_ADJACENT_MATRIX)[i][j] != 0:
                    mec_channel_state[i, temp] = (self.mec_adjacency_channel_pre_snr_matrix * MEC_ADJACENT_MATRIX)[i][j]
                    temp = temp + 1
        mec_channel_state = mec_channel_state.reshape((MEC_NUM, 1, -1))
        ue_channel_state = self.ue_mec_channel_pre_snr_matrix.reshape((MEC_NUM, 1, -1))
        task_state = self.get_environment_task_state().reshape((MEC_NUM, 1, -1))
        self.environment_state = torch.cat([torch.cat([torch.cat([torch.from_numpy(resource_state), mec_channel_state], dim=2), torch.from_numpy(ue_channel_state)], dim=2), torch.from_numpy(task_state)], dim=2)
        self.environment_state = self.environment_state.reshape((MEC_NUM, self.environment_state.shape[-1]))
        return self.environment_state

    def generate_task_matrix(self):
        for i in range(self.task_generation_matrix.shape[0]):
            index_list = np.random.randint(0, int(UE_NUM_PER_CYBERTWIN), int(TASK_NUM_PER_STEP))
            for j in range(self.task_generation_matrix.shape[1]):
                if j in index_list:
                    self.task_generation_matrix[0][j] == 1

    def generate_ue_distance_matrix(self):
        #  UE之间的理论最大距离应为直径，但是假设两者距离圆心（MEC）的距离视线确定，则两个UE之间的最大距离应为UE距离MEC的相加
        #  MEC_NUM x UE_NUM_PER_CYBERTWIN
        for i in range(self.ue_distance_matrix.shape[0]):
            for j in range(self.ue_distance_matrix.shape[1]):
                for p in range(self.ue_distance_matrix.shape[2]):
                    # 对角元素不做处理
                    if j < p:
                        self.ue_distance_matrix[i][j][p] = random.randint(1, self.n_ues[i][j].to_mec_distance + self.n_ues[i][p].to_mec_distance)
                        self.ue_distance_matrix[i][p][j] = self.ue_distance_matrix[i][j][p]

    def generate_ue_channel_matrix(self):
        for i in range(self.ue_mec_channel_matrix.shape[0]):
            for j in range(self.ue_mec_channel_matrix.shape[1]):
                for p in range(self.ue_mec_channel_matrix.shape[3]):
                    self.ue_mec_channel_matrix[i][j][0][p] = complex(random.gauss(0, math.sqrt(0.5)), random.gauss(0, math.sqrt(0.5)))  # Complex Gaussian distribution

    def computing_ue_mec_pre_snr(self):
        for i in range(self.ue_mec_channel_matrix.shape[0]):
            for j in range(self.ue_mec_channel_matrix.shape[1]):
                self.ue_mec_channel_pre_snr_matrix[i][j] = abs(np.dot(self.ue_mec_channel_matrix[i][j], self.ue_mec_channel_matrix[i][j].T.conjugate())[0][0]) / UE_CHANNEL_DISCOUNT_FACTOR

    def generate_mec_adjacency_channel_matrix(self):
        wavelength = 3 * math.pow(10, 8) / MEC_BETWEEN_CARRIER_FREQUENCY  # 单位m
        # 内含对称化操作，外层矩阵对称化, 实际在使用过程中只会同时应用上三角/下三角
        for i in range(self.mec_adjacency_channel_matrix.shape[0]):
            for j in range(self.mec_adjacency_channel_matrix.shape[1]):
                if i <= j:
                    if self.mec_adjacency_matrix[i][j] == 0:
                        continue
                    else:
                        for p in range(self.mec_adjacency_channel_matrix.shape[2]):
                            for k in range(self.mec_adjacency_channel_matrix.shape[3]):
                                los_channel_weight = (wavelength / (4 * math.pi * DISTANCE_BETWEEN_MEC)) * np.exp(complex(0, (-2 * math.pi * DISTANCE_BETWEEN_MEC)/wavelength))  # LOS channel
                                self.mec_adjacency_channel_matrix[i][j][p][k] = math.sqrt(RICIAN_FACTOR/(1 + RICIAN_FACTOR)) * complex(random.gauss(0, math.sqrt(0.5)), random.gauss(0, math.sqrt(0.5))) + math.sqrt(1/(1 + RICIAN_FACTOR)) * los_channel_weight  # NLOS + LOS channel
                        self.mec_adjacency_channel_matrix[j][i] = self.mec_adjacency_channel_matrix[i][j]

    def computing_mec_pre_snr(self):
        for i in range(self.mec_adjacency_channel_matrix.shape[0]):
            for j in range(self.mec_adjacency_channel_matrix.shape[1]):
                if self.mec_adjacency_matrix[i][j] == 1:
                    self.mec_adjacency_channel_pre_snr_matrix[i][j] = abs(np.linalg.det(np.dot(self.mec_adjacency_channel_matrix[i][j], self.mec_adjacency_channel_matrix[i][j].T.conjugate()))) / MEC_CHANNEL_DISCOUNT_FACTOR

    def get_environment_resources_state(self):
        for i in range(self.resources_information.shape[0]):
            near_index = []
            for mec_index in range(len(MEC_ADJACENT_MATRIX[i])):
                if MEC_ADJACENT_MATRIX[i][mec_index] == 1:
                    near_index.append(mec_index)
            for j in range(self.resources_information.shape[1]):
                if j < UE_NUM_PER_CYBERTWIN:
                    self.resources_information[i][j][0] = self.n_ues[i][j].available_computing / UE_COMPUTING
                    self.resources_information[i][j][1] = self.n_ues[i][j].available_caching / UE_CACHE
                elif j == UE_NUM_PER_CYBERTWIN:
                    self.resources_information[i][j][0] = self.n_mecs[i].available_computing / MEC_COMPUTING
                    self.resources_information[i][j][1] = self.n_mecs[i].available_caching / MEC_CACHE
                elif j == UE_NUM_PER_CYBERTWIN + 1:
                    self.resources_information[i][j][0] = self.n_mecs[i].available_mec_mec_bandwidth / MEC_BETWEEN_BANDWIDTH
                    self.resources_information[i][j][1] = self.n_mecs[i].available_uplink_bandwidth / UPLINK_BANDWIDTH
                elif j == UE_NUM_PER_CYBERTWIN + 2:
                    self.resources_information[i][j][0] = self.cloud.available_computing / CLOUD_COMPUTING
                    self.resources_information[i][j][1] = self.cloud.available_caching / CLOUD_CACHE
                elif j == UE_NUM_PER_CYBERTWIN + 2 + MEC_ADJACENT_NUM - 1:
                    self.resources_information[i][j][0] = self.n_mecs[near_index[0]].available_computing / MEC_COMPUTING
                    self.resources_information[i][j][1] = self.n_mecs[near_index[0]].available_caching / MEC_CACHE
                elif j == UE_NUM_PER_CYBERTWIN + 2 + MEC_ADJACENT_NUM:
                    self.resources_information[i][j][0] = self.n_mecs[near_index[MEC_ADJACENT_NUM - 1]].available_computing / MEC_COMPUTING
                    self.resources_information[i][j][1] = self.n_mecs[near_index[MEC_ADJACENT_NUM - 1]].available_caching / MEC_CACHE
        return self.resources_information

    def get_environment_task_state(self):
        for i in range(self.task_information.shape[0]):
            for j in range(self.task_information.shape[1]):
                self.task_information[i][j][0] = (self.tasks[i][j].task_size - MIN_TASK_SIZE) / (MAX_TASK_SIZE - MIN_TASK_SIZE)
                self.task_information[i][j][1] = (self.tasks[i][j].tolerance_delay - MIN_MAX_TASK_TOLERANCE_DELAY) / (MAX_MAX_TASK_TOLERANCE_DELAY - MIN_MAX_TASK_TOLERANCE_DELAY)
        return self.task_information

    def get_environment_state(self):
        self.get_environment_resources_state()
        self.generate_ue_channel_matrix()
        self.computing_ue_mec_pre_snr()
        self.generate_mec_adjacency_channel_matrix()
        self.computing_mec_pre_snr()
        self.generate_task_matrix()
        resource_state = self.resources_information.reshape((MEC_NUM, 1, -1))
        mec_channel_state = torch.zeros((MEC_NUM, MEC_ADJACENT_NUM))
        for i in range(MEC_NUM):
            temp = 0
            for j in range(MEC_NUM):
                if (self.mec_adjacency_channel_pre_snr_matrix * MEC_ADJACENT_MATRIX)[i][j] != 0:
                    mec_channel_state[i, temp] = (self.mec_adjacency_channel_pre_snr_matrix * MEC_ADJACENT_MATRIX)[i][j] / 10e150
                    temp = temp + 1
        mec_channel_state = mec_channel_state.reshape((MEC_NUM, 1, -1))
        ue_channel_state = self.ue_mec_channel_pre_snr_matrix.reshape((MEC_NUM, 1, -1))
        task_state = self.get_environment_task_state().reshape((MEC_NUM, 1, -1))
        self.environment_state = torch.cat([torch.cat([torch.cat([torch.from_numpy(resource_state), mec_channel_state], dim=2), torch.from_numpy(ue_channel_state)], dim=2), torch.from_numpy(task_state)], dim=2)
        self.environment_state = self.environment_state.reshape((MEC_NUM, self.environment_state.shape[-1]))
        return self.environment_state

    def set_action(self, action):
        rewards = torch.zeros(MEC_NUM, 1)
        self.cloud.pop_execute_buffer(self.now)
        for action_index in range(action.shape[0]):
            self.n_mecs[action_index].pop_execute_buffer(self.now)
            self.n_mecs[action_index].pop_mec_mec_bandwidth_buffer(self.now)
            self.n_mecs[action_index].pop_uplink_bandwidth(self.now)
        for action_index in range(action.shape[0]):
            # 子卸载失败 时间趋近于无穷大 +inf
            D2D_offloading_time = 0
            D2D_offloading_energy = 0
            local_offloading_time = 0
            local_offloading_energy = 0
            local_mec_offloading_time = 0
            local_mec_offloading_energy = 0
            near_mec_offloading_time_1 = 0
            near_mec_offloading_energy_1 = 0
            near_mec_offloading_time_2 = 0
            near_mec_offloading_energy_2 = 0
            cloud_offloading_time = 0
            cloud_offloading_energy = 0
            near_mec_bandwidth_1 = action[action_index, -2] * self.n_mecs[action_index].available_mec_mec_bandwidth / UE_NUM_PER_CYBERTWIN
            near_mec_bandwidth_2 = action[action_index, -1] * self.n_mecs[action_index].available_mec_mec_bandwidth / UE_NUM_PER_CYBERTWIN
            reward_sum = 0
            D2D_UE_ENERGY = 0
            local_ue_energy = 0
            local_mec_energy = 0
            near_mec_energy_1 = 0
            near_mec_energy_2 = 0
            cloud_energy = 0
            for j in range(UE_NUM_PER_CYBERTWIN):
                self.n_ues[action_index][j].pop_execute_buffer(self.now)
                local_task_size = self.tasks[action_index][j].task_size * action[action_index, j]
                local_computing_resource = action[action_index, j + UE_NUM_PER_CYBERTWIN] * self.n_ues[action_index][j].available_computing
                D2D_task_size = self.tasks[action_index][j].task_size * action[action_index, j + UE_NUM_PER_CYBERTWIN * 2]
                UE_uplink_power = action[action_index, j + UE_NUM_PER_CYBERTWIN * 4] * UE_MAX_UPLINK_POWER
                UE_uplink_bandwidth = action[action_index, j + UE_NUM_PER_CYBERTWIN * 5] * self.n_mecs[action_index].available_uplink_bandwidth
                local_mec_task_size = self.tasks[action_index][j].task_size * action[action_index, j + UE_NUM_PER_CYBERTWIN * 6]
                local_mec_computing = action[action_index, j + UE_NUM_PER_CYBERTWIN * 7] * self.n_mecs[action_index].available_computing
                near_mec_task_size_1 = self.tasks[action_index][j].task_size * action[action_index, j + UE_NUM_PER_CYBERTWIN * 8]
                near_mec_task_size_2 = self.tasks[action_index][j].task_size * action[action_index, j + UE_NUM_PER_CYBERTWIN * 10]
                cloud_task_size = self.tasks[action_index][j].task_size * action[action_index, j + UE_NUM_PER_CYBERTWIN * 12]
                cloud_computing = self.cloud.available_computing * action[action_index, j + UE_NUM_PER_CYBERTWIN * 13]
                self.n_ues[action_index][j].ue_uplink_power = UE_uplink_power
                self.n_ues[action_index][j].uplink_bandwidth = UE_uplink_bandwidth
                # 查询当前UE的D2D对象
                if action[action_index, j] + action[action_index, j + UE_NUM_PER_CYBERTWIN * 2] != 1:
                    if UE_uplink_bandwidth == 0 or UE_uplink_power == 0:
                        reward = -270
                        reward_sum = reward_sum + reward
                        continue
                if near_mec_task_size_1 != 0:
                    if near_mec_bandwidth_1 == 0:
                        reward = -270
                        reward_sum = reward_sum + reward
                        continue
                if near_mec_task_size_2 != 0:
                    if near_mec_bandwidth_2 == 0:
                        reward = -270
                        reward_sum = reward_sum + reward
                        continue
                if local_task_size != 0:
                    if local_computing_resource == 0:
                        reward = -270
                        reward_sum = reward_sum + reward
                        continue
                    self.n_ues[action_index][j].push_execute_buffer(local_task_size, local_computing_resource, self.now, 0)
                    local_offloading_time = local_task_size * TASK_COMPUTING_DENSITY / local_computing_resource
                    local_offloading_energy = ENERGY_COEFFICIENT * local_task_size * TASK_COMPUTING_DENSITY * local_computing_resource * local_computing_resource
                    local_ue_energy = local_ue_energy + local_offloading_energy.detach().numpy().tolist()
                if D2D_task_size != 0:
                    distance_list = self.ue_distance_matrix[action_index][j][:].tolist()
                    min_distance = [item for item in distance_list if item != 0]
                    min_distance = min(min_distance)
                    min_index = 0
                    for item_index in range(len(distance_list)):
                        if distance_list[item_index] == min_distance:
                            min_index = item_index
                            break
                    D2D_computing_resource = action[action_index, j + UE_NUM_PER_CYBERTWIN * 3] * self.n_ues[action_index][min_index].available_computing
                    if D2D_computing_resource == 0:
                        reward = -270
                        reward_sum = reward_sum + reward
                        continue
                    D2D_transmission_rate = UE_D2D_BANDWIDTH * math.log(1 + UE_D2D_POWER * math.pow(c_speed/(UE_D2D_CARRIER_FREQUENCY*4*math.pi*min_distance), 2) / (NOISE_VARIANCE*UE_D2D_BANDWIDTH), 2)
                    D2D_communication_delay = 1000 * D2D_task_size / D2D_transmission_rate  # 单位：ms
                    self.n_ues[action_index][min_index].push_execute_buffer(D2D_task_size, D2D_computing_resource, self.now, D2D_communication_delay)
                    D2D_offloading_time = D2D_communication_delay + D2D_task_size * TASK_COMPUTING_DENSITY / D2D_computing_resource
                    D2D_offloading_energy = self.n_ues[action_index][j].d2d_power * D2D_communication_delay + ENERGY_COEFFICIENT * D2D_task_size * TASK_COMPUTING_DENSITY * math.pow(D2D_computing_resource, 2)
                    local_ue_energy = local_ue_energy + (self.n_ues[action_index][j].d2d_power * D2D_communication_delay).detach().numpy().tolist()
                    D2D_UE_ENERGY = (ENERGY_COEFFICIENT * D2D_task_size * TASK_COMPUTING_DENSITY * math.pow(D2D_computing_resource, 2)).detach().numpy().tolist()
                    # 非本地计算任务上传时间计算
                # 需要判断带宽分配的合理性 大于零
                uplink_transmission_rate = self.n_ues[action_index][j].uplink_bandwidth * math.log(1 + (self.n_ues[action_index][j].ue_uplink_power * self.ue_mec_channel_pre_snr_matrix[action_index][j] * UE_CHANNEL_DISCOUNT_FACTOR) / (NOISE_VARIANCE*self.n_ues[action_index][j].uplink_bandwidth), 2)
                uplink_transmission_delay = (self.tasks[action_index][j].task_size - D2D_task_size) / uplink_transmission_rate
                self.n_mecs[action_index].push_uplink_bandwidth(self.n_ues[action_index][j].uplink_bandwidth, self.now, uplink_transmission_delay)
                if local_mec_task_size != 0:
                    if local_mec_computing == 0:
                        reward = -270
                        reward_sum = reward_sum + reward
                        continue
                    local_mec_offloading_time = uplink_transmission_delay + local_mec_task_size * TASK_COMPUTING_DENSITY / local_mec_computing
                    local_mec_offloading_energy = self.n_ues[action_index][j].ue_uplink_power * local_mec_task_size / uplink_transmission_rate + ENERGY_COEFFICIENT * local_mec_task_size * TASK_COMPUTING_DENSITY * math.pow(local_mec_computing, 2)
                    self.n_mecs[action_index].push_execute_buffer(local_mec_task_size, local_mec_computing, self.now, uplink_transmission_delay)
                    local_ue_energy = local_ue_energy + (self.n_ues[action_index][j].ue_uplink_power * local_mec_task_size / uplink_transmission_rate).detach().numpy().tolist()
                    local_mec_energy = local_mec_energy + (ENERGY_COEFFICIENT * local_mec_task_size * TASK_COMPUTING_DENSITY * math.pow(local_mec_computing, 2)).detach().numpy().tolist()
                if near_mec_task_size_1 != 0:
                    if near_mec_bandwidth_1 == 0:
                        reward = -270
                        reward_sum = reward_sum + reward
                        continue
                    for i in range(len(MEC_ADJACENT_MATRIX[action_index])):
                        if MEC_ADJACENT_MATRIX[action_index][i] != 0:
                            near_mec_computing_1 = self.n_mecs[i].available_computing * action[action_index, j + UE_NUM_PER_CYBERTWIN * 9]
                            if near_mec_computing_1 == 0:
                                break
                            near_mec_transmission_rate_1 = near_mec_bandwidth_1 * (math.log((self.n_mecs[action_index].mec_power * self.mec_adjacency_channel_pre_snr_matrix[action_index][i]) / (NOISE_VARIANCE*near_mec_bandwidth_1), 2) + math.log(MEC_CHANNEL_DISCOUNT_FACTOR, 2)) # 约等于
                            near_mec_communication_time_1 = uplink_transmission_delay + near_mec_task_size_1 / near_mec_transmission_rate_1
                            self.n_mecs[i].push_execute_buffer(near_mec_task_size_1, near_mec_computing_1,
                                                                          self.now, near_mec_communication_time_1)
                            near_mec_offloading_time_1 = near_mec_communication_time_1 + near_mec_task_size_1 * TASK_COMPUTING_DENSITY / near_mec_computing_1
                            near_mec_offloading_energy_1 = self.n_ues[action_index][
                                                               j].ue_uplink_power * near_mec_task_size_1 / uplink_transmission_rate + \
                                                           self.n_mecs[
                                                               action_index].mec_power * near_mec_task_size_1 / near_mec_transmission_rate_1 + ENERGY_COEFFICIENT * near_mec_task_size_1 * TASK_COMPUTING_DENSITY * math.pow(
                                near_mec_computing_1, 2)
                            local_ue_energy = local_ue_energy + (self.n_ues[action_index][
                                j].ue_uplink_power * near_mec_task_size_1 / uplink_transmission_rate).detach().numpy().tolist()
                            local_mec_energy = local_mec_energy + (self.n_mecs[
                                action_index].mec_power * near_mec_task_size_1 / near_mec_transmission_rate_1).detach().numpy().tolist()
                            near_mec_energy_1 = (ENERGY_COEFFICIENT * near_mec_task_size_1 * TASK_COMPUTING_DENSITY * math.pow(
                                near_mec_computing_1, 2)).detach().numpy().tolist()
                            self.n_mecs[action_index].push_mec_mec_bandwidth(near_mec_bandwidth_1, self.now, near_mec_communication_time_1)
                            break
                    try:
                        if near_mec_computing_1 == 0:
                            reward = -270
                            reward_sum = reward_sum + reward
                            continue
                    except:
                        pass
                if near_mec_task_size_2 != 0:
                    if near_mec_bandwidth_2 == 0:
                        reward = -270
                        reward_sum = reward_sum + reward
                        continue
                    mec_adjacent_index = 0
                    for i in range(len(MEC_ADJACENT_MATRIX[action_index])):
                        if MEC_ADJACENT_MATRIX[action_index][i] != 0:
                            if mec_adjacent_index == 1:
                                near_mec_computing_2 = self.n_mecs[i].available_computing * action[action_index, j + UE_NUM_PER_CYBERTWIN * 11]
                                if near_mec_computing_2 == 0:
                                    break
                                near_mec_transmission_rate_2 = near_mec_bandwidth_2 * (math.log((self.n_mecs[action_index].mec_power * self.mec_adjacency_channel_pre_snr_matrix[action_index][i]) / (NOISE_VARIANCE*near_mec_bandwidth_2), 2) + math.log(MEC_CHANNEL_DISCOUNT_FACTOR, 2)) # 约等于
                                near_mec_communication_time_2 = uplink_transmission_delay + near_mec_task_size_2 / near_mec_transmission_rate_2
                                self.n_mecs[i].push_execute_buffer(near_mec_task_size_2,
                                                                              near_mec_computing_2, self.now,
                                                                              near_mec_communication_time_2)
                                near_mec_offloading_time_2 = near_mec_communication_time_2 + near_mec_task_size_2 * TASK_COMPUTING_DENSITY / near_mec_computing_2
                                near_mec_offloading_energy_2 = self.n_ues[action_index][
                                                                   j].ue_uplink_power * near_mec_task_size_2 / uplink_transmission_rate + \
                                                               self.n_mecs[
                                                                   action_index].mec_power * near_mec_task_size_2 / near_mec_transmission_rate_2 + ENERGY_COEFFICIENT * near_mec_task_size_2 * TASK_COMPUTING_DENSITY * math.pow(
                                    near_mec_computing_2, 2)
                                local_ue_energy = local_ue_energy + (self.n_ues[action_index][
                                    j].ue_uplink_power * near_mec_task_size_2 / uplink_transmission_rate).detach().numpy().tolist()
                                local_mec_energy = local_mec_energy + (self.n_mecs[
                                    action_index].mec_power * near_mec_task_size_2 / near_mec_transmission_rate_2).detach().numpy().tolist()
                                near_mec_energy_2 = (ENERGY_COEFFICIENT * near_mec_task_size_2 * TASK_COMPUTING_DENSITY * math.pow(
                                    near_mec_computing_2, 2)).detach().numpy().tolist()
                                self.n_mecs[action_index].push_mec_mec_bandwidth(near_mec_bandwidth_2, self.now,
                                                                                 near_mec_communication_time_2)
                                break
                            mec_adjacent_index = mec_adjacent_index + 1
                    try:
                        if near_mec_computing_1 == 0:
                            reward = -270
                            reward_sum = reward_sum + reward
                            continue
                    except:
                        pass
                if cloud_task_size != 0:
                    if cloud_computing == 0:
                        reward = -270
                        reward_sum = reward_sum + reward
                        continue
                    cloud_wired_rate = self.cloud.generate_wired_rate()
                    cloud_communication_time = uplink_transmission_delay + cloud_task_size / cloud_wired_rate
                    cloud_offloading_time = uplink_transmission_delay + cloud_communication_time + cloud_task_size * TASK_COMPUTING_DENSITY / cloud_computing
                    cloud_offloading_energy = self.n_ues[action_index][j].ue_uplink_power * cloud_task_size / uplink_transmission_rate + ENERGY_COEFFICIENT * cloud_task_size * TASK_COMPUTING_DENSITY * math.pow(cloud_computing, 2)
                    self.cloud.push_execute_buffer(cloud_task_size, cloud_computing, self.now, cloud_communication_time)
                    local_ue_energy = local_ue_energy + (self.n_ues[action_index][j].ue_uplink_power * cloud_task_size / uplink_transmission_rate).detach().numpy().tolist()
                    cloud_energy = cloud_energy + (ENERGY_COEFFICIENT * cloud_task_size * TASK_COMPUTING_DENSITY * math.pow(cloud_computing, 2)).detach().numpy().tolist()

                UE_Consumption_Energy = local_ue_energy + D2D_UE_ENERGY
                MEC_Consumption_Energy = local_mec_energy + near_mec_energy_1 + near_mec_energy_2
                CLOUD_Consumption_Energy = cloud_energy
                system_loss = 1.2 * UE_Consumption_Energy + 1.0 * MEC_Consumption_Energy + 0.8 * CLOUD_Consumption_Energy + max(local_offloading_time, D2D_offloading_time, local_mec_offloading_time, near_mec_offloading_time_1, near_mec_offloading_time_2, cloud_offloading_time)
                if max(local_offloading_time, D2D_offloading_time, local_mec_offloading_time, near_mec_offloading_time_1, near_mec_offloading_time_2, cloud_offloading_time) > self.tasks[action_index][j].tolerance_delay:
                    reward = - math.pow(10, 7) * (system_loss / self.tasks[action_index][j].task_size) - 100
                else:
                    reward = - math.pow(10, 7) * (system_loss / self.tasks[action_index][j].task_size)
                reward_sum = reward_sum + reward
            rewards[action_index] = reward_sum
        return rewards

    def get_environment_state_dim(self):
        # UE(资源、信道、任务）MEC（资源，信道，带宽）cloud（资源）
        return UE_NUM_PER_CYBERTWIN * (2 + 1 + 2) + (MEC_ADJACENT_NUM + 1) * 2 + MEC_ADJACENT_NUM * 1 + 1 + 1 + CLOUD_NUM * 2

    def get_action_dim(self):
        # 由于UE相邻矩阵为稀疏矩阵，在这里考虑，UE只会向一个UE进行D2D通信卸载，进而降低动作矩阵的稀疏性
        # UE的任务分配（本地，D2D，MEC，nearby MECs，cloud）+ 上行功率，上行带宽 + 相邻MEC带宽
        return UE_NUM_PER_CYBERTWIN * (2 * (1 + 1 + 1 + 2 + 1)) + UE_NUM_PER_CYBERTWIN + UE_NUM_PER_CYBERTWIN + 2
# test_env = Env()
# test_state = test_env.reset(0)

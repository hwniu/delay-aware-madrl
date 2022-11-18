import torch
import os
import numpy as np
from torch.distributions import Normal

from environment.env_config import MEC_NUM
from utils.logger import Logger
from copy import deepcopy
from torch.optim import Adam
from algorithm_agent.critic import Critic
from algorithm_agent.actor import Actor
from algorithm_agent.experience_buffer import AugReplayMemory
from algorithm_agent.experience_buffer import Aug_experience
from algorithm_agent.experience_buffer import ActionBuffer
from environment.env_config import AVERAGE_POISSON_DELAY
from environment.env_config import STEP
from environment.env_config import UE_NUM_PER_CYBERTWIN

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class Delay_Aware_Matd3(object):
    def __init__(self, state_dim, action_dim, args, average_downlink_delay=AVERAGE_POISSON_DELAY, n_agents=MEC_NUM):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.aug_state_item_dim = state_dim + action_dim
        self.n_agents = n_agents
        self.average_downlink_delay = average_downlink_delay  # 实验过程中，智能体决策过程中，不能调用此值
        self.sending_action_buffer = []
        self.action_buffer = []
        self.last_predict_action_delay = 0  # 用lstm预测值进行替代

        self.now = 0  # 记录整个过程中的step

        self.actors = [Actor(self.aug_state_item_dim, self.action_dim).cuda().to(dtype=torch.float16) for _ in range(self.n_agents)]
        self.critics1 = [Critic(self.aug_state_item_dim, self.action_dim, self.n_agents).cuda().to(dtype=torch.float16) for _ in range(self.n_agents)]
        self.critics2 = [Critic(self.aug_state_item_dim, self.action_dim, self.n_agents).cuda().to(dtype=torch.float16) for _ in range(self.n_agents)]

        self.actors_target = deepcopy(self.actors)
        self.critics1_target = deepcopy(self.critics1)
        self.critics2_target = deepcopy(self.critics2)

        self.use_cuda = torch.cuda.is_available()
        # 相关参数
        self.args = args
        self.tau = self.args.tau
        self.actor_learn_rate = self.args.a_lr
        self.critic_learn_rate = self.args.c_lr
        self.gamma = self.args.gamma
        self.actor_target_update_interval = self.args.actor_target_update_interval
        self.experience_buffer = AugReplayMemory(capacity=self.args.memory_length)
        self.batch_size = self.args.batch_size
        self.episodes_before_train = self.args.episodes_before_train

        self.actors_optimizer = [Adam(item.parameters(), lr=self.actor_learn_rate) for item in self.actors]
        self.critics1_optimizer = [Adam(item.parameters(), lr=self.critic_learn_rate) for item in self.critics1]
        self.critics2_optimizer = [Adam(item.parameters(), lr=self.critic_learn_rate) for item in self.critics2]

        self.actors_loss = []
        self.critics1_loss = []
        self.critics2_loss = []

        self.target_noise_scale = self.args.target_noise_scale
        self.explore_noise_scale = self.args.explore_noise_scale

    def load_model(self):
        path_flag = True
        for idx in range(self.n_agents):
            path_flag = path_flag and os.path.exists("model/delay_aware_matd3/actor_[" + str(idx) + "]_" + ".pth") and os.path.exists("model/delay_aware_matd3/critic1_[" + str(idx) + "]_"+".pth") and os.path.exists("model/delay_aware_matd3/critic2_[" + str(idx) + "]_" + ".pth")
        if path_flag:
            Logger.logger.info("------------------ Load Delay-Aware-Matd3 Model successfully-------------------")
            for idx in range(self.n_agents):
                if self.use_cuda:
                    actor = torch.load("model/delay_aware_matd3/actor_[" + str(idx) + "]_" + ".pth")
                    critic1 = torch.load("model/delay_aware_matd3/critic1_[" + str(idx) + "]_"+".pth")
                    critic2 = torch.load("model/delay_aware_matd3/critic2_[" + str(idx) + "]_" + ".pth")
                else:
                    actor = torch.load("model/delay_aware_matd3/actor_[" + str(idx) + "]_" + ".pth", map_location=torch.device('cpu'))
                    critic1 = torch.load("model/delay_aware_matd3/critic1_[" + str(idx) + "]_"+".pth", map_location=torch.device('cpu'))
                    critic2 = torch.load("model/delay_aware_matd3/critic2_[" + str(idx) + "]_" + ".pth", map_location=torch.device('cpu'))
                self.actors[idx].load_state_dict(actor.state_dict())
                self.critics1[idx].load_state_dict(critic1.state_dict())
                self.critics2[idx].load_state_dict(critic2.state_dict())

        self.actors_target = deepcopy(self.actors)
        self.critics1_target = deepcopy(self.critics1)
        self.critics2_target = deepcopy(self.critics2)

    def save_model(self):
        if not os.path.exists("./model/delay_aware_matd3/"):
            os.makedirs("./model/delay_aware_matd3/")
        for i in range(self.n_agents):
            torch.save(self.actors[i], 'model/delay_aware_matd3/actor_[' + str(i) + ']' + '_' + '.pth')
            torch.save(self.critics1[i], 'model/delay_aware_matd3/critic1_[' + str(i) + ']' + '_' + '.pth')
            torch.save(self.critics2[i], 'model/delay_aware_matd3/critic2_[' + str(i) + ']' + '_' + '.pth')

    def update_parameter(self, train_episode):
        if train_episode <= self.episodes_before_train:
            return None, None

        # 根据是否GPU可用来选择tensor类型
        bool_tensor = torch.cuda.BoolTensor if self.use_cuda else torch.BoolTensor
        float_tensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        aug_transitions = self.experience_buffer.resample(batch_size=self.batch_size)
        aug_transitions_batch_data = Aug_experience(*zip(*aug_transitions))

        if train_episode % self.actor_target_update_interval == 0:
            self.critics1_loss.clear()
            self.critics2_loss.clear()
            self.actors_loss.clear()

        for agent_id in range(self.n_agents):
            augment_state_batch = torch.stack(aug_transitions_batch_data.aug_state).type(float_tensor)
            action_batch = torch.stack(aug_transitions_batch_data.actions).type(float_tensor)
            reward_batch = torch.stack(aug_transitions_batch_data.rewards).type(float_tensor)
            next_augment_state_batch = torch.stack([aug_state for aug_state in aug_transitions_batch_data.next_aug_state if aug_state is not None]).type(float_tensor)
            pre_state_batch = torch.stack(aug_transitions_batch_data.pre_state).type(float_tensor)

            self.actors_optimizer[agent_id].zero_grad()
            self.critics1_optimizer[agent_id].zero_grad()
            self.critics2_optimizer[agent_id].zero_grad()
            self.actors[agent_id].zero_grad()
            self.critics1[agent_id].zero_grad()
            self.critics2[agent_id].zero_grad()

            current_value1 = self.critics1[agent_id](augment_state_batch, action_batch)  # TODO
            current_value2 = self.critics2[agent_id](augment_state_batch, action_batch)  # TODO
            real_value1 = self.critics1[agent_id](pre_state_batch, action_batch)
            real_value2 = self.critics2[agent_id](pre_state_batch, action_batch)
            next_actions = []
            for i in range(self.n_agents):
                action_item = self.actors_target[i](next_augment_state_batch[:, i, :].cuda().to(dtype=torch.float16))
                normal = Normal(0, 1)
                target_noise_scale = 2 * self.target_noise_scale
                normal_noise = normal.sample(action_item.shape) * self.target_noise_scale
                normal_noise = torch.clamp(normal_noise, -target_noise_scale, target_noise_scale)
                action_item = action_item + normal_noise.to(device)
                action_item = torch.clamp(action_item, 0, 1)
                uplink_bandwidth_proportion_sum = sum(action_item[:, 0, UE_NUM_PER_CYBERTWIN * 5:UE_NUM_PER_CYBERTWIN * 6])
                for ue_index in range(UE_NUM_PER_CYBERTWIN):
                    sum_temp = action_item[:, 0, ue_index] + action_item[:, 0, ue_index + UE_NUM_PER_CYBERTWIN * 2] + \
                               action_item[:, 0, ue_index + UE_NUM_PER_CYBERTWIN * 6] + action_item[:, 0, ue_index + UE_NUM_PER_CYBERTWIN * 8] + action_item[:, 0, ue_index + UE_NUM_PER_CYBERTWIN * 10] + action_item[:, 0, ue_index + UE_NUM_PER_CYBERTWIN * 12]
                    action_item[:, 0, ue_index] = action_item[:, 0, ue_index] / sum_temp
                    action_item[:, 0, ue_index + UE_NUM_PER_CYBERTWIN * 2] = action_item[:, 0, ue_index + UE_NUM_PER_CYBERTWIN * 2] / sum_temp
                    action_item[:, 0, ue_index + UE_NUM_PER_CYBERTWIN * 6] = action_item[:, 0, ue_index + UE_NUM_PER_CYBERTWIN * 6] / sum_temp
                    action_item[:, 0, ue_index + UE_NUM_PER_CYBERTWIN * 8] = action_item[:, 0, ue_index + UE_NUM_PER_CYBERTWIN * 8] / sum_temp
                    action_item[:, 0, ue_index + UE_NUM_PER_CYBERTWIN * 10] = action_item[:, 0, ue_index + UE_NUM_PER_CYBERTWIN * 10] / sum_temp
                    action_item[:, 0, ue_index + UE_NUM_PER_CYBERTWIN * 12] = action_item[:, 0, ue_index + UE_NUM_PER_CYBERTWIN * 12] / sum_temp
                    action_item[:, 0, ue_index + UE_NUM_PER_CYBERTWIN * 5] = action_item[:, 0, ue_index + UE_NUM_PER_CYBERTWIN * 5] / uplink_bandwidth_proportion_sum
                # 相邻MEC带宽
                adjacent_proportion_sum = (action_item[:, 0, -1] + action_item[:, 0, -2])
                if adjacent_proportion_sum == 0:
                    action_item[:, 0, -2] = np.random.uniform(0, 1)
                    action_item[:, 0, -1] = 1 - action_item[:, 0, -2]
                else:
                    action_item[:, 0, -2] = action_item[:, 0, -2] / adjacent_proportion_sum
                    action_item[:, 0, -1] = action_item[:, 0, -1] / adjacent_proportion_sum
                next_actions.append(action_item)
            next_actions = torch.stack(next_actions)
            target_value1 = self.critics1_target[agent_id](next_augment_state_batch, next_actions)
            target_value2 = self.critics2_target[agent_id](next_augment_state_batch, next_actions)
            target_value = torch.min(target_value1, target_value2)
            target_value = target_value * self.gamma + reward_batch[:, agent_id]
            loss_value1 = torch.nn.MSELoss()[current_value1, target_value.detach()] + torch.nn.MSELoss()[current_value1, real_value1]
            loss_value2 = torch.nn.MSELoss()[current_value2, target_value.detach()] + torch.nn.MSELoss()[current_value2, real_value2]
            loss_value1.backward()
            loss_value2.backward()
            torch.nn.utils.clip_grad_norm_(self.critics1[agent_id].parameters(), 1)
            torch.nn.utils.clip_grad_norm_(self.critics2[agent_id].parameters(), 1)
            self.critics1_optimizer[agent_id].step()
            self.critics2_optimizer[agent_id].step()
            if train_episode % self.actor_target_update_interval == 0:
                self.actors_optimizer[agent_id].zero_grad()
                self.critics1_optimizer[agent_id].zero_grad()
                self.actors[agent_id].zero_grad()
                self.critics1[agent_id].zero_grad()
                aug_state_i = augment_state_batch[:, agent_id]
                action_i = self.actors[agent_id](aug_state_i)
                ac = action_batch.clone()
                ac[:, agent_id] = action_i
                actor_loss = -self.critics1[agent_id]()  # TODO
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), 1)
                torch.nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), 1)
                self.critics1_loss.append(loss_value1)
                self.critics2_loss.append(loss_value2)
                self.actors_loss.append(actor_loss)
                soft_update_parameter(self.critics1_target[agent_id], self.critics1[agent_id], self.tau)
                soft_update_parameter(self.critics2_target[agent_id], self.critics2[agent_id], self.tau)
                soft_update_parameter(self.actors_target[agent_id], self.actors[agent_id], self.tau)

    def generate_action(self, aug_state, noisy=True):
        actions = torch.zeros(self.n_agents, self.action_dim)
        for i in range(self.n_agents):
            aug_state_agent = aug_state[i].detach()
            action_item = self.actors[i](aug_state_agent.cuda().to(dtype=torch.float16))
            if noisy:
                normal = Normal(0, 1)
                explore_noise_clip = 2 * self.explore_noise_scale
                normal_noise = normal.sample(action_item.shape) * self.explore_noise_scale
                normal_noise = torch.clamp(normal_noise, -explore_noise_clip, explore_noise_clip)
                action_item = action_item + normal_noise.to(device)
            action_item = torch.clamp(action_item, 0, 1)
            # action mask, make it meet proportion relationship
            uplink_bandwidth_proportion_sum = sum(action_item[0][UE_NUM_PER_CYBERTWIN * 5:UE_NUM_PER_CYBERTWIN * 6])
            for ue_index in range(UE_NUM_PER_CYBERTWIN):
                # task proportion
                sum_temp = action_item[0, ue_index] + action_item[0, ue_index + UE_NUM_PER_CYBERTWIN * 2] + action_item[0, ue_index + UE_NUM_PER_CYBERTWIN * 6] + action_item[0, ue_index + UE_NUM_PER_CYBERTWIN * 8] + action_item[0, ue_index + UE_NUM_PER_CYBERTWIN * 10] + action_item[0, ue_index + UE_NUM_PER_CYBERTWIN * 12]
                action_item[0, ue_index] = action_item[0][ue_index] / sum_temp
                action_item[0, ue_index + UE_NUM_PER_CYBERTWIN * 2] = action_item[0, ue_index + UE_NUM_PER_CYBERTWIN * 2] / sum_temp
                action_item[0, ue_index + UE_NUM_PER_CYBERTWIN * 6] = action_item[0, ue_index + UE_NUM_PER_CYBERTWIN * 6] / sum_temp
                action_item[0, ue_index + UE_NUM_PER_CYBERTWIN * 8] = action_item[0, ue_index + UE_NUM_PER_CYBERTWIN * 8] / sum_temp
                action_item[0, ue_index + UE_NUM_PER_CYBERTWIN * 10] = action_item[0, ue_index + UE_NUM_PER_CYBERTWIN * 10] / sum_temp
                action_item[0, ue_index + UE_NUM_PER_CYBERTWIN * 12] = action_item[0, ue_index + UE_NUM_PER_CYBERTWIN * 12] / sum_temp
                # uplink bandwidth
                action_item[0, ue_index + UE_NUM_PER_CYBERTWIN * 5] = action_item[0, ue_index + UE_NUM_PER_CYBERTWIN * 5] / uplink_bandwidth_proportion_sum
            # 相邻MEC带宽
            adjacent_proportion_sum = (action_item[0, -1] + action_item[0, -2])
            if adjacent_proportion_sum == 0:
                action_item[0, -2] = np.random.uniform(0, 1)
                action_item[0, -1] = 1 - action_item[0, -2]
            else:
                action_item[0, -2] = action_item[0, -2] / adjacent_proportion_sum
                action_item[0, -1] = action_item[0, -1] / adjacent_proportion_sum
            actions[i, :] = action_item
        generation_action_delay = np.random.poisson(self.average_downlink_delay, 1)[0]
        return Delay_Action(generation_time=self.now, action=actions, delay=generation_action_delay)

    def add_action_buffer(self, action):
        self.action_buffer.push(action)

    def add_sending_action_buffer(self, action):
        self.sending_action_buffer.append(action)

    def pop_sending_action_buffer(self, actions):
        # 动作与时间具有一一对应关系
        pop_index = []
        for sent_action in actions:
            for i in range(len(self.sending_action_buffer)):
                if sent_action.generation_time == self.sending_action_buffer[i].generation_time:
                    pop_index.append(i)
                    break
        self.sending_action_buffer = [item for i, item in enumerate(self.sending_action_buffer) if i not in pop_index]

    def set_time(self, time):
        self.now = time

    def get_last_three_action_float_delay(self):
        delay_data = []
        if len(self.action_buffer) != 0:
            for action_item in reversed(self.action_buffer):
                if action_item.known_delay == True:
                    delay_data.append(action_item.float_delay)
                if len(delay_data) == 3:
                    return delay_data
        return None

class Delay_Action(object):
    def __init__(self, generation_time, action, delay):
        self.generation_time = generation_time
        self.action = action
        self.delay = delay
        self.known_delay = False
        self.float_delay = np.random.randint((self.delay - 1) * STEP, self.delay * STEP)

    def set_delay_know(self):
        self.known_delay = True

def soft_update_parameter(target, source, t):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)

def hard_update_parameter(target, source):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)
